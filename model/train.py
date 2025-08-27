from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from model.model import CNN, train_svm, train_xgboost
from evaluate.evaluate import evaluate_model

def get_features(model, data_loader, device='cpu'):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in data_loader:
            images = images.to(device)
            feats = model(images, return_features=True).cpu().numpy()
            features.append(feats)
            labels.append(lbls.cpu().numpy())
    return np.concatenate(features), np.concatenate(labels)

def train_and_evaluate(train_loader, val_loader, num_classes, output_dir: Path = Path('results'), device='cpu'):
    drive.mount('/content/drive')
    output_dir = Path('/content/drive/My Drive/cnn/results')

    output_dir.mkdir(parents=True, exist_ok=True)
    activations = ['relu', 'leaky_relu', 'elu']
    filter_configs = [[16, 32, 64], [32, 64, 128]]
    stride_configs = [[1, 1, 1], [2, 1, 1]]
    results = []

    for filters in filter_configs:
        for strides in stride_configs:
            for activation in activations:
                config_str = f"{filters}_{strides}_{activation}"
                print(f"\nTraining with filters={filters}, strides={strides}, activation={activation}...")

                model = CNN(num_classes=num_classes, activation=activation, filters=filters, strides=strides).to(device)
                class_counts = [len(list((Path('/content/drive/My Drive/cnn/data/train') / cls).glob('*.[jp][pn]g'))) for cls in sorted(Path('/content/drive/My Drive/cnn/data/train').iterdir()) if cls.is_dir()]
                class_weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float, device=device)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                optimizer = optim.Adam(model.parameters(), lr=0.0001)

                epochs = 30
                patience = 5
                best_val_loss = float('inf')
                patience_counter = 0
                history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
                y_true_all, y_score_all = [], []

                for epoch in range(epochs):
                    model.train()
                    train_loss = 0
                    train_correct = 0
                    train_total = 0
                    for images, labels in train_loader:
                        images, labels = images.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        train_total += labels.size(0)
                        train_correct += (predicted == labels).sum().item()

                    train_loss /= len(train_loader)
                    train_acc = train_correct / train_total

                    model.eval()
                    val_loss = 0
                    y_true, y_score = [], []
                    with torch.no_grad():
                        for images, labels in val_loader:
                            images, labels = images.to(device), labels.to(device)
                            outputs = model(images)
                            val_loss += criterion(outputs, labels).item()
                            y_true.extend(labels.cpu().numpy())
                            y_score.extend(torch.softmax(outputs, dim=1).cpu().numpy())
                    val_loss /= len(val_loader)

                    val_metrics = evaluate_model(model, val_loader, model_type='cnn', device=device)
                    history['train_loss'].append(train_loss)
                    history['val_loss'].append(val_loss)
                    history['train_acc'].append(train_acc)
                    history['val_acc'].append(val_metrics['accuracy'])

                    # Compute ROC AUC for multiclass
                    y_true_all.extend(y_true)
                    y_score_all.extend(y_score)
                    y_true_np = np.array(y_true_all)
                    y_score_np = np.array(y_score_all)
                    # Per-class and micro-average AUC
                    roc_auc = roc_auc_score(y_true_np, y_score_np, multi_class='ovr', average=None)  # Per-class AUCs
                    roc_auc_micro = roc_auc_score(y_true_np, y_score_np, multi_class='ovr', average='micro')  # Micro-average AUC

                    print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Micro ROC-AUC={roc_auc_micro:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        torch.save(model.state_dict(), output_dir / f'best_model_{config_str}.pth')
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Early stopping at epoch {epoch+1}")
                            break

                model.load_state_dict(torch.load(output_dir / f'best_model_{config_str}.pth'))

                # Final ROC plot (approximated based on micro-average AUC)
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.plot(history['train_loss'], label='Train Loss')
                plt.plot(history['val_loss'], label='Val Loss')
                plt.title(f'{config_str} - Loss')
                plt.legend()
                plt.subplot(1, 3, 2)
                plt.plot(history['train_acc'], label='Train Acc')
                plt.plot(history['val_acc'], label='Val Acc')
                plt.title(f'{config_str} - Accuracy')
                plt.legend()
                plt.subplot(1, 3, 3)
                # Approximate ROC curve based on micro-average AUC
                fpr = np.linspace(0, 1, 100)
                tpr = np.interp(fpr, [0, roc_auc_micro, 1], [0, roc_auc_micro, 1])
                plt.plot(fpr, tpr, label=f'Micro-average ROC (AUC = {roc_auc_micro:.4f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{config_str} - ROC Curve')
                plt.legend(loc="lower right")
                plt.savefig(output_dir / f'training_history_{config_str}.png')
                plt.close()

                X_train, y_train = get_features(model, train_loader, device)
                X_val, y_val = get_features(model, val_loader, device)

                svm_model = train_svm(X_train, y_train, X_val, y_val)
                xgb_model = train_xgboost(X_train, y_train, X_val, y_val)

                cnn_metrics = evaluate_model(model, val_loader, model_type='cnn', device=device)
                svm_metrics = evaluate_model(svm_model, val_loader, model_type='svm', feature_model=model, device=device)
                xgb_metrics = evaluate_model(xgb_model, val_loader, model_type='xgboost', feature_model=model, device=device)

                results.append({
                    'Config': config_str,
                    'Activation': activation,
                    'Model': 'CNN + Softmax',
                    **cnn_metrics,
                    'roc_auc': roc_auc_micro
                })
                results.append({
                    'Config': config_str,
                    'Activation': activation,
                    'Model': 'CNN + SVM',
                    **svm_metrics,
                    'roc_auc': roc_auc_micro
                })
                results.append({
                    'Config': config_str,
                    'Activation': activation,
                    'Model': 'CNN + XGBoost',
                    **xgb_metrics,
                    'roc_auc': roc_auc_micro
                })

                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.plot(history['train_loss'], label='Train Loss')
                plt.plot(history['val_loss'], label='Val Loss')
                plt.title(f'{config_str} - Loss')
                plt.legend()
                plt.subplot(1, 2, 2)
                plt.plot(history['train_acc'], label='Train Acc')
                plt.plot(history['val_acc'], label='Val Acc')
                plt.title(f'{config_str} - Accuracy')
                plt.legend()
                plt.savefig(output_dir / f'training_history_{config_str}_summary.png')
                plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'results.csv', index=False)
    print(f"Results saved to {output_dir}")

    return results_df