from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
import torch
import time

def evaluate_model(model, data_loader, model_type='cnn', feature_model=None, device='cpu'):
    """Evaluate model performance and return metrics."""
    start_time = time.time()
    model.eval() if model_type == 'cnn' else None
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            y_true.extend(labels.cpu().numpy())

            if model_type == 'cnn':
                outputs = model(images)
                y_pred.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            else:
                features = feature_model(images, return_features=True).cpu().numpy()
                if model_type == 'svm':
                    probs = model.predict_proba(features)
                else:  # xgboost
                    probs = model.predict_proba(features)
                y_pred.extend(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_labels = np.argmax(y_pred, axis=1)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_labels),
        'f1_score': f1_score(y_true, y_pred_labels, average='weighted'),
        'precision': precision_score(y_true, y_pred_labels, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred_labels, average='weighted'),
        'roc_auc': roc_auc_score(y_true, y_pred, multi_class='ovr'),
        'training_time': time.time() - start_time
    }
    return metrics