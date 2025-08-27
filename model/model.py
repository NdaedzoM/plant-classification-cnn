import torch
import torch.nn as nn
from xgboost import XGBClassifier
from sklearn.svm import SVC

class CNN(nn.Module):
    def __init__(self, num_classes=12, activation='relu', filters=[16, 32, 64], strides=[1, 1, 1], input_size=(32, 32)):
        super(CNN, self).__init__()
        self.filters = filters
        self.strides = strides
        self.input_size = input_size

        # Compute output size layer by layer
        h, w = input_size
        for i, stride in enumerate(strides):
            # Conv output: (h + 2*padding - kernel_size) // stride + 1
            h = (h + 2 * 1 - 3) // stride + 1
            w = (w + 2 * 1 - 3) // stride + 1
            if i < len(strides) - 1:  # Pooling after each conv except last
                h = h // 2
                w = w // 2
        # After final conv, apply last pooling
        h = h // 2
        w = w // 2
        self.feature_map_size = (h, w)
        linear_input_size = filters[-1] * h * w

        print(f"Config: filters={filters}, strides={strides}, feature_map_size={self.feature_map_size}, linear_input_size={linear_input_size}")

        self.features = nn.Sequential(
            nn.Conv2d(3, filters[0], kernel_size=3, stride=strides[0], padding=1),
            nn.BatchNorm2d(filters[0]),
            self._get_activation(activation),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(filters[0], filters[1], kernel_size=3, stride=strides[1], padding=1),
            nn.BatchNorm2d(filters[1]),
            self._get_activation(activation),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(filters[1], filters[2], kernel_size=3, stride=strides[2], padding=1),
            nn.BatchNorm2d(filters[2]),
            self._get_activation(activation),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_input_size, num_classes)
        )

    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif activation == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x, return_features=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = x if return_features else None
        x = self.classifier(x)
        return x if not return_features else features

def train_svm(X_train, y_train, X_val, y_val):
    from sklearn.svm import SVC
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train, y_train)
    return svm

def train_xgboost(X_train, y_train, X_val, y_val):
    from xgboost import XGBClassifier
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb.fit(X_train, y_train)
    return xgb