"""This module defines the frames of the neural networks and training/testing functions"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score


class MultiLayerPerceptron(nn.Module):
    def __init__(self, num_features, activation_fn=F.relu):
        super().__init__()
        self.activation_fn = activation_fn

        self.fc1 = nn.Linear(num_features, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.6)

        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.dropout1(self.bn1(self.activation_fn(self.fc1(x))))
        x = self.dropout2(self.bn2(self.activation_fn(self.fc2(x))))
        x = self.activation_fn(self.fc3(x))
        return self.fc4(x).squeeze()


class Bagging(nn.Module):
    def __init__(self, num_features, n_estimators=50):
        super().__init__()
        self.n_estimators = min(n_estimators, 10)

        def make_base_model():
            return nn.Sequential(
                nn.Linear(num_features, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        self.models = nn.ModuleList([make_base_model() for _ in range(self.n_estimators)])

    def forward(self, x):
        preds = [model(x).squeeze() for model in self.models]
        preds = torch.stack(preds, dim=0)
        return preds.mean(dim=0)


class TabularCNN(nn.Module):
    def __init__(self, num_features, activation_fn=F.relu):
        super().__init__()
        self.activation_fn = activation_fn
        self.reshape = lambda x: x.unsqueeze(-1)  # reshape (B, F) to (B, F, 1)

        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv1d(32, 16, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(0.6)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, F)
        x = self.dropout1(self.bn1(self.activation_fn(self.conv1(x))))
        x = self.dropout2(self.bn2(self.activation_fn(self.conv2(x))))
        x = self.global_pool(x).squeeze(-1)
        x = self.activation_fn(self.fc1(x))
        return self.fc2(x).squeeze()


# Early stop function blueprint by Michael Mior (SO)
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# Training function with defined loss and optimiser objects
def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=20, early_stop=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        val_loss_total = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                preds = model(xb)
                loss = loss_fn(preds, yb)

                # Convert logits to probabilities for metrics
                val_loss_total += loss.item()
                val_probs = torch.sigmoid(preds)
                # Move to CPU before converting to numpy
                val_preds.extend(val_probs.cpu().numpy())
                val_targets.extend(yb.cpu().numpy())

        val_auc = roc_auc_score(val_targets, val_preds)
        val_loss_avg = val_loss_total / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {running_loss:.4f},\
            Val Loss = {val_loss_avg:.4f}, Val AUC = {val_auc:.4f}")

        if early_stop and early_stop.early_stop(val_loss_avg):
            print(f"Stopped early at epoch {epoch+1}")
            break


def eval_model(model, test_loader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            logits = model(xb)
            probs = torch.sigmoid(logits)
            # Move to CPU before converting to numpy
            preds.extend(probs.cpu().numpy())
            targets.extend(yb.cpu().numpy())

    preds_bin = [1 if p >= 0.5 else 0 for p in preds]
    print("Test Accuracy:", accuracy_score(targets, preds_bin))
    print("Precision:", precision_score(targets, preds_bin))
    print("Recall:", recall_score(targets, preds_bin))
    print("ROC AUC:", roc_auc_score(targets, preds))
