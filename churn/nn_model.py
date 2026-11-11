"""A PyTorch neural network wrapped as a scikit-learn classifier.

Wrapping it this way lets the network sit inside the same Pipeline,
GridSearchCV and cross-validation code as the classical models.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn


class ChurnMLP(nn.Module):
    """Feed-forward network: [Linear -> BatchNorm -> ReLU -> Dropout] x N -> Linear."""

    def __init__(self, n_features: int, hidden_dims: tuple[int, ...] = (64, 32), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))  # single logit for binary classification
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """scikit-learn compatible wrapper around :class:`ChurnMLP`.

    Uses BCEWithLogitsLoss, the Adam optimiser and early stopping on a
    small validation split taken from the training data.
    """

    def __init__(self, hidden_dims=(64, 32), dropout=0.2, lr=1e-3, weight_decay=1e-4,
                 epochs=100, batch_size=128, patience=10, random_state=42):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        # hold out 10% of the training rows for early stopping
        idx = rng.permutation(len(X))
        n_val = max(1, int(0.1 * len(X)))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        X_tr, y_tr = torch.from_numpy(X[tr_idx]), torch.from_numpy(y[tr_idx])
        X_val, y_val = torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx])

        self.model_ = ChurnMLP(X.shape[1], tuple(self.hidden_dims), self.dropout)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        best_loss, best_state, bad_epochs = float("inf"), None, 0
        self.history_ = []
        for epoch in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(len(X_tr))
            for start in range(0, len(X_tr), self.batch_size):
                batch = perm[start:start + self.batch_size]
                if len(batch) < 2:  # BatchNorm needs >1 sample
                    continue
                optimizer.zero_grad()
                loss = loss_fn(self.model_(X_tr[batch]), y_tr[batch])
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(X_val), y_val).item()
            self.history_.append(val_loss)
            if val_loss < best_loss - 1e-4:
                best_loss, bad_epochs = val_loss, 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.n_epochs_trained_ = epoch + 1
        return self

    def predict_proba(self, X):
        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.model_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.model_(X)).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
