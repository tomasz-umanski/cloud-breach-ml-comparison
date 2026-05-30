"""Deep-learning models implemented in PyTorch.

Three architectures are provided, all trained for binary classification
(benign vs. attack):

* ``MLP``    - a fully connected feed-forward network,
* ``CNN1D``  - a 1D convolutional network that treats the feature vector as a
  short signal,
* ``LSTMNet``- a recurrent network reading the features as a sequence.

A single :class:`DeepModel` wrapper handles the common training loop, early
stopping, class-imbalance weighting and probability prediction so all three
share an identical interface with the classical models.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .. import config


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CNN1D(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = self.features(x)
        return self.classifier(x).squeeze(-1)


class LSTMNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)  # (batch, seq=features, input=1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # final time step (both directions)
        return self.classifier(last).squeeze(-1)


_ARCHITECTURES = {"MLP": MLP, "CNN1D": CNN1D, "LSTM": LSTMNet}
_DISPLAY_NAMES = {"MLP": "MLP", "CNN1D": "CNN 1D", "LSTM": "LSTM"}


class DeepModel:
    """Common training/prediction wrapper for the PyTorch architectures."""

    is_deep = True

    def __init__(self, arch: str, in_dim: int, cfg: config.DeepConfig, seed: int) -> None:
        if arch not in _ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {arch}")
        torch.manual_seed(seed)
        self.arch = arch
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _ARCHITECTURES[arch](in_dim, cfg.hidden_dim, cfg.dropout).to(self.device)
        self.history: dict = {"train_loss": [], "val_loss": [], "val_auc": []}

    @property
    def name(self) -> str:
        return _DISPLAY_NAMES[self.arch]

    def _loader(self, X, y, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=shuffle)

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        from sklearn.metrics import roc_auc_score

        train_loader = self._loader(X_train, y_train, shuffle=True)
        val_loader = self._loader(X_val, y_val, shuffle=False)

        # Weight the positive class to counter benign-majority imbalance.
        pos = float((y_train == 1).sum())
        neg = float((y_train == 0).sum())
        pos_weight = torch.tensor([neg / max(pos, 1.0)], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        best_auc = -np.inf
        best_state = None
        patience = self.cfg.early_stopping_patience
        stale = 0

        for epoch in range(self.cfg.epochs):
            self.net.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(train_loader.dataset)

            val_loss, val_probs = self._evaluate_loss(val_loader, criterion)
            try:
                val_auc = roc_auc_score(y_val, val_probs)
            except ValueError:
                val_auc = float("nan")

            self.history["train_loss"].append(epoch_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(float(val_auc))

            if np.isnan(val_auc) or val_auc > best_auc:
                best_auc = val_auc if not np.isnan(val_auc) else best_auc
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

    def _evaluate_loss(self, loader: DataLoader, criterion) -> tuple[float, np.ndarray]:
        self.net.eval()
        total_loss = 0.0
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.net(xb)
                total_loss += criterion(logits, yb).item() * xb.size(0)
                probs.append(torch.sigmoid(logits).cpu().numpy())
        return total_loss / len(loader.dataset), np.concatenate(probs)

    def predict_proba(self, X) -> np.ndarray:
        self.net.eval()
        loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32)),
            batch_size=self.cfg.batch_size,
            shuffle=False,
        )
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                probs.append(torch.sigmoid(self.net(xb)).cpu().numpy())
        return np.concatenate(probs)

    def complexity(self) -> dict:
        n_params = sum(p.numel() for p in self.net.parameters())
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        return {"n_parameters": int(n_params), "trainable_parameters": int(trainable)}
