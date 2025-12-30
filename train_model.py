# train_model.py
"""
Train SystemStateNet from training_data.csv.

Requirements:
    pip install torch scikit-learn pandas numpy
"""

import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from model import SystemStateNet

DATA_FILE = "training_data.csv"
MODEL_FILE = "model.pth"
CLASSES_FILE = "classes.json"

BATCH_SIZE = 256
EPOCHS = 25
LR = 1e-3


def load_dataset():
    df = pd.read_csv(DATA_FILE)

    feature_cols = [c for c in df.columns if c not in ("timestamp", "label")]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y_text = df["label"].to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(y_text).astype(np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    return train_ds, val_ds, le, X.shape[1]


def train():
    train_ds, val_ds, le, input_dim = load_dataset()
    num_classes = len(le.classes_)

    model = SystemStateNet(input_dim, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        # ---- eval ----
        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.numel()
        val_acc = correct / max(1, total)
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} val_acc={val_acc:.3f}")

    # Save model + label mapping
    torch.save(model.state_dict(), MODEL_FILE)
    with open(CLASSES_FILE, "w", encoding="utf-8") as f:
        json.dump(list(le.classes_), f, indent=2)
    print(f"Saved {MODEL_FILE} and {CLASSES_FILE}")


if __name__ == "__main__":
    train()
