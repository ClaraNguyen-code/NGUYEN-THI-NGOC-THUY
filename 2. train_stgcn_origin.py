# === File: train_stgcn_origin.py ===
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from model_stgcn_origin import STGCN_Model

# === PARAMS ===
DATA_DIR = "data"
MODEL_DIR = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQUENCE_LENGTH = 30
NUM_KEYPOINTS = 17
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# === COCO Skeleton ===
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 11), (6, 12)
]

def create_edge_index(skeleton, num_keypoints=17):
    adj = np.zeros((num_keypoints, num_keypoints))
    for i, j in skeleton:
        adj[i, j] = 1
        adj[j, i] = 1
    edge_index = np.array(np.where(adj))
    return torch.tensor(edge_index, dtype=torch.long)

# === Load Dataset ===
def load_dataset():
    X = np.load(os.path.join(DATA_DIR, "processed_features.npy"))  # (N, 30, 17, 2)
    Y = np.load(os.path.join(DATA_DIR, "processed_features_labels.npy"))
    with open(os.path.join(DATA_DIR, "label_mapping.pkl"), "rb") as f:
        label_map = pickle.load(f)

    labels = sorted(set(label_map.values()))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    Y_encoded = np.array([label_to_idx[y] for y in Y])

    action_counts = pd.read_csv(os.path.join(DATA_DIR, "action_counts.csv"))
    total = action_counts["count"].sum()
    class_weights = []
    for label in labels:
        row = action_counts[action_counts["label"] == label]
        count = row["count"].values[0] if not row.empty else 1
        class_weights.append(total / (len(labels) * count))

    return X, Y_encoded, labels, np.array(class_weights), label_to_idx

# === Train ===
def train():
    X, Y, label_names, class_weights, label_to_idx = load_dataset()
    edge_index = create_edge_index(COCO_SKELETON).to(DEVICE)

    # Normalize shape
    X = X.reshape(X.shape[0], SEQUENCE_LENGTH, NUM_KEYPOINTS * 2)  # (B, T, 34)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)

    model = STGCN_Model(
        in_channels=2,
        num_class=len(label_names),
        edge_index=edge_index,
        num_keypoints=NUM_KEYPOINTS
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_acc, val_acc, train_loss, val_loss = [], [], [], []
    best_val = 0
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "stgcn_origin_model.pth")
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix_origin.png")

    for epoch in range(NUM_EPOCHS):
        model.train()
        correct, total, loss_sum = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            pred = out.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

        acc = correct / total
        model.eval()
        with torch.no_grad():
            out_val = model(X_val.to(DEVICE))
            val_loss_val = criterion(out_val, Y_val.to(DEVICE)).item()
            val_acc_val = (out_val.argmax(1) == Y_val.to(DEVICE)).float().mean().item()

        train_acc.append(acc)
        train_loss.append(loss_sum / len(train_loader))
        val_acc.append(val_acc_val)
        val_loss.append(val_loss_val)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Acc: {acc:.4f} Val Acc: {val_acc_val:.4f}")

        if val_acc_val > best_val:
            best_val = val_acc_val
            torch.save(model.state_dict(), model_path)
            print("✅ Saved best model.")

    # === Report ===
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        pred = model(X_val.to(DEVICE)).argmax(1).cpu().numpy()
        true = Y_val.cpu().numpy()

    print("\nClassification Report:")
    print(classification_report(true, pred, target_names=label_names))

    cm = confusion_matrix(true, pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.title("Confusion Matrix (%)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.show()

if __name__ == "__main__":
    train()
