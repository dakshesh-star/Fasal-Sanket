"""
Fasal Sanket - Sequence-Based LSTM Model Training with Progressive Prefix Learning

Implements a lightweight PyTorch LSTM sequence crop classification model.
Includes:
- Training on independent field splits with cloud imputation.
- Prefix sequence sampling during training so the recurrent cells learn
  both early-season (20%, 40%) and late-season (80%, 100%) phenological dynamics.
- Validation tracking with early stopping.
- Progressive checkpoint evaluation (20%, 40%, 60%, 80%, 100%) without future leakage.
- Serialization of model weights to backend/models/lstm_classifier.pt
"""

import os
import json
from typing import Optional, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

import sys
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from app.pipeline import PreprocessingPipeline
from app.taxonomy import TAXONOMY_TREE
from data.generator import CROPS, CROP_TO_ID


class CropLSTM(nn.Module):
    """
    Lightweight bidirectional or unidirectional LSTM sequence classifier.
    Computes representation over observed timesteps and classifies into 5 crops.
    """
    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.2
    ):
        super(CropLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param x: Tensor of shape (Batch, Timesteps, Features)
        :param lengths: Optional tensor of observed step counts per sequence
        """
        batch_size, seq_len, _ = x.shape
        lstm_out, _ = self.lstm(x)  # lstm_out: (Batch, Seq, Hidden)

        if lengths is not None:
            # Clamp length index between 0 and seq_len - 1
            idx = torch.clamp(lengths - 1, min=0, max=seq_len - 1).view(-1, 1).expand(-1, self.hidden_size).unsqueeze(1)
            last_out = lstm_out.gather(1, idx).squeeze(1)
        else:
            last_out = lstm_out[:, -1, :]

        feat = self.dropout(self.relu(self.fc1(last_out)))
        logits = self.classifier(feat)
        return logits


def evaluate_taxonomy_levels(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    lvl4_acc = float(accuracy_score(y_true, y_pred))
    true_crops = [CROPS[idx] for idx in y_true]
    pred_crops = [CROPS[idx] for idx in y_pred]

    lvl1_true = [TAXONOMY_TREE[c]["level_1"] for c in true_crops]
    lvl1_pred = [TAXONOMY_TREE[c]["level_1"] for c in pred_crops]
    lvl1_acc = float(accuracy_score(lvl1_true, lvl1_pred))

    lvl2_true = [TAXONOMY_TREE[c]["level_2"] for c in true_crops]
    lvl2_pred = [TAXONOMY_TREE[c]["level_2"] for c in pred_crops]
    lvl2_acc = float(accuracy_score(lvl2_true, lvl2_pred))

    lvl3_true = [TAXONOMY_TREE[c]["level_3"] for c in true_crops]
    lvl3_pred = [TAXONOMY_TREE[c]["level_3"] for c in pred_crops]
    lvl3_acc = float(accuracy_score(lvl3_true, lvl3_pred))

    return {
        "level_1": round(lvl1_acc, 4),
        "level_2": round(lvl2_acc, 4),
        "level_3": round(lvl3_acc, 4),
        "level_4": round(lvl4_acc, 4)
    }


def train_lstm_model():
    models_dir = os.path.join(backend_dir, "models")
    data_dir = os.path.join(backend_dir, "data")
    npz_path = os.path.join(data_dir, "synthetic_crops.npz")

    if not os.path.exists(npz_path):
        from data.generator import generate_and_save
        generate_and_save(data_dir)

    print("Loading synthetic field dataset for LSTM...")
    data = np.load(npz_path)
    X_train, y_train, f_train = data["X_train"], data["y_train"], data["f_train"]
    X_val, y_val, f_val = data["X_val"], data["y_val"], data["f_val"]
    X_test, y_test, f_test = data["X_test"], data["y_test"], data["f_test"]

    # Preprocessing with imputation
    pipeline = PreprocessingPipeline()
    X_train_clean = pipeline.impute_missing(pipeline.validate_input(X_train))
    X_val_clean = pipeline.impute_missing(pipeline.validate_input(X_val))
    X_test_clean = pipeline.impute_missing(pipeline.validate_input(X_test))

    # Multi-stage prefix data augmentation so LSTM learns early through late season
    # Checkpoints: 20% (4 steps), 40% (8 steps), 60% (12 steps), 80% (16 steps), 100% (20 steps)
    checkpoints = [4, 8, 12, 16, 20]
    X_train_aug = []
    y_train_aug = []
    lens_train_aug = []

    for length in checkpoints:
        X_trunc = np.zeros_like(X_train_clean)
        X_trunc[:, :length, :] = X_train_clean[:, :length, :]
        X_train_aug.append(X_trunc)
        y_train_aug.append(y_train)
        lens_train_aug.append(np.full((len(y_train),), length, dtype=np.int64))

    X_train_final = np.vstack(X_train_aug)
    y_train_final = np.concatenate(y_train_aug)
    lens_train_final = np.concatenate(lens_train_aug)

    train_dataset = TensorDataset(
        torch.tensor(X_train_final, dtype=torch.float32),
        torch.tensor(y_train_final, dtype=torch.long),
        torch.tensor(lens_train_final, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_clean, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
        torch.full((len(y_val),), 20, dtype=torch.long)
    )

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")
    print(f"Instantiating CropLSTM (samples={len(train_dataset)}) on device: {device}")
    model = CropLSTM(input_size=9, hidden_size=64, num_layers=2, num_classes=len(CROPS), dropout=0.2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_weights = None
    patience = 8
    patience_counter = 0
    epochs = 30

    print("Beginning LSTM training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y, batch_lens in train_loader:
            batch_x, batch_y, batch_lens = batch_x.to(device), batch_y.to(device), batch_lens.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x, lengths=batch_lens)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)

        train_loss = total_loss / len(train_dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        with torch.no_grad():
            for batch_x, batch_y, batch_lens in val_loader:
                batch_x, batch_y, batch_lens = batch_x.to(device), batch_y.to(device), batch_lens.to(device)
                outputs = model(batch_x, lengths=batch_lens)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * len(batch_y)
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        val_loss = val_loss / len(val_dataset)
        val_acc = accuracy_score(y_val, val_preds)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or patience_counter == 0:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1%}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    lstm_model_path = os.path.join(models_dir, "lstm_classifier.pt")
    torch.save(model.state_dict(), lstm_model_path)
    print(f"Trained LSTM weights saved to {lstm_model_path}")

    # Full Progressive Evaluation (20%, 40%, 60%, 80%, 100%)
    model.eval()
    stages = [0.2, 0.4, 0.6, 0.8, 1.0]
    progressive_val_evidence = {}
    progressive_test_metrics = {}

    print("\n--- Progressive Sequence Evaluation for LSTM ---")
    with torch.no_grad():
        for stg in stages:
            stg_key = f"{int(stg * 100)}%"

            # Val fields progressive
            X_val_stg, steps_v = pipeline.prepare_lstm_sequence(X_val, fraction=stg)
            lens_v = torch.full((len(X_val),), steps_v, dtype=torch.long)
            val_out = model(torch.tensor(X_val_stg, dtype=torch.float32).to(device), lengths=lens_v)
            val_pred = torch.argmax(val_out, dim=1).cpu().numpy()
            tax_metrics = evaluate_taxonomy_levels(y_val, val_pred)
            progressive_val_evidence[stg_key] = tax_metrics

            # Test fields progressive
            X_test_stg, steps_t = pipeline.prepare_lstm_sequence(X_test, fraction=stg)
            lens_t = torch.full((len(X_test),), steps_t, dtype=torch.long)
            test_out = model(torch.tensor(X_test_stg, dtype=torch.float32).to(device), lengths=lens_t)
            test_pred = torch.argmax(test_out, dim=1).cpu().numpy()
            t_acc = float(accuracy_score(y_test, test_pred))
            t_f1 = float(f1_score(y_test, test_pred, average="macro"))

            progressive_test_metrics[stg_key] = {
                "accuracy": round(t_acc, 4),
                "macro_f1": round(t_f1, 4),
                "taxonomy_accuracy": tax_metrics
            }

            print(f"Stage {stg_key:>4}: Level 1 (Clade)={tax_metrics['level_1']:.1%}, "
              f"Level 2 (Family)={tax_metrics['level_2']:.1%}, "
              f"Level 3 (Tribe)={tax_metrics['level_3']:.1%}, "
              f"Level 4 (Species Acc)={tax_metrics['level_4']:.1%}")

    final_test_acc = progressive_test_metrics["100%"]["accuracy"]
    final_test_f1 = progressive_test_metrics["100%"]["macro_f1"]

    metadata = {
        "model_type": "CropLSTM",
        "input_size": 9,
        "hidden_size": 64,
        "num_layers": 2,
        "num_classes": len(CROPS),
        "crops": CROPS,
        "best_val_loss": round(float(best_val_loss), 4),
        "full_season_test_accuracy": round(float(final_test_acc), 4),
        "full_season_test_macro_f1": round(float(final_test_f1), 4),
        "progressive_val_evidence": progressive_val_evidence,
        "progressive_test_metrics": progressive_test_metrics,
        "is_synthetic": True
    }

    meta_path = os.path.join(models_dir, "lstm_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"LSTM metadata and validation evidence saved to {meta_path}")
    return model, metadata


if __name__ == "__main__":
    train_lstm_model()
