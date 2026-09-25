"""
training/train.py

Trains DermatologyVisionModel on DermaCon-IN using the actual provided
train_split.csv / test_split.csv (subject-wise split, not recreated here).

Run (after scripts/setup_dermacon.py and data/inspect_dataset.py have been
run successfully at least once):

    python training/train.py --epochs 10

Useful flags:
    --epochs N
    --batch-size N
    --learning-rate F
    --image-size N
    --num-workers N
    --model timm_model_name
    --device auto|cuda|cpu
    --resume                 (resume from models/checkpoints/last_model.pth)
    --freeze-backbone        (only train the classifier head)

Outputs (in models/checkpoints/):
    class_mapping.json       (Disease_label -> int, built from train split)
    best_model.pth           (lowest validation loss so far)
    last_model.pth           (most recent epoch, for --resume)
    training_history.json    (per-epoch train/val loss + accuracy)
    confusion_matrix.png
    classification_report.txt
    class_support_report.csv
    unseen_test_classes.csv
    fairness_by_fitzpatrick.csv
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import DermaConDataset, build_and_save_class_mapping, build_master_table
from data.transforms import get_eval_transforms, get_train_transforms
from models.vision_model import build_model, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train DermaCon-IN vision model")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "training" / "config.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_config(config: dict, args: argparse.Namespace) -> dict:
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.image_size is not None:
        config["training"]["image_size"] = args.image_size
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    if args.model is not None:
        config["model"]["name"] = args.model
    if args.freeze_backbone:
        config["model"]["freeze_backbone"] = True
    return config


def verify_no_large_scale_missing_images(train_df, test_df):
    """
    build_master_table() already drops unresolvable rows and raises if the
    fraction is too large (see data/dataset.py). This is a final, explicit
    checkpoint before training starts so failures are never silent.
    """
    if len(train_df) == 0:
        raise RuntimeError("Train split has 0 usable rows after image resolution. Cannot train.")
    if len(test_df) == 0:
        raise RuntimeError("Test split has 0 usable rows after image resolution. Cannot evaluate.")
    print(f"[OK] {len(train_df)} usable training images, {len(test_df)} usable test images.")


def compute_class_weights(train_df, class_mapping: dict, device, power: float, max_weight: float) -> torch.Tensor:
    counts = Counter(train_df["Disease_label"].dropna().tolist())
    num_classes = len(class_mapping)
    weights = torch.ones(num_classes, dtype=torch.float32)
    for label, idx in class_mapping.items():
        n = counts.get(label, 0)
        weights[idx] = (1.0 / (n ** power)) if n > 0 else 0.0
    # Normalize so mean weight (over classes actually present) is 1.0
    present = weights[weights > 0]
    if len(present) > 0:
        weights = weights / present.mean()
        weights = torch.clamp(weights, max=max_weight)
    return weights.to(device)


def write_class_support_report(train_df, test_df, class_mapping, checkpoint_dir, rare_threshold):
    train_counts = Counter(train_df["Disease_label"].dropna().tolist())
    test_counts = Counter(test_df["Disease_label"].dropna().tolist())
    rows = []
    for label in sorted(set(train_counts) | set(test_counts)):
        train_count = train_counts.get(label, 0)
        rows.append({
            "Disease_label": label,
            "train_count": train_count,
            "test_count": test_counts.get(label, 0),
            "in_training_mapping": label in class_mapping,
            "support_band": "rare" if train_count < rare_threshold else "supported",
        })
    report_path = checkpoint_dir / "class_support_report.csv"
    pd.DataFrame(rows).to_csv(report_path, index=False)
    unseen_rows = [
        {
            "Disease_label": row["Disease_label"],
            "test_count": row["test_count"],
            "status": "unseen_in_training",
        }
        for row in rows
        if not row["in_training_mapping"]
    ]
    pd.DataFrame(unseen_rows).to_csv(checkpoint_dir / "unseen_test_classes.csv", index=False)
    rare_count = sum(row["train_count"] < rare_threshold for row in rows if row["in_training_mapping"])
    unseen_count = sum(not row["in_training_mapping"] for row in rows)
    print(f"Class support: {rare_count} mapped labels have fewer than {rare_threshold} training images.")
    print(f"Unseen test-only labels: {unseen_count}; report saved to {report_path}")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            valid = labels >= 0
            if not valid.any():
                continue
            loss = criterion(outputs[valid], labels[valid])

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * valid.sum().item()
            preds = outputs[valid].argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels[valid].cpu().numpy().tolist())

    if not all_labels:
        raise RuntimeError("No known disease labels were available for this epoch.")
    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels


def main():
    args = parse_args()
    config = load_config(args.config)
    config = merge_config(config, args)

    checkpoint_dir = PROJECT_ROOT / config["paths"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print(f"Using device: {device}")

    # ---- Data ----
    print("Building dataset tables from train_split.csv / test_split.csv ...")
    train_df, test_df = build_master_table()
    verify_no_large_scale_missing_images(train_df, test_df)

    class_mapping = build_and_save_class_mapping(train_df)
    num_classes = len(class_mapping)
    print(f"Number of learned disease classes: {num_classes}")
    write_class_support_report(
        train_df,
        test_df,
        class_mapping,
        checkpoint_dir,
        config["training"]["rare_class_threshold"],
    )

    image_size = config["training"]["image_size"]
    train_dataset = DermaConDataset(train_df, class_mapping, transform=get_train_transforms(image_size))
    test_dataset = DermaConDataset(test_df, class_mapping, transform=get_eval_transforms(image_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    # ---- Model ----
    model = build_model(
        model_name=config["model"]["name"],
        num_classes=num_classes,
        pretrained=config["model"]["pretrained"],
        freeze_backbone=config["model"]["freeze_backbone"],
    ).to(device)

    if config["training"]["use_class_weights"]:
        class_weights = compute_class_weights(
            train_df,
            class_mapping,
            device,
            config["training"]["class_weight_power"],
            config["training"]["class_weight_max"],
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # ---- Resume ----
    start_epoch = 0
    history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    last_ckpt_path = checkpoint_dir / "last_model.pth"
    if args.resume and last_ckpt_path.exists():
        print(f"Resuming from {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        history_path = PROJECT_ROOT / config["paths"]["history_file"]
        if history_path.exists():
            with open(history_path, encoding="utf-8") as f:
                history = json.load(f)
        print(f"Resumed at epoch {start_epoch}")

    total_epochs = config["training"]["epochs"]

    # ---- Training loop ----
    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_preds, val_labels = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch + 1}/{total_epochs} "
            f"| train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"| {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "elapsed_sec": elapsed,
        })

        # Save "last" checkpoint every epoch (for --resume)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config,
        }, last_ckpt_path)

        # Save "best" checkpoint + early stopping tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "config": config,
            }, checkpoint_dir / "best_model.pth")
            print(f"  [OK] New best model saved (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1

        with open(PROJECT_ROOT / config["paths"]["history_file"], "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if epochs_without_improvement >= config["training"]["early_stopping_patience"]:
            print(f"Early stopping: no val_loss improvement for {epochs_without_improvement} epochs.")
            break

    # ---- Final evaluation report on test set (using the LAST epoch's weights) ----
    print("\nRunning final evaluation on test set...")
    idx_to_label = {v: k for k, v in class_mapping.items()}
    present_labels = sorted(set(val_labels) | set(val_preds))
    target_names = [idx_to_label.get(i, str(i)) for i in present_labels]

    report = classification_report(
        val_labels, val_preds, labels=present_labels, target_names=target_names,
        zero_division=0,
    )
    report_path = checkpoint_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

    cm = confusion_matrix(val_labels, val_preds, labels=present_labels)
    np.save(checkpoint_dir / "confusion_matrix.npy", cm)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_size = max(8, len(present_labels) * 0.2)
        plt.figure(figsize=(fig_size, fig_size))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Confusion matrix (test set, final epoch)")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(checkpoint_dir / "confusion_matrix.png", dpi=150)
        plt.close()
        print(f"Confusion matrix image saved to {checkpoint_dir / 'confusion_matrix.png'}")
    except Exception as e:
        print(f"[WARN] Could not render confusion matrix image: {e}")

    print("\nTraining complete.")
    print(f"Checkpoints and reports are in: {checkpoint_dir}")
    print("Next step:")
    print("  python evaluation/evaluate.py")
    print("  python evaluation/fairness_report.py")


if __name__ == "__main__":
    main()
