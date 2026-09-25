"""
evaluation/evaluate.py

Loads models/checkpoints/best_model.pth and re-runs evaluation on the
test split, independent of training/train.py. Useful for re-checking a
checkpoint without retraining.

Run:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --checkpoint models/checkpoints/last_model.pth
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import DermaConDataset, build_master_table, load_class_mapping
from data.transforms import get_eval_transforms
from models.vision_model import build_model, get_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(PROJECT_ROOT / "models" / "checkpoints" / "best_model.pth"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[FATAL] Checkpoint not found: {ckpt_path}")
        print("        Run training/train.py first.")
        sys.exit(1)

    device = get_device(args.device)
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    model_cfg = config.get("model", {"name": "efficientnet_b0"})
    image_size = config.get("training", {}).get("image_size", 224)

    class_mapping = load_class_mapping()
    num_classes = len(class_mapping)

    model = build_model(
        model_name=model_cfg.get("name", "efficientnet_b0"),
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, test_df = build_master_table()
    test_dataset = DermaConDataset(test_df, class_mapping, transform=get_eval_transforms(image_size))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_preds, all_labels = [], []
    unseen_labels = Counter()
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            valid = labels >= 0
            all_preds.extend(preds[valid].numpy().tolist())
            all_labels.extend(labels[valid].numpy().tolist())
            for label, label_idx in zip(batch["disease_label"], labels.tolist()):
                if label_idx < 0:
                    unseen_labels[label] += 1

    idx_to_label = {v: k for k, v in class_mapping.items()}
    present_labels = sorted(set(all_labels) | set(all_preds))
    target_names = [idx_to_label.get(i, str(i)) for i in present_labels]

    report = classification_report(
        all_labels, all_preds, labels=present_labels, target_names=target_names, zero_division=0
    )
    print(report)

    out_dir = ckpt_path.parent
    with open(out_dir / "evaluate_classification_report.txt", "w", encoding="utf-8") as f:
        f.write("Learned-class evaluation excludes test-only disease labels.\n")
        f.write("Those labels have no training examples and cannot be learned from this split.\n\n")
        f.write(report)

    unseen_report = pd.DataFrame(
        [{"Disease_label": label, "test_count": count, "status": "unseen_in_training"}
         for label, count in sorted(unseen_labels.items())]
    )
    unseen_report.to_csv(out_dir / "unseen_test_classes.csv", index=False)
    print(f"Unseen test-only classes: {sum(unseen_labels.values())} images across {len(unseen_labels)} labels")
    print(f"Unseen-class report saved to {out_dir / 'unseen_test_classes.csv'}")

    cm = confusion_matrix(all_labels, all_preds, labels=present_labels)
    np.save(out_dir / "evaluate_confusion_matrix.npy", cm)

    print(f"\nReport saved to {out_dir / 'evaluate_classification_report.txt'}")
    print(f"Confusion matrix array saved to {out_dir / 'evaluate_confusion_matrix.npy'}")


if __name__ == "__main__":
    main()
