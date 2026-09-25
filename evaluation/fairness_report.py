"""
evaluation/fairness_report.py

Loads models/checkpoints/best_model.pth and reports test-set accuracy
broken down by Fitzpatrick type and by Monk Skin Tone.

This does NOT claim the model is fair or unbiased. It only measures and
reports the differences that are actually present in the checkpoint's
predictions. Groups with fewer than MIN_SAMPLES test images are reported
as "insufficient data" rather than given a misleading accuracy number.

Run:
    python evaluation/fairness_report.py
"""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import DermaConDataset, build_master_table, load_class_mapping
from data.transforms import get_eval_transforms
from models.vision_model import build_model, get_device

MIN_SAMPLES = 10  # minimum test images in a group before we report an accuracy number


def main():
    ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "best_model.pth"
    if not ckpt_path.exists():
        print(f"[FATAL] {ckpt_path} not found. Run training/train.py first.")
        sys.exit(1)

    device = get_device("auto")
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
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    records = []
    unseen_labels = Counter()
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()

            for i in range(len(labels)):
                if labels[i].item() < 0:
                    unseen_labels[batch["disease_label"][i]] += 1
                    continue
                records.append({
                    "correct": bool(preds[i].item() == labels[i].item()),
                    "fitzpatrick": batch["fitzpatrick"][i],
                    "monk_skin_tone": batch["monk_skin_tone"][i],
                })

    df = pd.DataFrame(records)

    print(
        f"Excluded {sum(unseen_labels.values())} test-only images across "
        f"{len(unseen_labels)} unseen disease labels from learned-class fairness accuracy."
    )

    print("=" * 60)
    print("Fairness report - accuracy by Fitzpatrick type")
    print("=" * 60)

    fitz_report = []
    for fitz_value, group in df.groupby("fitzpatrick"):
        n = len(group)
        acc = group["correct"].mean() if n > 0 else None
        if n >= MIN_SAMPLES:
            print(f"  Fitzpatrick {fitz_value:10s}: n={n:5d}  accuracy={acc:.4f}")
            fitz_report.append({"Fitzpatrick": fitz_value, "n": n, "accuracy": acc})
        else:
            print(f"  Fitzpatrick {fitz_value:10s}: n={n:5d}  [insufficient data, < {MIN_SAMPLES} samples]")
            fitz_report.append({"Fitzpatrick": fitz_value, "n": n, "accuracy": None})

    print()
    print("=" * 60)
    print("Fairness report - accuracy by Monk Skin Tone")
    print("=" * 60)

    monk_report = []
    for monk_value, group in df.groupby("monk_skin_tone"):
        n = len(group)
        acc = group["correct"].mean() if n > 0 else None
        if n >= MIN_SAMPLES:
            print(f"  Monk {monk_value:10s}: n={n:5d}  accuracy={acc:.4f}")
            monk_report.append({"Monk_skin_tone": monk_value, "n": n, "accuracy": acc})
        else:
            print(f"  Monk {monk_value:10s}: n={n:5d}  [insufficient data, < {MIN_SAMPLES} samples]")
            monk_report.append({"Monk_skin_tone": monk_value, "n": n, "accuracy": None})

    out_dir = PROJECT_ROOT / "models" / "checkpoints"
    pd.DataFrame(fitz_report).to_csv(out_dir / "fairness_by_fitzpatrick.csv", index=False)
    pd.DataFrame(monk_report).to_csv(out_dir / "fairness_by_monk_skin_tone.csv", index=False)

    print()
    print(f"Saved: {out_dir / 'fairness_by_fitzpatrick.csv'}")
    print(f"Saved: {out_dir / 'fairness_by_monk_skin_tone.csv'}")
    print()
    print("NOTE: This report measures accuracy differences observed on the test")
    print("      set. It does not certify the model as fair or unbiased, and small")
    print("      per-group sample sizes limit how much can be concluded.")


if __name__ == "__main__":
    main()
