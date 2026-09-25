"""
data/dataset.py

PyTorch Dataset for DermaCon-IN.

Uses the ACTUAL provided train_split.csv / test_split.csv (subject-wise,
already split by Sub_class - we do not recreate this split).

Target label: Disease_label (per project instructions - fine-grained,
~245 classes). Fitzpatrick, Monk_skin_tone, Subject_ID, and Image_name
are carried through on every sample for fairness evaluation later -
they are NOT used as model inputs and NOT used as prediction targets.

Class index (Disease_label -> integer) is built from the TRAIN split
only, and saved to models/checkpoints/class_mapping.json for
reproducibility. Disease labels that appear in test but never in train
are reported separately and receive evaluation target -1; they are never
treated as a learned disease class.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "DermaCon-IN"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"

def find_metadata_dir() -> Path:
    candidates = list(DATASET_ROOT.rglob("Skin_Metadata.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find Skin_Metadata.csv under {DATASET_ROOT}. "
            f"Run scripts/setup_dermacon.py first."
        )
    return candidates[0].parent


def load_image_index() -> pd.DataFrame:
    idx_path = PROCESSED_DIR / "image_index.csv"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"{idx_path} not found. Run scripts/setup_dermacon.py first."
        )
    return pd.read_csv(idx_path, dtype=str)


def build_master_table() -> tuple:
    """
    Merges Skin_Metadata.csv with train_split.csv / test_split.csv and the
    image index. Returns (train_df, test_df), each containing:
        Image_name, path, Subject_ID, Fitzpatrick, Monk_skin_tone,
        Body_part, Descriptors, Main_class, Sub_class, Disease_label,
        Confidence, Quality, Sex, Age, Gradability
    Rows whose image file could not be resolved are dropped, but the
    count of dropped rows is returned so callers can decide whether to
    fail loudly.
    """
    metadata_dir = find_metadata_dir()
    metadata = pd.read_csv(metadata_dir / "Skin_Metadata.csv", dtype=str)

    for required in ["Image_name", "Subject_ID", "Disease_label", "Fitzpatrick", "Monk_skin_tone"]:
        if required not in metadata.columns:
            raise ValueError(
                f"Skin_Metadata.csv is missing required column '{required}'. "
                f"Columns present: {list(metadata.columns)}"
            )

    train_split_path = metadata_dir / "train_split.csv"
    test_split_path = metadata_dir / "test_split.csv"
    if not train_split_path.exists() or not test_split_path.exists():
        raise FileNotFoundError(
            f"train_split.csv / test_split.csv not found in {metadata_dir}"
        )

    train_split = pd.read_csv(train_split_path, dtype=str)
    test_split = pd.read_csv(test_split_path, dtype=str)

    for name, split_df in [("train_split.csv", train_split), ("test_split.csv", test_split)]:
        if "Image_name" not in split_df.columns:
            raise ValueError(
                f"{name} does not contain an 'Image_name' column. "
                f"Columns present: {list(split_df.columns)}. "
                f"Cannot merge with Skin_Metadata.csv."
            )

    image_index = load_image_index()

    def build(split_df: pd.DataFrame, split_name: str):
        merged = split_df[["Image_name"]].merge(metadata, on="Image_name", how="left")
        n_before = len(merged)
        merged = merged.merge(image_index, on="Image_name", how="left")

        missing = merged[merged["path"].isna()]
        n_missing = len(missing)
        if n_missing > 0:
            pct = 100.0 * n_missing / n_before
            print(
                f"[WARN] {split_name}: {n_missing}/{n_before} rows ({pct:.2f}%) "
                f"have no resolvable image file and will be dropped."
            )
            if pct > 10.0:
                raise RuntimeError(
                    f"{split_name}: {pct:.1f}% of rows have no matching image file. "
                    f"This is too large a fraction to silently proceed with. "
                    f"Check data/reports/setup_report.txt and re-run scripts/setup_dermacon.py."
                )
        merged = merged.dropna(subset=["path"]).reset_index(drop=True)
        return merged

    train_df = build(train_split, "train_split.csv")
    test_df = build(test_split, "test_split.csv")

    return train_df, test_df


def build_and_save_class_mapping(train_df: pd.DataFrame) -> dict:
    """
    Builds Disease_label -> integer index mapping from the TRAIN split only,
    sorted alphabetically for reproducibility, and saves it to
    models/checkpoints/class_mapping.json
    """
    classes = sorted(train_df["Disease_label"].dropna().unique().tolist())
    mapping = {label: idx for idx, label in enumerate(classes)}
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_DIR / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    return mapping


def load_class_mapping() -> dict:
    path = CHECKPOINT_DIR / "class_mapping.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run training once to generate it, "
            f"or run data/inspect_dataset.py / training/train.py first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class DermaConDataset(Dataset):
    """
    df: one row per usable image (output of build_master_table()[0 or 1])
    class_mapping: Disease_label -> int, from build_and_save_class_mapping()
    transform: albumentations Compose (see data/transforms.py)
    """

    def __init__(self, df: pd.DataFrame, class_mapping: dict, transform=None):
        self.df = df.reset_index(drop=True)
        self.class_mapping = class_mapping
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["path"]).convert("RGB")
        image = image_to_numpy(image)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        disease_label = row.get("Disease_label", None)
        if pd.isna(disease_label) or disease_label not in self.class_mapping:
            label_idx = -1
            reported_label = "N/A" if pd.isna(disease_label) else disease_label
        else:
            label_idx = self.class_mapping[disease_label]
            reported_label = disease_label

        sample = {
            "image": image,
            "label": label_idx,
            "disease_label": reported_label,
            "image_name": row["Image_name"],
            "subject_id": row.get("Subject_ID", ""),
            "fitzpatrick": row.get("Fitzpatrick", "N/A") if not pd.isna(row.get("Fitzpatrick", None)) else "N/A",
            "monk_skin_tone": row.get("Monk_skin_tone", "N/A") if not pd.isna(row.get("Monk_skin_tone", None)) else "N/A",
            "body_part": row.get("Body_part", "") if not pd.isna(row.get("Body_part", None)) else "",
        }
        return sample


def image_to_numpy(pil_image):
    import numpy as np
    return np.array(pil_image)


if __name__ == "__main__":
    # Quick smoke test: build the tables and print shapes. Does not train anything.
    try:
        train_df, test_df = build_master_table()
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    print(f"Train usable rows: {len(train_df)}")
    print(f"Test usable rows:  {len(test_df)}")

    mapping = build_and_save_class_mapping(train_df)
    print(f"Classes in mapping: {len(mapping)}")
    print(f"Saved to {CHECKPOINT_DIR / 'class_mapping.json'}")
