"""
data/inspect_dataset.py

Reads:
    data/processed/image_index.csv     (built by scripts/setup_dermacon.py)
    data/raw/DermaCon-IN/METADATA/Skin_Metadata.csv
    data/raw/DermaCon-IN/METADATA/train_split.csv
    data/raw/DermaCon-IN/METADATA/test_split.csv

Produces (in data/reports/):
    inspection_report.txt      - human-readable summary
    images_per_disease.csv
    images_per_subclass.csv
    images_per_mainclass.csv
    fitzpatrick_distribution.csv
    monk_skin_tone_distribution.csv
    disease_x_fitzpatrick.csv
    rare_classes.csv

This script must be run (successfully) before training/train.py, because
train.py reuses build_master_table() from this file / data/dataset.py to
resolve images and will refuse to run if image_index.csv is missing.

Run:
    python data/inspect_dataset.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "DermaCon-IN"

RARE_CLASS_THRESHOLD = 5  # classes with fewer than this many total images
MIN_SAMPLES_FOR_FAIRNESS = 10  # used later by evaluation/fairness_report.py, reported here too

REQUIRED_METADATA_COLUMNS = [
    "Image_name", "Subject_ID", "Quality", "Sex", "Age", "Gradability",
    "Fitzpatrick", "Monk_skin_tone", "Body_part", "Descriptors",
    "Main_class", "Sub_class", "Disease_label", "Confidence",
]


def find_metadata_dir() -> Path:
    candidates = list(DATASET_ROOT.rglob("Skin_Metadata.csv"))
    if not candidates:
        print(f"[FATAL] Could not find Skin_Metadata.csv under {DATASET_ROOT}")
        print("        Run scripts/setup_dermacon.py first.")
        sys.exit(1)
    return candidates[0].parent


def load_metadata(metadata_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_dir / "Skin_Metadata.csv", dtype=str)
    missing_cols = [c for c in REQUIRED_METADATA_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"[FATAL] Skin_Metadata.csv is missing expected columns: {missing_cols}")
        print(f"        Columns present: {list(df.columns)}")
        sys.exit(1)
    return df


def load_split(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        print(f"[FATAL] {name} not found at {path}")
        sys.exit(1)
    df = pd.read_csv(path, dtype=str)
    if "Image_name" not in df.columns:
        print(f"[FATAL] {name} does not contain an 'Image_name' column.")
        print(f"        Columns present: {list(df.columns)}")
        print("        train.py / dataset.py need Image_name to merge this split")
        print("        with Skin_Metadata.csv. Inspect the file manually.")
        sys.exit(1)
    return df


def load_image_index() -> pd.DataFrame:
    idx_path = PROCESSED_DIR / "image_index.csv"
    if not idx_path.exists():
        print(f"[FATAL] {idx_path} not found.")
        print("        Run scripts/setup_dermacon.py first.")
        sys.exit(1)
    return pd.read_csv(idx_path, dtype=str)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metadata_dir = find_metadata_dir()
    metadata = load_metadata(metadata_dir)
    train_split = load_split(metadata_dir / "train_split.csv", "train_split.csv")
    test_split = load_split(metadata_dir / "test_split.csv", "test_split.csv")
    image_index = load_image_index()

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("DermaCon-IN dataset inspection report")
    log("=" * 60)
    log()

    # ---- Basic counts ----
    log(f"Total rows in Skin_Metadata.csv: {len(metadata)}")
    log(f"Total unique Image_name values:  {metadata['Image_name'].nunique()}")
    log(f"Total unique Subject_ID values:  {metadata['Subject_ID'].nunique()}")
    log(f"Total unique Disease_label values: {metadata['Disease_label'].nunique()}")
    log(f"Total unique Sub_class values:     {metadata['Sub_class'].nunique()}")
    log(f"Total unique Main_class values:    {metadata['Main_class'].nunique()}")
    log()

    # ---- Missing metadata ----
    log("Missing / empty values per column:")
    for col in REQUIRED_METADATA_COLUMNS:
        n_missing = metadata[col].isna().sum() + (metadata[col].astype(str).str.strip().isin(["", "N/A", "NA", "nan"])).sum()
        log(f"  {col:20s}: {n_missing}")
    log()

    # ---- Images resolvable on disk ----
    merged = metadata.merge(image_index, on="Image_name", how="left", indicator=True)
    missing_images = merged[merged["_merge"] == "left_only"]
    log(f"Metadata rows with NO matching image file on disk: {len(missing_images)}")
    if len(missing_images) > 0:
        pct = 100.0 * len(missing_images) / len(metadata)
        log(f"  ({pct:.2f}% of all rows)")
        if pct > 5.0:
            log("  [ACTION NEEDED] This is a large fraction - check scripts/setup_dermacon.py report.")
    log()

    # ---- Duplicate image paths ----
    dup_paths = image_index[image_index.duplicated("path", keep=False)]
    log(f"Duplicate resolved image paths (same file used by >1 Image_name): {dup_paths['path'].nunique() if len(dup_paths) else 0}")
    log()

    # ---- Images per disease ----
    per_disease = metadata["Disease_label"].value_counts().rename_axis("Disease_label").reset_index(name="count")
    per_disease.to_csv(REPORTS_DIR / "images_per_disease.csv", index=False)
    log(f"Wrote images_per_disease.csv ({len(per_disease)} disease labels)")

    # ---- Images per sub_class ----
    per_subclass = metadata["Sub_class"].value_counts().rename_axis("Sub_class").reset_index(name="count")
    per_subclass.to_csv(REPORTS_DIR / "images_per_subclass.csv", index=False)
    log(f"Wrote images_per_subclass.csv ({len(per_subclass)} sub-classes)")

    # ---- Images per main_class ----
    per_mainclass = metadata["Main_class"].value_counts().rename_axis("Main_class").reset_index(name="count")
    per_mainclass.to_csv(REPORTS_DIR / "images_per_mainclass.csv", index=False)
    log(f"Wrote images_per_mainclass.csv ({len(per_mainclass)} main classes)")
    log()

    # ---- Fitzpatrick distribution ----
    fitz_dist = metadata["Fitzpatrick"].fillna("MISSING").value_counts().rename_axis("Fitzpatrick").reset_index(name="count")
    fitz_dist.to_csv(REPORTS_DIR / "fitzpatrick_distribution.csv", index=False)
    log("Fitzpatrick distribution:")
    for _, row in fitz_dist.iterrows():
        log(f"  {row['Fitzpatrick']:10s}: {row['count']}")
    log()

    # ---- Monk skin tone distribution ----
    monk_dist = metadata["Monk_skin_tone"].fillna("MISSING").value_counts().rename_axis("Monk_skin_tone").reset_index(name="count")
    monk_dist.to_csv(REPORTS_DIR / "monk_skin_tone_distribution.csv", index=False)
    log("Monk Skin Tone distribution:")
    for _, row in monk_dist.iterrows():
        log(f"  {row['Monk_skin_tone']:10s}: {row['count']}")
    log()

    # ---- Disease x Fitzpatrick cross-tab ----
    cross = pd.crosstab(metadata["Disease_label"], metadata["Fitzpatrick"].fillna("MISSING"))
    cross.to_csv(REPORTS_DIR / "disease_x_fitzpatrick.csv")
    log(f"Wrote disease_x_fitzpatrick.csv ({cross.shape[0]} diseases x {cross.shape[1]} Fitzpatrick categories)")
    log()

    # ---- Rare classes ----
    rare = per_disease[per_disease["count"] < RARE_CLASS_THRESHOLD]
    rare.to_csv(REPORTS_DIR / "rare_classes.csv", index=False)
    log(f"Disease labels with fewer than {RARE_CLASS_THRESHOLD} total images: {len(rare)}")
    if len(rare) > 0:
        log("  These will be hard or impossible to learn reliably. Consider this during training.")
    log()

    # ---- Train/test subject overlap (subject leakage check) ----
    split_metadata = metadata[["Image_name", "Subject_ID", "Disease_label", "Sub_class"]]
    train_merged = train_split[["Image_name"]].merge(split_metadata, on="Image_name", how="left")
    test_merged = test_split[["Image_name"]].merge(split_metadata, on="Image_name", how="left")

    train_subjects = set(train_merged["Subject_ID"].dropna())
    test_subjects = set(test_merged["Subject_ID"].dropna())
    overlap = train_subjects & test_subjects

    log(f"Train split: {len(train_split)} rows, {len(train_subjects)} unique subjects")
    log(f"Test split:  {len(test_split)} rows, {len(test_subjects)} unique subjects")
    log(f"Subject_ID overlap between train and test: {len(overlap)}")
    if overlap:
        log("  [WARNING] Subject leakage detected between the supplied train/test split.")
        log(f"  Example overlapping subjects: {list(overlap)[:10]}")
    else:
        log("  No subject leakage detected. Good.")
    log()

    # ---- Train/test class coverage ----
    train_classes = set(train_merged["Disease_label"].dropna())
    test_classes = set(test_merged["Disease_label"].dropna())
    only_in_test = test_classes - train_classes
    only_in_train = train_classes - test_classes

    log(f"Disease_label classes in train: {len(train_classes)}")
    log(f"Disease_label classes in test:  {len(test_classes)}")
    log(f"Classes in test but NOT in train (cannot be predicted correctly, model never saw them): {len(only_in_test)}")
    if only_in_test:
        log(f"  {sorted(only_in_test)[:20]}{' ...' if len(only_in_test) > 20 else ''}")
    log(f"Classes in train but NOT in test: {len(only_in_train)}")
    log()

    # ---- Missing image files, split by train/test ----
    train_img_check = train_merged.merge(image_index, on="Image_name", how="left", indicator=True)
    test_img_check = test_merged.merge(image_index, on="Image_name", how="left", indicator=True)
    train_missing = (train_img_check["_merge"] == "left_only").sum()
    test_missing = (test_img_check["_merge"] == "left_only").sum()
    log(f"Train split rows with missing image file: {train_missing} / {len(train_split)}")
    log(f"Test split rows with missing image file:  {test_missing} / {len(test_split)}")
    log()

    log("=" * 60)
    log("Inspection complete. Full report written to data/reports/inspection_report.txt")
    log("Review data/reports/*.csv before running training.")

    with open(REPORTS_DIR / "inspection_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
