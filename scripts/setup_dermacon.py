"""
scripts/setup_dermacon.py

Locates the DermaCon-IN dataset that you have placed under:
    data/raw/DermaCon-IN/

Expected (as you described it) layout:
    data/raw/DermaCon-IN/
    ├── checkpoints/                     (ignored - we do not use these .pth files)
    ├── DATASET/
    │   ├── DATASET_0/DATASET_0/*.jpg
    │   └── DATASET_1/DATASET_1/*.jpg
    └── METADATA/
        ├── Skin_Metadata.csv
        ├── train_split.csv
        ├── test_split.csv
        └── Metadata_schema.md

This script does NOT move, copy, or delete your image files. It:
  1. Recursively finds every image file under DATASET/
  2. Builds an index: Image_name -> absolute_path
  3. Loads Skin_Metadata.csv and checks how many Image_name values resolve
     to an actual file on disk
  4. Writes data/processed/image_index.csv  (Image_name, path)
  5. Writes data/reports/setup_report.txt   (human-readable summary)

Run:
    python scripts/setup_dermacon.py

Optional:
    python scripts/setup_dermacon.py --source /custom/path/to/DermaCon-IN
"""

import argparse
import csv
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Filenames we deliberately do not use (pretrained checkpoints supplied with the dataset)
IGNORED_CHECKPOINTS = {
    "CBM_MC_best_model.pth",
    "CBM_SC_MC_type1.pth",
    "CBM_SC_MC_type2.pth",
    "Swin_MC_best_model.pth",
}


def find_project_root() -> Path:
    # scripts/setup_dermacon.py -> project root is parent of scripts/
    return Path(__file__).resolve().parent.parent


def find_metadata_dir(dataset_root: Path) -> Path:
    candidates = list(dataset_root.rglob("Skin_Metadata.csv"))
    if not candidates:
        print(f"[FATAL] Could not find Skin_Metadata.csv anywhere under {dataset_root}")
        print("        Make sure you have placed the DermaCon-IN folder (with its")
        print("        METADATA/ subfolder) inside data/raw/DermaCon-IN/")
        sys.exit(1)
    if len(candidates) > 1:
        print(f"[WARN] Found multiple Skin_Metadata.csv files, using the first:")
        for c in candidates:
            print(f"       - {c}")
    return candidates[0].parent


def find_dataset_image_dirs(dataset_root: Path) -> list:
    """
    Find directories that actually contain image files, under any 'DATASET' folder.
    We do not hardcode DATASET_0/DATASET_0 exactly, in case the nesting differs
    slightly on your machine (e.g. only one level deep) - we search and report
    what we actually found.
    """
    dataset_dir = dataset_root / "DATASET"
    if not dataset_dir.exists():
        print(f"[FATAL] Expected a DATASET/ folder under {dataset_root}, not found.")
        sys.exit(1)

    dirs_with_images = []
    for sub in dataset_dir.rglob("*"):
        if sub.is_dir():
            has_image = any(
                f.suffix.lower() in IMAGE_EXTENSIONS
                for f in sub.iterdir()
                if f.is_file()
            )
            if has_image:
                dirs_with_images.append(sub)
    return dirs_with_images


def build_image_index(image_dirs: list) -> dict:
    """
    Returns dict: filename (as it appears on disk, e.g. 'IMG_0001.jpg') -> absolute path (str)
    Also tracks duplicate filenames found in more than one location.
    """
    index = {}
    duplicates = []
    for d in image_dirs:
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                if f.name in index and index[f.name] != str(f.resolve()):
                    duplicates.append((f.name, index[f.name], str(f.resolve())))
                index[f.name] = str(f.resolve())
    return index, duplicates


def try_match(image_name: str, index: dict):
    """
    Skin_Metadata.csv's Image_name may or may not include the file extension.
    Try a direct match first, then try matching by stem against indexed filenames.
    """
    if image_name in index:
        return index[image_name]

    # Try common extensions if Image_name has none
    stem = Path(image_name).stem
    for ext in IMAGE_EXTENSIONS:
        candidate = stem + ext
        if candidate in index:
            return index[candidate]

    return None


def main():
    parser = argparse.ArgumentParser(description="Verify and index the DermaCon-IN dataset")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to the DermaCon-IN folder. Defaults to data/raw/DermaCon-IN",
    )
    args = parser.parse_args()

    project_root = find_project_root()
    dataset_root = Path(args.source).resolve() if args.source else project_root / "data" / "raw" / "DermaCon-IN"

    print(f"Project root:   {project_root}")
    print(f"Dataset source: {dataset_root}")

    if not dataset_root.exists():
        print(f"[FATAL] {dataset_root} does not exist.")
        print("        Copy/move your DermaCon-IN folder there first, e.g.:")
        print(f"        {dataset_root}/DATASET/...")
        print(f"        {dataset_root}/METADATA/...")
        sys.exit(1)

    # 1. Locate metadata
    metadata_dir = find_metadata_dir(dataset_root)
    print(f"[OK] Metadata folder found: {metadata_dir}")

    skin_metadata_path = metadata_dir / "Skin_Metadata.csv"
    train_split_path = metadata_dir / "train_split.csv"
    test_split_path = metadata_dir / "test_split.csv"
    schema_path = metadata_dir / "Metadata_schema.md"

    for p, label in [
        (train_split_path, "train_split.csv"),
        (test_split_path, "test_split.csv"),
    ]:
        if not p.exists():
            print(f"[FATAL] Expected {label} at {p} but it was not found.")
            sys.exit(1)
        print(f"[OK] Found {label}")

    if not schema_path.exists():
        print(f"[WARN] Metadata_schema.md not found at {schema_path} (non-fatal)")

    # 2. Locate image directories
    image_dirs = find_dataset_image_dirs(dataset_root)
    if not image_dirs:
        print(f"[FATAL] No directories containing images were found under {dataset_root / 'DATASET'}")
        sys.exit(1)

    print(f"[OK] Found {len(image_dirs)} directories containing images:")
    for d in image_dirs:
        n = sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
        print(f"     - {d}  ({n} image files)")

    # 3. Build the filename -> path index
    index, duplicates = build_image_index(image_dirs)
    print(f"[OK] Indexed {len(index)} unique image filenames on disk")
    if duplicates:
        print(f"[WARN] {len(duplicates)} filename collisions across directories (see report)")

    # 4. Load Skin_Metadata.csv and check Image_name resolution
    with open(skin_metadata_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if "Image_name" not in (fieldnames or []):
        print(f"[FATAL] Skin_Metadata.csv does not contain an 'Image_name' column.")
        print(f"        Columns found: {fieldnames}")
        sys.exit(1)

    matched = []
    unmatched = []
    for row in rows:
        image_name = row["Image_name"]
        path = try_match(image_name, index)
        if path:
            matched.append((image_name, path))
        else:
            unmatched.append(image_name)

    print(f"[RESULT] {len(matched)} / {len(rows)} Skin_Metadata.csv rows resolved to an image file")
    if unmatched:
        print(f"[WARN] {len(unmatched)} rows could NOT be matched to an image file")

    # 5. Write image_index.csv (only for matched rows, keyed by the metadata's Image_name)
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    index_out_path = processed_dir / "image_index.csv"
    with open(index_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image_name", "path"])
        for image_name, path in matched:
            writer.writerow([image_name, path])
    print(f"[OK] Wrote {index_out_path}")

    # 6. Write a human-readable report
    reports_dir = project_root / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "setup_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DermaCon-IN setup / verification report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset source: {dataset_root}\n")
        f.write(f"Metadata folder: {metadata_dir}\n\n")
        f.write(f"Image directories found ({len(image_dirs)}):\n")
        for d in image_dirs:
            f.write(f"  - {d}\n")
        f.write(f"\nTotal unique filenames indexed on disk: {len(index)}\n")
        f.write(f"Total rows in Skin_Metadata.csv: {len(rows)}\n")
        f.write(f"Matched to an image file: {len(matched)}\n")
        f.write(f"NOT matched (missing image file): {len(unmatched)}\n\n")

        if duplicates:
            f.write(f"Filename collisions across directories ({len(duplicates)}):\n")
            for name, p1, p2 in duplicates:
                f.write(f"  - {name}: kept {p2} (overwrote {p1})\n")
            f.write("\n")

        if unmatched:
            f.write("Unmatched Image_name values (first 200 shown):\n")
            for name in unmatched[:200]:
                f.write(f"  - {name}\n")
            if len(unmatched) > 200:
                f.write(f"  ... and {len(unmatched) - 200} more\n")

    print(f"[OK] Wrote {report_path}")

    print()
    if unmatched:
        pct = 100.0 * len(unmatched) / len(rows)
        if pct > 5.0:
            print(f"[ACTION NEEDED] {pct:.1f}% of metadata rows have no matching image file.")
            print("                This is large enough that something is likely misconfigured.")
            print("                Review the report before proceeding to training.")
        else:
            print(f"[NOTE] {pct:.1f}% of metadata rows have no matching image file.")
            print("       This is a small enough fraction to proceed, but review the report.")
    else:
        print("[SUCCESS] Every row in Skin_Metadata.csv resolved to an image file.")

    print()
    print("Next step:")
    print("    python data/inspect_dataset.py")


if __name__ == "__main__":
    main()
