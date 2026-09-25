# Dermatology Assistant (local prototype)

College project prototype: a locally-trained dermatology vision model, plus
scaffolding for a RAG-style explanation layer using a local Ollama model.
No paid APIs. No cloud LLM calls.

Pipeline:

```
image --> vision model (trained on DermaCon-IN) --> disease predictions
                                                            |
questionnaire  ---------------------------------------------
                                                            |
safety rules (deterministic, not LLM) -----------------------
                                                            |
local knowledge base (JSON, keyword retrieval) --------------
                                                            v
                                            local Ollama model
                                        (explanation + doctor summary)
                                                            v
                                                       Streamlit UI
```

## What's real vs. scaffolded

- **Real, working now (once you provide the dataset):** dataset setup/verification,
  dataset inspection reports, PyTorch Dataset, training loop, checkpointing,
  evaluation, per-Fitzpatrick / per-Monk-Skin-Tone fairness report.
- **Scaffolded, needs your input before real use:**
  - `knowledge_base/conditions/*.json` - every field is marked `PLACEHOLDER`.
    Replace with content from a real, cited medical source.
  - `llm/ollama_client.py` - works once you install Ollama and pull a model
    locally; nothing to configure in code beyond the model name.
  - `web/app.py` - functional Streamlit app, but its explanation quality is
    only as good as the (currently placeholder) knowledge base.

## 1. Put the dataset here

Copy (or cut) your `DermaCon-IN` folder so it looks like this:

```
dermatology_assistant/data/raw/DermaCon-IN/
├── DATASET/
│   ├── DATASET_0/DATASET_0/*.jpg
│   └── DATASET_1/DATASET_1/*.jpg
└── METADATA/
    ├── Skin_Metadata.csv
    ├── train_split.csv
    ├── test_split.csv
    └── Metadata_schema.md
```

The `checkpoints/` folder that ships with DermaCon-IN (the `.pth` files) is
not used anywhere in this project - you can leave it out or ignore it.

Do not delete your original DermaCon-IN copy until `scripts/setup_dermacon.py`
confirms everything resolved correctly (see below).

## 2. Environment setup

```bash
cd dermatology_assistant
python -m venv venv

# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

If you have an NVIDIA GPU and want CUDA acceleration, install the matching
PyTorch build from https://pytorch.org/get-started/locally/ instead of the
default CPU wheel that `pip install -r requirements.txt` pulls in.

## 3. Verify and index the dataset

```bash
python scripts/setup_dermacon.py
```

This finds your images and metadata, checks that `Image_name` values in
`Skin_Metadata.csv` resolve to real files, and writes:
- `data/processed/image_index.csv`
- `data/reports/setup_report.txt`

It does **not** move, copy, or delete your image files.

Read the printed summary. If a large fraction of rows fail to match, stop
and fix the dataset placement before continuing.

## 4. Inspect the dataset

```bash
python data/inspect_dataset.py
```

Produces, in `data/reports/`:
- `inspection_report.txt` (full summary)
- `images_per_disease.csv`, `images_per_subclass.csv`, `images_per_mainclass.csv`
- `fitzpatrick_distribution.csv`, `monk_skin_tone_distribution.csv`
- `disease_x_fitzpatrick.csv`
- `rare_classes.csv`

This also checks for subject leakage between the supplied train/test split
and for disease classes that appear in test but never in train.

**Read this report before training.** It tells you what you're actually
working with.

## 5. Train

```bash
python training/train.py --epochs 10
```

Common overrides:

```bash
python training/train.py --epochs 20 --batch-size 16 --model efficientnet_b0 --device cpu
python training/train.py --resume            # continue from models/checkpoints/last_model.pth
python training/train.py --freeze-backbone   # only train the classifier head
```

Outputs in `models/checkpoints/`:
- `class_mapping.json` - Disease_label -> integer index (built from train split)
- `best_model.pth`, `last_model.pth`
- `training_history.json`
- `classification_report.txt`, `confusion_matrix.png` / `.npy`

## 6. Evaluate

```bash
python evaluation/evaluate.py
python evaluation/fairness_report.py
```

`fairness_report.py` gives you per-Fitzpatrick-type and per-Monk-Skin-Tone
accuracy, with a minimum sample-size cutoff before it reports a number. It
does not claim the model is fair - it reports what it measured.

## 7. (Optional) Local LLM explanation layer

Install Ollama (https://ollama.com), then:

```bash
ollama pull llama3.1     # or a smaller model like `phi3` if your laptop is limited
ollama serve              # if it isn't already running as a service
```

## 8. Run the Streamlit app

```bash
streamlit run web/app.py
```

---

## Known gaps you need to close yourself

1. **Knowledge base content** (`knowledge_base/conditions/*.json`) is
   placeholder only. Fill it in from a real dermatology reference and
   record the source in `knowledge_base/sources.json`.
2. **Only one dataset is wired in** (DermaCon-IN), by design, per project
   scope. Adding more datasets means writing an additional loader and
   deciding how to reconcile label schemas - not done here.
3. **No de-identification / consent pipeline** for any future data
   collection - out of scope for this prototype.
