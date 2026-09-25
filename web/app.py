"""
web/app.py

Streamlit scaffold tying together:
  image upload -> vision model inference -> questionnaire -> safety rules
  -> knowledge base retrieval -> local Ollama explanation -> doctor summary

Run:
    streamlit run web/app.py

Requires:
  - models/checkpoints/best_model.pth to exist (train first) for real predictions
  - Ollama running locally for the explanation step (optional - app still
    shows vision predictions + safety flags + KB context without it)

This is intentionally a straightforward, single-file Streamlit app -
no extra web framework.
"""

import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import load_class_mapping
from data.transforms import get_eval_transforms
from models.vision_model import build_model, get_device
from knowledge_base.retrieval import KnowledgeBaseRetriever
from knowledge_base.formatter import format_kb_results_for_prompt
from questionnaire.questions import QUESTIONS, validate_answers
from safety.emergency_rules import check_emergency_flags, get_general_emergency_guidance
from llm.ollama_client import OllamaClient
from llm.prompt_engine import build_prompt


st.set_page_config(page_title="Dermatology Assistant (local prototype)", layout="wide")


@st.cache_resource
def load_model():
    ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "best_model.pth"
    if not ckpt_path.exists():
        return None, None, None

    device = get_device("auto")
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    model_cfg = config.get("model", {"name": "efficientnet_b0"})
    image_size = config.get("training", {}).get("image_size", 224)

    class_mapping = load_class_mapping()
    idx_to_label = {v: k for k, v in class_mapping.items()}

    model = build_model(
        model_name=model_cfg.get("name", "efficientnet_b0"),
        num_classes=len(class_mapping),
        pretrained=False,
        freeze_backbone=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, idx_to_label, (device, image_size)


@st.cache_resource
def load_kb():
    return KnowledgeBaseRetriever()


def run_inference(model, idx_to_label, device_and_size, pil_image, top_k=5):
    device, image_size = device_and_size
    transform = get_eval_transforms(image_size)

    import numpy as np
    img_array = np.array(pil_image.convert("RGB"))
    tensor = transform(image=img_array)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, k=min(top_k, probs.shape[0]))
    results = []
    for p, i in zip(top_probs.cpu().numpy(), top_idxs.cpu().numpy()):
        label = idx_to_label.get(int(i), str(i))
        results.append({"disease_label": label, "confidence": float(p)})
    return results


def main():
    st.title("Dermatology Assistant - local prototype")
    st.caption(
        "College project prototype. Not a medical device. Does not provide a diagnosis. "
        "Vision predictions come from a locally trained classifier; explanations (if enabled) "
        "come from a local Ollama model. Knowledge base content in this prototype is placeholder "
        "and NOT medically verified - see knowledge_base/conditions/*.json."
    )

    model, idx_to_label, device_and_size = load_model()
    kb = load_kb()

    if model is None:
        st.warning(
            "No trained model checkpoint found at models/checkpoints/best_model.pth. "
            "Run training/train.py first. You can still fill out the questionnaire below "
            "to see the safety-flag and knowledge-base scaffolding working."
        )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Upload an image")
        uploaded_file = st.file_uploader("Skin image", type=["jpg", "jpeg", "png"])
        pil_image = None
        vision_results = []
        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file)
            st.image(pil_image, caption="Uploaded image", use_column_width=True)
            if model is not None:
                vision_results = run_inference(model, idx_to_label, device_and_size, pil_image)
                st.write("**Vision model predictions:**")
                for r in vision_results:
                    st.write(f"- {r['disease_label']}: {r['confidence']:.1%}")

        st.subheader("2. Questionnaire")
        answers = {}
        free_text = st.text_area("Anything else you want to describe about the area?", "")
        for q in QUESTIONS:
            if q["type"] == "boolean":
                answers[q["id"]] = st.checkbox(q["text"], key=q["id"])
            elif q["type"] == "number":
                answers[q["id"]] = st.number_input(q["text"], min_value=0, max_value=120, step=1, key=q["id"])
            elif q["type"] == "single_choice":
                answers[q["id"]] = st.selectbox(q["text"], q["options"], key=q["id"])
            else:
                answers[q["id"]] = st.text_input(q["text"], key=q["id"])

    with col2:
        st.subheader("3. Safety check")
        missing = validate_answers(answers)
        if missing:
            st.info(f"Optional: fill in these fields for a more complete assessment: {', '.join(missing)}")

        flags = check_emergency_flags(answers, free_text)
        if flags:
            st.error("**Urgent care may be needed:**")
            for f in flags:
                st.error(f"- {f}")
        else:
            st.success("No deterministic emergency flags triggered by your answers.")

        with st.expander("Always seek urgent care if you notice any of the following"):
            for item in get_general_emergency_guidance():
                st.write(f"- {item}")

        st.subheader("4. Knowledge base context (placeholder content)")
        if pil_image is not None or free_text:
            query = free_text or (vision_results[0]["disease_label"] if vision_results else "")
            if query:
                kb_results = kb.retrieve(query, top_k=3)
                skin_hint = ""
                kb_text = format_kb_results_for_prompt(kb_results, skin_tone_hint=skin_hint)
                st.text(kb_text)
            else:
                st.write("No query available yet - upload an image or describe the area above.")

        st.subheader("5. Local LLM explanation (Ollama)")
        st.caption(
            "Requires Ollama running locally (`ollama serve`) with a model pulled "
            "(see llm/ollama_client.py). This step is optional."
        )
        if st.button("Generate explanation with local Ollama model"):
            client = OllamaClient()
            if not client.is_available():
                st.error(
                    "Could not reach a local Ollama server at http://localhost:11434. "
                    "Install Ollama, run `ollama serve`, and pull a model (e.g. `ollama pull llama3.1`)."
                )
            elif not vision_results:
                st.warning("Upload an image and get vision predictions first.")
            else:
                query = vision_results[0]["disease_label"]
                kb_results = kb.retrieve(query, top_k=3)
                kb_text = format_kb_results_for_prompt(kb_results)
                prompt = build_prompt(vision_results, answers, kb_text, flags)
                with st.spinner("Generating explanation locally..."):
                    try:
                        response = client.generate(prompt)
                        st.write(response)
                    except Exception as e:
                        st.error(f"Ollama request failed: {e}")


if __name__ == "__main__":
    main()
