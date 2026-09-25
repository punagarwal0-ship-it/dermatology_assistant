"""
llm/prompt_engine.py

Builds the prompt sent to the local Ollama model, combining:
  - vision model output (disease predictions + confidence)
  - patient questionnaire answers
  - retrieved knowledge base context (see knowledge_base/)
  - any deterministic safety flags raised (see safety/emergency_rules.py)

The LLM is used ONLY to explain, contextualize, ask follow-up questions,
and draft a doctor-visit summary. It must never be the source of the
disease probability numbers themselves - those come from the vision
model. This module keeps that separation explicit.
"""

from typing import Dict, List


SYSTEM_PREAMBLE = """You are a dermatology patient-education assistant running locally.
You are NOT a doctor and must not present any of this as a confirmed diagnosis.
Use the vision model's predictions and the knowledge base context below as your
evidence. Do not invent medical facts that are not supported by the provided context.
If the safety flags below indicate an emergency, your first priority is to tell the
user to seek urgent in-person medical care - do not attempt to reassure them out of it.
"""


def build_prompt(
    vision_predictions: List[Dict],
    questionnaire_answers: Dict,
    kb_context_text: str,
    safety_flags: List[str],
) -> str:
    """
    vision_predictions: list of {"disease_label": str, "confidence": float}, sorted desc
    questionnaire_answers: dict of question_id -> answer (see questionnaire/questions.py)
    kb_context_text: output of knowledge_base.formatter.format_kb_results_for_prompt()
    safety_flags: output of safety.emergency_rules.check_emergency_flags()
    """
    lines = [SYSTEM_PREAMBLE, ""]

    if safety_flags:
        lines.append("SAFETY FLAGS RAISED (deterministic rules, not from the LLM):")
        for flag in safety_flags:
            lines.append(f"  - {flag}")
        lines.append("")

    lines.append("VISION MODEL PREDICTIONS (ranked, from the trained classifier):")
    for pred in vision_predictions:
        lines.append(f"  - {pred['disease_label']}: {pred['confidence']:.1%}")
    lines.append("")

    lines.append("PATIENT QUESTIONNAIRE ANSWERS:")
    for question_id, answer in questionnaire_answers.items():
        lines.append(f"  - {question_id}: {answer}")
    lines.append("")

    lines.append("KNOWLEDGE BASE CONTEXT (local, possibly placeholder content - see notes):")
    lines.append(kb_context_text)
    lines.append("")

    lines.append("Now produce:")
    lines.append("1. A short plain-language explanation of what the top predictions mean")
    lines.append("2. General, safe first-aid guidance only (no prescription-strength advice)")
    lines.append("3. A restatement of any emergency warning signs the patient should watch for")
    lines.append("4. A concise doctor-visit summary the patient could show a clinician")
    lines.append("5. 2-3 follow-up questions that would help narrow things down further")

    return "\n".join(lines)
