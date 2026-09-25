"""
safety/emergency_rules.py

Deterministic, rule-based emergency detection. Deliberately kept SEPARATE
from the LLM (see llm/prompt_engine.py) - these checks run first and are
never generated or altered by the language model. If any flag fires, the
UI (web/app.py) must surface an urgent-care recommendation regardless of
what the vision model or LLM says.

This is intentionally simple keyword/boolean matching, not a medical
device. It is a coarse safety net, not a diagnostic system.
"""

from typing import Dict, List

# Keywords in free-text answers that should trigger an emergency flag.
# Kept lowercase for case-insensitive matching.
EMERGENCY_KEYWORDS = {
    "difficulty breathing": ["difficulty breathing", "can't breathe", "cant breathe", "shortness of breath", "trouble breathing"],
    "facial swelling": ["face swelling", "facial swelling", "lips swelling", "throat swelling", "tongue swelling"],
    "rapidly spreading rash": ["spreading fast", "spreading rapidly", "spreading within hours", "getting worse fast"],
    "black or dying skin": ["black skin", "dying skin", "skin turning black", "necrosis", "necrotic"],
    "severe burns": ["severe burn", "third degree burn", "3rd degree burn", "burn covering"],
}


def check_emergency_flags(questionnaire_answers: Dict, free_text_description: str = "") -> List[str]:
    """
    questionnaire_answers: dict from questionnaire/questions.py answers
    free_text_description: any additional free-text the patient typed

    Returns a list of human-readable emergency flag strings. Empty list
    means no deterministic emergency rule fired (this does NOT mean the
    situation is safe - it only means these specific rules did not match).
    """
    flags = []

    combined_text = " ".join([
        str(free_text_description or ""),
        str(questionnaire_answers.get("new_medications", "")),
        str(questionnaire_answers.get("family_history", "")),
    ]).lower()

    for flag_name, keywords in EMERGENCY_KEYWORDS.items():
        if any(kw in combined_text for kw in keywords):
            flags.append(f"Possible {flag_name} mentioned in patient description - seek urgent care.")

    # Boolean/structured answer based rules
    if questionnaire_answers.get("fever") is True and questionnaire_answers.get("spreading") is True:
        flags.append("Fever combined with a rapidly spreading rash - seek urgent medical evaluation.")

    if questionnaire_answers.get("immune_disorder") is True and questionnaire_answers.get("fever") is True:
        flags.append("Fever in a patient with an immune disorder / immunosuppression - seek prompt medical evaluation.")

    return flags


def get_general_emergency_guidance() -> List[str]:
    """Static list of conditions that always warrant urgent in-person care, for display in the UI regardless of flags."""
    return [
        "Difficulty breathing or swelling of the face, lips, or throat",
        "Rapidly spreading rash, especially with fever",
        "High fever together with a new or worsening rash",
        "Skin that appears black, dying, or is spreading necrosis",
        "Severe burns, especially covering a large area or on the face/hands/genitals",
        "Signs of a severe allergic reaction (hives with breathing difficulty, dizziness, throat tightness)",
    ]


if __name__ == "__main__":
    demo_answers = {"fever": True, "spreading": True}
    flags = check_emergency_flags(demo_answers, "the rash is spreading fast and I have difficulty breathing")
    for f in flags:
        print(f"[FLAG] {f}")
