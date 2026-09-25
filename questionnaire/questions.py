"""
questionnaire/questions.py

Static definition of the patient questionnaire. Kept as plain Python data
(not a framework) so it's trivial to render in Streamlit or elsewhere.

Each question has:
    id       - stable key used in answers dict
    text     - question shown to the user
    type     - "text" | "number" | "boolean" | "single_choice" | "multi_choice"
    options  - for single_choice / multi_choice
    required - bool
"""

QUESTIONS = [
    {"id": "age", "text": "What is your age?", "type": "number", "required": True},
    {"id": "sex", "text": "Sex", "type": "single_choice", "options": ["Male", "Female", "Other", "Prefer not to say"], "required": False},
    {"id": "duration", "text": "How long have you had this?", "type": "text", "required": True},
    {"id": "itching", "text": "Is it itchy?", "type": "boolean", "required": True},
    {"id": "pain", "text": "Is it painful?", "type": "boolean", "required": True},
    {"id": "fever", "text": "Do you currently have a fever?", "type": "boolean", "required": True},
    {"id": "spreading", "text": "Has it spread since you first noticed it?", "type": "boolean", "required": True},
    {"id": "new_medications", "text": "Have you started any new medications recently?", "type": "text", "required": False},
    {"id": "known_allergies", "text": "Do you have any known allergies?", "type": "text", "required": False},
    {"id": "recent_travel", "text": "Have you traveled recently?", "type": "text", "required": False},
    {"id": "insect_bite", "text": "Could this be related to an insect bite?", "type": "boolean", "required": False},
    {"id": "sun_exposure", "text": "Was the area recently exposed to a lot of sun?", "type": "boolean", "required": False},
    {"id": "family_history", "text": "Does anyone in your family have a similar skin condition?", "type": "text", "required": False},
    {"id": "chronic_illness", "text": "Do you have any chronic illnesses (e.g. diabetes, autoimmune conditions)?", "type": "text", "required": False},
    {"id": "pregnancy", "text": "Are you currently pregnant?", "type": "boolean", "required": False},
    {"id": "immune_disorder", "text": "Do you have any known immune system disorders, or are you on immunosuppressive treatment?", "type": "boolean", "required": False},
    {"id": "new_products", "text": "Have you used any new soaps, cosmetics, or skincare products recently?", "type": "text", "required": False},
]


def get_required_questions():
    return [q for q in QUESTIONS if q["required"]]


def validate_answers(answers: dict) -> list:
    """Returns a list of missing required question ids, empty list if all present."""
    missing = []
    for q in get_required_questions():
        if q["id"] not in answers or answers[q["id"]] in (None, ""):
            missing.append(q["id"])
    return missing
