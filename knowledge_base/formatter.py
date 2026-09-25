"""
knowledge_base/formatter.py

Formats retrieval results (from knowledge_base/retrieval.py) into a plain
text block suitable for inserting into a local LLM prompt (see
llm/prompt_engine.py).
"""

from typing import List, Dict


def format_kb_results_for_prompt(results: List[Dict], skin_tone_hint: str = None) -> str:
    """
    results: output of KnowledgeBaseRetriever.retrieve()
    skin_tone_hint: e.g. "dark" / "light" / a Fitzpatrick or Monk value,
                    used to pick which appearance description to surface first.
    """
    if not results:
        return "No matching entries found in the local knowledge base."

    blocks = []
    for r in results:
        entry = r["entry"]
        name = entry.get("condition_name", "Unknown condition")
        relevance = r.get("relevance", 0.0)

        appearance = entry.get("appearance", {})
        general = appearance.get("general", "")
        light = appearance.get("light_skin", "")
        dark = appearance.get("dark_skin", "")

        appearance_lines = [f"  General appearance: {general}"] if general else []
        if skin_tone_hint and "dark" in skin_tone_hint.lower():
            if dark:
                appearance_lines.append(f"  Appearance on darker skin: {dark}")
            if light:
                appearance_lines.append(f"  Appearance on lighter skin (for contrast): {light}")
        else:
            if light:
                appearance_lines.append(f"  Appearance on lighter skin: {light}")
            if dark:
                appearance_lines.append(f"  Appearance on darker skin: {dark}")

        block = [
            f"- {name} (match score: {relevance:.2f})",
            f"  Description: {entry.get('description', '')}",
        ] + appearance_lines + [
            f"  Symptoms: {', '.join(entry.get('symptoms', []))}",
            f"  Common lookalikes: {', '.join(entry.get('common_lookalikes', []))}",
            f"  Warning signs: {', '.join(entry.get('warning_signs', []))}",
        ]

        if entry.get("_PLACEHOLDER"):
            block.append("  [NOTE: this entry is placeholder content and has not been medically verified]")

        blocks.append("\n".join(block))

    return "\n\n".join(blocks)
