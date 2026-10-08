# knowledge_base/formatter.py

from typing import List, Dict

def format_kb_results_for_prompt(kb_results: List[Dict], skin_tone_hint: str = "") -> str:
    """
    Format retrieved KB results into readable text for LLM.
    
    Args:
        kb_results: List of dicts with "condition" key (from retriever.retrieve())
        skin_tone_hint: User's skin tone ("light", "dark", etc.) for appearance selection
    
    Returns:
        Formatted string to include in LLM prompt
    """
    if not kb_results:
        return "[No medical context found in knowledge base]"
    
    formatted_text = ""
    
    for idx, result in enumerate(kb_results, 1):
        condition = result.get("condition", {})
        
        if not condition:
            continue
        
        # Extract fields (with fallbacks)
        condition_name = condition.get("condition_name", "Unknown")
        description = condition.get("description", "")
        symptoms = condition.get("symptoms", [])
        appearance = condition.get("appearance", {})
        first_aid = condition.get("general_first_aid", [])
        warnings = condition.get("warning_signs", [])
        sources = condition.get("sources", [])
        
        # Choose skin-tone-specific appearance
        appearance_text = ""
        if skin_tone_hint and skin_tone_hint.lower() in ["dark", "darker"]:
            appearance_text = appearance.get("dark_skin", appearance.get("light_skin", ""))
        else:
            appearance_text = appearance.get("light_skin", "")
        
        # Build formatted section
        formatted_text += f"\n{'='*60}\n"
        formatted_text += f"CONDITION {idx}: {condition_name}\n"
        formatted_text += f"{'='*60}\n"
        
        if description:
            formatted_text += f"DESCRIPTION:\n{description}\n\n"
        
        if symptoms:
            formatted_text += f"COMMON SYMPTOMS:\n"
            for sym in symptoms:
                formatted_text += f"  • {sym}\n"
            formatted_text += "\n"
        
        if appearance_text:
            formatted_text += f"APPEARANCE:\n{appearance_text}\n\n"
        
        if first_aid:
            formatted_text += f"GENERAL FIRST AID:\n"
            for aid in first_aid:
                formatted_text += f"  • {aid}\n"
            formatted_text += "\n"
        
        if warnings:
            formatted_text += f"⚠️ WARNING SIGNS (SEEK URGENT CARE):\n"
            for warn in warnings:
                formatted_text += f"  • {warn}\n"
            formatted_text += "\n"
        
        if sources:
            formatted_text += f"SOURCES: {', '.join(sources)}\n"
    
    return formatted_text


# Test formatter
if __name__ == "__main__":
    from knowledge_base.retrieval import KnowledgeBaseRetriever
    
    retriever = KnowledgeBaseRetriever()
    results = retriever.retrieve("eczema", top_k=1)
    
    formatted = format_kb_results_for_prompt(results, skin_tone_hint="dark")
    print(formatted)
