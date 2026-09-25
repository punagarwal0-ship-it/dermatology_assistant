"""
knowledge_base/retrieval.py

Simple keyword-overlap retrieval over the local JSON condition files in
knowledge_base/conditions/. No FAISS, no LangChain, no embeddings yet -
this is intentionally the simplest thing that works, structured behind
a small interface (KnowledgeBaseRetriever) so an embedding-based backend
can be substituted later without changing callers (llm/prompt_engine.py).

Usage:
    from knowledge_base.retrieval import KnowledgeBaseRetriever
    kb = KnowledgeBaseRetriever()
    results = kb.retrieve("itchy red patches on arm", top_k=3)
"""

import json
import re
from pathlib import Path
from typing import List, Dict

CONDITIONS_DIR = Path(__file__).resolve().parent / "conditions"

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "on", "in", "with", "for", "to",
    "is", "are", "was", "were", "it", "this", "that", "my", "me", "i",
}


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def _load_all_conditions() -> List[Dict]:
    entries = []
    for path in sorted(CONDITIONS_DIR.glob("*.json")):
        if path.name.startswith("_TEMPLATE"):
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["_source_file"] = path.name
        entries.append(data)
    return entries


def _entry_text_blob(entry: Dict) -> str:
    """Flatten the searchable text fields of a condition entry into one string."""
    parts = [
        entry.get("condition_name", ""),
        entry.get("description", ""),
        " ".join(entry.get("symptoms", [])),
        " ".join(entry.get("causes", [])),
        " ".join(entry.get("risk_factors", [])),
        entry.get("appearance", {}).get("general", ""),
        entry.get("appearance", {}).get("light_skin", ""),
        entry.get("appearance", {}).get("dark_skin", ""),
        " ".join(entry.get("common_lookalikes", [])),
    ]
    return " ".join(parts)


class KnowledgeBaseRetriever:
    def __init__(self):
        self.entries = _load_all_conditions()
        self._blobs = [_tokenize(_entry_text_blob(e)) for e in self.entries]

    def reload(self):
        self.__init__()

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Returns up to top_k condition entries, ranked by simple token
        overlap with the query. Each result includes a 'relevance' score
        in [0, 1] (fraction of query tokens matched) and the full entry.

        This is a placeholder ranking method. To upgrade to embeddings later,
        implement a new class with the same retrieve() signature (e.g. using
        sentence-transformers + FAISS) and swap it in at the call site.
        """
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []

        scored = []
        for entry, blob_tokens in zip(self.entries, self._blobs):
            blob_set = set(blob_tokens)
            overlap = query_tokens & blob_set
            score = len(overlap) / len(query_tokens) if query_tokens else 0.0
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, entry in scored[:top_k]:
            results.append({
                "relevance": round(score, 3),
                "entry": entry,
            })
        return results

    def get_by_name(self, condition_name: str) -> Dict:
        for entry in self.entries:
            if entry.get("condition_name", "").lower() == condition_name.lower():
                return entry
        return None


if __name__ == "__main__":
    kb = KnowledgeBaseRetriever()
    print(f"Loaded {len(kb.entries)} condition entries (placeholder content).")
    demo_results = kb.retrieve("itchy red patches", top_k=3)
    for r in demo_results:
        print(f"  {r['relevance']:.2f}  {r['entry'].get('condition_name')}  [{r['entry'].get('_source_file')}]")
