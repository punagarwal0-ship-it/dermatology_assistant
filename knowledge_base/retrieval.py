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

# knowledge_base/retrieval.py

import json
import os
from pathlib import Path
from typing import List, Dict

class KnowledgeBaseRetriever:
    def __init__(self, kb_root: str = "knowledge_base/conditions"):
        """
        Initialize knowledge base from JSON files.
        
        Args:
            kb_root: Path to folder containing condition JSONs
        """
        self.kb_root = Path(kb_root)
        self.conditions = {}
        self._load_kb()
    
    def _load_kb(self):
        """Load all JSON files from kb_root into memory."""
        if not self.kb_root.exists():
            print(f"⚠️ KB folder not found: {self.kb_root}")
            return
        
        json_files = list(self.kb_root.glob("*.json"))
        print(f"🔄 Loading {len(json_files)} KB entries...")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Skip placeholder template
                if json_file.name == "_TEMPLATE.json":
                    continue
                
                # Skip entries marked as placeholder
                if data.get("_PLACEHOLDER", True):
                    continue
                
                # Get condition name (required field)
                condition_name = data.get("condition_name", "")
                if not condition_name:
                    print(f"  ⚠️ Skipping (no condition_name): {json_file.name}")
                    continue
                
                # Use condition_name as key
                key = condition_name.lower().replace(" ", "_")
                self.conditions[key] = data
                print(f"  ✅ Loaded: {condition_name}")
            
            except json.JSONDecodeError as e:
                print(f"  ❌ JSON error in {json_file.name}: {e}")
            except Exception as e:
                print(f"  ❌ Error in {json_file.name}: {e}")
        
        print(f"✅ Total conditions loaded: {len(self.conditions)}\n")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve matching conditions. Handles various naming formats.
        
        Args:
            query: Disease label from vision model (e.g., "Psoriasis", "Infected Eczema")
            top_k: Number of results to return
        
        Returns:
            List of matching conditions
        """
        if not self.conditions:
            print(f"⚠️ No conditions loaded in KB")
            return []
        
        # Normalize query: lowercase, replace spaces with underscores
        query_normalized = query.lower().replace(" ", "_")
        results = []
        
        # Try exact match first (fastest)
        if query_normalized in self.conditions:
            return [{"condition": self.conditions[query_normalized], "score": 100}]
        
        # Try substring matching on normalized names
        for key, condition in self.conditions.items():
            condition_name = condition.get("condition_name", "").lower().replace(" ", "_")
            
            # Full match
            if query_normalized == condition_name:
                results.append({"condition": condition, "score": 100})
            # Substring match (query is part of condition name)
            elif query_normalized in condition_name:
                results.append({"condition": condition, "score": 80})
            # Reverse substring match (condition name is part of query)
            elif condition_name in query_normalized:
                results.append({"condition": condition, "score": 70})
        
        # Remove duplicates and sort by score
        seen = set()
        unique_results = []
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            key = r["condition"].get("condition_name", "")
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        
        return unique_results[:top_k]
    
    def get_condition_by_name(self, condition_name: str) -> Dict:
        """
        Direct lookup by condition name.
        
        Args:
            condition_name: Disease name (e.g., "eczema_atopic")
        
        Returns:
            Condition dictionary or empty dict if not found
        """
        key = condition_name.lower().replace(" ", "_")
        return self.conditions.get(key, {})


# Test
if __name__ == "__main__":
    retriever = KnowledgeBaseRetriever()
    print(f"Loaded {len(retriever.conditions)} conditions") content).")
