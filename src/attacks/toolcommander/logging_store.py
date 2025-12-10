# src/attacks/toolcommander/logging_store.py
from __future__ import annotations

from typing import List
from dataclasses import dataclass, field


@dataclass
class QueryLogStore:
    """Simple in-memory store for queries stolen by manipulator tools."""
    queries: List[str] = field(default_factory=list)
    max_tokens: int = 512

    def log(self, query: str) -> None:
        if not query:
            return
        # Very rough length cap
        if len(query) > self.max_tokens:
            query = query[: self.max_tokens] + "..."
        self.queries.append(query)

    def unique_queries(self) -> List[str]:
        # Keep order, deduplicate
        seen = set()
        out = []
        for q in self.queries:
            if q not in seen:
                seen.add(q)
                out.append(q)
        return out