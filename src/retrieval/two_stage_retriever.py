# src/retrieval/two_stage_retriever.py
from __future__ import annotations

from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever

import os

TOOL_TOP_K = int(os.getenv("TOOL_TOP_K", "8"))
TOOL_STAGE1_K = int(os.getenv("TOOL_STAGE1_K", "256"))
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


class ToolRetriever2Stage:
    """
    Stage 1: BM25 over all tool schemas (fast, lexical).
    Stage 2: Ollama embeddings over the BM25 candidates (semantic rerank).
    Returns the top-k tool schemas for a given query.
    """

    def __init__(
        self,
        tool_schemas: List[Dict[str, Any]],
        embedding_model: str = OLLAMA_EMBED_MODEL,
        stage1_k: int = TOOL_STAGE1_K,
    ):
        if not tool_schemas:
            raise ValueError("ToolRetriever2Stage requires a non-empty list of tool_schemas")

        self.tool_schemas = tool_schemas
        self.embedding_model_name = embedding_model
        self.stage1_k = max(1, stage1_k)

        # Build BM25 corpus
        docs: List[Document] = []

        for t in tool_schemas:
            fn = t.get("function", {})
            name = fn.get("name", t.get("name", "unknown_tool"))
            description = fn.get("description", "") or t.get("description", "")

            desc_short = description[:256]
            text = f"{name}: {desc_short}" if desc_short else name

            docs.append(
                Document(
                    page_content=text,
                    metadata={"name": name, "schema": t},
                )
            )

        print(f"[INFO] Building BM25 retriever over {len(docs)} tools")
        self.bm25 = BM25Retriever.from_documents(docs)
        self.bm25.k = self.stage1_k

        print(f"[INFO] Initializing Ollama embeddings '{self.embedding_model_name}' for stage 2 reranking")
        self.embedding = OllamaEmbeddings(model=self.embedding_model_name)

    def get_top_k_tools(self, query: str, k: int = TOOL_TOP_K) -> List[Dict[str, Any]]:
        if k <= 0:
            return []

        # Stage 1: BM25
        candidates = self.bm25.invoke(query)  # List[Document]

        if not candidates:
            return []

        # Stage 2: embeddings
        candidate_texts = [d.page_content for d in candidates]
        candidate_schemas = [d.metadata["schema"] for d in candidates]

        query_emb = self.embedding.embed_query(query)
        doc_embs = self.embedding.embed_documents(candidate_texts)

        def cosine(u, v):
            num = sum(a * b for a, b in zip(u, v))
            norm_u = sum(a * a for a in u) ** 0.5
            norm_v = sum(b * b for b in v) ** 0.5
            if norm_u == 0 or norm_v == 0:
                return 0.0
            return num / (norm_u * norm_v)

        scored = [(idx, cosine(query_emb, emb)) for idx, emb in enumerate(doc_embs)]
        scored.sort(key=lambda x: x[1], reverse=True)

        top_indices = [idx for idx, _ in scored[:k]]
        selected = [candidate_schemas[i] for i in top_indices]

        # Deduplicate by name
        seen = set()
        unique_tools = []
        for t in selected:
            fn = t.get("function", {})
            name = fn.get("name", t.get("name", "unknown_tool"))
            if name not in seen:
                seen.add(name)
                unique_tools.append(t)

        return unique_tools