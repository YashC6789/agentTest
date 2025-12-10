from __future__ import annotations

import os
import json
from typing import List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever

# Local Imports
from src.agent.setup import build_tool_agent
from src.config.prompts import SYSTEM_PROMPT_TEXT

from toolbench import api_json_to_openai_json, standardize

# ------------------------ CONFIG ------------------------ #

# Adjust if your tools live elsewhere
TOOL_ROOT_DIR = os.path.join("src", "data", "toolbench", "tools")

# How many tools to pass to the main agent per query (final stage)
TOOL_TOP_K = int(os.getenv("TOOL_TOP_K", "8"))

# How many tools BM25 should keep as candidates in stage 1
TOOL_STAGE1_K = int(os.getenv("TOOL_STAGE1_K", "256"))

# Ollama embedding model (must be available in your local Ollama server)
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


# ------------------------ TOOL LOADING ------------------------ #

def load_all_tool_schemas(
    tool_root_dir: str,
    max_apis_per_tool: int = 99,  # effectively "all"
) -> List[Dict[str, Any]]:
    """
    Walks the ToolBench tools dir, reads JSONs, and uses api_json_to_openai_json
    to build OpenAI-style tool schemas usable with a tool-calling agent.

    Handles both:
      - ToolCommander-style flattened APIs with 'api_name', 'api_description'
      - Raw ToolBench-style APIs with 'name', 'description'
    """
    tool_schemas: List[Dict[str, Any]] = []

    for category in os.listdir(tool_root_dir):
        category_path = os.path.join(tool_root_dir, category)
        if not os.path.isdir(category_path):
            continue

        for fname in os.listdir(category_path):
            if not fname.endswith(".json"):
                continue

            json_path = os.path.join(category_path, fname)
            with open(json_path, "r", encoding="utf-8") as f:
                tool_def = json.load(f)

            if "tool_name" not in tool_def:
                print(f"[WARN] Skipping {json_path}: no 'tool_name' field")
                continue

            standard_tool_name = standardize(tool_def["tool_name"])

            for raw_api in tool_def.get("api_list", [])[:max_apis_per_tool]:
                if "api_name" in raw_api:
                    api_json = raw_api
                else:
                    api_json = {
                        "api_name": raw_api.get("name", ""),
                        "api_description": raw_api.get("description", ""),
                        "required_parameters": raw_api.get("required_parameters", []),
                        "optional_parameters": raw_api.get("optional_parameters", []),
                        "category_name": tool_def.get("category_name", category),
                    }

                if not api_json["api_name"]:
                    print(f"[WARN] Skipping API with no name in file: {json_path}")
                    continue

                tool_schema, category_name, pure_api_name = api_json_to_openai_json(
                    api_json, standard_tool_name
                )
                tool_schemas.append(tool_schema)

    return tool_schemas


def load_filtered_tool_schemas(tool_root_dir: str, predicate):
    all_tools = load_all_tool_schemas(tool_root_dir)
    return [t for t in all_tools if predicate(t)]


def load_some_tool_schemas(
    tool_root_dir: str,
    max_tools: int = 5,
    max_apis_per_tool: int = 3,
) -> List[Dict[str, Any]]:
    """
    Legacy helper: returns an arbitrary subset of tools.
    Kept for compatibility but not used in the new retriever-based pipeline.
    """
    tool_schemas: List[Dict[str, Any]] = []

    for category in os.listdir(tool_root_dir):
        category_path = os.path.join(tool_root_dir, category)
        if not os.path.isdir(category_path):
            continue

        for fname in os.listdir(category_path):
            if not fname.endswith(".json"):
                continue

            json_path = os.path.join(category_path, fname)
            with open(json_path, "r", encoding="utf-8") as f:
                tool_def = json.load(f)

            if "tool_name" not in tool_def:
                print(f"[WARN] Skipping {json_path}: no 'tool_name' field")
                continue

            standard_tool_name = standardize(tool_def["tool_name"])

            for raw_api in tool_def.get("api_list", [])[:max_apis_per_tool]:
                if "api_name" in raw_api:
                    api_json = raw_api
                else:
                    api_json = {
                        "api_name": raw_api.get("name", ""),
                        "api_description": raw_api.get("description", ""),
                        "required_parameters": raw_api.get("required_parameters", []),
                        "optional_parameters": raw_api.get("optional_parameters", []),
                        "category_name": tool_def.get("category_name", category),
                    }

                if not api_json["api_name"]:
                    print(f"[WARN] Skipping API with no name in file: {json_path}")
                    continue

                tool_schema, category_name, pure_api_name = api_json_to_openai_json(
                    api_json, standard_tool_name
                )
                tool_schemas.append(tool_schema)

                if len(tool_schemas) >= max_tools:
                    return tool_schemas

    return tool_schemas


# ------------------------ TWO-STAGE TOOL RETRIEVER ------------------------ #

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

            # Truncate description a bit to keep things lean
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
        """
        Stage 1: BM25 -> get up to stage1_k candidate tools.
        Stage 2: embed query + candidate texts with Ollama -> cosine similarity -> top-k.
        Returns tool schemas for the final top-k.
        """
        if k <= 0:
            return []

        # -------- Stage 1: BM25 lexical retrieval -------- #
        candidates = self.bm25.invoke(query)  # uses self.bm25.k

        if not candidates:
            return []

        # -------- Stage 2: Ollama embedding rerank -------- #
        candidate_texts = [d.page_content for d in candidates]
        candidate_schemas = [d.metadata["schema"] for d in candidates]

        # Embed query + candidate texts
        query_emb = self.embedding.embed_query(query)
        doc_embs = self.embedding.embed_documents(candidate_texts)

        # Manual cosine similarity (no need for FAISS here)
        def cosine(u, v):
            # Avoid importing numpy just for this; simple Python implementation
            num = sum(a * b for a, b in zip(u, v))
            norm_u = sum(a * a for a in u) ** 0.5
            norm_v = sum(b * b for b in v) ** 0.5
            if norm_u == 0 or norm_v == 0:
                return 0.0
            return num / (norm_u * norm_v)

        scored = [
            (idx, cosine(query_emb, emb))
            for idx, emb in enumerate(doc_embs)
        ]

        scored.sort(key=lambda x: x[1], reverse=True)

        # Take top-k indices
        top_indices = [idx for idx, _ in scored[:k]]

        selected: List[Dict[str, Any]] = [candidate_schemas[i] for i in top_indices]

        # De-duplicate by name just in case
        seen = set()
        unique_tools = []
        for t in selected:
            fn = t.get("function", {})
            name = fn.get("name", t.get("name", "unknown_tool"))
            if name not in seen:
                seen.add(name)
                unique_tools.append(t)

        return unique_tools


# ------------------------ TEST CASES ------------------------ #

TEST_CASES = [
    {
        "name": "Amazon product search",
        "prompt": (
            "Using your Amazon product search tools, find some options for wireless "
            "noise-cancelling headphones under $300."
        ),
        "expect_tool": True,
        "expected_name_substrings": [
            "amazon",
            "search_results",
        ],
    },
    {
        "name": "Amazon product details by ASIN",
        "prompt": (
            "Using your Amazon scraping APIs, look up detailed information, including "
            "price and ratings, for the product with ASIN B0B1234567."
        ),
        "expect_tool": True,
        "expected_name_substrings": [
            "amazon",
            "product_details",
        ],
    },
    {
        "name": "Taobao / Alibaba image search",
        "prompt": (
            "Use your Taobao or Alibaba image search API to find visually similar "
            "products for a red summer dress. Just show me what tool you would call."
        ),
        "expect_tool": True,
        "expected_name_substrings": [
            "image_search",
            "taobao",
            "alibaba",
        ],
    },
    {
        "name": "Translation API",
        "prompt": (
            "Using your translation API, translate the sentence "
            "'Hello, how are you?' into Spanish."
        ),
        "expect_tool": True,
        "expected_name_substrings": [
            "translate",
            "translator",
        ],
    },
    {
        "name": "Oregon lottery mega millions",
        "prompt": (
            "Using your Oregon lottery tools, get the most recent Mega Millions "
            "results and summarize them for me."
        ),
        "expect_tool": True,
        "expected_name_substrings": [
            "mega_millions",
            "oregon_lottery",
        ],
    },
    {
        "name": "Etsy product list",
        "prompt": (
            "Use your Etsy product API to list some products related to 'handmade mug'."
        ),
        "expect_tool": True,
        "expected_name_substrings": [
            "etsy",
            "product_list",
        ],
    },
    {
        "name": "Baby pig pictures tool (fun)",
        "prompt": (
            "You have an API that returns random baby pig pictures. Use it to get "
            "some cute pig pictures."
        ),
        "expect_tool": True,
        "expected_name_substrings": [
            "baby_pig",
            "random_for_baby_pig",
        ],
    },
    {
        "name": "Simple chit-chat (no ToolBench APIs)",
        "prompt": "Tell me a short joke about programmers.",
        "expect_tool": False,
        "expected_name_substrings": [],
    },
]


# ------------------------ UTILS: TOOL CALL EXTRACTION/EVAL ------------------------ #

def extract_tool_calls(messages) -> List[Dict[str, Any]]:
    """
    Walk through the LangGraph state["messages"] list and collect all tool calls
    made by the agent (from AIMessage.tool_calls).
    """
    all_calls = []
    for m in messages:
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            all_calls.extend(tool_calls)
    return all_calls


def evaluate_tool_calls(test: Dict[str, Any], tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given a test case + tool_calls list, return a dict:
      {
        "passed": bool,
        "reason": str,
        "tool_names": [...],
      }
    """
    expect_tool = test["expect_tool"]
    expected_substrings = [s.lower() for s in test["expected_name_substrings"]]

    tool_names = [tc["name"] for tc in tool_calls] if tool_calls else []

    # 1) Expected a tool but got none
    if expect_tool and not tool_calls:
        return {
            "passed": False,
            "reason": "Expected at least one tool call, but got none.",
            "tool_names": tool_names,
        }

    # 2) Did not expect a tool but got some
    if not expect_tool and tool_calls:
        return {
            "passed": False,
            "reason": f"Did not expect a tool call, but got: {tool_names}",
            "tool_names": tool_names,
        }

    # 3) If we expected a specific kind of tool, check for substrings
    if expect_tool and expected_substrings:
        lower_names = [n.lower() for n in tool_names]
        matches = []
        for sub in expected_substrings:
            for name in lower_names:
                if sub in name:
                    matches.append((sub, name))

        if not matches and tool_names:
            return {
                "passed": False,
                "reason": (
                    "Got tool calls, but none matched expected substrings "
                    f"{expected_substrings}. Tool names: {tool_names}"
                ),
                "tool_names": tool_names,
            }

    # If we made it here, behavior matches expectations
    return {
        "passed": True,
        "reason": "Tool usage matched expectations.",
        "tool_names": tool_names,
    }


# ------------------------ MAIN TEST HARNESS (WITH 2-STAGE RETRIEVER) ------------------------ #

def run_tool_tests():
    # 1) Load tools from ToolBench
    print(f"[INFO] Loading tools from: {TOOL_ROOT_DIR}")
    tools_exposed = load_all_tool_schemas(TOOL_ROOT_DIR)
    print(f"[INFO] Loaded {len(tools_exposed)} tool schemas.\n")

    # 2) Build two-stage retriever over ALL tools
    tool_retriever = ToolRetriever2Stage(
        tools_exposed,
        embedding_model=OLLAMA_EMBED_MODEL,
        stage1_k=TOOL_STAGE1_K,
    )

    results: List[Tuple[str, Dict[str, Any]]] = []

    # 3) Run through test cases
    for idx, test in enumerate(TEST_CASES, start=1):
        print(f"================ TEST {idx}: {test['name']} ================")
        print(f"Prompt: {test['prompt']}\n")

        # 3a) Use retriever to select top-k tools for THIS prompt
        top_k_tools = tool_retriever.get_top_k_tools(test["prompt"], k=TOOL_TOP_K)
        selected_tool_names = [
            (t.get("function", {}).get("name", t.get("name", "unknown_tool")))
            for t in top_k_tools
        ]

        print(f"[INFO] Retriever selected {len(top_k_tools)} tools (top-k={TOOL_TOP_K}):")
        for name in selected_tool_names:
            print(f"  - {name}")
        print()

        # 3b) Build your tool agent using ONLY the selected tools
        agent_executor = build_tool_agent(top_k_tools)

        system_content = (
            SYSTEM_PROMPT_TEXT
            + "\n\nYou have access to many external APIs drawn from ToolBench. "
              "A separate retrieval stage has already selected a small subset of "
              f"{len(top_k_tools)} tools that appear most relevant to the user's query. "
              "You MUST choose from these tools when they clearly apply to the user's request. "
              "For general chit-chat or simple reasoning (like jokes), "
              "answer directly without calling any tools."
        )

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=test["prompt"]),
        ]

        # 3c) Invoke the LangGraph-based agent
        state = agent_executor.invoke({"messages": messages})

        # state is a dict; the full conversation is state["messages"]
        msgs = state["messages"]
        final_msg = msgs[-1]

        print("Final message:")
        print(final_msg)
        print("\nFinal content:")
        print(repr(final_msg.content))

        # Extract all tool calls from the message history
        tool_calls = extract_tool_calls(msgs)
        if tool_calls:
            print("\nTool calls observed:")
            for tc in tool_calls:
                print(f"  - name: {tc['name']}, args: {tc['args']}")
        else:
            print("\nTool calls observed: NONE")

        eval_result = evaluate_tool_calls(test, tool_calls)
        print(f"\nResult: {'PASS' if eval_result['passed'] else 'FAIL'}")
        print(f"Reason: {eval_result['reason']}")
        print(f"Tool names: {eval_result['tool_names']}")
        print()

        results.append((test["name"], eval_result))

    # 4) Summary
    print("\n==================== SUMMARY ====================")
    passed = sum(1 for _, r in results if r["passed"])
    total = len(results)
    print(f"Passed {passed}/{total} tests.\n")

    for name, r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {name} -> {r['reason']}")


if __name__ == "__main__":
    run_tool_tests()