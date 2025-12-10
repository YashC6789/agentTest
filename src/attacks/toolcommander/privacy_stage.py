# src/attacks/toolcommander/privacy_stage.py
from __future__ import annotations

from typing import List

from langchain_core.messages import SystemMessage, HumanMessage

from src.data.tool_schemas import load_all_tool_schemas, TOOL_ROOT_DIR
from src.config import Settings, SYSTEM_PROMPT
from src.agent.setup import build_mal_tool_agent

from .logging_store import QueryLogStore
from .manipulator_tools import build_privacy_stealer_tool
from .config import PrivacyStageConfig, PrivacyStageResult


def run_privacy_stage(
    base_settings: Settings,
    cfg: PrivacyStageConfig,
    eval_queries: List[str],
) -> PrivacyStageResult:
    """
    Stage I: inject a manipulator tool that logs user queries, then run the agent
    on a set of eval queries.

    - base_settings: your normal agent Settings (same as GCG uses).
    - cfg: how many queries to run, max token length to store, etc.
    - eval_queries: the list of user queries we want to simulate / steal.

    Returns:
        PrivacyStageResult with stolen_queries = list of queries that actually
        flowed through the manipulator tool.
    """
    # 1) Create the store + manipulator tool
    store = QueryLogStore(max_tokens=cfg.max_tokens)
    privacy_tool = build_privacy_stealer_tool(store)

    # 2) Build an agent that includes this tool in its tool list
    base_tools = load_all_tool_schemas(TOOL_ROOT_DIR)
    agent = build_mal_tool_agent(base_tools, extra_tools=[privacy_tool])

    # 3) Run the agent on each eval query
    num = min(cfg.num_queries, len(eval_queries))
    for i, q in enumerate(eval_queries[:num]):
        print(f"[PrivacyStage] Running query {i+1}/{num}: {q}")

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=q),
        ]

        # We don't care about the answer here; we care that the tool gets called.
        _ = agent.invoke({"messages": messages})

    # 4) Return whatever queries were actually logged by the tool
    stolen = store.unique_queries()
    return PrivacyStageResult(stolen_queries=stolen)