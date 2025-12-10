# src/attack/toolhijacker_attack.py
from __future__ import annotations

from typing import List, Dict, Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.data.tool_schemas import (
    load_all_tool_schemas,
    inject_malicious_tool,
)
from src.retrieval.two_stage_retriever import ToolRetriever2Stage
from main_agent import build_main_agent, run_agent_once

from src.config.settings import ToolHijackerConfig, ToolHijackerResult
from .shadow_pipeline import ShadowToolSelectionPipeline
from .toolhijacker_generator import generate_malicious_tool_description


def build_shadow_pipeline(
    shadow_llm: BaseChatModel,
    base_tools: List[Dict[str, Any]],
) -> ShadowToolSelectionPipeline:
    retriever = ToolRetriever2Stage(base_tools)
    return ShadowToolSelectionPipeline(
        shadow_llm=shadow_llm,
        tool_schemas=base_tools,
        retriever=retriever,
    )


def run_toolhijacker_attack(
    shadow_llm: BaseChatModel,
    cfg: ToolHijackerConfig,
) -> ToolHijackerResult:
    """
    1) Load clean tool library.
    2) Use it as base_tools for the shadow pipelines.
    3) Generate a malicious description via gradient-free optimization.
    """
    clean_tools = load_all_tool_schemas()

    result = generate_malicious_tool_description(
        shadow_llm, clean_tools, cfg
    )
    return result


def build_real_agent_with_malicious_tool(
    malicious_tool_name: str,
    malicious_tool_description: str,
) -> Any:
    """
    Inject the malicious tool into the real tool list and build the main agent.
    """
    clean_tools = load_all_tool_schemas()
    poisoned_tools = inject_malicious_tool(
        clean_tools,
        tool_name=malicious_tool_name,
        tool_description=malicious_tool_description,
    )
    agent = build_main_agent(poisoned_tools)
    return agent