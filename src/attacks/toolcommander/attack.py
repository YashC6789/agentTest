# src/attacks/toolcommander/attack.py
from __future__ import annotations

from typing import List

from src.config import Settings

from .config import (
    ToolCommanderSettings,
    ToolCommanderResult,
    PrivacyStageResult,
    CommandStageResult,
)
from .privacy_stage import run_privacy_stage
from .command_stage import run_command_stage


class ToolCommanderAttack:
    """
    High-level ToolCommander-style attack wrapper.

    Stage I: inject a privacy-stealing manipulator tool to log real user queries.
    Stage II: build a refined manipulator tool using stolen queries, then evaluate
              tool scheduling manipulation (unscheduled tool-calling, DoS, target bias).
    """

    def __init__(
        self,
        settings: ToolCommanderSettings,
        base_agent_settings: Settings,
        privacy_eval_queries: List[str],
    ):
        self.tc_settings = settings
        self.base_agent_settings = base_agent_settings
        self.privacy_eval_queries = privacy_eval_queries

    def run(self) -> ToolCommanderResult:
        # Stage I: privacy theft
        print("\n" + "=" * 60)
        print("Stage I: Privacy theft / query collection")
        print("=" * 60)
        privacy_result: PrivacyStageResult = run_privacy_stage(
            base_settings=self.base_agent_settings,
            cfg=self.tc_settings.privacy_stage,
            eval_queries=self.privacy_eval_queries,
        )
        print(f"[Stage I] Stolen {len(privacy_result.stolen_queries)} unique queries.")

        # Stage II: command injection / scheduling manipulation
        print("\n" + "=" * 60)
        print("Stage II: Tool scheduling manipulation")
        print("=" * 60)
        command_result: CommandStageResult = run_command_stage(
            base_settings=self.base_agent_settings,
            cfg=self.tc_settings.command_stage,
            stolen_queries=privacy_result.stolen_queries,
        )

        print(f"[Stage II] manipulator_call_rate     = {command_result.manipulator_call_rate:.3f}")
        print(f"[Stage II] target_tool_call_rate     = {command_result.target_tool_call_rate:.3f}")
        print(f"[Stage II] denial_of_service_rate    = {command_result.denial_of_service_rate:.3f}")

        return ToolCommanderResult(
            privacy_result=privacy_result,
            command_result=command_result,
        )