# src/attacks/toolcommander/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class PrivacyStageConfig:
    """Config for Stage I: privacy theft / query logging."""
    num_queries: int = 20              # how many eval queries to run
    max_tokens: int = 512              # cap on stored query length


@dataclass
class CommandStageConfig:
    """Config for Stage II: tool manipulation / unscheduled tool-calling."""
    target_tool_name: str              # the tool we want to promote or use for DoS
    eval_queries: List[str]            # queries used to measure success
    max_iters: int = 5                 # gradient-free refinement steps
    injection_strength: float = 1.0    # how aggressively we prompt-inject


@dataclass
class ToolCommanderSettings:
    """Top-level attack config."""
    privacy_stage: PrivacyStageConfig
    command_stage: CommandStageConfig


@dataclass
class PrivacyStageResult:
    """Metrics for Stage I."""
    stolen_queries: List[str] = field(default_factory=list)


@dataclass
class CommandStageResult:
    """Metrics for Stage II."""
    # Fraction of eval queries where manipulator tool is called at least once
    manipulator_call_rate: float

    # Fraction where the *target* tool is chosen (business bias / unscheduled call)
    target_tool_call_rate: float

    # Fraction where *legitimate* tools are never used (DoS-like behavior)
    denial_of_service_rate: float

    # Final manipulator tool description used in Stage II
    manipulator_description: str


@dataclass
class ToolCommanderResult:
    """End-to-end ToolCommander attack result."""
    privacy_result: PrivacyStageResult
    command_result: CommandStageResult