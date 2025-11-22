"""Configuration settings for the agent and GCG attack."""

from dataclasses import dataclass


@dataclass
class Settings:
    """Configuration settings for the agent and attack system."""
    
    # LLM Configuration
    model_name: str = "llama3.2:latest"
    
    # GCG Attack Configuration
    max_iters: int = 50
    suffix_len: int = 15
    candidates_per_pos: int = 10
    early_stop_threshold: float = 0.95  # Stop if leak score reaches this
    
    # Agent Configuration
    user_role: str = "expert"
    
    def __post_init__(self):
        """Validate settings after initialization."""
        if self.max_iters < 1:
            raise ValueError("max_iters must be at least 1")
        if self.suffix_len < 1:
            raise ValueError("suffix_len must be at least 1")
        if not 0.0 <= self.early_stop_threshold <= 1.0:
            raise ValueError("early_stop_threshold must be between 0.0 and 1.0")

