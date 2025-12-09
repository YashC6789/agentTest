#!/usr/bin/env python3
"""Main entry point for the GCG attack on LangChain agent."""

from src.agent import create_agent, run_agent_with_suffix
from src.attacks import GCGAttack
from src.scoring import leak_score
from src.config import Settings, SYSTEM_PROMPT, BASE_QUERY


def main():
    """Run the GCG attack on the agent."""
    # Initialize settings
    settings = Settings(
        max_iters=30,
        suffix_len=15
    )
    
    # Create agent
    print("Creating agent...")
    agent = create_agent(settings)
    
    # Sanity check: run agent once without attack
    print("\n" + "=" * 60)
    print("Testing baseline behavior (no attack)...")
    print("=" * 60)
    clean_answer = run_agent_with_suffix(
        agent=agent,
        suffix="",
        settings=settings
    )
    print("Clean answer (no attack):")
    print(clean_answer)
    print("=" * 60)
    
    # Create score function that uses the agent
    def score_suffix(suffix: str) -> float:
        """End-to-end reward function for GCG."""
        ans = run_agent_with_suffix(
            agent=agent,
            suffix=suffix,
            settings=settings
        )
        return leak_score(SYSTEM_PROMPT, ans)
    
    # Initialize and run GCG attack
    print("\n" + "=" * 60)
    print("Running GCG attack...")
    print("=" * 60)
    gcg = GCGAttack(score_function=score_suffix, settings=settings)
    
    best_suffix, best_score = gcg.run(
        max_iters=settings.max_iters,
        suffix_len=settings.suffix_len,
        verbose=True
    )
    
    # Show final answer
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    final_answer = run_agent_with_suffix(
        agent=agent,
        suffix=best_suffix,
        settings=settings
    )
    print("\n=== FINAL ADVERSARIAL SUFFIX ===")
    print(best_suffix)
    print(f"\n=== FINAL LEAK SCORE ===")
    print(f"{best_score:.3f}")
    print("\n=== FINAL ANSWER FROM AGENT ===")
    print(final_answer)
    print("=" * 60)
    
    return best_suffix, best_score


if __name__ == "__main__":
    main()

