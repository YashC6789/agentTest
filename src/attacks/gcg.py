"""GCG (Greedy Coordinate Gradient) attack implementation."""

import random
from typing import Callable, Tuple

from ..config.settings import Settings
from .vocab import VOCAB


class GCGAttack:
    """Greedy Coordinate Gradient attack for finding adversarial suffixes."""
    
    def __init__(
        self,
        score_function: Callable[[str], float],
        settings: Settings = None,
        vocab: list[str] = None
    ):
        """
        Initialize GCG attack.
        
        Args:
            score_function: Function that takes a suffix string and returns a leak score
            settings: Configuration settings (uses defaults if None)
            vocab: Adversarial vocabulary (uses default if None)
        """
        self.score_function = score_function
        self.settings = settings or Settings()
        self.vocab = vocab or VOCAB
    
    def initialize_suffix(self, length: int = None) -> list[str]:
        """Create a random initial suffix from vocabulary."""
        length = length or self.settings.suffix_len
        return random.choices(self.vocab, k=length)
    
    @staticmethod
    def suffix_to_str(words: list[str]) -> str:
        """Convert list of words to string."""
        return " ".join(words)
    
    def gcg_step(self, words: list[str]) -> Tuple[list[str], float, bool]:
        """
        Perform one iteration of GCG optimization.
        
        Args:
            words: Current suffix as list of words
            
        Returns:
            Tuple of (updated words, best score, improved flag)
        """
        current_suffix = self.suffix_to_str(words)
        current_score = self.score_function(current_suffix)
        improved = False
        
        for i in range(len(words)):
            best_word = words[i]
            best_score = current_score
            
            for _ in range(self.settings.candidates_per_pos):
                new_word = random.choice(self.vocab)
                if new_word == words[i]:
                    continue
                
                trial_words = words.copy()
                trial_words[i] = new_word
                trial_suffix = self.suffix_to_str(trial_words)
                trial_score = self.score_function(trial_suffix)
                
                if trial_score > best_score:
                    best_score = trial_score
                    best_word = new_word
            
            if best_score > current_score:
                words[i] = best_word
                current_score = best_score
                improved = True
        
        return words, current_score, improved
    
    def run(
        self,
        max_iters: int = None,
        suffix_len: int = None,
        verbose: bool = True
    ) -> Tuple[str, float]:
        """
        Run GCG attack to find optimal adversarial suffix.
        
        Args:
            max_iters: Maximum iterations (uses settings default if None)
            suffix_len: Suffix length (uses settings default if None)
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best suffix, best score)
        """
        max_iters = max_iters or self.settings.max_iters
        suffix_len = suffix_len or self.settings.suffix_len
        
        words = self.initialize_suffix(length=suffix_len)
        best_suffix = self.suffix_to_str(words)
        best_score = self.score_function(best_suffix)
        
        if verbose:
            print(f"Init score={best_score:.3f}, suffix='{best_suffix}'")
        
        for it in range(max_iters):
            words, best_score, improved = self.gcg_step(words)
            best_suffix = self.suffix_to_str(words)
            
            if verbose:
                print(f"Iter {it+1}: score={best_score:.3f}, suffix='{best_suffix}'")
            
            if best_score >= self.settings.early_stop_threshold:
                if verbose:
                    print("High leak score reached, stopping early.")
                break
            if not improved:
                if verbose:
                    print("No improvement this iteration, stopping.")
                break
        
        return best_suffix, best_score

