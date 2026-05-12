"""
View selection policies for EA-AVS-MVP v0.1.1.

Important change in v0.1.1:
    OursPolicy compares the current view with valid candidate views.
    If the current view already has the best predicted quality, the robot stays still.
"""

from typing import List
import numpy as np

from .candidate_sampler import CandidateView


def _get_q(score: dict) -> float:
    """Return predicted quality score with backward compatibility."""
    if not score:
        return float("-inf")
    if "Q_pred" in score:
        return float(score["Q_pred"])
    if "Q" in score:
        return float(score["Q"])
    return float("-inf")


class FixedPolicy:
    """Baseline policy: keep the initial viewpoint."""

    name = "Fixed"

    def select(self, current_view: CandidateView,
               candidates: List[CandidateView]) -> CandidateView:
        return current_view


class RandomPolicy:
    """Randomly select one valid candidate. Fallback to current_view if none exists."""

    name = "Random"

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def select(self, current_view: CandidateView,
               candidates: List[CandidateView]) -> CandidateView:
        valid = [c for c in candidates if c.is_valid]
        if not valid:
            return current_view
        idx = self.rng.randint(len(valid))
        return valid[idx]


class NearestPolicy:
    """Select the valid candidate with the shortest geodesic distance."""

    name = "Nearest"

    def select(self, current_view: CandidateView,
               candidates: List[CandidateView]) -> CandidateView:
        valid = [c for c in candidates if c.is_valid]
        if not valid:
            return current_view
        return min(valid, key=lambda c: c.geodesic_distance)


class OursPolicy:
    """
    Select the view with the best predicted quality.

    Unlike v0.1, this policy does not force movement. It compares current_view
    and all valid candidates, then returns the one with the highest predicted Q.
    """

    name = "Ours"

    def select(self, current_view: CandidateView,
               candidates: List[CandidateView]) -> CandidateView:
        valid = [c for c in candidates if c.is_valid and _get_q(c.score) != float("-inf")]
        pool = [current_view] + valid
        return max(pool, key=lambda c: _get_q(c.score))
