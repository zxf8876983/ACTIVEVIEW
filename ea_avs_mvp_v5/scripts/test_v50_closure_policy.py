#!/usr/bin/env python
"""Pure-Python closure tests for v5.0 policy validity and stay behavior."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ea_avs_v5.candidate_sampler import CandidateView
from ea_avs_v5.policies import OursPolicy
from ea_avs_v5.ablation_policies import FullOursPolicy


def view(cid, q, valid=True, current=False):
    return CandidateView(cid, np.zeros(3), 0.0, 0.0, 1.0, True, pred_score={
        "Q_pred": q, "is_occlusion_valid_pred": valid,
    })


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"PASS: {name}")


def main():
    ours = OursPolicy()
    # Current wins over lower-Q candidates.
    cur = view(-1, .90); a = view(1, .70); b = view(2, .80)
    check("current wins", ours.select(cur, [a, b]) is cur)
    # Candidate wins when its valid Q is higher.
    cur = view(-1, .60); a = view(1, .85); b = view(2, .75)
    check("candidate wins", ours.select(cur, [a, b]) is a)
    # Invalid high-Q candidate is excluded.
    cur = view(-1, .80); a = view(1, .99, False); b = view(2, .70)
    check("invalid high-Q excluded", ours.select(cur, [a, b]) is cur)
    check("excluded count", ours.last_selection_stats["excluded_invalid_occ_count"] == 1)
    # All invalid: safe fallback to current.
    cur = view(-1, .80, False); a = view(1, .99, False)
    check("all invalid fallback", ours.select(cur, [a]) is cur)
    check("fallback reason", ours.last_selection_stats["fell_back_to_current"])
    # FullOurs follows the same current-vs-candidate validity semantics.
    cur = view(-1, .90); a = view(1, .70)
    check("FullOurs agrees", FullOursPolicy().select(cur, [a]) is cur)
    print("PASS: all v5.0 closure policy tests")


if __name__ == "__main__":
    main()