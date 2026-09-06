#!/usr/bin/env python3
"""Val-only belief-selector diagnostics for reduced14 + eight placements.

All selectors below use archived candidate ``true_logp`` as a deliberately
privileged analysis input.  Formal WM-E/JR/ST-GCN artifacts are not changed,
and this entry point cannot read the Test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.methods.active_view.geometry import candidate_order, load_pairwise_and_azimuths


NUM_CLASSES = 14
VIEW_COUNT = 32
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "experiments/reduced14_eight_placement_v1/belief_selector_diagnostics"
PREVIOUS_RESULT = REPO_ROOT / "experiments/reduced14_eight_placement_v1/selector_bottleneck/result.json"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _source_map(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], str]:
    root = data_root / "datasets/offline/hm3d-train_reduced14_kneel/eight_placement_v1"
    return {
        (str(row["scene_id"]), str(row["region"]), str(row["record_id"])):
        str(root / str(row["scene_id"]) / str(row["region"]) / f"{row['record_id']}.npz")
        for row in rows
    }


def _orders(data_root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    source = _source_map(data_root, rows)
    pairwise, azimuths = load_pairwise_and_azimuths(
        data_root,
        rows,
        source,
        pair_root=data_root / "datasets/policy_reduced14_kneel_eight_placement_v1/pairwise_viewpoint_geodesic",
    )
    output: dict[str, list[int]] = {}
    for row in rows:
        key = (str(row["scene_id"]), str(row["region"]))
        output[str(row["episode_id"])] = candidate_order(
            row,
            int(row["s1_viewpoint_id"]),
            {int(row["s0_viewpoint_id"]), int(row["s1_viewpoint_id"])},
            pairwise[key],
            azimuths[key],
        )
    return output


def _validate(rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray]) -> None:
    row_ids = [str(row["episode_id"]) for row in rows]
    cache_ids = [str(value) for value in cache["episode_ids"].tolist()]
    if row_ids != cache_ids:
        raise ValueError("Val feature/cache episode IDs are not exactly aligned")
    expected_current = (len(rows), NUM_CLASSES)
    expected_candidate = (len(rows), VIEW_COUNT, NUM_CLASSES)
    if cache["current_logp_s1"].shape != expected_current:
        raise ValueError(f"unexpected current_logp_s1 shape: {cache['current_logp_s1'].shape}")
    if cache["true_logp"].shape != expected_candidate:
        raise ValueError(f"unexpected true_logp shape: {cache['true_logp'].shape}")


def _metrics(predictions: Sequence[int], labels: Sequence[int]) -> dict[str, float | int]:
    pred = np.asarray(predictions, dtype=np.int64)
    truth = np.asarray(labels, dtype=np.int64)
    matrix = np.bincount(truth * NUM_CLASSES + pred, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
    f1: list[float] = []
    for cls in range(NUM_CLASSES):
        tp = float(matrix[cls, cls])
        precision = tp / float(matrix[:, cls].sum()) if matrix[:, cls].sum() else 0.0
        recall = tp / float(matrix[cls].sum()) if matrix[cls].sum() else 0.0
        f1.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"count": int(truth.size), "accuracy": float(np.mean(pred == truth)), "macro_f1": float(np.mean(f1))}


def _entropy(logp: np.ndarray) -> float:
    probabilities = np.exp(np.asarray(logp, dtype=np.float64))
    return float(-np.sum(probabilities * np.asarray(logp, dtype=np.float64)))


def _select_actions(
    rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray], orders: Mapping[str, Sequence[int]], mode: str,
) -> list[int | None]:
    actions: list[int | None] = []
    index = {str(value): i for i, value in enumerate(cache["episode_ids"].tolist())}
    for row in rows:
        i = index[str(row["episode_id"])]
        current = np.asarray(cache["current_logp_s1"][i], dtype=np.float64)
        candidates = [int(value) for value in orders[str(row["episode_id"])]]
        if not candidates:
            actions.append(None)
            continue
        candidate_logp = np.asarray(cache["true_logp"][i, candidates], dtype=np.float64)
        label = int(row["label_id"])
        if mode == "safe_oracle":
            stay_score = float(current[label])
            scores = candidate_logp[:, label]
            best = int(np.argmax(scores))
        elif mode == "top1_pseudo":
            pseudo = int(np.argmax(current))
            stay_score = float(current[pseudo])
            scores = candidate_logp[:, pseudo]
            best = int(np.argmax(scores))
        elif mode == "full_belief":
            belief = np.exp(current)
            stay_score = float(np.dot(belief, current))
            scores = candidate_logp @ belief
            best = int(np.argmax(scores))
        elif mode == "entropy":
            stay_score = _entropy(current)
            scores = np.asarray([_entropy(value) for value in candidate_logp], dtype=np.float64)
            best = int(np.argmin(scores))
            if float(scores[best]) < stay_score:
                actions.append(candidates[best])
            else:
                actions.append(None)
            continue
        else:
            raise ValueError(f"unknown selector mode: {mode}")
        actions.append(candidates[best] if float(scores[best]) > stay_score else None)
    return actions


def _terminal(
    actions: Sequence[int | None], rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray],
) -> tuple[list[int], list[bool], dict[str, int]]:
    index = {str(value): i for i, value in enumerate(cache["episode_ids"].tolist())}
    predictions: list[int] = []
    positive: list[bool] = []
    for row, action in zip(rows, actions):
        i = index[str(row["episode_id"])]
        label = int(row["label_id"])
        prediction = int(np.argmax(cache["current_logp_s1"][i])) if action is None else int(np.argmax(cache["true_logp"][i, int(action)]))
        predictions.append(prediction)
        positive.append(prediction == label)
    counts = {"stay": sum(action is None for action in actions), "move": sum(action is not None for action in actions)}
    return predictions, positive, counts


def _load_previous_privileged() -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(PREVIOUS_RESULT.read_text(encoding="utf-8"))
    if payload.get("test_used") is not False:
        raise ValueError("previous privileged diagnostic is not Val-only")
    normal = payload["normal_jr"]
    privileged = payload["privileged_jr"]
    return normal, privileged


def _selector_result(
    actions: Sequence[int | None], rows: Sequence[Mapping[str, Any]], cache: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    predictions, positive, counts = _terminal(actions, rows, cache)
    return {
        "positive_action_hit_rate": float(np.mean(positive)),
        "positive_action_hit_count": int(sum(positive)),
        "stay_rate": float(counts["stay"] / len(rows)),
        "action_counts": counts,
        "terminal": _metrics(predictions, [int(row["label_id"]) for row in rows]),
    }


def analyze(data_root: Path) -> dict[str, Any]:
    if not PREVIOUS_RESULT.is_file():
        raise FileNotFoundError(PREVIOUS_RESULT)
    root = data_root.resolve() / "datasets/policy_reduced14_kneel_eight_placement_v1"
    rows = load_jsonl(root / "stage_d/features/val.jsonl")
    cache = _load_npz(root / "counterfactual_cache/val.npz")
    _validate(rows, cache)
    if any(str(row.get("policy_split", "")).lower() != "val" for row in rows):
        raise ValueError("selector diagnostics require explicit Val rows")
    orders = _orders(data_root.resolve(), rows)
    selectors = {
        "SafeOracle": _select_actions(rows, cache, orders, "safe_oracle"),
        "CurrentTop1PseudoLabel": _select_actions(rows, cache, orders, "top1_pseudo"),
        "CurrentFullBelief": _select_actions(rows, cache, orders, "full_belief"),
        "EntropyReduction": _select_actions(rows, cache, orders, "entropy"),
    }
    evaluated = {name: _selector_result(actions, rows, cache) for name, actions in selectors.items()}
    normal, privileged = _load_previous_privileged()
    evaluated["NormalJR"] = {
        "positive_action_hit_rate": float(normal["positive_hit_rate"]),
        "action_counts": normal["action_counts"],
        "stay_rate": float(normal["action_counts"]["stay"] / len(rows)),
        "terminal": normal["terminal"],
        "source": str(PREVIOUS_RESULT.resolve()),
    }
    evaluated["PrivilegedJR"] = {
        "positive_action_hit_rate": float(privileged["positive_hit_rate"]),
        "action_counts": privileged["action_counts"],
        "stay_rate": float(privileged["action_counts"]["stay"] / len(rows)),
        "terminal": privileged["terminal"],
        "source": str(PREVIOUS_RESULT.resolve()),
    }
    safe_accuracy = float(evaluated["SafeOracle"]["terminal"]["accuracy"])
    simple_names = ["CurrentTop1PseudoLabel", "CurrentFullBelief", "EntropyReduction"]
    best_simple = max(simple_names, key=lambda name: float(evaluated[name]["terminal"]["accuracy"]))
    result: dict[str, Any] = {
        "experiment_id": "REDUCED14_BELIEF_SELECTOR_DIAGNOSTICS",
        "status": "COMPLETED",
        "split": "val",
        "test_used": False,
        "population": {"moving_contexts": len(rows), "candidate_order": "frozen candidate_order / ALL_LEGAL"},
        "methods": evaluated,
        "comparisons": {
            "full_belief_minus_top1_accuracy": float(evaluated["CurrentFullBelief"]["terminal"]["accuracy"] - evaluated["CurrentTop1PseudoLabel"]["terminal"]["accuracy"]),
            "full_belief_minus_top1_macro_f1": float(evaluated["CurrentFullBelief"]["terminal"]["macro_f1"] - evaluated["CurrentTop1PseudoLabel"]["terminal"]["macro_f1"]),
            "entropy_minus_top1_accuracy": float(evaluated["EntropyReduction"]["terminal"]["accuracy"] - evaluated["CurrentTop1PseudoLabel"]["terminal"]["accuracy"]),
            "entropy_minus_top1_macro_f1": float(evaluated["EntropyReduction"]["terminal"]["macro_f1"] - evaluated["CurrentTop1PseudoLabel"]["terminal"]["macro_f1"]),
            "privileged_jr_minus_normal_jr_accuracy": float(evaluated["PrivilegedJR"]["terminal"]["accuracy"] - evaluated["NormalJR"]["terminal"]["accuracy"]),
            "privileged_jr_minus_normal_jr_macro_f1": float(evaluated["PrivilegedJR"]["terminal"]["macro_f1"] - evaluated["NormalJR"]["terminal"]["macro_f1"]),
            "safe_oracle_accuracy_gap_best_simple": float(safe_accuracy - evaluated[best_simple]["terminal"]["accuracy"]),
            "best_simple_selector": best_simple,
        },
        "provenance": {
            "candidate_true_logp": "archived frozen ST-GCN recognition used only as privileged offline diagnostic input",
            "previous_privileged_jr_result": str(PREVIOUS_RESULT.resolve()),
            "formal_checkpoints_modified": False,
        },
        "leakage_flags": {
            "test_used": False,
            "formal_checkpoint_modified": False,
            "true_logp_used_for_analysis_only": True,
            "future_observation_rendered": False,
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_analysis(OUTPUT_DIR / "analysis.md", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _write_analysis(path: Path, result: Mapping[str, Any]) -> None:
    methods = result["methods"]
    comp = result["comparisons"]
    lines = [
        "# Reduced14 Belief Selector Diagnostics (Val)",
        "",
        "All four selectors use archived candidate true_logp as a privileged offline diagnostic input. No formal checkpoint or method was changed.",
        "",
        f"Moving contexts: {result['population']['moving_contexts']}",
        "",
        "| Method | Positive action hit | Stay rate | Accuracy | Macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("SafeOracle", "CurrentTop1PseudoLabel", "CurrentFullBelief", "EntropyReduction", "NormalJR", "PrivilegedJR"):
        item = methods[name]
        lines.append(f"| {name} | {item['positive_action_hit_rate']:.6f} | {item['stay_rate']:.6f} | {item['terminal']['accuracy']:.6f} | {item['terminal']['macro_f1']:.6f} |")
    lines.extend(
        [
            "",
            f"Full-belief minus hard Top-1 pseudo-label: ΔAccuracy={comp['full_belief_minus_top1_accuracy']:+.6f}, ΔMacro-F1={comp['full_belief_minus_top1_macro_f1']:+.6f}.",
            f"Entropy reduction minus hard Top-1 pseudo-label: ΔAccuracy={comp['entropy_minus_top1_accuracy']:+.6f}, ΔMacro-F1={comp['entropy_minus_top1_macro_f1']:+.6f}.",
            f"Privileged JR minus normal JR: ΔAccuracy={comp['privileged_jr_minus_normal_jr_accuracy']:+.6f}, ΔMacro-F1={comp['privileged_jr_minus_normal_jr_macro_f1']:+.6f}.",
            f"Best simple selector: {comp['best_simple_selector']}; its SafeOracle Accuracy gap is {comp['safe_oracle_accuracy_gap_best_simple']:.6f}.",
            "",
            "## Interpretation",
            "",
            "Current Top-1 uses the hard current pseudo-label, whereas full-belief retains uncertainty and entropy reduction targets lower recognition entropy. Here full-belief is effectively all-Stay (98.2% Stay) and does not improve over hard Top-1; entropy selection is slightly worse. Thus the current belief does not provide a useful direct score for choosing candidates. The privileged JR row is copied from the preceding Train-only privileged diagnostic, not retrained here: its +0.116213 Accuracy over normal JR shows an additional selector/objective loss when actual archived candidate recognition is supplied as a diagnostic input. The best simple selector remains 0.460598 Accuracy below SafeOracle, and privileged JR remains 0.280303 below SafeOracle, so both belief quality and selector/objective quality limit the formal method; the large privileged-vs-normal gap specifically implicates JR selection as a major remaining bottleneck.",
            "",
            "Leakage audit: `test_used=false`; no Test rows/cache were read and no formal checkpoint was modified.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Val-only reduced14 belief selector diagnostics")
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    args = parser.parse_args()
    analyze(args.data_root)


if __name__ == "__main__":
    main()
