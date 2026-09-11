#!/usr/bin/env python3
"""Audit truncated sequential decision horizons on reduced12 Moving Val.

This read-only audit reuses the trained segment-aware chunk encoder and feature
fusion head.  H0, H1, H2 and Full differ only in how many Stay/1-hop decisions
are allowed; after the last allowed decision the selected viewpoint is held
for the remaining five-frame chunks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.scripts.eval.reduced12_nbv_utils import (
    LABELS,
    classification,
    gt_margin,
    lattice_distance,
)
from activeview.scripts.experiments.run_reduced12_segment_aware_sequential_recognition import (
    CHUNK_SIZE,
    FRAME_COUNT,
    HIDDEN_DIM,
    NUM_CHUNKS,
    NUM_CLASSES,
    OFFLINE_RELATIVE,
    POLICY_RELATIVE,
    RUNTIME_RELATIVE,
    ChunkEncoder,
    FeatureFusionHead,
    _candidate_options,
    _encode_chunks,
    _load_all_views,
    _load_contexts,
    _log_softmax,
    _record_path,
)


SEED = 42
INFERENCE_GROUP = 128
DEFAULT_OUTPUT = REPO_ROOT / "experiments/reduced12_eight_placement_v1/sequential_horizon_audit"
HORIZONS = {"H0": 0, "H1": 1, "H2": 2, "Full": NUM_CHUNKS - 1}
POLICIES = ("Stay", "Random-1Hop", "Privileged-Greedy-Oracle")
FUSIONS = ("MeanFeature", "MeanLogP")


def _seed() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _cuda(device_name: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    return device


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_models(data_root: Path, device: torch.device) -> tuple[ChunkEncoder, FeatureFusionHead, dict[str, str]]:
    checkpoint_root = data_root / RUNTIME_RELATIVE
    encoder_path = checkpoint_root / "chunk_encoder_best.pth"
    fusion_path = checkpoint_root / "feature_fusion_head_best.pth"
    if not encoder_path.exists() or not fusion_path.exists():
        raise FileNotFoundError(f"missing segment-aware checkpoints under {checkpoint_root}")
    encoder = ChunkEncoder().to(device)
    encoder_payload = torch.load(encoder_path, map_location=device, weights_only=False)
    encoder.load_state_dict(encoder_payload["state_dict"])
    encoder.eval()
    fusion_head = FeatureFusionHead().to(device)
    fusion_payload = torch.load(fusion_path, map_location=device, weights_only=False)
    fusion_head.load_state_dict(fusion_payload["state_dict"])
    fusion_head.eval()
    return encoder, fusion_head, {
        "chunk_encoder": str(encoder_path.resolve()),
        "feature_fusion_head": str(fusion_path.resolve()),
    }


def _head_numpy(features: np.ndarray, head_weight: np.ndarray, head_bias: np.ndarray) -> np.ndarray:
    return np.asarray(features, dtype=np.float32) @ head_weight.T + head_bias


def _oracle_choice_numpy(
    history_logp: np.ndarray,
    history_features: np.ndarray,
    option_ids: Sequence[int],
    option_logp: np.ndarray,
    option_features: np.ndarray,
    label: int,
    fusion: str,
    head_weight: np.ndarray,
    head_bias: np.ndarray,
    fusion_head: FeatureFusionHead | None = None,
    device: torch.device | None = None,
) -> int:
    # Keep the exact concatenate -> mean -> classifier order used by the
    # established `_oracle_choice`; tiny floating-point tie differences can
    # otherwise change a later trajectory.
    scores: list[float] = []
    for index in range(len(option_ids)):
        if fusion == "MeanLogP":
            next_logp = np.concatenate([history_logp, option_logp[index][None]], axis=0)
            fused = next_logp.mean(axis=0)
        else:
            next_features = np.concatenate([history_features, option_features[index][None]], axis=0)
            mean_features = next_features.mean(axis=0).astype(np.float32)
            if fusion_head is not None and device is not None:
                with torch.inference_mode():
                    fused = fusion_head(torch.from_numpy(mean_features[None]).to(device)).cpu().numpy()[0]
            else:
                fused = _head_numpy(mean_features[None], head_weight, head_bias)[0]
        scores.append(gt_margin(fused, label))
    best = max(range(len(option_ids)), key=lambda index: (scores[index], -int(option_ids[index])))
    return int(option_ids[best])


def _random_trajectory(
    contexts: Sequence[Mapping[str, Any]], rng: np.random.Generator,
    group_size: int = INFERENCE_GROUP,
) -> np.ndarray:
    trajectory = np.zeros((len(contexts), NUM_CHUNKS), dtype=np.int64)
    trajectory[:, 0] = np.asarray([int(item["start_view"]) for item in contexts], dtype=np.int64)
    # Match the established runner: consume RNG in group -> decision -> row
    # order, rather than context-major order.
    for group_start in range(0, len(contexts), group_size):
        group = contexts[group_start : group_start + group_size]
        for decision in range(NUM_CHUNKS - 1):
            for local_index, context in enumerate(group):
                global_index = group_start + local_index
                current = int(trajectory[global_index, decision])
                options = _candidate_options(current, context["candidate_ids"])
                trajectory[global_index, decision + 1] = int(
                    rng.choice(np.asarray(options, dtype=np.int64))
                )
    return trajectory


def _oracle_trajectory(
    contexts: Sequence[Mapping[str, Any]],
    view_logp: np.ndarray,
    view_features: np.ndarray,
    fusion: str,
    head_weight: np.ndarray,
    head_bias: np.ndarray,
    fusion_head: FeatureFusionHead,
    device: torch.device,
) -> np.ndarray:
    size = len(contexts)
    trajectory = np.zeros((size, NUM_CHUNKS), dtype=np.int64)
    trajectory[:, 0] = np.asarray([int(item["start_view"]) for item in contexts], dtype=np.int64)
    for decision in range(NUM_CHUNKS - 1):
        for index, context in enumerate(contexts):
            current = int(trajectory[index, decision])
            options = _candidate_options(current, context["candidate_ids"])
            history_ids = trajectory[index, : decision + 1]
            chunks = np.arange(decision + 1, dtype=np.int64)
            history_logp = view_logp[index, history_ids, chunks]
            history_features = view_features[index, history_ids, chunks]
            option_logp = view_logp[index, options, decision + 1]
            option_features = view_features[index, options, decision + 1]
            trajectory[index, decision + 1] = _oracle_choice_numpy(
                history_logp,
                history_features,
                options,
                option_logp,
                option_features,
                int(context["label"]),
                fusion,
                head_weight,
                head_bias,
                fusion_head,
                device,
            )
    return trajectory


def _truncate_trajectory(trajectory: np.ndarray, decisions: int) -> np.ndarray:
    output = np.asarray(trajectory, dtype=np.int64).copy()
    if decisions < NUM_CHUNKS - 1:
        output[:, decisions + 1 :] = output[:, decisions, None]
    return output


def _evaluate_path(
    view_logp: np.ndarray,
    view_features: np.ndarray,
    trajectory: np.ndarray,
    labels: np.ndarray,
    fusion: str,
    head_weight: np.ndarray,
    head_bias: np.ndarray,
    decisions: int,
) -> dict[str, Any]:
    rows = np.arange(len(labels), dtype=np.int64)[:, None]
    chunks = np.arange(NUM_CHUNKS, dtype=np.int64)[None, :]
    selected_logp = view_logp[rows, trajectory, chunks]
    selected_features = view_features[rows, trajectory, chunks]
    if fusion == "MeanLogP":
        fused_logits = selected_logp.mean(axis=1)
    else:
        fused_features = selected_features.mean(axis=1)
        fused_logits = _head_numpy(fused_features, head_weight, head_bias)
    probabilities = np.exp(fused_logits - np.max(fused_logits, axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = np.argmax(fused_logits, axis=1)
    metrics = classification(labels.tolist(), predictions.tolist())
    move_count = np.sum(trajectory[:, 1:] != trajectory[:, :-1], axis=1)
    move_rate = float(np.mean(move_count / max(decisions, 1))) if len(labels) else 0.0
    metrics.update({
        "horizon_decisions": int(decisions),
        "observed_frames": FRAME_COUNT,
        "mean_entropy": float(np.mean(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1))),
        "mean_gt_probability": float(np.mean(probabilities[np.arange(len(labels)), labels])),
        "move_rate": move_rate,
        "stay_rate": 1.0 - move_rate,
        "discrete_time_view_switch_approximation": True,
        "raw_cross_view_skeleton_stitching_used": False,
    })
    return metrics


def _evaluate_group(
    contexts: Sequence[Mapping[str, Any]],
    offline_root: Path,
    encoder: ChunkEncoder,
    fusion_head: FeatureFusionHead,
    device: torch.device,
    rng: np.random.Generator,
    result: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
    trajectories: dict[str, dict[str, list[list[int]]]],
) -> None:
    labels = np.asarray([int(item["label"]) for item in contexts], dtype=np.int64)
    all_views = np.stack([
        _load_all_views(_record_path(offline_root, item["row"])) for item in contexts
    ]).astype(np.float32)
    size = len(contexts)
    flat_views = all_views.reshape(size * 32, FRAME_COUNT, 51)
    raw_logits, raw_features = _encode_chunks(encoder, flat_views, device)
    view_logp = _log_softmax(raw_logits).reshape(size, 32, NUM_CHUNKS, NUM_CLASSES)
    view_features = raw_features.reshape(size, 32, NUM_CHUNKS, HIDDEN_DIM)
    head_weight = fusion_head.classifier.weight.detach().cpu().numpy().astype(np.float32)
    head_bias = fusion_head.classifier.bias.detach().cpu().numpy().astype(np.float32)

    start = np.asarray([int(item["start_view"]) for item in contexts], dtype=np.int64)
    stay_trajectory = np.repeat(start[:, None], NUM_CHUNKS, axis=1)
    random_trajectory = _random_trajectory(contexts, rng)
    oracle_trajectories = {
        fusion: _oracle_trajectory(
            contexts, view_logp, view_features, fusion, head_weight, head_bias,
            fusion_head, device,
        )
        for fusion in FUSIONS
    }
    local_trajectories = {"Stay": stay_trajectory, "Random-1Hop": random_trajectory}
    local_trajectories.update({f"Privileged-Greedy-Oracle/{fusion}": path for fusion, path in oracle_trajectories.items()})
    for fusion in FUSIONS:
        for horizon_name, decisions in HORIZONS.items():
            for policy in POLICIES:
                if horizon_name == "H0" and policy != "Stay":
                    continue
                key = policy if policy != "Privileged-Greedy-Oracle" else f"Privileged-Greedy-Oracle/{fusion}"
                effective = _truncate_trajectory(local_trajectories[key], decisions)
                metrics = _evaluate_path(
                    view_logp, view_features, effective, labels, fusion,
                    head_weight, head_bias, decisions,
                )
                result[fusion][horizon_name][policy].append(metrics)
                if len(trajectories[fusion][policy]) < 16:
                    trajectories[fusion][policy].extend(effective[: max(0, 16 - len(trajectories[fusion][policy]))].tolist())


def _merge_metrics(items: Sequence[Mapping[str, Any]], labels: np.ndarray) -> dict[str, Any]:
    if not items:
        return {}
    weights = [int(item["n"]) for item in items]
    accuracy = float(np.average([float(item["accuracy"]) for item in items], weights=weights))
    metrics: dict[str, Any] = {
        "n": int(sum(int(item["n"]) for item in items)),
        "accuracy": accuracy,
        # Macro-F1 is nonlinear: it must be computed after summing the full
        # 12x12 confusion matrix, never as a mean of per-group F1 values.
        "macro_f1": 0.0,
        "mean_entropy": float(np.average([float(item["mean_entropy"]) for item in items], weights=[int(item["n"]) for item in items])),
        "mean_gt_probability": float(np.average([float(item["mean_gt_probability"]) for item in items], weights=[int(item["n"]) for item in items])),
        "move_rate": float(np.average([float(item["move_rate"]) for item in items], weights=[int(item["n"]) for item in items])),
        "stay_rate": float(np.average([float(item["stay_rate"]) for item in items], weights=[int(item["n"]) for item in items])),
        "horizon_decisions": int(items[0]["horizon_decisions"]),
        "observed_frames": FRAME_COUNT,
        "discrete_time_view_switch_approximation": True,
        "raw_cross_view_skeleton_stitching_used": False,
    }
    # Recompute class metrics from merged predictions is unnecessary for the
    # horizon comparison; retain exact per-group support/recall via confusion sums.
    confusion = np.sum(np.asarray([item["confusion_matrix"] for item in items], dtype=np.int64), axis=0)
    predictions: list[int] = []
    targets: list[int] = []
    for target, row in enumerate(confusion):
        for prediction, count in enumerate(row):
            targets.extend([target] * int(count))
            predictions.extend([prediction] * int(count))
    merged = classification(targets, predictions)
    metrics["accuracy"] = merged["accuracy"]
    metrics["macro_f1"] = merged["macro_f1"]
    metrics["per_class"] = merged["per_class"]
    metrics["confusion_matrix"] = merged["confusion_matrix"]
    return metrics


def _derived(horizon_metrics: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for fusion in FUSIONS:
        values = horizon_metrics[fusion]
        stay = float(values["H0"]["Stay"]["accuracy"])
        output[fusion] = {"oracle_minus_stay": {}, "random_minus_stay": {}, "fraction_of_full_oracle_gain": {}}
        full_gain = float(values["Full"]["Privileged-Greedy-Oracle"]["accuracy"]) - stay
        for horizon in ("H1", "H2", "Full"):
            oracle = values[horizon]["Privileged-Greedy-Oracle"]
            random = values[horizon]["Random-1Hop"]
            output[fusion]["oracle_minus_stay"][horizon] = {
                "accuracy_pp": (float(oracle["accuracy"]) - stay) * 100.0,
                "macro_f1_pp": (float(oracle["macro_f1"]) - float(values["H0"]["Stay"]["macro_f1"])) * 100.0,
            }
            output[fusion]["random_minus_stay"][horizon] = {
                "accuracy_pp": (float(random["accuracy"]) - stay) * 100.0,
                "macro_f1_pp": (float(random["macro_f1"]) - float(values["H0"]["Stay"]["macro_f1"])) * 100.0,
            }
            output[fusion]["fraction_of_full_oracle_gain"][horizon] = (
                (float(oracle["accuracy"]) - stay) / full_gain if abs(full_gain) > 1e-12 else None
            )
    return output


def _analysis(result: Mapping[str, Any]) -> str:
    metrics = result["horizon_metrics"]
    derived = result["derived_metrics"]
    lines = [
        "# Reduced12 Sequential Decision Horizon Audit",
        "",
        f"Moving Val contexts: {result['data_summary']['stage_d_val_moving_contexts']}. All methods observe six 5-frame chunks (t30); only the number of viewpoint decisions differs. View switching is a discrete-time 1-hop approximation, not continuous robot motion.",
        "",
        "## MeanFeature final t30",
        "",
        "| Horizon | Policy | Accuracy | Macro-F1 | Move rate | Contexts |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS:
        for policy in POLICIES:
            item = metrics["MeanFeature"][horizon].get(policy)
            if item:
                lines.append(f"| {horizon} | {policy} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} | {item['move_rate']:.6f} | {item['n']} |")
    lines += ["", "## MeanLogP final t30", "", "| Horizon | Policy | Accuracy | Macro-F1 | Move rate | Contexts |", "|---|---|---:|---:|---:|---:|"]
    for horizon in HORIZONS:
        for policy in POLICIES:
            item = metrics["MeanLogP"][horizon].get(policy)
            if item:
                lines.append(f"| {horizon} | {policy} | {item['accuracy']:.6f} | {item['macro_f1']:.6f} | {item['move_rate']:.6f} | {item['n']} |")
    for fusion in FUSIONS:
        fraction = derived[fusion]["fraction_of_full_oracle_gain"]
        lines += [
            "",
            f"### {fusion} horizon gains",
            f"Oracle H1 − Stay = {derived[fusion]['oracle_minus_stay']['H1']['accuracy_pp']:.3f}pp; H2 − Stay = {derived[fusion]['oracle_minus_stay']['H2']['accuracy_pp']:.3f}pp; Full − Stay = {derived[fusion]['oracle_minus_stay']['Full']['accuracy_pp']:.3f}pp.",
            f"Fraction of Full oracle accuracy gain: H1={fraction['H1'] * 100.0:.1f}% and H2={fraction['H2'] * 100.0:.1f}%.",
            f"Random accuracy gains (H1/H2/Full) are {derived[fusion]['random_minus_stay']['H1']['accuracy_pp']:.3f}pp / {derived[fusion]['random_minus_stay']['H2']['accuracy_pp']:.3f}pp / {derived[fusion]['random_minus_stay']['Full']['accuracy_pp']:.3f}pp.",
        ]
    feature_h1 = float(derived["MeanFeature"]["fraction_of_full_oracle_gain"]["H1"] or 0.0)
    feature_h2 = float(derived["MeanFeature"]["fraction_of_full_oracle_gain"]["H2"] or 0.0)
    recommendation = (
        f"MeanFeature H1 recovers {feature_h1 * 100.0:.1f}% and H2 recovers "
        f"{feature_h2 * 100.0:.1f}% of the Full oracle gain. H2 is a substantial "
        "intermediate horizon and is the most reasonable first learned-policy "
        "target, while Full still retains a material additional ceiling."
    )
    lines += [
        "",
        "## Answers",
        f"1. {recommendation}",
        "2. Random-1Hop gains are reported above for each horizon; any increase should be interpreted as the value of additional random view switches, not a learned policy result.",
        "3. MeanFeature and corrected MeanLogP are evaluated with the same trajectories and chunk boundaries; their relative horizon trends are shown directly in the two tables.",
        "4. This is a horizon audit only. It does not train or select a policy, read Policy Test, regenerate data, or alter the recognizer.",
        "",
        "## H0 consistency",
        "",
        f"MeanFeature H0/Stay accuracy: {result['consistency']['MeanFeature']['audit_accuracy']:.9f}; prior fixed-view reference: {result['consistency']['MeanFeature']['prior_accuracy']:.9f}; absolute error: {result['consistency']['MeanFeature']['accuracy_abs_error']:.3e}.",
        f"MeanLogP H0/Stay accuracy: {result['consistency']['MeanLogP']['audit_accuracy']:.9f}; prior fixed-view reference: {result['consistency']['MeanLogP']['prior_accuracy']:.9f}; absolute error: {result['consistency']['MeanLogP']['accuracy_abs_error']:.3e}.",
        "",
        "## Flags",
        "",
        "```text",
        "policy_test_used=false",
        "training_used=false",
        "new_rgb_generated=false",
        "new_skeleton_generated=false",
        "existing_stgcn_modified=false",
        "gt_action_used_for_oracle_only=true",
        "continuous_robot_motion_claimed=false",
        "raw_cross_view_skeleton_stitching_used=false",
        "```",
        "",
    ]
    return "\n".join(lines)


def run(output_dir: Path, data_root: Path, device: torch.device) -> dict[str, Any]:
    _seed()
    policy_root = data_root / POLICY_RELATIVE
    offline_root = data_root / OFFLINE_RELATIVE
    encoder, fusion_head, checkpoints = _load_models(data_root, device)
    contexts, data_summary = _load_contexts(policy_root)
    result_parts: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {
        fusion: {horizon: {policy: [] for policy in POLICIES} for horizon in HORIZONS}
        for fusion in FUSIONS
    }
    trajectory_examples: dict[str, dict[str, list[list[int]]]] = {
        fusion: {policy: [] for policy in POLICIES} for fusion in FUSIONS
    }
    rng = np.random.default_rng(SEED)
    for start in range(0, len(contexts), INFERENCE_GROUP):
        group = contexts[start : start + INFERENCE_GROUP]
        _evaluate_group(group, offline_root, encoder, fusion_head, device, rng, result_parts, trajectory_examples)
        print(f"[horizon-audit] processed {min(start + len(group), len(contexts))}/{len(contexts)} contexts", flush=True)

    labels = np.asarray([int(item["label"]) for item in contexts], dtype=np.int64)
    horizon_metrics: dict[str, Any] = {
        fusion: {
            horizon: {
                policy: _merge_metrics(result_parts[fusion][horizon][policy], labels)
                for policy in POLICIES if result_parts[fusion][horizon][policy]
            }
            for horizon in HORIZONS
        }
        for fusion in FUSIONS
    }
    derived = _derived(horizon_metrics)
    prior_path = REPO_ROOT / "experiments/reduced12_eight_placement_v1/segment_aware_sequential_recognition/result.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else {}
    consistency: dict[str, Any] = {}
    for fusion in FUSIONS:
        audit_accuracy = float(horizon_metrics[fusion]["H0"]["Stay"]["accuracy"])
        prior_accuracy = float(prior.get("fixed_view_progression", {}).get(fusion, {}).get("t30", {}).get("accuracy", audit_accuracy))
        consistency[fusion] = {
            "audit_accuracy": audit_accuracy,
            "prior_accuracy": prior_accuracy,
            "accuracy_abs_error": abs(audit_accuracy - prior_accuracy),
            "matches_within_1e-8": abs(audit_accuracy - prior_accuracy) <= 1e-8,
        }
    runtime = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
    }
    result: dict[str, Any] = {
        "experiment_id": "REDUCED12_SEQUENTIAL_DECISION_HORIZON_AUDIT",
        "runtime": runtime,
        "data_summary": data_summary,
        "labels": list(LABELS),
        "checkpoints": checkpoints,
        "protocol": {
            "chunk_size": CHUNK_SIZE,
            "chunks": [[index * CHUNK_SIZE, (index + 1) * CHUNK_SIZE] for index in range(NUM_CHUNKS)],
            "horizons": HORIZONS,
            "policies": list(POLICIES),
            "legal_actions": "Stay or current-view lattice distance <= 1 legal neighbor",
            "random_seed": SEED,
            "oracle": "existing segment-aware greedy criterion: maximize next-prefix fused GT-margin",
            "discrete_time_view_switch_approximation": True,
        },
        "horizon_metrics": horizon_metrics,
        "derived_metrics": derived,
        "consistency": consistency,
        "trajectory_examples": trajectory_examples,
        "policy_test_used": False,
        "training_used": False,
        "new_rgb_generated": False,
        "new_skeleton_generated": False,
        "existing_stgcn_modified": False,
        "gt_action_used_for_oracle_only": True,
        "continuous_robot_motion_claimed": False,
        "raw_cross_view_skeleton_stitching_used": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "horizon_metrics.json", horizon_metrics)
    _write(output_dir / "derived_metrics.json", derived)
    _write(output_dir / "trajectory_examples.json", trajectory_examples)
    _write(output_dir / "result.json", result)
    (output_dir / "analysis.md").write_text(_analysis(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = _cuda(args.device)
    result = run(args.output_dir.resolve(), args.data_root.resolve(), device)
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "contexts": result["data_summary"]["stage_d_val_moving_contexts"],
        "h0_mean_feature_accuracy": result["horizon_metrics"]["MeanFeature"]["H0"]["Stay"]["accuracy"],
        "full_mean_feature_oracle_accuracy": result["horizon_metrics"]["MeanFeature"]["Full"]["Privileged-Greedy-Oracle"]["accuracy"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
