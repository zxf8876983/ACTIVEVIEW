#!/usr/bin/env python3
"""Train and evaluate a privileged GT-conditioned Action-Discriminative WM-E.

This is a Train/Val-only diagnostic.  It initializes the existing
Action-Discriminative WM-E from its frozen checkpoint, adds a small one-hot
ground-truth action conditioning branch, and writes a separate checkpoint and
diagnostic report.  The formal WM-E/JR/ST-GCN artifacts are never modified and
the Test split is intentionally unreachable from this entry point.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.core.paths import get_data_root
from activeview.data.preprocessing.cache import load_jsonl
from activeview.methods.joint_revision.history_aware import HistoryIdentityEncoder
from activeview.methods.world_model.model import CandidateObservationWorldModel, world_model_loss
from activeview.recognition.stgcn.model import load_checkpoint
from activeview.scripts.eval.analyze_reduced14_selector_bottleneck import (
    _load_npz,
    _validate_rows_cache,
)
from activeview.scripts.eval.analyze_reduced14_wm_e import _correlation
from activeview.scripts.train.train_reduced14_action_discriminative_world_model import (
    CHECKPOINT_DIR,
    DATASET_NAME,
    FEATURE_WEIGHT,
    BELIEF_WEIGHT,
    NUM_CLASSES,
    REC_WEIGHT,
    SEED,
    WM_BATCH_SIZE,
    WM_EPOCHS,
    WM_LR,
    WM_WEIGHT_DECAY,
    _build_loader,
    _filtered_rgb_lookup,
    _load_history_identity,
    _run_h1,
    _sha256,
    _label_names,
)


OUTPUT_DIR = REPO_ROOT / "experiments/reduced14_eight_placement_v1/gt_conditioned_world_model"
GT_CHECKPOINT_NAME = "wm_e_gt_conditioned_best.pth"


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_gt_model(old_checkpoint: Path, device: torch.device) -> CandidateObservationWorldModel:
    payload = torch.load(old_checkpoint, map_location=device, weights_only=False)
    state = payload.get("model_state_dict") or payload.get("state_dict")
    if state is None:
        raise ValueError(f"invalid Action-Discriminative WM-E checkpoint: {old_checkpoint}")
    model = CandidateObservationWorldModel(
        use_belief=True,
        use_rgb=True,
        residual=False,
        num_classes=NUM_CLASSES,
        use_action_discriminative_heads=True,
        gt_action_conditioning_dim=NUM_CLASSES,
    ).to(device)
    loaded = model.load_state_dict(state, strict=False)
    expected = {
        "gt_action_encoder.0.weight",
        "gt_action_encoder.0.bias",
        "gt_action_encoder.2.weight",
        "gt_action_encoder.2.bias",
        "condition_fusion.weight",
        "condition_fusion.bias",
    }
    if set(loaded.missing_keys) != expected or loaded.unexpected_keys:
        raise ValueError(f"unexpected old WM-E state mismatch: {loaded}")
    return model


def _model_kwargs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: batch[name].to(device, non_blocking=True)
        for name in (
            "history_skeleton",
            "history_descriptor",
            "candidate_descriptor",
            "history_belief",
            "history_rgb",
        )
    }


def _gt_actions(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    labels = labels.to(device=device, dtype=torch.long)
    if labels.ndim != 1 or bool(torch.any((labels < 0) | (labels >= NUM_CLASSES))):
        raise ValueError("batch labels must be valid reduced14 class IDs")
    return F.one_hot(labels, NUM_CLASSES).to(dtype=torch.float32)


def _train_step_gt(
    model: CandidateObservationWorldModel,
    teacher: torch.nn.Module,
    identity: HistoryIdentityEncoder,
    batch: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    kwargs = _model_kwargs(batch, device)
    target = batch["target_skeleton"].to(device, non_blocking=True)
    valid = batch["candidate_mask"].to(device)
    labels = batch["label_id"]
    prediction, predicted_feature, predicted_logits = model(
        **kwargs,
        ground_truth_action=_gt_actions(labels, device),
        return_action_discriminative=True,
    )
    batch_size, candidate_count = prediction.shape[:2]
    pred_flat = predicted_feature.reshape(batch_size, candidate_count, 256)[valid]
    logits_flat = predicted_logits.reshape(batch_size, candidate_count, NUM_CLASSES)[valid]
    prediction_flat = prediction[valid]
    target_flat = target[valid]
    pose_total, pose, velocity = world_model_loss(prediction_flat, target_flat)
    with torch.no_grad():
        true_feature = teacher.forward_features(target_flat)
        true_logp = torch.log_softmax(teacher.fc(true_feature), dim=-1)
        s0_feature = teacher.forward_features(kwargs["history_skeleton"][:, 0])
        s0_logp = torch.log_softmax(teacher.fc(s0_feature), dim=-1)
    pred_logp = torch.log_softmax(logits_flat, dim=-1)
    rec = F.kl_div(pred_logp, true_logp.exp(), reduction="batchmean")
    feat = (1.0 - F.cosine_similarity(pred_flat, true_feature, dim=-1)).mean()
    repeated_s0_feature = s0_feature[:, None, :].expand(-1, candidate_count, -1)[valid]
    repeated_s0_logp = s0_logp[:, None, :].expand(-1, candidate_count, -1)[valid]
    real_input = torch.cat([repeated_s0_feature, true_feature, repeated_s0_logp, true_logp], dim=-1)
    pred_input = torch.cat([repeated_s0_feature, pred_flat, repeated_s0_logp, pred_logp], dim=-1)
    with torch.no_grad():
        real_belief = torch.softmax(identity(real_input)[1], dim=-1)
    pred_belief_logp = torch.log_softmax(identity(pred_input)[1], dim=-1)
    belief = F.kl_div(pred_belief_logp, real_belief, reduction="batchmean")
    loss = pose_total + REC_WEIGHT * rec + FEATURE_WEIGHT * feat + BELIEF_WEIGHT * belief
    return loss, {
        "loss": float(loss.detach().cpu()),
        "pose_loss": float(pose.detach().cpu()),
        "velocity_loss": float(velocity.detach().cpu()),
        "recognition_kl": float(rec.detach().cpu()),
        "feature_cosine_loss": float(feat.detach().cpu()),
        "belief_kl": float(belief.detach().cpu()),
    }


def _evaluate_wm_gt(
    model: CandidateObservationWorldModel,
    teacher: torch.nn.Module,
    identity: HistoryIdentityEncoder,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    agreement: list[bool] = []
    skeleton_agreement: list[bool] = []
    pred_scores: list[float] = []
    true_scores: list[float] = []
    cosines: list[float] = []
    belief_kls: list[float] = []
    pred_entropies: list[float] = []
    true_entropies: list[float] = []
    pred_margins: list[float] = []
    true_margins: list[float] = []
    top_hits: list[tuple[bool, bool]] = []
    oracle_exists: list[bool] = []
    with torch.inference_mode():
        for batch in loader:
            kwargs = _model_kwargs(batch, device)
            labels = batch["label_id"].to(device)
            prediction, predicted_feature, predicted_logits = model(
                **kwargs,
                ground_truth_action=_gt_actions(labels, device),
                return_action_discriminative=True,
            )
            target = batch["target_skeleton"].to(device, non_blocking=True)
            batch_size, candidate_count = prediction.shape[:2]
            valid = batch["candidate_mask"].to(device) & batch["legal_candidate_mask"].to(device)
            target_flat = target.reshape(-1, 3, 30, 17)
            pred_flat = prediction.reshape(-1, 3, 30, 17)
            true_feature_flat = teacher.forward_features(target_flat)
            true_logp = torch.log_softmax(teacher.fc(true_feature_flat), dim=-1).reshape(batch_size, candidate_count, NUM_CLASSES)
            skeleton_logp = torch.log_softmax(teacher(pred_flat), dim=-1).reshape(batch_size, candidate_count, NUM_CLASSES)
            pred_logp = torch.log_softmax(predicted_logits, dim=-1)
            predicted_feature_grid = predicted_feature.reshape(batch_size, candidate_count, 256)
            true_feature_grid = true_feature_flat.reshape(batch_size, candidate_count, 256)
            s0_feature = teacher.forward_features(kwargs["history_skeleton"][:, 0])
            s0_logp = torch.log_softmax(teacher.fc(s0_feature), dim=-1)
            repeated_s0_feature = s0_feature[:, None, :].expand(-1, candidate_count, -1).reshape(-1, 256)
            repeated_s0_logp = s0_logp[:, None, :].expand(-1, candidate_count, -1).reshape(-1, NUM_CLASSES)
            pred_belief = torch.softmax(identity(torch.cat([repeated_s0_feature, predicted_feature_grid.reshape(-1, 256), repeated_s0_logp, pred_logp.reshape(-1, NUM_CLASSES)], dim=-1))[1], dim=-1).reshape(batch_size, candidate_count, NUM_CLASSES)
            real_belief = torch.softmax(identity(torch.cat([repeated_s0_feature, true_feature_grid.reshape(-1, 256), repeated_s0_logp, true_logp.reshape(-1, NUM_CLASSES)], dim=-1))[1], dim=-1).reshape(batch_size, candidate_count, NUM_CLASSES)
            for index in range(batch_size):
                mask = valid[index]
                if not bool(mask.any()):
                    continue
                p = pred_logp[index, mask]
                t = true_logp[index, mask]
                sk = skeleton_logp[index, mask]
                label = int(labels[index])
                agreement.extend((p.argmax(dim=-1) == t.argmax(dim=-1)).cpu().tolist())
                skeleton_agreement.extend((sk.argmax(dim=-1) == t.argmax(dim=-1)).cpu().tolist())
                pred_scores.extend(p[:, label].cpu().tolist())
                true_scores.extend(t[:, label].cpu().tolist())
                cosines.extend(F.cosine_similarity(predicted_feature_grid[index, mask], true_feature_grid[index, mask], dim=-1).cpu().tolist())
                pred_b = pred_belief[index, mask]
                real_b = real_belief[index, mask]
                belief_kls.extend((real_b * (real_b.clamp_min(1e-12).log() - pred_b.clamp_min(1e-12).log())).sum(dim=-1).cpu().tolist())
                p_entropy = -(pred_b * pred_b.clamp_min(1e-12).log()).sum(dim=-1).cpu().numpy()
                t_entropy = -(real_b * real_b.clamp_min(1e-12).log()).sum(dim=-1).cpu().numpy()
                p_margin = torch.sort(pred_b, dim=-1).values[:, -1] - torch.sort(pred_b, dim=-1).values[:, -2]
                t_margin = torch.sort(real_b, dim=-1).values[:, -1] - torch.sort(real_b, dim=-1).values[:, -2]
                pred_entropies.extend(p_entropy.tolist())
                true_entropies.extend(t_entropy.tolist())
                pred_margins.extend(p_margin.cpu().numpy().tolist())
                true_margins.extend(t_margin.cpu().numpy().tolist())
                positive = t.argmax(dim=-1).cpu().numpy() == label
                ranking = np.argsort(-p[:, label].cpu().numpy(), kind="stable")
                top_hits.append((bool(positive[ranking[0]]), bool(np.any(positive[ranking[: min(3, len(ranking))]]))))
                oracle_exists.append(bool(positive.any()))
    pearson, spearman = _correlation(np.asarray(pred_scores), np.asarray(true_scores))
    entropy_pearson, entropy_spearman = _correlation(np.asarray(pred_entropies), np.asarray(true_entropies))
    margin_pearson, margin_spearman = _correlation(np.asarray(pred_margins), np.asarray(true_margins))
    top1 = np.asarray([item[0] for item in top_hits], dtype=bool)
    top3 = np.asarray([item[1] for item in top_hits], dtype=bool)
    exists = np.asarray(oracle_exists, dtype=bool)
    return {
        "legal_candidate_samples": len(pred_scores),
        "recognition_agreement": float(np.mean(agreement)) if agreement else 0.0,
        "skeleton_recognition_agreement": float(np.mean(skeleton_agreement)) if skeleton_agreement else 0.0,
        "pearson": pearson,
        "spearman": spearman,
        "feature_cosine_similarity": float(np.mean(cosines)) if cosines else 0.0,
        "belief_kl": float(np.mean(belief_kls)) if belief_kls else 0.0,
        "belief_entropy_pearson": entropy_pearson,
        "belief_entropy_spearman": entropy_spearman,
        "belief_margin_pearson": margin_pearson,
        "belief_margin_spearman": margin_spearman,
        "top1_positive_hit": float(np.mean(top1)) if top1.size else 0.0,
        "top3_positive_hit": float(np.mean(top3)) if top3.size else 0.0,
        "oracle_positive_contexts": int(exists.sum()),
        "top1_when_oracle_positive": float(np.mean(top1[exists])) if exists.any() else None,
        "top3_when_oracle_positive": float(np.mean(top3[exists])) if exists.any() else None,
        "contexts": int(len(top_hits)),
    }


def _rename_gt_selectors(selectors: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(selectors)
    for source, target in (
        (
            "Action_Discriminative_WM_imagined_Min_entropy_H1",
            "GT_conditioned_WM_imagined_Min_entropy_H1",
        ),
        (
            "Action_Discriminative_WM_imagined_Max_margin_H1",
            "GT_conditioned_WM_imagined_Max_margin_H1",
        ),
    ):
        if source in output:
            output[target] = output.pop(source)
    return output


def _write_analysis(path: Path, result: Mapping[str, Any]) -> None:
    wm = result["wm_val"]
    selectors = result["h1"]["selectors"]
    lines = [
        "# GT-conditioned World Model (reduced14 Train/Val)",
        "",
        f"Train contexts: {result['population']['train_contexts']}; Val moving contexts: {result['population']['val_moving_contexts']}. Test was not read.",
        "",
        "The model is a privileged diagnostic initialized from the existing Action-Discriminative WM-E. A reduced14 one-hot ground-truth action is fused into the candidate condition; the old checkpoint and all formal methods remain untouched.",
        "",
        "## WM-E Val diagnostics",
        "",
        "| Metric | Action-Discriminative WM-E | GT-conditioned WM-E |",
        "|---|---:|---:|",
    ]
    old = result["comparison"]["action_discriminative_wm_e"]
    for key in (
        "recognition_agreement",
        "pearson",
        "spearman",
        "feature_cosine_similarity",
        "belief_kl",
        "belief_entropy_pearson",
        "belief_entropy_spearman",
        "belief_margin_pearson",
        "belief_margin_spearman",
        "top1_positive_hit",
        "top3_positive_hit",
    ):
        lines.append(f"| {key} | {old.get(key)} | {wm.get(key)} |")
    lines.extend([
        "",
        "## Imagined H1 identity after real selected observation",
        "",
        "| Selector | Accuracy | Macro-F1 | Mean entropy | s0 correction |",
        "|---|---:|---:|---:|---:|",
    ])
    selector_order = (
        "Frozen_current_H1",
        "Real_Min_entropy_H1",
        "Real_Max_margin_H1",
        "Action_Discriminative_WM_imagined_Min_entropy_H1",
        "Action_Discriminative_WM_imagined_Max_margin_H1",
        "GT_conditioned_WM_imagined_Min_entropy_H1",
        "GT_conditioned_WM_imagined_Max_margin_H1",
        "IdentityOracle_H1",
    )
    for name in selector_order:
        item = selectors.get(name)
        if item is None:
            continue
        lines.append(
            f"| {name} | {item['identity']['accuracy']:.6f} | {item['identity']['macro_f1']:.6f} | {item['mean_entropy']:.6f} | {item['s0_error_correction_rate']:.6f} |"
        )
    gt_min = selectors.get("GT_conditioned_WM_imagined_Min_entropy_H1", {}).get("identity", {}).get("accuracy")
    gt_max = selectors.get("GT_conditioned_WM_imagined_Max_margin_H1", {}).get("identity", {}).get("accuracy")
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"GT-conditioned WM imagined H1 Accuracy is {gt_min} (min-entropy) and {gt_max} (max-margin).",
        "A clear fidelity and H1 gain would support hypothesis-conditioned future prediction as the next method direction. If fidelity improves but imagined H1 remains below Frozen H1, the remaining limitation is in the imagined selector/target interface rather than hypothesis-agnostic future prediction.",
        "The GT action is used only as privileged Train/Val diagnostic conditioning and is not available to a deployable policy.",
        "",
        "Leakage audit: `test_used=false`; no Test rows, caches, or files were read. ST-GCN and History Identity were frozen, the formal WM-E/JR artifacts were not modified, and the old Action-Discriminative WM-E checkpoint was not overwritten.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(data_root: Path, device: torch.device, evaluate_only: bool = False) -> dict[str, Any]:
    _seed()
    started = time.perf_counter()
    data_root = data_root.resolve()
    policy_root = data_root / "datasets" / DATASET_NAME
    train_rows = load_jsonl(policy_root / "stage_d/features/train.jsonl")
    val_rows = load_jsonl(policy_root / "stage_d/features/val.jsonl")
    if any(str(row.get("policy_split", "")).lower() != "train" for row in train_rows):
        raise ValueError("Train split contamination")
    if any(str(row.get("policy_split", "")).lower() != "val" for row in val_rows):
        raise ValueError("Val split contamination")
    rgb_lookup = _filtered_rgb_lookup(data_root, [*train_rows, *val_rows])
    old_checkpoint = data_root / "checkpoints" / CHECKPOINT_DIR / "wm_e_action_discriminative_best.pth"
    new_checkpoint = data_root / "checkpoints" / CHECKPOINT_DIR / GT_CHECKPOINT_NAME
    model = _load_gt_model(old_checkpoint, device)
    teacher_checkpoint = data_root / "checkpoints/stgcn_reduced14_kneel_babel_diversity_v1/stgcn_reduced14_kneel_best.pth"
    teacher, _ = load_checkpoint(teacher_checkpoint, NUM_CLASSES, str(device))
    identity, identity_checkpoint = _load_history_identity(data_root, device)
    val_loader = _build_loader(data_root, val_rows, rgb_lookup, WM_BATCH_SIZE, 2, False)
    train_loader: DataLoader | None = None
    train_history: list[dict[str, float]] = []
    val_history: list[dict[str, Any]] = []
    best_epoch = 0
    started_train = time.perf_counter()
    if not evaluate_only:
        train_loader = _build_loader(data_root, train_rows, rgb_lookup, WM_BATCH_SIZE, 4, True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=WM_LR, weight_decay=WM_WEIGHT_DECAY)
        best_key = (-np.inf, -np.inf)
        for epoch in range(1, WM_EPOCHS + 1):
            model.train()
            batches: list[dict[str, float]] = []
            assert train_loader is not None
            for batch in train_loader:
                loss, stats = _train_step_gt(model, teacher, identity, batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                batches.append(stats)
            train_stats = {key: float(np.mean([item[key] for item in batches])) for key in batches[0]}
            train_stats["epoch"] = float(epoch)
            train_history.append(train_stats)
            val_stats = _evaluate_wm_gt(model, teacher, identity, val_loader, device)
            val_stats["epoch"] = epoch
            val_history.append(val_stats)
            key = (
                float(val_stats["belief_entropy_pearson"] or -1.0),
                float(val_stats["belief_margin_spearman"] or -1.0),
            )
            if key > best_key:
                best_key = key
                best_epoch = epoch
                new_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "state_dict": model.state_dict(),
                        "variant": "E_gt_conditioned_action_discriminative",
                        "epoch": epoch,
                        "seed": SEED,
                        "num_classes": NUM_CLASSES,
                        "gt_action_conditioning_dim": NUM_CLASSES,
                        "loss_weights": {
                            "recognition": REC_WEIGHT,
                            "feature": FEATURE_WEIGHT,
                            "belief": BELIEF_WEIGHT,
                        },
                        "val_metrics": val_stats,
                    },
                    new_checkpoint,
                )
            print(
                f"GT-conditioned WM epoch {epoch}/{WM_EPOCHS} loss={train_stats['loss']:.6f} val_spearman={val_stats['spearman']}",
                flush=True,
            )
    else:
        if not new_checkpoint.is_file():
            raise FileNotFoundError(f"evaluate-only requested but checkpoint is missing: {new_checkpoint}")
    payload = torch.load(new_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    val_cache = _load_npz(policy_root / "counterfactual_cache/val.npz")
    _validate_rows_cache(val_rows, val_cache, "val")
    if not np.array_equal(np.asarray(val_cache["label_id"], dtype=np.int64), np.asarray([int(row["label_id"]) for row in val_rows], dtype=np.int64)):
        raise ValueError("Val cache labels are not aligned with Val feature rows")
    final_wm_val = _evaluate_wm_gt(model, teacher, identity, val_loader, device)
    final_wm_val["epoch"] = int(payload.get("epoch", best_epoch or 0))
    names = _label_names(data_root)
    val_labels = np.asarray([int(row["label_id"]) for row in val_rows], dtype=np.int64)
    h1_result, _ = _run_h1(
        data_root,
        val_rows,
        val_cache,
        rgb_lookup,
        model,
        device,
        names,
        ground_truth_labels=val_labels,
    )
    selectors = _rename_gt_selectors(h1_result["selectors"])
    previous_result_path = REPO_ROOT / "experiments/reduced14_eight_placement_v1/action_discriminative_wm_e/result.json"
    previous = json.loads(previous_result_path.read_text(encoding="utf-8")) if previous_result_path.is_file() else {}
    previous_selectors = previous.get("h1", {}).get("selectors", {})
    for name in (
        "Action_Discriminative_WM_imagined_Min_entropy_H1",
        "Action_Discriminative_WM_imagined_Max_margin_H1",
    ):
        if name in previous_selectors:
            selectors[name] = previous_selectors[name]
    old_wm_val = previous.get("wm_val", {})
    result: dict[str, Any] = {
        "experiment_id": "REDUCED14_GT_CONDITIONED_WORLD_MODEL",
        "status": "COMPLETED",
        "split": "train_val",
        "test_used": False,
        "population": {
            "train_contexts": len(train_rows),
            "val_moving_contexts": len(val_rows),
            "val_candidate_hypotheses": h1_result["inference"]["candidate_hypotheses"],
        },
        "training": {
            "performed": not evaluate_only,
            "epochs": WM_EPOCHS,
            "batch_size": WM_BATCH_SIZE,
            "workers": 4,
            "learning_rate": WM_LR,
            "weight_decay": WM_WEIGHT_DECAY,
            "seed": SEED,
            "train_history": train_history,
            "val_history": val_history,
            "best_epoch": int(payload.get("epoch", best_epoch or 0)),
            "final_train_loss": train_history[-1]["loss"] if train_history else None,
            "elapsed_seconds": time.perf_counter() - started_train,
            "checkpoint": str(new_checkpoint.resolve()),
            "checkpoint_sha256": _sha256(new_checkpoint),
        },
        "wm_val": final_wm_val,
        "h1": {
            "selectors": selectors,
            "belief_alignment": {"GT_conditioned_WM": h1_result["belief_alignment"]["Action_Discriminative_WM"]},
            "inference": h1_result["inference"],
            "real_candidate_stats": h1_result["real_candidate_stats"],
        },
        "comparison": {
            "action_discriminative_wm_e": old_wm_val,
            "gt_conditioned_wm_e": final_wm_val,
            "h1_previous_action_discriminative": previous_selectors,
        },
        "protocol": {
            "conditioning": "reduced14 ground-truth action one-hot (14-D) fused with candidate-conditioned latent",
            "loss": "L_pose + 0.1*KL(true ST-GCN distribution || predicted logits) + 0.1*(1-cosine(predicted_feature,true_feature)) + 0.2*KL(real HistoryIdentity belief || predicted HistoryIdentity belief)",
            "predicted_feature": "256-D candidate-conditioned decoder latent head",
            "predicted_logits": "14-D candidate-conditioned recognition head",
            "teacher": "frozen ST-GCN",
            "history_identity": "frozen pretrained History Identity",
            "candidate_budget": "ALL_LEGAL",
            "diagnostic_only": True,
        },
        "artifacts": {
            "base_wm_checkpoint": str(old_checkpoint.resolve()),
            "base_wm_untouched": True,
            "new_checkpoint": str(new_checkpoint.resolve()),
            "stgcn_checkpoint": str(teacher_checkpoint.resolve()),
            "history_identity_checkpoint": str(identity_checkpoint.resolve()),
        },
        "leakage_flags": {
            "test_used": False,
            "train_val_only": True,
            "formal_wm_modified": False,
            "formal_jr_modified": False,
            "formal_stgcn_modified": False,
            "gt_action_available_only_as_diagnostic_conditioning": True,
            "real_candidate_archive_used_as_training_target": True,
            "future_candidate_observation_used_as_wm_input": False,
            "rgb_lookup_scope": "Train/Val visited s0/s1 keys only",
        },
        "runtime": {"device": str(device), "elapsed_seconds": time.perf_counter() - started},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_analysis(OUTPUT_DIR / "analysis.md", result)
    (OUTPUT_DIR / "training.json").write_text(json.dumps(result["training"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=get_data_root())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GT-conditioned WM-E requires CUDA; CPU fallback is disabled")
    run(args.data_root.resolve(), device, evaluate_only=args.evaluate_only)


if __name__ == "__main__":
    main()
