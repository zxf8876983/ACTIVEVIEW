"""Val-only prediction helpers for Stage C-v2 policy architectures."""

from __future__ import annotations

from typing import Any, Dict, Mapping
import torch

from activeview.active_view.stage_c_evaluation import _candidate_choice


def forward_policy(model: torch.nn.Module, model_type: str, batch: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    """Dispatch a v2 model without exposing any candidate perception."""
    geometry = batch["candidate_geometry"].to(device)
    mask = batch["candidate_mask"].to(device)
    semantic = batch["semantic_context"].to(device)
    if model_type in {"joint_aware_set_ranker", "candidate_conditioned_attention"}:
        return model(batch["joint_tokens"].to(device), semantic, geometry, mask)
    if model_type == "skeleton_policy_transformer":
        return model(batch["skeleton"].to(device), semantic, geometry, mask)
    raise ValueError(f"Unsupported Stage C-v2 model type: {model_type}")


def predict_dataset_v2(
    model: torch.nn.Module,
    model_type: str,
    loader,
    stage_b_lookup: Mapping[str, Mapping[str, Any]],
    device: torch.device,
) -> list[Dict[str, Any]]:
    model.eval()
    rows: list[Dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            predicted = forward_policy(model, model_type, batch, device).cpu().numpy()
            targets = batch["utility_targets"].numpy()
            valid_mask = batch["candidate_mask"].numpy()
            geodesic = batch["candidate_geodesic"].numpy()
            for index, episode_id in enumerate(batch["episode_id"]):
                ids = batch["candidate_ids"][index]
                target_values = [float(value) for value in targets[index][valid_mask[index]]]
                predicted_values = [float(value) for value in predicted[index][valid_mask[index]]]
                geo_values = [float(value) for value in geodesic[index][valid_mask[index]]]
                stage_b = stage_b_lookup[str(episode_id)]
                by_id = {int(item["viewpoint_id"]): item for item in stage_b["candidates"]}
                predicted_id, max_predicted = _candidate_choice(predicted_values, ids, geo_values)
                predicted_stays = max_predicted <= 0.0
                current_id = int(stage_b["current"]["viewpoint_id"])
                selected_id = current_id if predicted_stays else predicted_id
                selected = stage_b["current"] if predicted_stays else by_id[selected_id]
                oracle = stage_b["oracle"]
                oracle_id = int(oracle["candidate_oracle_viewpoint_id"])
                safe_id = int(oracle["safe_oracle_viewpoint_id"])
                candidate_oracle = by_id[oracle_id]
                safe_stays = bool(oracle["safe_oracle_stays"])
                safe_item = stage_b["current"] if safe_stays else by_id[safe_id]
                rows.append({
                    "episode_id": str(episode_id), "record_id": str(batch["record_id"][index]),
                    "policy_split": str(batch["policy_split"][index]), "scene_id": str(batch["scene_id"][index]),
                    "region": str(batch["region"][index]), "label_id": int(batch["label_id"][index]),
                    "current_viewpoint_id": current_id, "candidate_viewpoint_ids": ids,
                    "utility_targets": target_values, "predicted_utilities": predicted_values,
                    "predicted_candidate_viewpoint_id": predicted_id,
                    "predicted_action": "stay" if predicted_stays else f"candidate:{predicted_id}",
                    "predicted_stays": predicted_stays,
                    "selected_true_utility": 0.0 if predicted_stays else float(selected["utility"]),
                    "selected_predicted_label_id": int(selected["predicted_label_id"]),
                    "selected_entropy": float(selected["entropy"]),
                    "current_predicted_label_id": int(stage_b["current"]["predicted_label_id"]),
                    "current_entropy": float(stage_b["current"]["entropy"]),
                    "current_margin": float(batch["current_margin"][index]),
                    "current_pose_confidence": float(batch["current_pose_confidence"][index]),
                    "candidate_oracle_viewpoint_id": oracle_id,
                    "candidate_oracle_predicted_label_id": int(candidate_oracle["predicted_label_id"]),
                    "candidate_oracle_entropy": float(candidate_oracle["entropy"]),
                    "safe_oracle_viewpoint_id": safe_id, "safe_oracle_stays": safe_stays,
                    "safe_oracle_action": "stay" if safe_stays else f"candidate:{safe_id}",
                    "safe_oracle_utility": float(oracle["safe_oracle_utility"]),
                    "safe_oracle_predicted_label_id": int(safe_item["predicted_label_id"]),
                    "safe_oracle_entropy": float(safe_item["entropy"]),
                    "regret": float(oracle["safe_oracle_utility"]) - (0.0 if predicted_stays else float(selected["utility"])),
                })
    return rows
