"""Frozen initial-policy prediction and evaluation helpers."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

from activeview.active_view.geometry import order_candidates
from activeview.active_view.metrics import summarize_policy_predictions


def load_stage_b_lookup(path) -> Dict[str, Mapping[str, Any]]:
    import json
    return {str(row["episode_id"]): row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def _candidate_choice(predicted: Sequence[float], ids: Sequence[int], geodesic: Sequence[float]) -> tuple[int, float]:
    ordered = order_candidates(predicted, ids, geodesic)
    selected = int(ordered[0])
    return selected, float(predicted[list(ids).index(selected)])


def move_stay_decision(move_probability: float) -> bool:
    """Return whether to Stay for a sigmoid Move probability below 0.5."""
    return float(move_probability) < 0.5


def predict_dataset(
    model: torch.nn.Module,
    loader,
    stage_b_lookup: Mapping[str, Mapping[str, Any]],
    device: torch.device,
    model_type: str = "",
) -> list[Dict[str, Any]]:
    model.eval()
    rows: list[Dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            current = batch["current_feature"].to(device)
            geometry = batch["candidate_geometry"].to(device)
            mask = batch["candidate_mask"].to(device)
            predicted = model(current, geometry, mask).cpu().numpy()
            move_probabilities = None
            if model_type == "move_stay_set_ranker":
                if not hasattr(model, "move_logits"):
                    raise TypeError("move_stay_set_ranker must expose move_logits")
                move_probabilities = torch.sigmoid(model.move_logits(current)).cpu().numpy()
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
                predicted_candidate_id, max_predicted = _candidate_choice(predicted_values, ids, geo_values)
                if move_probabilities is None:
                    predicted_stays = max_predicted <= 0.0
                    move_probability = None
                else:
                    move_probability = float(move_probabilities[index])
                    predicted_stays = move_stay_decision(move_probability)
                current_id = int(stage_b["current"]["viewpoint_id"])
                selected_id = current_id if predicted_stays else predicted_candidate_id
                selected_stage_b = stage_b["current"] if predicted_stays else by_id[selected_id]
                oracle_id = int(stage_b["oracle"]["candidate_oracle_viewpoint_id"])
                safe_id = int(stage_b["oracle"]["safe_oracle_viewpoint_id"])
                oracle_item = by_id[oracle_id]
                safe_item = stage_b["current"] if bool(stage_b["oracle"]["safe_oracle_stays"]) else by_id[safe_id]
                rows.append({
                    "episode_id": str(episode_id), "record_id": str(batch["record_id"][index]),
                    "policy_split": str(batch["policy_split"][index]), "scene_id": str(batch["scene_id"][index]),
                    "region": str(batch["region"][index]), "label_id": int(batch["label_id"][index]),
                    "current_viewpoint_id": current_id,
                    "candidate_viewpoint_ids": ids, "utility_targets": target_values,
                    "predicted_utilities": predicted_values,
                    "predicted_candidate_viewpoint_id": predicted_candidate_id,
                    "predicted_action": "stay" if predicted_stays else f"candidate:{predicted_candidate_id}",
                    "predicted_stays": predicted_stays,
                    "move_probability": move_probability,
                    "selected_true_utility": 0.0 if predicted_stays else float(by_id[selected_id]["utility"]),
                    "selected_predicted_label_id": int(selected_stage_b["predicted_label_id"]),
                    "selected_entropy": float(selected_stage_b["entropy"]),
                    "current_predicted_label_id": int(stage_b["current"]["predicted_label_id"]),
                    "current_entropy": float(stage_b["current"]["entropy"]),
                    "candidate_oracle_viewpoint_id": oracle_id,
                    "candidate_oracle_predicted_label_id": int(oracle_item["predicted_label_id"]),
                    "candidate_oracle_entropy": float(oracle_item["entropy"]),
                    "safe_oracle_viewpoint_id": safe_id,
                    "safe_oracle_stays": bool(stage_b["oracle"]["safe_oracle_stays"]),
                    "safe_oracle_action": "stay" if bool(stage_b["oracle"]["safe_oracle_stays"]) else f"candidate:{safe_id}",
                    "safe_oracle_utility": float(stage_b["oracle"]["safe_oracle_utility"]),
                    "safe_oracle_predicted_label_id": int(safe_item["predicted_label_id"]),
                    "safe_oracle_entropy": float(safe_item["entropy"]),
                    "regret": float(stage_b["oracle"]["safe_oracle_utility"]) - (0.0 if predicted_stays else float(by_id[selected_id]["utility"])),
                })
    return rows


def evaluate_predictions(rows: Sequence[Mapping[str, Any]], categories: Sequence[str]) -> Dict[str, Any]:
    return summarize_policy_predictions(rows, categories)
