from activeview.data.preprocessing.policy_data import RecordBalancedSampler, collate_episode_batch
import torch


def _row(record_id, index):
    return {"record_id": record_id, "current_feature": [0.0] * 275, "candidate_geometry": [[0.0] * 11] * (index + 1), "utility_targets": [0.0] * (index + 1), "candidate_viewpoint_ids": list(range(index + 1)), "candidate_geodesic": [1.0] * (index + 1), "episode_id": f"e-{record_id}-{index}", "policy_split": "train", "scene_id": "s", "region": "r", "label_id": 0}


def test_record_balanced_sampler_and_padding_collate():
    rows = [_row("a", 0), _row("a", 1), _row("b", 0)]
    sampler = RecordBalancedSampler(rows, episodes_per_record=2, seed=42)
    sampled = list(iter(sampler))
    assert len(sampled) == 4
    assert sum(rows[index]["record_id"] == "a" for index in sampled) == 2
    collate_rows = []
    for row in (rows[0], rows[1]):
        collate_rows.append({**row, "current_feature": torch.tensor(row["current_feature"]), "candidate_ids": row["candidate_viewpoint_ids"], "candidate_geometry": torch.tensor(row["candidate_geometry"]), "utility_targets": torch.tensor(row["utility_targets"]), "candidate_geodesic": torch.tensor(row["candidate_geodesic"])})
    batch = collate_episode_batch(collate_rows)
    assert batch["candidate_geometry"].shape == (2, 2, 11)
    assert torch.equal(batch["candidate_mask"], torch.tensor([[True, False], [True, True]]))
