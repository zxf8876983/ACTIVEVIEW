#!/usr/bin/env python3
"""Combine frozen control and EXP052-DH Val artifacts into the experiment record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/stage_d/EXP052_diverse_history_world_model"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    result = json.loads((EXP / "result.json").read_text())
    original = json.loads(Path("/tmp/activeview_exp052_r2_original3/result.json").read_text())
    diverse = json.loads(Path("/tmp/activeview_exp052_r2_dh3/result.json").read_text())
    original_actions = original["h2_action_signatures_moving"]; diverse_actions = diverse["h2_action_signatures_moving"]
    original_terminal = original["h2_terminal_predictions_moving"]; diverse_terminal = diverse["h2_terminal_predictions_moving"]; labels = original["moving_labels"]
    changed = [a != b for a, b in zip(original_actions, diverse_actions)]
    original_correct = [p == y for p, y in zip(original_terminal, labels)]; diverse_correct = [p == y for p, y in zip(diverse_terminal, labels)]
    changed_rescued = sum(c and not old and new for c, old, new in zip(changed, original_correct, diverse_correct))
    changed_harmful = sum(c and old and not new for c, old, new in zip(changed, original_correct, diverse_correct))
    result.update({"status": "COMPLETED", "split": ["train", "val"], "val_control": {"h1": original["h1_real"], "h2": original["h2_real"], "fused": original["fused"], "history_shift_fidelity": original["history_shift_fidelity"]}, "val_diverse_history": {"h1": diverse["h1_real"], "h2": diverse["h2_real"], "fused": diverse["fused"], "history_shift_fidelity": diverse["history_shift_fidelity"]}, "h2_action_audit": {"changed_action_count": int(sum(changed)), "rescued": int(changed_rescued), "harmful": int(changed_harmful), "net": int(changed_rescued - changed_harmful)}, "test_used": False, "training_performed": True})
    (EXP / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    fidelity = {"original_wm_e": original["history_shift_fidelity"], "diverse_history_wm_e": diverse["history_shift_fidelity"], "delta_diverse_minus_original": {name: {metric: (None if original["history_shift_fidelity"][name][metric] is None or diverse["history_shift_fidelity"][name][metric] is None else diverse["history_shift_fidelity"][name][metric] - original["history_shift_fidelity"][name][metric]) for metric in ("top1_agreement", "true_label_logp_pearson", "true_label_logp_spearman", "kl_true_to_imagined", "probability_l1")} for name in ("h0", "h1")}}
    (EXP / "fidelity_comparison.json").write_text(json.dumps(fidelity, indent=2) + "\n")
    h2 = {"original_wm_e": {"moving": original["h2_real"]["moving_subset"], "full": original["h2_real"]["full"], "fused": original["fused"]}, "diverse_history_wm_e": {"moving": diverse["h2_real"]["moving_subset"], "full": diverse["h2_real"]["full"], "fused": diverse["fused"]}, "action_audit": result["h2_action_audit"]}
    (EXP / "h2_comparison.json").write_text(json.dumps(h2, indent=2) + "\n")
    control = Path("/tmp/activeview_exp042r1_E/last.pth"); checkpoint = Path(result["train"]["checkpoint"])
    (EXP / "checkpoint_manifest.json").write_text(json.dumps({"control_checkpoint": str(control), "control_sha256": _sha256(control), "diverse_history_checkpoint": str(checkpoint), "diverse_history_sha256": _sha256(checkpoint), "test_used": False}, indent=2) + "\n")
    o_h0, o_h1 = original["history_shift_fidelity"]["h0"], original["history_shift_fidelity"]["h1"]; d_h0, d_h1 = diverse["history_shift_fidelity"]["h0"], diverse["history_shift_fidelity"]["h1"]
    analysis = f"""# EXP052 — Diverse-History WM-E\n\n## Protocol\n\nWM-E-DH uses the frozen WM-E architecture, recognition-aware loss (pose + 0.10 ST-GCN KL), and DINO spatial history features. Train sampling is deterministic (seed 42): 4 distinct ordered real history pairs × 4 excluded candidate targets per context. No Joint Revision, ST-GCN, perception, Habitat or Test artifact was changed.\n\n## Fidelity\n\n| Model | History | Agreement | Pearson | Spearman | KL | L1 |\n|---|---|---:|---:|---:|---:|---:|\n| Original WM-E | h0 [s0,s1] | {o_h0['top1_agreement']:.6f} | {o_h0['true_label_logp_pearson']:.6f} | {o_h0['true_label_logp_spearman']:.6f} | {o_h0['kl_true_to_imagined']:.6f} | {o_h0['probability_l1']:.6f} |\n| WM-E-DH | h0 [s0,s1] | {d_h0['top1_agreement']:.6f} | {d_h0['true_label_logp_pearson']:.6f} | {d_h0['true_label_logp_spearman']:.6f} | {d_h0['kl_true_to_imagined']:.6f} | {d_h0['probability_l1']:.6f} |\n| Original WM-E | h1 [s1,v1] | {o_h1['top1_agreement']:.6f} | {o_h1['true_label_logp_pearson']:.6f} | {o_h1['true_label_logp_spearman']:.6f} | {o_h1['kl_true_to_imagined']:.6f} | {o_h1['probability_l1']:.6f} |\n| WM-E-DH | h1 [s1,v1] | {d_h1['top1_agreement']:.6f} | {d_h1['true_label_logp_pearson']:.6f} | {d_h1['true_label_logp_spearman']:.6f} | {d_h1['kl_true_to_imagined']:.6f} | {d_h1['probability_l1']:.6f} |\n\nWM-E-DH decreases both h0 and h1 fidelity rather than correcting recurrent shift.\n\n## H2 real terminal recognition\n\n| Population | Original H2 Acc/F1 | WM-E-DH H2 Acc/F1 |\n|---|---:|---:|\n| Moving (9,742) | {original['h2_real']['moving_subset']['accuracy']:.6f} / {original['h2_real']['moving_subset']['macro_f1']:.6f} | {diverse['h2_real']['moving_subset']['accuracy']:.6f} / {diverse['h2_real']['moving_subset']['macro_f1']:.6f} |\n| Full (13,987) | {original['h2_real']['full']['accuracy']:.6f} / {original['h2_real']['full']['macro_f1']:.6f} | {diverse['h2_real']['full']['accuracy']:.6f} / {diverse['h2_real']['full']['macro_f1']:.6f} |\n\nSecond-step action changed in {sum(changed)} episodes; DH rescued {changed_rescued}, harmed {changed_harmful}, net {changed_rescued - changed_harmful}.\n\n## Decision\n\n**Case C (with an h0 regression):** simple diverse-history sampling did not improve h1 fidelity and reduced initial-history fidelity. It provides no evidence that this intervention resolves recurrent distribution shift; a history-aware architecture or mixed canonical/recurrent training would require separate human authorization.\n\n`test_used=false`; no H3 was run.\n"""
    (EXP / "analysis.md").write_text(analysis)


if __name__ == "__main__": main()
