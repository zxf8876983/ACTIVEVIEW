"""
BABEL 老人动作候选集筛选器 —— select_babel_elderly_actions.py
================================================================

功能：
    解析 BABEL annotation (train, val, extra_train, extra_val)，
    根据配置中的 token-aware 规则精准筛选面向室内老人助老场景的 5 类典型动作：
        1. standing
        2. sitting
        3. bending / picking
        4. reaching
        5. fall_related (区分高置信 fall 与需人工确认的 lie)
    生成完整候选清单与第一轮 Motion Feasibility Set。
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import yaml

from .data_paths import (
    get_babel_dir,
    get_cache_dir,
    get_repo_root,
)


def load_query_config(config_path: Optional[Path] = None) -> dict:
    """加载动作关键词配置。"""
    if config_path is None:
        config_path = get_repo_root() / "tools" / "motion_assets" / "elderly_action_queries.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tokenize(text: str) -> List[str]:
    """将文本标准化并拆分为 token 列表。"""
    if not text:
        return []
    cleaned = re.sub(r"[^a-zA-Z0-9\s_\-/]", " ", str(text).lower())
    return [t.strip() for t in cleaned.split() if t.strip()]


def contains_phrase_or_token(text: str, phrase_or_token: str) -> bool:
    """检查文本中是否包含指定 token 或多词短语 (token-aware, 支持标准单词边界与基础词形后缀)."""
    if not text or not phrase_or_token:
        return False
    target = phrase_or_token.strip().lower()
    text_clean = " ".join(tokenize(text))
    target_clean = " ".join(tokenize(target))
    if not target_clean:
        return False
    # 精确匹配整个词/短语或带标准词形后缀 (s/ed/ing/d)
    pattern = r"\b" + re.escape(target_clean) + r"(?:s|ed|ing|d)?\b"
    return bool(re.search(pattern, text_clean))


def extract_amass_dataset(feat_p: str) -> str:
    """从 BABEL feat_p 提取第一级 AMASS 子数据集名称。"""
    if not feat_p:
        return "UNKNOWN"
    norm = feat_p.replace("\\", "/").strip().lstrip("/")
    parts = norm.split("/")
    return parts[0] if parts else "UNKNOWN"


def match_action(
    act_cat: List[str],
    proc_label: str,
    raw_label: str,
    query_cfg: dict,
) -> Optional[Tuple[str, bool, Optional[str]]]:
    """判定一段标注是否匹配 5 类动作中的某一类。

    返回值：
        None: 不匹配任何目标动作或被显式排除
        (target_class, needs_manual_review, match_source): 匹配成功
    """
    classes_cfg = query_cfg.get("action_classes", {})
    all_texts = []
    if act_cat:
        all_texts.extend(act_cat)
    if proc_label:
        all_texts.append(proc_label)
    if raw_label:
        all_texts.append(raw_label)
    full_text = " ; ".join(all_texts).lower()

    # 1. Fall-related (最高优先级分类)
    fall_cfg = classes_cfg.get("fall_related", {})
    # 检查 fall 排除词 (如 sleep, sunbathing 等)
    for exc in fall_cfg.get("exclude", []):
        if contains_phrase_or_token(full_text, exc):
            return None

    # 高置信 fall
    for inc in fall_cfg.get("high_confidence_include", []):
        if contains_phrase_or_token(full_text, inc):
            return ("fall_related", False, f"high_confidence:{inc}")

    # 弱候选 lie (需要人工确认是否为真实跌倒/倒地)
    for inc in fall_cfg.get("weak_include", []):
        if contains_phrase_or_token(full_text, inc):
            return ("fall_related", True, f"weak_candidate:{inc}")

    # 2. Bending / Picking
    bend_cfg = classes_cfg.get("bending", {})
    is_bend_excluded = any(contains_phrase_or_token(full_text, exc) for exc in bend_cfg.get("exclude", []))
    if not is_bend_excluded:
        for inc in bend_cfg.get("include", []):
            if contains_phrase_or_token(full_text, inc):
                return ("bending", False, f"include:{inc}")

    # 3. Reaching
    reach_cfg = classes_cfg.get("reaching", {})
    is_reach_excluded = any(contains_phrase_or_token(full_text, exc) for exc in reach_cfg.get("exclude", []))
    if not is_reach_excluded:
        for inc in reach_cfg.get("include", []):
            if contains_phrase_or_token(full_text, inc):
                return ("reaching", False, f"include:{inc}")

    # 4. Sitting
    sit_cfg = classes_cfg.get("sitting", {})
    is_sit_excluded = any(contains_phrase_or_token(full_text, exc) for exc in sit_cfg.get("exclude", []))
    if not is_sit_excluded:
        for inc in sit_cfg.get("include", []):
            if contains_phrase_or_token(full_text, inc):
                return ("sitting", False, f"include:{inc}")

    # 5. Standing
    stand_cfg = classes_cfg.get("standing", {})
    is_stand_excluded = any(contains_phrase_or_token(full_text, exc) for exc in stand_cfg.get("exclude", []))
    if not is_stand_excluded:
        for inc in stand_cfg.get("include", []):
            if contains_phrase_or_token(full_text, inc):
                return ("standing", False, f"include:{inc}")

    return None


def parse_babel_split(
    split_name: str,
    json_path: Path,
    query_cfg: dict,
) -> List[dict]:
    """解析单个 BABEL JSON 文件并提取所有符合条件的动作片段。"""
    if not json_path.exists():
        print(f"[Warning] BABEL annotation file not found: {json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = []

    for sid, entry in data.items():
        babel_sid = entry.get("babel_sid", sid)
        url = entry.get("url", "")
        feat_p = entry.get("feat_p", "")
        seq_dur = float(entry.get("dur", 0.0))
        amass_dataset = extract_amass_dataset(feat_p)

        # 1. 检查 frame_ann / frame_anns (精准帧级片段)
        frame_ann_list = []
        if entry.get("frame_ann"):
            frame_ann_list.append(entry["frame_ann"])
        if entry.get("frame_anns"):
            if isinstance(entry["frame_anns"], list):
                frame_ann_list.extend(entry["frame_anns"])
            elif isinstance(entry["frame_anns"], dict):
                frame_ann_list.append(entry["frame_anns"])

        for f_ann in frame_ann_list:
            if not f_ann or "labels" not in f_ann:
                continue
            for lbl in f_ann.get("labels", []):
                act_cat = lbl.get("act_cat", [])
                proc_label = lbl.get("proc_label", "")
                raw_label = lbl.get("raw_label", "")
                start_t = float(lbl.get("start_t", 0.0))
                end_t = float(lbl.get("end_t", seq_dur))
                seg_dur = max(0.0, end_t - start_t)

                match_res = match_action(act_cat, proc_label, raw_label, query_cfg)
                if match_res is not None:
                    target_class, needs_review, match_src = match_res
                    candidates.append({
                        "target_class": target_class,
                        "source_split": split_name,
                        "babel_sid": babel_sid,
                        "annotation_level": "frame_level",
                        "act_cat": act_cat,
                        "proc_label": proc_label,
                        "raw_label": raw_label,
                        "start_t": round(start_t, 3),
                        "end_t": round(end_t, 3),
                        "segment_duration": round(seg_dur, 3),
                        "sequence_duration": round(seq_dur, 3),
                        "feat_p": feat_p,
                        "amass_dataset": amass_dataset,
                        "babel_render_url": url,
                        "needs_manual_review": needs_review,
                        "match_source": match_src,
                    })

        # 2. 检查 seq_ann / seq_anns (整段序列级标注)
        seq_ann_list = []
        if entry.get("seq_ann"):
            seq_ann_list.append(entry["seq_ann"])
        if entry.get("seq_anns"):
            if isinstance(entry["seq_anns"], list):
                seq_ann_list.extend(entry["seq_anns"])
            elif isinstance(entry["seq_anns"], dict):
                seq_ann_list.append(entry["seq_anns"])

        for s_ann in seq_ann_list:
            if not s_ann or "labels" not in s_ann:
                continue
            for lbl in s_ann.get("labels", []):
                act_cat = lbl.get("act_cat", [])
                proc_label = lbl.get("proc_label", "")
                raw_label = lbl.get("raw_label", "")
                start_t = 0.0
                end_t = seq_dur
                seg_dur = seq_dur

                match_res = match_action(act_cat, proc_label, raw_label, query_cfg)
                if match_res is not None:
                    target_class, needs_review, match_src = match_res
                    candidates.append({
                        "target_class": target_class,
                        "source_split": split_name,
                        "babel_sid": babel_sid,
                        "annotation_level": "sequence_level",
                        "act_cat": act_cat,
                        "proc_label": proc_label,
                        "raw_label": raw_label,
                        "start_t": round(start_t, 3),
                        "end_t": round(end_t, 3),
                        "segment_duration": round(seg_dur, 3),
                        "sequence_duration": round(seq_dur, 3),
                        "feat_p": feat_p,
                        "amass_dataset": amass_dataset,
                        "babel_render_url": url,
                        "needs_manual_review": needs_review,
                        "match_source": match_src,
                    })

    return candidates


def select_feasibility_set(
    candidates: List[dict],
    quotas: Optional[Dict[str, int]] = None,
) -> List[dict]:
    """从候选池中选出第一轮高质量 Motion Feasibility Set。

    选择规则：
        1. 优先选择 frame_level 精准片段；
        2. 优先选择典型/纯粹动作标签 (如 stand, sit down, pick up, bend down, reach forward, fall to the ground)；
        3. 时长合理 (1.5s <= duration <= 8.0s 优先)；
        4. feat_p 严格去重 (每条 sequence 最多选一个代表 segment)；
        5. fall_related 必须优先选择 high confidence (needs_manual_review == False) 的真实 fall/trip/fallen；
        6. 优先选择主流代表性 AMASS 子库 (如 CMU, KIT, EyesJapanDataset, BMLrub, MPIHDM05, BMLmovi)。
    """
    if quotas is None:
        quotas = {
            "standing": 3,
            "sitting": 3,
            "bending": 3,
            "reaching": 3,
            "fall_related": 5,
        }

    canonical_label_keywords = {
        "standing": ["stand", "stand still", "stand in place", "just stand"],
        "sitting": ["sit", "sit down", "sit on chair", "seated"],
        "bending": ["pick up", "bend down", "bend over", "pick up object", "bend forward", "pick"],
        "reaching": ["reach for", "reach forward", "reach right hand", "reach left hand", "reach upwards", "reach"],
        "fall_related": ["fall to the ground", "fall down", "fall forward", "fall back", "trip", "walk and trip", "fall"],
    }

    preferred_amass_datasets = {"CMU", "KIT", "EyesJapanDataset", "BMLrub", "MPIHDM05", "BMLmovi", "EKUT", "ACCAD"}

    selected = []
    used_feat_p = set()

    for target_class, quota in quotas.items():
        class_cands = [c for c in candidates if c["target_class"] == target_class]
        canonical_kws = canonical_label_keywords.get(target_class, [])

        def sort_key(item):
            review_penalty = 1 if item["needs_manual_review"] else 0
            level_penalty = 0 if item["annotation_level"] == "frame_level" else 1

            # 动作标签典型度打分
            proc = item["proc_label"].lower()
            raw = item["raw_label"].lower()
            label_score = 10
            for idx, kw in enumerate(canonical_kws):
                if proc == kw or raw == kw:
                    label_score = idx
                    break
                elif proc.startswith(kw) or raw.startswith(kw):
                    label_score = idx + 2
                    break
                elif kw in proc or kw in raw:
                    label_score = idx + 4
                    break

            # 子库偏好打分
            ds = item["amass_dataset"]
            ds_penalty = 0 if ds in preferred_amass_datasets else 1

            # 时长下限惩罚 (严禁优先选取 < 1.0s 的超短碎片)
            dur = item["segment_duration"]
            is_too_short = 1 if dur < 1.0 else 0

            if 1.5 <= dur <= 6.0:
                dur_penalty = abs(dur - 3.2)
            else:
                dur_penalty = abs(dur - 3.2) + 10.0

            return (review_penalty, is_too_short, level_penalty, ds_penalty, label_score, dur_penalty)

        class_cands.sort(key=sort_key)

        class_picked = 0
        for cand in class_cands:
            fp = cand["feat_p"]
            if fp in used_feat_p:
                continue

            # 对 fall_related，第一轮 feasibility 严格要求 high confidence
            if target_class == "fall_related" and cand["needs_manual_review"]:
                continue

            selected.append(cand)
            used_feat_p.add(fp)
            class_picked += 1
            if class_picked >= quota:
                break

    return selected


def main():
    parser = argparse.ArgumentParser(description="Select Elderly Action Motions from BABEL Annotations")
    parser.add_argument("--config", type=str, default=None, help="Path to elderly_action_queries.yaml")
    parser.add_argument("--babel-dir", type=str, default=None, help="Path to BABEL annotations directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output cache directory")
    args = parser.parse_args()

    query_cfg = load_query_config(Path(args.config) if args.config else None)
    babel_dir = Path(args.babel_dir) if args.babel_dir else get_babel_dir()
    out_dir = Path(args.output_dir) if args.output_dir else get_cache_dir("babel_selection")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[BABEL Selector] BABEL Annotation Dir: {babel_dir}")
    print(f"[BABEL Selector] Output Cache Dir:     {out_dir}")

    splits = ["train", "val", "extra_train", "extra_val"]
    all_candidates = []

    for split in splits:
        json_file = babel_dir / f"{split}.json"
        cands = parse_babel_split(split, json_file, query_cfg)
        print(f"  - Parsed {split}.json: {len(cands)} candidate action segments")
        all_candidates.extend(cands)

    print(f"[BABEL Selector] Total raw candidate action segments: {len(all_candidates)}")

    # 1. 导出 full candidate json & csv
    cand_json_path = out_dir / "elderly_action_candidates.json"
    with open(cand_json_path, "w", encoding="utf-8") as f:
        json.dump(all_candidates, f, indent=2, ensure_ascii=False)

    cand_csv_path = out_dir / "elderly_action_candidates.csv"
    if all_candidates:
        fieldnames = list(all_candidates[0].keys())
        with open(cand_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_candidates)

    # 2. 导出 manual review 列表 (弱候选 lie 等)
    manual_review_items = [c for c in all_candidates if c["needs_manual_review"]]
    manual_csv_path = out_dir / "manual_review_fall_related.csv"
    if manual_review_items:
        with open(manual_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manual_review_items[0].keys()))
            writer.writeheader()
            writer.writerows(manual_review_items)

    # 3. 统计 summary
    classes = ["standing", "sitting", "bending", "reaching", "fall_related"]
    summary = {}
    amass_dataset_counts = {}

    for cls in classes:
        cls_items = [c for c in all_candidates if c["target_class"] == cls]
        unique_seqs = sorted(list(set(c["feat_p"] for c in cls_items)))
        cls_amass = sorted(list(set(c["amass_dataset"] for c in cls_items)))
        high_conf_count = sum(1 for c in cls_items if not c["needs_manual_review"])
        weak_count = sum(1 for c in cls_items if c["needs_manual_review"])

        summary[cls] = {
            "candidate_segments": len(cls_items),
            "unique_sequences": len(unique_seqs),
            "high_confidence_segments": high_conf_count,
            "manual_review_segments": weak_count,
            "amass_datasets": cls_amass,
        }

        for c in cls_items:
            ds = c["amass_dataset"]
            amass_dataset_counts[ds] = amass_dataset_counts.get(ds, 0) + 1

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 4. 导出所有 AMASS 子库列表与 sequence 列表
    sorted_amass = sorted(amass_dataset_counts.items(), key=lambda x: x[1], reverse=True)
    subds_path = out_dir / "amass_subdatasets.txt"
    with open(subds_path, "w", encoding="utf-8") as f:
        for ds, cnt in sorted_amass:
            f.write(f"{ds}\t{cnt}\n")

    all_seq_paths = sorted(list(set(c["feat_p"] for c in all_candidates)))
    seq_paths_file = out_dir / "amass_sequence_paths.txt"
    with open(seq_paths_file, "w", encoding="utf-8") as f:
        for p in all_seq_paths:
            f.write(f"{p}\n")

    # 5. 选取第一轮 Motion Feasibility Set
    feasibility_set = select_feasibility_set(all_candidates)
    feas_json_path = out_dir / "feasibility_manifest.json"
    with open(feas_json_path, "w", encoding="utf-8") as f:
        json.dump(feasibility_set, f, indent=2, ensure_ascii=False)

    feas_csv_path = out_dir / "feasibility_manifest.csv"
    if feasibility_set:
        with open(feas_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(feasibility_set[0].keys()))
            writer.writeheader()
            writer.writerows(feasibility_set)

    # 6. 生成 feasibility 所需的 AMASS 子库与 sequences
    feas_subdatasets = sorted(list(set(c["amass_dataset"] for c in feasibility_set)))
    feas_subds_path = out_dir / "required_amass_subdatasets_feasibility.txt"
    with open(feas_subds_path, "w", encoding="utf-8") as f:
        for ds in feas_subdatasets:
            f.write(f"{ds}\n")

    feas_seqs = sorted(list(set(c["feat_p"] for c in feasibility_set)))
    feas_seqs_path = out_dir / "required_amass_sequences_feasibility.txt"
    with open(feas_seqs_path, "w", encoding="utf-8") as f:
        for s in feas_seqs:
            f.write(f"{s}\n")

    print("\n[BABEL Selector] Summary Statistics:")
    for cls, st in summary.items():
        print(f"  - {cls.ljust(14)}: {st['candidate_segments']} segments, {st['unique_sequences']} sequences (HighConf: {st['high_confidence_segments']}, ManualReview: {st['manual_review_segments']})")

    print(f"\n[BABEL Selector] Selected Feasibility Set: {len(feasibility_set)} sequences")
    for item in feasibility_set:
        print(f"    [{item['target_class'].ljust(12)}] sid={item['babel_sid']} | {item['amass_dataset']} | {item['proc_label']} | dur={item['segment_duration']}s | {item['feat_p']}")

    print(f"\n[BABEL Selector] Feasibility Set required AMASS subdatasets: {feas_subdatasets}")


if __name__ == "__main__":
    main()
