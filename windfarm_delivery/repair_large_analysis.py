"""Frozen primary HV/paired statistics plus repair-cost diagnostics."""
from pathlib import Path
import sys
import json
import csv
import hashlib
import datetime
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "windfarm_research_v4/posthoc"))
from confirm_analysis import run as primary_analysis


def table(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run(src, cfg):
    summaries = [json.loads(p.read_text()) for p in src.glob("*/DONE.json") if (p.parent / "evaluations.npz").exists()]
    assert len(summaries) == 60
    for seed in cfg["seeds"]:
        group = [r for r in summaries if r["seed"] == seed]
        assert len(group) == 6
        for field in ("initial_keys_sha256", "initial_layouts_sha256", "initial_metrics_sha256"):
            assert len({r[field] for r in group}) == 1, field
    primary_analysis(src, cfg)
    out = src / "analysis"
    with (out / "per_run_metrics.csv").open(encoding="utf-8-sig", newline="") as f:
        metrics = list(csv.DictReader(f))
    by_pair = {(r["variant"], int(r["seed"])): float(r["HV"]) for r in metrics}
    differences = []
    independently_checked = []
    for treatment, control in cfg["paired_contrasts"]:
        delta = np.array([by_pair[treatment, seed]-by_pair[control, seed] for seed in cfg["seeds"]])
        # Independently enumerate signed sums by recursive doubling rather than
        # the matrix-product implementation used by primary_analysis.
        sums = np.array([0.])
        for d in delta:
            sums = np.r_[sums+d, sums-d]
        p = float(np.mean(np.abs(sums/len(delta)) >= abs(float(delta.mean()))-1e-14))
        independently_checked.append(dict(treatment=treatment, control=control, independent_p=p))
        for seed, d in zip(cfg["seeds"], delta):
            differences.append(dict(seed=seed, treatment=treatment, control=control,
                treatment_HV=by_pair[treatment, seed], control_HV=by_pair[control, seed], difference=float(d)))
    order = sorted(range(4), key=lambda i: independently_checked[i]["independent_p"])
    maxp = 0.
    for rank, i in enumerate(order):
        maxp = max(maxp, min(1., independently_checked[i]["independent_p"]*(4-rank)))
        independently_checked[i]["independent_Holm_p"] = maxp
    with (out / "paired_comparisons.csv").open(encoding="utf-8-sig", newline="") as f:
        original = list(csv.DictReader(f))
    for row in independently_checked:
        expected = next(r for r in original if r["treatment"] == row["treatment"] and r["control"] == row["control"])
        assert abs(row["independent_p"]-float(expected["paired_signflip_p"])) < 1e-14
        assert abs(row["independent_Holm_p"]-float(expected["Holm_adjusted_p"])) < 1e-14
    table(out / "individual_paired_differences.csv", differences)
    table(out / "independent_statistical_verification.csv", independently_checked)
    diagnostic_fields = ["offspring_raw_feasible_rate", "offspring_post_feasible_rate", "unique_feasible_layouts",
        "offspring_feasible_duplicates", "offspring_infeasible_duplicates", "offspring_greedy_calls",
        "offspring_greedy_failures", "offspring_backtrack_calls", "offspring_backtrack_attempts",
        "offspring_backtrack_successes", "offspring_decode_seconds", "offspring_physical_evaluations",
        "offspring_velocity_coordinates_reset", "initialization_seconds", "elapsed_seconds", "checkpoint_restore_seconds"]
    diagnostic_rows = []
    for r in summaries:
        diagnostic_rows.append(dict(variant=r["variant"], seed=r["seed"],
            **{k:r[k] for k in diagnostic_fields},
            active_seconds_including_initialization_and_restore=r["elapsed_seconds"]+r["initialization_seconds"]+r["checkpoint_restore_seconds"]))
    table(out / "repair_diagnostics_per_run.csv", diagnostic_rows)
    aggregate = []
    for variant in cfg["variants"]:
        rr = [r for r in diagnostic_rows if r["variant"] == variant]
        row = dict(variant=variant, n=len(rr))
        for field in diagnostic_fields + ["active_seconds_including_initialization_and_restore"]:
            row[field+"_mean"] = float(np.mean([r[field] for r in rr]))
            row[field+"_sd"] = float(np.std([r[field] for r in rr], ddof=1))
        aggregate.append(row)
    table(out / "repair_diagnostics_aggregate.csv", aggregate)
    bounds = json.loads((out / "normalization.json").read_text())
    for row in bounds.values():
        row["zero_range_axes"] = [i for i, (a,b) in enumerate(zip(row["ideal"],row["nadir"])) if a == b]
        row["normalization_width"] = [b-a if b>a else 1. for a,b in zip(row["ideal"],row["nadir"])]
    (out / "normalization.json").write_text(json.dumps(bounds, indent=2), encoding="utf-8")
    note = (
        "本批只比较LMOCSO、DGEA适配中的三种间距修复策略。前100个可行初始化在所有组完全一致，计入10000请求预算。\n"
        "修复后可行率、重复、调用次数等带offspring前缀的指标只统计第101—10000次请求。\n"
        "elapsed_seconds是初始化后活动运行时间；另列初始化和断点装载时间。停机/排队时间不算算法计算时间。\n"
        "若贪心和有限回退结果相同，结合回退调用次数判断，保留零效应，不调整实例寻找优势。\n"
        "修复策略会连带改变回写及已有速度清零响应，属于整个策略的效果，不冒称纯几何操作的独立因果效果。\n"
        "主检验预先固定为四组配对比较并统一Holm校正。初始可行解不代表优化已解决；以可行Pareto解集HV及物理目标权衡判断质量。\n"
        "仅一个实例的局部机制证据；不能推广为所有风况、地形或原论文算法的普遍优劣。\n")
    (out / "结果解释边界.txt").write_text(note, encoding="utf-8")
    result = json.loads((out / "DONE.json").read_text())
    result.update(all_initial_keys_layouts_and_metrics_matched=True, independently_verified_tests=4,
                  diagnostics_complete=True, analysis_finished_at=datetime.datetime.now().astimezone().isoformat())
    (out / "DONE.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
