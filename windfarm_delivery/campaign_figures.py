"""Traceable final figures; requires a complete campaign and its analysis."""
from pathlib import Path
import csv
import datetime
import gc
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "windfarm_research_v4"))
import bootstrap_v4
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
from algorithms import NDS
from fast_audit import independent_check
from evaluator import Scenario


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csvrows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def table(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def savefig(fig, folder, name):
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{ext}", dpi=210, bbox_inches="tight", pad_inches=.14)
    plt.close(fig)


def runtime_accounting(src, row):
    """Log filename timestamps bracket launches, including any restart gaps."""
    logs = sorted((src / row["name"]).glob("worker_*.log"))
    assert logs, "Runtime provenance is missing"
    start = datetime.datetime.strptime(logs[0].stem.removeprefix("worker_"),
                                       "%Y%m%d_%H%M%S_%f").astimezone()
    finish = datetime.datetime.fromisoformat(row["finished_at"])
    wall = (finish - start).total_seconds()
    timer = float(row["elapsed_seconds"])
    assert wall >= timer - 1, "Unexpected timer/launch provenance"
    return dict(run=row["name"], variant=row["variant"], seed=int(row["seed"]),
        worker_recorded_seconds=timer, first_launch_to_finish_seconds=wall,
        launch_log_count=len(logs), first_launch_log=str(logs[0]),
        first_launch_timestamp=start.isoformat(), finished_at=finish.isoformat(),
        timer_scope="After initialization/restoration; search, checkpoints, final audit and serialization",
        wall_scope="Launch/import/setup to finish; includes gaps if restarted, not active CPU time",
        concurrent_load="Shared host; not an isolated runtime benchmark")


def check_complete(src):
    assert (src / "DONE.json").exists(), "Campaign is incomplete; do not publish partial rankings"
    assert (src / "analysis/DONE.json").exists(), "Campaign analysis is incomplete"
    cfg = read(src / "frozen_config.json")
    rows = csvrows(src / "analysis/per_run_metrics.csv")
    expected = len(cfg["cases"]) * len(cfg["seeds"]) * len(cfg["variants"])
    assert len(rows) == expected
    assert len({r["name"] for r in rows}) == expected
    assert all(int(r["requests"]) == cfg["evaluations"] for r in rows)
    manifest = read(src / "manifest.json")
    for path, digest in manifest.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    return cfg, rows


def case_figures(src, out, case, rows, cfg, bounds, curves):
    folder = out / case
    folder.mkdir(exist_ok=True)
    variants = cfg["variants"]
    seeds = cfg["seeds"]
    assert {(r["variant"], int(r["seed"])) for r in rows} == {
        (v, s) for v in variants for s in seeds}
    for seed in seeds:
        assert len({r["initial_layouts_sha256"] for r in rows if int(r["seed"]) == seed}) == 1
    # Curves and feasibility are derived from every seed, not a selected winner.
    fig, ax = plt.subplots(figsize=(7.2, 4.9), layout="constrained")
    for index, variant in enumerate(variants):
        selected = [r for r in curves if r["case"] == case and r["variant"] == variant]
        times = sorted({int(r["requests"]) for r in selected})
        mean, sd = [], []
        for t in times:
            vals = [float(r["HV"]) for r in selected if int(r["requests"]) == t]
            assert len(vals) == len(seeds)
            mean.append(np.mean(vals)); sd.append(np.std(vals, ddof=1))
        mean = np.array(mean); sd = np.array(sd)
        color = {
            "NSGA3_base": "#1F77B4", "CCwake_base": "#D57B21", "CCrandom_base": "#2C8C4C",
            "LM_base": "#BE3B38", "LM_range_APD": "#8655A5", "LM_fill_population": "#815A41",
            "DGEA_base": "#C15D93", "DGEA_no_writeback": "#59646F",
        }.get(variant, plt.get_cmap("tab10")(index % 10))
        line, = ax.plot(times, mean, label=variant, color=color, lw=1.5)
        ax.fill_between(times, mean - sd, mean + sd, color=line.get_color(), alpha=.08)
    ax.set(xlabel="评价请求数（包含重复布局）", ylabel="共同归一化 HV", title=case)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.19), ncol=2, fontsize=8, frameon=False)
    ax.grid(alpha=.15)
    savefig(fig, folder, "convergence")

    summary = []
    for v in variants:
        rr = [r for r in rows if r["variant"] == v]
        summary.append(dict(variant=v, **{
            col: float(np.mean([float(r[col]) for r in rr])) for col in (
                "raw_feasible_rate", "post_feasible_rate", "duplicate_rate",
                "unique_feasible_layouts", "mean_population", "elapsed_seconds", "HV")}))
    fig, ax = plt.subplots(figsize=(7.2, max(3.7, .38 * len(variants) + 1)), layout="constrained")
    yy = np.arange(len(variants))
    ax.barh(yy - .18, [r["raw_feasible_rate"] * 100 for r in summary], height=.34,
            color="#BD8B50", label="解码后 修复前")
    ax.barh(yy + .18, [r["post_feasible_rate"] * 100 for r in summary], height=.34,
            color="#267C8D", label="修复后")
    ax.set(yticks=yy, yticklabels=variants, xlabel="可行评价占比（%）", xlim=(0, 104),
           title=case)
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.14), ncol=2, frameon=False)
    savefig(fig, folder, "feasibility")
    table(folder / "case_summary.csv", summary)
    table(folder / "runtime_accounting.csv", [runtime_accounting(src, r) for r in rows])

    # Pool only within the same candidate set and physical model.
    pooled, idslist, provenance, seen = [], [], [], set()
    candidate_xy = None
    for row in rows:
        with np.load(src / row["name"] / "evaluations.npz") as z:
            xy = z["candidate_xy"]
            if candidate_xy is None:
                candidate_xy = xy.copy()
            else:
                assert np.array_equal(xy, candidate_xy)
            for i in z["pareto_request_indices"]:
                assert z["cv"][i] == 0
                selected = z["selected"][i]
                packed = selected.tobytes()
                if packed in seen:
                    continue
                seen.add(packed)
                pooled.append(z["metrics"][i].copy())
                idslist.append(selected.copy())
                provenance.append(dict(run=row["name"], variant=row["variant"],
                    seed=int(row["seed"]), request_index_zero_based=int(i)))
    metrics = np.array(pooled)
    f = metrics[:, :3] * [-1, 1, 1]
    nd = NDS.do(f, only_non_dominated_front=True)
    assert len(nd) > 0
    lo, hi = np.array(bounds["ideal"]), np.array(bounds["nadir"])
    scaled = (f - lo) / np.where(hi > lo, hi - lo, 1)
    choices = dict(
        compromise=int(nd[np.argmin(np.linalg.norm(scaled[nd], axis=1))]),
        maximum_energy=int(nd[np.argmax(metrics[nd, 0])]),
        minimum_lcoe=int(nd[np.argmin(metrics[nd, 1])]),
        minimum_area=int(nd[np.argmin(metrics[nd, 2])]))
    representative = {}
    first = rows[0]
    terrain_kind = first.get("terrain")
    if terrain_kind:
        sys.path.insert(0, str(ROOT / "windfarm_terrain_v5"))
        from model_terrain import TerrainFarm
        farm = TerrainFarm(first["scenario"], int(first["candidates"]), int(first["turbines"]),
            terrain_kind=terrain_kind, resolution=cfg["resolution_m"],
            slope_limit_deg=cfg["slope_limit_deg"])
    else:
        from model_v4 import Farm
        farm = Farm(first["scenario"], int(first["candidates"]), int(first["turbines"]))
    assert np.array_equal(farm.xy, candidate_xy)
    for label, i in choices.items():
        selected = idslist[i]
        xy = candidate_xy[selected]
        assert len(np.unique(selected)) == int(first["turbines"])
        assert independent_check(farm.scene, xy)
        if terrain_kind:
            from shapely.geometry import Point
            assert all(not farm.terrain.excluded.covers(Point(p)) for p in xy)
        # Re-evaluate the exact saved layout using frozen physical inputs.
        keys = np.linspace(.6, .99, len(candidate_xy))
        keys[selected] = np.linspace(.01, .4, len(selected))
        _, _, cv = farm.evaluate(keys)
        assert cv == 0
        assert np.array_equal(np.frombuffer(farm.records[-1][0], dtype="<u2"), selected)
        assert np.allclose(farm.records[-1][1], metrics[i], rtol=1e-10, atol=1e-7)
        record = dict(provenance[i], criterion=label, AEP_GWh=float(metrics[i, 0]),
            LCOE_USD2022_MWh=float(metrics[i, 1]), effective_area_km2=float(metrics[i, 2]),
            connection_length_m=float(metrics[i, 3]), gross_area_km2=float(metrics[i, 4]),
            selected_candidate_indices=selected.tolist(),
            independent_geometry_passed=True, fixed_layout_reevaluation_passed=True)
        representative[label] = record
        table(folder / f"{label}_coordinates.csv", [
            dict(candidate_index_zero_based=int(j), x_m=float(candidate_xy[j, 0]),
                 y_m=float(candidate_xy[j, 1]),
                 synthetic_elevation_m=float(farm.terrain.sample(candidate_xy[j])) if terrain_kind else "")
            for j in selected])
    (folder / "representative_layouts.json").write_text(json.dumps(dict(case=case,
        scope="Nondominated among visited layouts in this case, not global Pareto optimality",
        compromise_rule="Minimum equal-scale Euclidean distance to empirical ideal",
        unique_pooled_nondominated=len(nd), layouts=representative), ensure_ascii=False, indent=2), encoding="utf-8")
    table(folder / "pooled_pareto.csv", [
        dict(**provenance[i], AEP_GWh=float(metrics[i, 0]), LCOE_USD2022_MWh=float(metrics[i, 1]),
             effective_area_km2=float(metrics[i, 2]), gross_area_km2=float(metrics[i, 4]))
        for i in nd])
    labels = ["年发电量（GWh/年）", "基准 LCOE（2022年美元/MWh）", "有效包络面积（km²）"]
    fig, axes = plt.subplots(3, 1, figsize=(6.7, 8.2), layout="constrained")
    for ax, (a, b) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        ax.scatter(metrics[nd, a], metrics[nd, b], s=12, alpha=.65, color="#327C8C")
        i = choices["compromise"]
        ax.scatter([metrics[i, a]], [metrics[i, b]], marker="*", s=70, color="#C66B36")
        ax.set(xlabel=labels[a], ylabel=labels[b])
        ax.grid(alpha=.15)
    axes[0].set_title(case + " 已访问解的非支配投影")
    savefig(fig, folder, "pareto_projections")
    fig = plt.figure(figsize=(6.7, 5.2))
    ax = fig.add_axes([.15, .12, .77, .77], projection="3d")
    ax.scatter(metrics[nd, 0], metrics[nd, 1], metrics[nd, 2], s=14, color="#327C8C")
    ax.set(xlabel="AEP（GWh/年）", ylabel="LCOE（美元/MWh）")
    ax.tick_params(labelsize=8, pad=2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.label.set_fontsize(9); ax.yaxis.label.set_fontsize(9)
    ax.xaxis.labelpad = 10; ax.yaxis.labelpad = 10
    ax.view_init(elev=22, azim=-125)
    fig.text(.035, .54, "有效包络面积（km²）", rotation=90, va="center", ha="center")
    fig.suptitle(case, y=.98)
    savefig(fig, folder, "pareto_3d")

    p = representative["compromise"]
    fig, ax = plt.subplots(figsize=(6.8, 6), layout="constrained")
    extra = []
    if terrain_kind:
        t = farm.terrain
        image = ax.pcolormesh(t.x / 1000, t.y / 1000, t.z, shading="auto", cmap="terrain", alpha=.65)
        fig.colorbar(image, ax=ax, label="人工高程（m）", shrink=.55, pad=.06)
        mask = np.ma.masked_where(~t.forbidden_cells, np.ones_like(t.forbidden_cells, float))
        ax.pcolormesh(t.x / 1000, t.y / 1000, mask, shading="flat",
                     cmap=ListedColormap(["#BD3840"]), alpha=.65)
        extra.append(Patch(facecolor="#BD3840", alpha=.65, label="坡度禁布区"))
    for k, (a, b, c, d) in enumerate(farm.scene.obstacles):
        ax.add_patch(Rectangle((a / 1000, b / 1000), (c - a) / 1000, (d - b) / 1000,
            facecolor="#B5BBC2", edgecolor="#555D68", label="原始禁布区" if k == 0 else None))
    ax.scatter(candidate_xy[:, 0] / 1000, candidate_xy[:, 1] / 1000, s=3,
               color="#8E9AA7", alpha=.5, label="候选位置")
    xy = candidate_xy[p["selected_candidate_indices"]]
    ax.scatter(xy[:, 0] / 1000, xy[:, 1] / 1000, s=8, color="#063F69", label="选中风机")
    ax.set(xlim=(0, 7), ylim=(0, 14), xlabel="局部横坐标（km）", ylabel="局部纵坐标（km）",
           aspect="equal", title=case + "\n一份经验折中布局")
    handles, names = ax.get_legend_handles_labels()
    ax.legend(handles + extra, names + [p.get_label() for p in extra],
              loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2, fontsize=8, frameon=False)
    savefig(fig, folder, "representative_layout")
    result = dict(case=case, runs=len(rows), pooled_nondominated=len(nd),
                  representative_rules=len(choices), fixed_layout_reevaluations=len(choices))
    (folder / "DONE.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    del farm
    gc.collect()
    return result


def run(src, out):
    src, out = Path(src), Path(out)
    cfg, rows = check_complete(src)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 10,
                         "axes.unicode_minus": False})
    bounds = read(src / "analysis/normalization.json")
    curves = csvrows(src / "analysis/convergence.csv")
    cases = sorted({r["case"] for r in rows})
    assert len(cases) == len(cfg["cases"])
    reports = []
    for case in cases:
        print("CASE START", case, flush=True)
        reports.append(case_figures(src, out, case, [r for r in rows if r["case"] == case],
                                    cfg, bounds[case], curves))
        print("CASE COMPLETE", case, flush=True)
    (out / "CAPTIONS.txt").write_text(
        "所有图表须连同场景、M、N、种子数和预算说明使用。\n"
        "收敛实线为各种子均值，阴影为上下一个样本标准差，不是置信区间。\n"
        "不同算例采用各自共同经验归一化，不能横向比较绝对HV。\n"
        "前沿是该算例已访问布局的合并非支配集，不是全局Pareto前沿。\n"
        "三幅二维投影来自同一组三目标非支配布局，投影后可能出现支配。\n"
        "折中布局按等尺度归一化后的理想点距离选取，不是唯一工程最优解。\n"
        "可行率同时展示修复前后。候选筛选已排除了场界和禁布区。\n"
        "人工高程是自建模拟数据，非真实测绘；坡度阈值15度是场景假设。\n"
        "地形改变允许区域，跨地形比较有效面积时必须同时考虑总包络面积。\n"
        "LCOE使用2022年美元基准参数，不代表本地工程报价。\n"
        "所有代表布局均经独立几何检查和冻结模型固定布局重新评价。\n"
        "worker记录耗时从初始化或恢复完成后开始；runtime_accounting另列启动至完成墙钟包络。\n"
        "墙钟包络在重启情况下包含停机间隔，两者都不是CPU用量；并发负载使其不宜作为独占性能排名。\n"
        "自动导出完成不等于文档视觉验收；应逐图查看后再嵌入报告。\n",
        encoding="utf-8")
    return dict(status="complete", source=str(src), runs=len(rows), cases=reports,
                new_optimization_runs=0, visual_review_required=True)
