"""Matched initial-layout comparison; no new search or hypothesis-test claim."""
from pathlib import Path
import sys
import json
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "windfarm_research_v4"))
import bootstrap_v4
import numpy as np
import matplotlib.pyplot as plt
from campaign_figures import csvrows, table, check_complete, savefig


def paired_initial_layouts(src, m, seed, population=100):
    prefix = f"obs_00_{{terrain}}_M{m}_N400_NSGA3_base_s{seed}"
    arrays = {}
    for kind in ("flat", "rolling"):
        folder = src / prefix.format(terrain=kind)
        assert (folder / "DONE.json").exists()
        with np.load(folder / "evaluations.npz") as z:
            arrays[kind] = dict(xy=z["candidate_xy"].copy(),
                selected=z["selected"][:population].copy(),
                metrics=z["metrics"][:population].copy(), cv=z["cv"][:population].copy())
    a, b = arrays["flat"], arrays["rolling"]
    assert np.array_equal(a["xy"], b["xy"]), "Different candidate positions cannot be paired"
    assert np.array_equal(a["selected"], b["selected"]), "Initial repaired layouts differ"
    assert np.all(a["cv"] == 0) and np.all(b["cv"] == 0)
    assert len(a["selected"]) == population
    fa, fb = a["metrics"], b["metrics"]
    # The long-wave rolling DEM has no >15-degree forbidden cells, so the
    # allowed mask and both planar envelopes must remain exactly the same.
    assert np.allclose(fa[:, [2, 4]], fb[:, [2, 4]], rtol=0, atol=1e-10)
    assert np.all(fb[:, 0] >= fa[:, 0] - 1e-8)
    assert np.all(fb[:, 3] >= fa[:, 3] - 1e-6)
    # Fixed N and shared economics give the exact relative LCOE identity.
    predicted_lcoe_ratio = (fb[:, 7] / fa[:, 7]) / (fb[:, 0] / fa[:, 0])
    assert np.allclose(fb[:, 1] / fa[:, 1], predicted_lcoe_ratio, rtol=1e-12, atol=1e-12)
    rows = []
    for i in range(population):
        rows.append(dict(M=m, seed=seed, initial_request_index_zero_based=i,
            flat_run=prefix.format(terrain="flat"), rolling_run=prefix.format(terrain="rolling"),
            AEP_flat_GWh=float(fa[i, 0]), AEP_rolling_GWh=float(fb[i, 0]),
            AEP_change_percent=float((fb[i, 0] / fa[i, 0] - 1) * 100),
            LCOE_flat_USD2022_MWh=float(fa[i, 1]), LCOE_rolling_USD2022_MWh=float(fb[i, 1]),
            LCOE_change_percent=float((fb[i, 1] / fa[i, 1] - 1) * 100),
            route_flat_m=float(fa[i, 3]), route_rolling_m=float(fb[i, 3]),
            route_change_percent=float((fb[i, 3] / fa[i, 3] - 1) * 100),
            annual_cost_change_percent=float((fb[i, 7] / fa[i, 7] - 1) * 100),
            effective_area_difference_km2=float(fb[i, 2] - fa[i, 2]),
            gross_area_difference_km2=float(fb[i, 4] - fa[i, 4])))
    return rows


def run(src, out):
    src, out = Path(src), Path(out)
    cfg, _ = check_complete(src)
    assert cfg["stage"] == "synthetic_terrain_screening"
    rows = []
    for m in (800, 1600):
        for seed in cfg["seeds"]:
            rows.extend(paired_initial_layouts(src, m, seed, cfg["population"]))
    expected = 2 * len(cfg["seeds"]) * cfg["population"]
    assert len(rows) == expected
    summary = []
    for m in (800, 1600):
        for seed in cfg["seeds"]:
            rr = [r for r in rows if r["M"] == m and r["seed"] == seed]
            summary.append(dict(M=m, seed=seed, paired_layouts=len(rr), **{
                field + "_mean": float(np.mean([r[field] for r in rr]))
                for field in ("AEP_change_percent", "route_change_percent",
                              "annual_cost_change_percent", "LCOE_change_percent")}))
    out.mkdir(parents=True, exist_ok=True)
    table(out / "matched_initial_layouts.csv", rows)
    table(out / "per_seed_changes.csv", summary)
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 10,
                         "axes.unicode_minus": False})
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7), layout="constrained")
    for ax, field, label in zip(axes,
        ("AEP_change_percent_mean", "route_change_percent_mean", "LCOE_change_percent_mean"),
        ("年发电量变化（%）", "地表连接长度变化（%）", "基准LCOE变化（%）")):
        for j, seed in enumerate(cfg["seeds"]):
            values = [next(r[field] for r in summary if r["M"] == m and r["seed"] == seed)
                      for m in (800, 1600)]
            ax.scatter(np.array([0, 1]) + (j - 1) * .055, values, label=f"种子 {seed}", s=32)
        ax.axhline(0, color="#888888", lw=.8)
        ax.set(xticks=[0, 1], xticklabels=["M=800", "M=1600"], ylabel=label)
        ax.grid(axis="y", alpha=.15)
    axes[0].set_title("丘陵相对平坦  相同初始布局的配对检查")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[-1].set_xlabel("每点为一个种子100个配对初始布局的均值")
    savefig(fig, out, "matched_terrain_effects")
    explanation = (
        "本分析只使用NSGA-III保存的初始100份布局，每个M和种子各取一次；"
        "不重复计入其他方法共享的同一初始布局。两种地形的候选坐标、"
        "修复后选择索引和可行性逐一相同。因此这些差异来自已声明的模型变化，"
        "不是两种优化结果选择不同造成的差异。\n\n"
        "在固定水平位置、相同来流、相同塔高和扩散参数下，三维判据"
        "y²+Δz²<(R+kt)²比平面判据y²<(R+kt)²更严格。当前亏损幅值只依赖下游距离t，"
        "故高差只能使某些尾流对失活，不能新添尾流对。RSS亏损不会增大。"
        "对本项目的Weibull参数和功率曲线，配对检查验证AEP未下降；"
        "注意功率曲线含切出段，因此“亏损减小必然提高所有可能风况的AEP”"
        "并非一般数学定理，不应脱离当前风况泛化。\n\n"
        "地表路径积分不短于相同端点的水平距离，每一条候选边权都不减小，"
        "因此使用相同顶点集的地表MST最优值也不小于平面MST。"
        "LCOE的相对变化由年成本比除以AEP比决定，两个方向相反的因素共同作用。\n\n"
        "丘陵坡度未触发15度禁布，因此有效和总包络面积保持一致。"
        "山脊禁布改变候选池，不能拿不同位置索引当作相同物理布局配对，故未列入本分析。\n\n"
        "这些是人工几何模型下的结果，不证明真实丘陵场地更优。"
        "模型没有求解地形加速、分离、流线弯曲和湍流变化。"
        "同一随机种子内100份布局不视作100个独立重复；仅作描述性配对，不给显著性p值。\n"
    )
    (out / "模型效应解释.txt").write_text(explanation, encoding="utf-8")
    return dict(status="complete", paired_layouts=len(rows), seeds=cfg["seeds"],
        case_sizes=[800, 1600], identical_candidate_coordinates_verified=True,
        identical_repaired_layouts_verified=True, planar_areas_identical=True,
        AEP_nondecrease_observed_for_these_winds=True,
        surface_MST_nondecrease_verified=True, exact_LCOE_ratio_verified=True,
        new_optimization_runs=0, statistical_test_performed=False,
        scope="Fixed identical initial layouts, flat versus rolling; not real terrain validation")
