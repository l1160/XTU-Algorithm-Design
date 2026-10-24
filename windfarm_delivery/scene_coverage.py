"""Audit actual wind/terrain/geometry coverage without adding experiments."""
from pathlib import Path
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/"windfarm_terrain_v5"))
import bootstrap_v5
import numpy as np
from evaluator import Scenario
from terrain import Terrain
from shapely.geometry import box
from shapely.ops import unary_union
from campaign_figures import table, check_complete


def run(src,out):
    check_complete(Path(src))
    out=Path(out)
    v4=json.loads((ROOT/"windfarm_research_v4/posthoc/confirmation_config.json").read_text(encoding="utf-8"))
    v5=json.loads((ROOT/"windfarm_terrain_v5/terrain_config.json").read_text(encoding="utf-8"))
    kinds=["flat","rolling","ridge"]
    scenes={name:Scenario.load(ROOT/"windflo_standard/data/Scenarios"/(name+".xml"))
            for name in sorted({c["scenario"] for c in v4["cases"]+v5["cases"]})}
    windhashes={name:hashlib.sha256(np.vstack([s.c,s.shape,s.omega/s.omega.sum()]).astype("<f8").tobytes()).hexdigest()
                for name,s in scenes.items()}
    assert len(set(windhashes.values()))==2
    assert all(np.array_equal(s.obstacles,scenes["obs_00"].obstacles) for s in scenes.values())
    original=unary_union([box(*ob) for ob in scenes["obs_00"].obstacles])
    site=box(0,0,7000,14000)
    geometry={}
    candidates={}
    rows=[]
    for kind in kinds:
        terrain=Terrain(kind,50,15)
        saved=json.loads((ROOT/f"windfarm_terrain_v5/data/{kind}_50m/metadata.json").read_text(encoding="utf-8"))
        assert terrain.digest==saved["sha256"]
        steep=terrain.excluded
        union=original.union(steep)
        allowed=site.difference(union)
        assert np.isclose(steep.area,int(terrain.forbidden_cells.sum())*50**2)
        assert np.isclose(union.area,original.area+steep.area-original.intersection(steep).area)
        geometry[kind]=allowed
        for size in (800,1600):
            xy=np.load(ROOT/f"windfarm_terrain_v5/data/{kind}_50m/candidates_M{size}.npy")
            assert xy.shape==(size,2)
            assert terrain.allowed_mask(xy).all()
            assert scenes["obs_00"].site_mask(xy,closed_obstacles=True).all()
            candidates[kind,size]=xy
        assert np.array_equal(candidates[kind,800],candidates[kind,1600][:800])
        rows.append(dict(terrain=kind,source="synthetic",resolution_m=50,
            minimum_height_m=saved["minimum_height_m"],maximum_height_m=saved["maximum_height_m"],
            original_forbidden_km2=original.area/1e6,slope_forbidden_km2=steep.area/1e6,
            overlap_km2=original.intersection(steep).area/1e6,
            combined_forbidden_km2=union.area/1e6,allowed_km2=allowed.area/1e6,
            terrain_sha256=terrain.digest,
            candidate_M800_sha256=hashlib.sha256(candidates[kind,800].astype("<f8").tobytes()).hexdigest(),
            candidate_M1600_sha256=hashlib.sha256(candidates[kind,1600].astype("<f8").tobytes()).hexdigest()))
    assert geometry["flat"].equals(geometry["rolling"])
    assert not geometry["flat"].equals(geometry["ridge"])
    assert all(np.array_equal(candidates["flat",m],candidates["rolling",m]) for m in (800,1600))
    assert all(not np.array_equal(candidates["flat",m],candidates["ridge",m]) for m in (800,1600))
    coverage=[]
    for campaign,cfg in (("V4_confirmation",v4),("V5_terrain_screening",v5)):
        for case in cfg["cases"]:
            terrain=case.get("terrain","flat")
            coverage.append(dict(campaign=campaign,wind=case["scenario"],terrain=terrain,
                M=case["candidates"],N=case["turbines"],seeds=len(cfg["seeds"]),
                algorithms_or_variants=len(cfg["variants"]),requests_per_run=cfg["evaluations"],
                geometry_class="original_plus_slope" if terrain=="ridge" else "original_rectangles",
                wind_sha256=windhashes[case["scenario"]]))
    physical=set((r["wind"],r["terrain"]) for r in coverage)
    assert len(physical)==4
    table(out/"terrain_geometry_coverage.csv",rows)
    table(out/"campaign_coverage.csv",coverage)
    text=(
        "覆盖计数来自冻结配置与实际数据，而不是按运行总数推测。\n\n"
        "更正后的V4确认采用obs_00、obs_01两个不同风况，二者原始场界和矩形禁布几何相同。\n"
        "V5采用obs_00下的平坦、丘陵、山脊三种人工曲面。平坦与丘陵允许区完全相同，"
        "山脊坡度禁布引入第二种允许区。因此总计两种风况、三种人工曲面、两种允许区几何。\n"
        "合并两阶段，实际覆盖四个风况—地形组合，每个有800和1600两个M。"
        "这不是两个风况乘三个地形的全交叉设计。V4平坦与V5平坦在不同种子、预算下分别运行，"
        "不应重复计算为两个不同地形场景。\n\n"
        "高程、候选坐标、坡度禁布与原矩形区的重叠面积已逐一核查。"
        "坡度面积不能直接加到原禁布面积上，必须扣除重叠。\n"
        "人工地形补齐了可控高程输入和几何对照，仍未提供真实场址测量、"
        "地形流场校准或教师对研究范围的认可。任务验收材料应明确这些事实。\n"
    )
    (out/"场景覆盖说明.txt").write_text(text,encoding="utf-8")
    return dict(status="complete",distinct_wind_distributions=2,synthetic_terrain_surfaces=3,
        distinct_allowed_geometries=2,wind_terrain_combinations=4,
        full_wind_terrain_factorial=False,candidate_counts=[800,1600],installed_turbines=400,
        all_candidate_masks_verified=True,all_nested_prefixes_verified=True,
        original_slope_overlap_accounted=True,new_optimization_runs=0)
