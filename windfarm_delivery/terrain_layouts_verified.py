"""Replot verified saved layouts with a common elevation scale, no evaluation."""
from pathlib import Path
import hashlib
import json
from campaign_figures import ROOT, check_complete, savefig
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, Patch
from evaluator import Scenario


def run(src, out):
    cfg, rows = check_complete(Path(src))
    previous = ROOT / "windfarm_delivery/results/terrain_figures"
    assert (previous / "DONE.json").exists()
    data = ROOT / "windfarm_terrain_v5/data"
    terrain = {}
    hashes = {}
    for kind in ("flat", "rolling", "ridge"):
        path = data / f"{kind}_50m/dem.npz"
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:
            terrain[kind] = {key: z[key].copy() for key in z.files}
    lo = 50 * np.floor(min(t["elevation_m"].min() for t in terrain.values()) / 50)
    hi = 50 * np.ceil(max(t["elevation_m"].max() for t in terrain.values()) / 50)
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 10, "axes.unicode_minus": False})
    outputs = []
    for case in sorted({r["case"] for r in rows}):
        info = next(r for r in rows if r["case"] == case)
        kind, m = info["terrain"], int(info["candidates"])
        t = terrain[kind]
        saved = previous / case / "representative_layouts.json"
        hashes[str(saved)] = hashlib.sha256(saved.read_bytes()).hexdigest()
        selected = json.loads(saved.read_text(encoding="utf-8"))["layouts"]["compromise"]
        assert selected["independent_geometry_passed"] and selected["fixed_layout_reevaluation_passed"]
        candidates = np.load(data / f"{kind}_50m/candidates_M{m}.npy")
        scene = Scenario.load(ROOT / "windflo_standard/data/Scenarios" / (info["scenario"] + ".xml"))
        xy = candidates[selected["selected_candidate_indices"]]
        assert len(xy) == 400
        folder = Path(out) / case
        folder.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.8, 6), layout="constrained")
        artist = ax.pcolormesh(t["x_m"] / 1000, t["y_m"] / 1000, t["elevation_m"],
                              shading="auto", cmap="terrain", vmin=lo, vmax=hi, alpha=.65)
        fig.colorbar(artist, ax=ax, label="人工高程（m，各地形统一色标）", shrink=.55, pad=.06)
        forbidden = t["forbidden_slope_cells"]
        extra = []
        if forbidden.any():
            mask = np.ma.masked_where(~forbidden, np.ones_like(forbidden, dtype=float))
            ax.pcolormesh(t["x_m"] / 1000, t["y_m"] / 1000, mask, shading="flat",
                          cmap=ListedColormap(["#BD3840"]), alpha=.65)
            extra.append(Patch(facecolor="#BD3840", alpha=.65, label="坡度禁布区"))
        for index, (a, b, c, d) in enumerate(scene.obstacles):
            ax.add_patch(Rectangle((a / 1000, b / 1000), (c-a) / 1000, (d-b) / 1000,
                facecolor="#B5BBC2", edgecolor="#555D68", label="原始禁布区" if index == 0 else None))
        ax.scatter(candidates[:, 0] / 1000, candidates[:, 1] / 1000, s=3,
                   color="#8E9AA7", alpha=.5, label="候选位置")
        ax.scatter(xy[:, 0] / 1000, xy[:, 1] / 1000, s=8, color="#063F69", label="选中风机")
        note = "平坦高程恒为100 m" if kind == "flat" else "人工地形"
        ax.set(xlim=(0, 7), ylim=(0, 14), aspect="equal", xlabel="局部横坐标（km）",
               ylabel="局部纵坐标（km）", title=case + "\n经验折中布局  " + note)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + extra, labels + [p.get_label() for p in extra], loc="upper center",
                  bbox_to_anchor=(.5, -.12), ncol=2, fontsize=8, frameon=False)
        savefig(fig, folder, "representative_layout")
        outputs.append(dict(case=case, source_layout=str(saved), selected_points=len(xy),
                            elevation_limits_m=[float(lo), float(hi)]))
        print("REPLOT", case, flush=True)
    (Path(out) / "sources.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    return dict(status="complete", cases=outputs, new_optimization_runs=0,
                new_physical_evaluations=0, original_figures_preserved=True,
                visual_review_required=True,
                purpose="Common elevation color scale and only actually present slope-mask legend")
