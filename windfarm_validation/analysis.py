"""Saved-result confirmation with independent geometry/front/HV audit."""
from pathlib import Path
import csv
import itertools
import math
import xml.etree.ElementTree as ET
import numpy as np
from pymoo.indicators.hv import HV
from windfarm_research_v6.compat import ROOT, legacy as A
from windfarm_research_v6.lifecycle import sha, now
from .runtime import METHODS, read, save_json


def table(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nondominated(points):
    """Independent exact minimization front: Fenwick prefix minimum sweep."""
    points = np.unique(np.asarray(points, float), axis=0)
    points = points[np.lexsort((points[:,2], points[:,1], points[:,0]))]
    ys = np.unique(points[:,1])
    tree = np.full(len(ys)+1, np.inf)
    front = []
    for point in points:
        index = int(np.searchsorted(ys, point[1]))+1
        cursor, best = index, np.inf
        while cursor:
            best = min(best, tree[cursor])
            cursor -= cursor & -cursor
        if best > point[2]:
            front.append(point)
        cursor = index
        while cursor < len(tree):
            tree[cursor] = min(tree[cursor], point[2])
            cursor += cursor & -cursor
    return np.asarray(front, float).reshape(-1,3)


def area2(points, ref):
    points = points[np.argsort(points[:,0], kind="stable")]
    area, best, previous = 0., float(ref[1]), float(points[0,0])
    for y, z in points:
        area += (float(y)-previous)*(float(ref[1])-best)
        best, previous = min(best, float(z)), float(y)
    return area+(float(ref[0])-previous)*(float(ref[1])-best)


def volume3(points, ref):
    points = np.unique(points, axis=0)
    if not len(points):
        return 0.
    assert np.all(points <= ref)
    cuts = np.unique(points[:,0])
    value = 0.
    for i, x in enumerate(cuts):
        right = cuts[i+1] if i+1 < len(cuts) else ref[0]
        value += float(right-x)*area2(points[points[:,0] <= x,1:], ref[1:])
    return value


def exact_p(differences):
    values = np.asarray(differences, float)
    signs = np.asarray(list(itertools.product((-1., 1.), repeat=len(values))))
    observed = abs(float(values.sum()))
    tolerance = 1e-12*max(1., float(np.abs(values).sum()))
    return float(np.mean(np.abs(signs @ values) >= observed-tolerance))


def holm(pvalues):
    values = np.asarray(pvalues, float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    maximum = 0.
    for rank, index in enumerate(order):
        maximum = max(maximum, (len(values)-rank)*values[index])
        adjusted[index] = min(1., maximum)
    return adjusted


def cover(a, b):
    return float(np.mean([np.any(np.all(a <= p, axis=1) & np.any(a < p, axis=1)) for p in b]))


def independent_geometry(xy, selected, scene, terrain):
    assert np.all((xy >= 0) & (xy <= [scene["width"], scene["height"]]))
    for a,b,c,d in scene["obstacles"]:
        assert not np.any((xy[:,0]>=a)&(xy[:,0]<=c)&(xy[:,1]>=b)&(xy[:,1]<=d))
    # Closed forbidden cells, including shared edges and corners. This does not
    # call Terrain.allowed_mask or the optimizer's conflict/repair functions.
    z, resolution = terrain["z"], 50.
    dx0, dx1 = np.diff(z, axis=1)[:-1]/resolution, np.diff(z, axis=1)[1:]/resolution
    dy0, dy1 = np.diff(z, axis=0)[:,:-1]/resolution, np.diff(z, axis=0)[:,1:]/resolution
    forbidden = np.hypot(np.maximum(abs(dx0), abs(dx1)), np.maximum(abs(dy0), abs(dy1))) > math.tan(math.radians(15.))
    np.testing.assert_array_equal(forbidden, terrain["forbidden"])
    x, y = xy[:,0]/resolution, xy[:,1]/resolution
    ix = np.minimum(x.astype(int), forbidden.shape[1]-1)
    iy = np.minimum(y.astype(int), forbidden.shape[0]-1)
    blocked = forbidden[iy, ix].copy()
    left, below = (abs(x-ix)<1e-12)&(ix>0), (abs(y-iy)<1e-12)&(iy>0)
    blocked |= left & forbidden[iy, np.maximum(ix-1,0)]
    blocked |= below & forbidden[np.maximum(iy-1,0), ix]
    blocked |= left & below & forbidden[np.maximum(iy-1,0), np.maximum(ix-1,0)]
    assert not blocked.any()
    assert np.all(selected[:,1:] > selected[:,:-1])
    assert selected.min() >= 0 and selected.max() < len(xy)
    squared = np.sum((xy[:,None]-xy[None,:])**2, axis=2)
    edges = np.where(np.triu(squared < scene["spacing"]**2,1))
    valid = np.ones(len(selected), bool)
    for start in range(0, len(selected), 128):
        block = selected[start:start+128]
        member = np.zeros((len(block), len(xy)), bool)
        member[np.arange(len(block))[:,None], block] = True
        valid[start:start+len(block)] = ~np.any(member[:,edges[0]] & member[:,edges[1]], axis=1)
    return valid


def analyze_case(plan, case):
    output = Path(plan["output"])
    folder = output/"analysis"/case["id"]
    folder.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in plan["jobs"] if j["case"] == case["id"]]
    assert len(jobs) == 60 and all((Path(j["output"])/"DONE.json").exists() for j in jobs)
    scene_xml = ET.parse(ROOT/"windflo_standard/data/Scenarios"/(case["scenario"]+".xml")).getroot()
    setup = output/"preparation"/case["id"]
    metadata = read(setup/"metadata.json")
    scene = dict(width=float(scene_xml.findtext('.//Width')), height=float(scene_xml.findtext('.//Height')),
                 obstacles=[tuple(float(o.attrib[k]) for k in ('xmin','ymin','xmax','ymax')) for o in scene_xml.findall('.//obstacle')],
                 spacing=metadata["minimum_spacing_m"])
    with np.load(setup/"geometry.npz", allow_pickle=False) as z:
        terrain = dict(z=z["elevation"].copy(), forbidden=z["forbidden"].copy())
        expected_xy = z["candidate_xy"].copy()
    economics = read(ROOT/"windfarm_study_v3/economics.json")
    initial = {}
    data = []
    audits = []
    for job in jobs:
        src = Path(job["output"])
        done = read(src/"DONE.json")
        assert done["status"] == "complete" and done["requests"] == 10000
        with np.load(src/"evaluations.npz", allow_pickle=False) as z:
            selected, metrics, cv = z["selected"].copy(), z["metrics"].copy(), z["cv"].copy()
            xy, front, hits = z["candidate_xy"].copy(), z["pareto_request_indices"].copy(), z["cache_hit"].copy()
        assert selected.shape == (10000, case["turbines"]) and metrics.shape == (10000,10)
        assert np.isfinite(metrics).all()
        np.testing.assert_array_equal(xy, expected_xy)
        valid = independent_geometry(xy, selected, scene, terrain)
        np.testing.assert_array_equal(valid, cv == 0)
        prefix = selected[:100], metrics[:100]
        if job["seed"] in initial:
            for a,b in zip(prefix, initial[job["seed"]]):
                np.testing.assert_array_equal(a, b)
        else:
            initial[job["seed"]] = tuple(a.copy() for a in prefix)
        seen = set()
        independent_hits = []
        for row in selected:
            key = row.tobytes()
            independent_hits.append(key in seen)
            seen.add(key)
        np.testing.assert_array_equal(hits, independent_hits)
        unique_feasible = len(np.unique(selected[valid], axis=0))
        assert unique_feasible == done["unique_feasible_layouts"]
        m = metrics[valid]
        capacity = case["turbines"]*metadata["rated_kw"]
        electrical = capacity*economics["electrical_infrastructure_capex_per_kw"]*m[:,3]/m[:,9]
        capex = capacity*(economics["total_capex_per_kw"]-economics["electrical_infrastructure_capex_per_kw"])+electrical
        annual = economics["real_fixed_charge_rate"]*capex+capacity*economics["annual_opex_per_kw"]
        np.testing.assert_allclose(m[:,8], electrical, rtol=1e-12, atol=1e-6)
        np.testing.assert_allclose(m[:,6], capex, rtol=1e-12, atol=1e-6)
        np.testing.assert_allclose(m[:,7], annual, rtol=1e-12, atol=1e-6)
        np.testing.assert_allclose(m[:,1], annual/(1000*m[:,0]), rtol=1e-12, atol=1e-10)
        f = metrics[:,:3]*[-1,1,1]
        exact_front = nondominated(f[valid])
        assert {tuple(p) for p in exact_front} == {tuple(p) for p in f[front]}
        assert valid[front].all()
        audit = dict(id=job["id"], layouts_checked=len(selected), feasible_layouts=int(valid.sum()),
                     unique_feasible_layouts=unique_feasible, exact_front_matched=True,
                     independent_geometry_passed=True, cost_identities_passed=True,
                     hashes={name:sha(src/name) for name in ("manifest.json","DONE.json","evaluations.npz","trace.json")})
        audits.append(audit)
        data.append(dict(job=job, done=done, f=f, metrics=metrics, valid=valid, front=front,
                         exact_front=exact_front, trace=read(src/"trace.json")))
        # Export each run's feasible nondominated layouts with physical decisions.
        np.savez_compressed(folder/(job["id"]+"_pareto.npz"), selected=selected[front], metrics=metrics[front], candidate_xy=xy,
                            request_indices=front)
    all_points = np.vstack([d["f"][d["valid"]] for d in data])
    low, high = all_points.min(0), all_points.max(0)
    width, ref = np.maximum(high-low,1e-12), np.full(3,1.1)
    save_json(folder/"normalization.json", dict(ideal=low.tolist(), nadir=high.tolist(), width=width.tolist(), reference=ref.tolist(),
                                               scope=plan["normalization"]))
    indicator = HV(ref_point=ref)
    def hv(points):
        return float(indicator((points-low)/width)) if len(points) else 0.
    rows, curves, paired = [], [], []
    indexed = {(d["job"]["method"],d["job"]["seed"]):d for d in data}
    for d, audit in zip(data, audits):
        job, done = d["job"], d["done"]
        d["HV"] = hv(d["exact_front"])
        independent_hv = volume3((d["exact_front"]-low)/width,ref)
        assert abs(independent_hv-d["HV"]) < 1e-10
        audit.update(independent_HV=independent_hv, HV_error=abs(independent_hv-d["HV"]))
        row = dict(case=case["id"], method=job["method"], seed=job["seed"], HV=d["HV"],
                   CPU_seconds=done["worker_cpu_seconds"], wall_seconds=done["worker_wall_seconds"],
                   feasible_rate=float(d["valid"].mean()), duplicate_rate=done["duplicate_request_rate"],
                   unique_layouts=done["unique_feasible_layouts"], pareto_layouts=len(d["front"]),
                   mean_working_population=float(np.mean([t["working_population"] for t in d["trace"]])),
                   restore_wall_seconds=done["restore_wall_seconds"],
                   local_requests=done["operator_counts"].get("local_exact_requests",0))
        for key, axis in (("AEP_GWh",0),("LCOE",1),("effective_area_km2",2),("gross_area_km2",4)):
            row[key+"_min"] = float(d["metrics"][d["front"],axis].min())
            row[key+"_max"] = float(d["metrics"][d["front"],axis].max())
        rows.append(row)
        previous, current, history = 0, np.empty((0,3)), []
        for point in d["trace"]:
            count = point["evaluations"]
            block = np.unique(np.vstack((current,d["f"][previous:count][d["valid"][previous:count]])),axis=0)
            current = block[A.NDS.do(block, only_non_dominated_front=True)] if len(block) else block
            value = hv(current)
            entry = dict(case=case["id"],method=job["method"],seed=job["seed"],requests=count,HV=value,
                         CPU_seconds=point["elapsed_cpu_seconds"],wall_seconds=point["elapsed_wall_seconds"],
                         feasible_rate=float(d["valid"][:count].mean()))
            history.append(entry)
            curves.append(entry)
            previous = count
        assert previous == 10000 and abs(history[-1]["HV"]-d["HV"]) < 1e-10
        assert np.all(np.diff([p["HV"] for p in history]) >= -1e-12)
        d["curve"] = history
    for seed in plan["seeds"]:
        for family in ("CCGDE3","LMOCSO","DGEA"):
            a,b = indexed[family,seed],indexed[family+"_problem",seed]
            limit = a["done"]["worker_cpu_seconds"]
            available = [p for p in b["curve"] if p["CPU_seconds"] <= limit]
            timed_hv = available[-1]["HV"] if available else 0.
            if b["done"]["worker_cpu_seconds"] <= limit:
                timed_hv = b["HV"]
            hits = [p for p in b["curve"] if p["HV"] >= a["HV"]-1e-10]
            paired.append(dict(case=case["id"], family=family, seed=seed,
                baseline_HV=a["HV"], improved_HV=b["HV"], HV_delta=b["HV"]-a["HV"],
                HV_change_percent=100*(b["HV"]/a["HV"]-1),
                improved_HV_at_baseline_CPU=timed_hv, fixed_CPU_HV_delta=timed_hv-a["HV"],
                baseline_CPU_seconds=limit, improved_CPU_seconds=b["done"]["worker_cpu_seconds"],
                CPU_change_percent=100*(b["done"]["worker_cpu_seconds"]/limit-1),
                baseline_quality_attained=bool(hits), first_attainment_CPU_seconds=hits[0]["CPU_seconds"] if hits else None,
                first_attainment_requests=hits[0]["requests"] if hits else None,
                improved_dominates_baseline_fraction=cover(b["exact_front"],a["exact_front"]),
                baseline_dominates_improved_fraction=cover(a["exact_front"],b["exact_front"])))
    table(folder/"runs.csv",rows)
    table(folder/"paired.csv",paired)
    table(folder/"convergence.csv",curves)
    save_json(folder/"independent_audit.json",dict(status="passed",runs=60,geometry_records=600000,
        shared_initialization_verified=True,all_candidate_sets_matched=True,rows=audits,
        methods="Independent spacing/closed-mask geometry, cost identities, Fenwick nondomination and slab-integrated HV"))
    figures_and_representatives(plan,case,folder,data,rows,curves,expected_xy,terrain,scene,low,width)
    save_json(folder/"DONE.json",dict(status="complete",case=case["id"],runs=60,finished_at=now(),paired=paired,rows=rows))
    return rows,paired


def figures_and_representatives(plan,case,folder,data,rows,curves,xy,terrain,scene,low,width):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from windfarm_research_v7.model import FastFarm
    from windfarm_research_v6.model import OverlapFarm
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False})
    colors = {m:plt.get_cmap("tab10")(i) for i,m in enumerate(METHODS)}
    fig,axes = plt.subplots(1,3,figsize=(16,4.7),layout="constrained")
    for method in METHODS:
        own = [d for d in data if d["job"]["method"] == method]
        for axis,xkey,end in ((axes[0],"requests",10000),(axes[1],"CPU_seconds",max(d["curve"][-1]["CPU_seconds"] for d in own))):
            grid = np.linspace(0,end,201)
            values = []
            for d in own:
                xx = np.array([p[xkey] for p in d["curve"]])
                yy = np.array([p["HV"] for p in d["curve"]])
                positions = np.searchsorted(xx,grid,side="right")-1
                values.append(np.where(positions>=0,yy[np.maximum(positions,0)],0.))
            values = np.array(values)
            mean,sd = values.mean(0),values.std(0,ddof=1)
            axis.plot(grid,mean,color=colors[method],label=method)
            axis.fill_between(grid,np.maximum(0,mean-sd),mean+sd,color=colors[method],alpha=.08)
        rate = [float(d["valid"].mean()) for d in own]
        axes[2].scatter([METHODS.index(method)]*10,rate,color=colors[method],alpha=.5)
    axes[0].set(xlabel="评价请求数",ylabel="HV",title="10个种子的均值±标准差")
    axes[1].set(xlabel="单工作进程 CPU 秒",ylabel="HV",title="各运行完成后保持最终质量")
    axes[2].set(xticks=range(6),xticklabels=METHODS,ylabel="可行率",ylim=(0,1.02),title="每次运行的可行率")
    axes[2].tick_params(axis="x",rotation=65)
    axes[0].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle(case["id"])
    fig.savefig(folder/"convergence_feasibility.png",dpi=160)
    plt.close(fig)
    fig=plt.figure(figsize=(13,5),layout="constrained")
    ax3=fig.add_subplot(121,projection="3d")
    ax2=fig.add_subplot(122)
    pooled=[]
    for d in data:
        for index in d["front"]:
            pooled.append((d,int(index)))
    points=np.array([d["metrics"][index,:3] for d,index in pooled])
    objective=points*[-1,1,1]
    front_ids=A.NDS.do(objective,only_non_dominated_front=True)
    for method in METHODS:
        p=np.vstack([d["metrics"][d["front"],:3] for d in data if d["job"]["method"]==method])
        p=np.unique(p,axis=0)
        p=p[A.NDS.do(p*[-1,1,1],only_non_dominated_front=True)]
        ax3.scatter(p[:,0],p[:,1],p[:,2],s=6,color=colors[method],label=method)
        ax2.scatter(p[:,0],p[:,1],s=8,color=colors[method],label=method)
    ax3.set(xlabel="AEP (GWh)",ylabel="LCOE",zlabel="有效面积 (km²)",title="各方法合并10次运行的前沿")
    ax2.set(xlabel="AEP (GWh)",ylabel="LCOE",title="能量—成本投影")
    ax2.legend(fontsize=8)
    fig.savefig(folder/"pareto_energy_cost.png",dpi=160)
    plt.close(fig)
    normalized=(objective[front_ids]-low)/width
    choices=[front_ids[np.argmin(objective[front_ids,i])] for i in range(3)]
    choices.append(front_ids[np.argmin(np.linalg.norm(normalized,axis=1))])
    labels=("最大发电量","最低度电成本","最小有效面积","归一化折中布局")
    fig,axes=plt.subplots(1,4,figsize=(15,7),layout="constrained")
    representatives=[]
    for ax,label,chosen in zip(axes,labels,choices):
        d,index=pooled[chosen]
        with np.load(Path(d["job"]["output"])/"evaluations.npz",allow_pickle=False) as z:
            ids=z["selected"][index].copy()
        ax.imshow(terrain["z"],origin="lower",extent=(0,scene["width"],0,scene["height"]),cmap="terrain",alpha=.45,aspect="equal")
        mask=np.ma.masked_where(~terrain["forbidden"],terrain["forbidden"])
        ax.imshow(mask,origin="lower",extent=(0,scene["width"],0,scene["height"]),cmap="Reds",vmin=0,vmax=1,alpha=.5,aspect="equal")
        for a,b,c,e in scene["obstacles"]:
            ax.add_patch(Rectangle((a,b),c-a,e-b,facecolor="red",alpha=.5))
        ax.scatter(xy[ids,0],xy[ids,1],s=3,c="#164369")
        ax.set(title=label+"\n"+d["job"]["method"],xlabel="x (m)",ylabel="y (m)")
        representatives.append(dict(label=label,run=d["job"]["id"],request_index=index,selected_indices=ids.tolist(),metrics=d["metrics"][index].tolist()))
    fig.savefig(folder/"representative_layouts.png",dpi=180)
    plt.close(fig)
    save_json(folder/"representative_layouts.json",representatives)
    # Three objective extremes from each method on the first prespecified seed:
    # complete wake sums and the older independent SciPy MST implementation.
    farm=FastFarm(case["scenario"],case["candidates"],case["turbines"],20261012,terrain_kind=case["terrain"])
    checks=[]
    for d in data:
        if d["job"]["seed"] != plan["seeds"][0]:continue
        indices={int(d["front"][np.argmin(d["f"][d["front"],axis])]) for axis in range(3)}
        with np.load(Path(d["job"]["output"])/"evaluations.npz",allow_pickle=False) as z:
            for index in sorted(indices):
                exact=OverlapFarm.physical_metrics(farm,z["selected"][index],force_full=True)
                np.testing.assert_allclose(exact,d["metrics"][index],rtol=1e-10,atol=1e-6)
                checks.append(dict(run=d["job"]["id"],request_index=index,maximum_relative_error=float(np.max(abs(exact-d["metrics"][index])/np.maximum(1.,abs(exact))))))
    save_json(folder/"full_objective_rechecks.json",dict(status="passed",checks=checks,scope="Full wake accumulation and independent MST, same declared engineering model"))


def run(plan):
    output=Path(plan["output"])
    folder=output/"analysis"
    folder.mkdir(exist_ok=True)
    for path,digest in plan["source_sha256"].items():assert sha(ROOT/path)==digest,path
    rows,paired=[],[]
    for case in plan["cases"]:
        print(now(),"ANALYZE",case["id"],flush=True)
        rr,pp=analyze_case(plan,case)
        rows.extend(rr);paired.extend(pp)
    table(folder/"all_runs.csv",rows)
    table(folder/"all_paired.csv",paired)
    tests=[]
    for case in plan["cases"]:
        for family in ("CCGDE3","LMOCSO","DGEA"):
            pp=[p for p in paired if p["case"]==case["id"] and p["family"]==family]
            for endpoint,key in (("fixed_requests","HV_delta"),("fixed_CPU","fixed_CPU_HV_delta")):
                values=np.array([p[key] for p in pp])
                tests.append(dict(case=case["id"],family=family,endpoint=endpoint,n=10,
                    mean_HV_delta=float(values.mean()),sd_HV_delta=float(values.std(ddof=1)),
                    wins=int((values>1e-10).sum()),ties=int((abs(values)<=1e-10).sum()),losses=int((values < -1e-10).sum()),
                    exact_two_sided_p=exact_p(values)))
    for test,p in zip(tests,holm([t["exact_two_sided_p"] for t in tests])):
        test.update(Holm_p=float(p),positive_after_Holm=bool(p<.05 and test["mean_HV_delta"]>0))
    table(folder/"primary_tests.csv",tests)
    summaries=[]
    for case in plan["cases"]:
        for method in METHODS:
            rr=[r for r in rows if r["case"]==case["id"] and r["method"]==method]
            row=dict(case=case["id"],method=method,runs=10)
            for key in ("HV","CPU_seconds","wall_seconds","feasible_rate","duplicate_rate","unique_layouts"):
                values=np.array([r[key] for r in rr])
                row[key+"_mean"]=float(values.mean());row[key+"_sd"]=float(values.std(ddof=1))
            summaries.append(row)
    table(folder/"summary.csv",summaries)
    lines=["# 冻结算法的正式确认结果","",f"完成时间：{now()}。180次运行，3个算例，每组10个新种子。",
           "所有方法使用相同的尾流、成本、面积模型和精确加速评价后端；所有局部搜索的精确评价均计入10000次请求预算。","",
           "| 算例 | 算法 | 同请求数HV差值（均值） | 校正p | 同CPU预算HV差值（均值） | 校正p | 达到基线最终质量 |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for case in plan["cases"]:
        for family in ("CCGDE3","LMOCSO","DGEA"):
            tt=[t for t in tests if t["case"]==case["id"] and t["family"]==family]
            pp=[p for p in paired if p["case"]==case["id"] and p["family"]==family]
            lines.append(f"| {case['id']} | {family} | {tt[0]['mean_HV_delta']:.6f} | {tt[0]['Holm_p']:.4f} | {tt[1]['mean_HV_delta']:.6f} | {tt[1]['Holm_p']:.4f} | {sum(p['baseline_quality_attained'] for p in pp)}/10 |")
    lines += ["","正差值表示改进版更好。统计方法为10个配对差值的精确双侧符号翻转检验，对18项主要比较进行Holm校正。正差值不等于所有目标逐点支配；配对覆盖率和未达到基线质量的次数均保留在all_paired.csv。",
              "","本报告自动核对全部180万条布局记录的约束、保存的前沿、HV和成本恒等式；每个算例另按三个目标极值抽查六种方法的完整尾流累加及独立最小生成树。",
              "","各子目录包含三维前沿与能量—成本投影、布局叠加禁布区、收敛与可行率图，以及每次运行的完整可行Pareto布局。summary.csv提供均值和标准差。",
              "","模型范围：转子重叠Jensen尾流近似、固定推力系数、24风向区间、合成地形，以及研究用成本和有效包络面积代理。CCGDE3为现有Python适配版。三个算例不足以证明普遍收敛或真实场址工程有效性。",
              "","ridge_1600同时改变风况和地形，只用于迁移验证。800和1600候选点为两个固定离散化，不保证网格嵌套。跨算例直接比较HV数值没有意义。",
              "","课题交付仍需将本结果与建模说明、算法伪代码、复杂度分析整合到最终课程报告，并明确研究目标2、3在哪些条件成立；实验完成不自动表示报告已提交或课题已验收。"]
    (folder/"validation_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    result=dict(status="complete",finished_at=now(),runs=180,geometry_records_checked=1800000,
                primary_comparisons=tests,source_hashes_verified=True,independent_audits_passed=True,
                report=str(folder/"validation_report.md"),scope="Prespecified benchmark confirmation; course report and submission are separate deliverables")
    save_json(folder/"DONE.json",result)
    return result
