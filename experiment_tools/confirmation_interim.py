"""Read-only balanced interim summary. Never starts optimization or reevaluation."""
from pathlib import Path
import argparse
import ctypes
import datetime
import json
import os
import sys
import traceback


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--campaign",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent.parent
    sys.path.insert(0,str(root))
    sys.path.insert(0,r"D:\suanfa\windfarm_proto\.deps")
    os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",MPLBACKEND="Agg")
    if os.name=="nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x00004000)
    import numpy as np
    from pymoo.indicators.hv import HV
    from windfarm_research_v6.lifecycle import FileLock,sha,now
    from windfarm_validation.runtime import read,save_json,METHODS
    from windfarm_validation.analysis import nondominated,volume3,cover,table
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    with FileLock(output/"summary.lock"):
        if (output/"DONE.json").exists():
            print("Already summarized; preserving output",flush=True)
            return
        try:
            campaign=Path(args.campaign).resolve()
            plan=read(campaign/"plan.json")
            for path,digest in plan["source_sha256"].items():assert sha(root/path)==digest,path
            seeds=plan["seeds"][:8]
            selected_jobs=[j for j in plan["jobs"] if j["seed"] in seeds]
            assert len(selected_jobs)==144
            assert all((Path(j["output"])/"DONE.json").exists() for j in selected_jobs)
            print(now(),"START balanced 144-run saved-result summary; no optimizer or objective evaluation",flush=True)
            save_json(output/"input_snapshot.json",dict(created_at=now(),source_plan_sha256=sha(campaign/"plan.json"),
                seed_rule="First eight prespecified seeds in all cases and all methods",seeds=seeds,run_ids=[j["id"] for j in selected_jobs],
                scope="Interim descriptive results only; no hypothesis testing, parameter changes, or early stopping"))
            all_rows=[];all_pairs=[];summaries=[];audits=[];norms={}
            for case in plan["cases"]:
                data=[];prefix={};common_packages=None;candidate_xy=None
                jobs=[j for j in selected_jobs if j["case"]==case["id"]]
                assert len(jobs)==48
                for job in jobs:
                    src=Path(job["output"])
                    manifest=read(src/"manifest.json")
                    done=read(src/"DONE.json")
                    config=read(job["config"])
                    assert sha(job["config"])==job["config_sha256"]
                    assert manifest["config"]==done["config"]==config
                    for path,digest in manifest["sources"].items():assert plan["source_sha256"][path]==digest,path
                    environment=(manifest["python"],manifest["packages"])
                    if common_packages is None:common_packages=environment
                    else:assert common_packages==environment
                    assert done["status"]=="complete" and done["requests"]==10000
                    assert done["independent_pareto_geometry_passed"]
                    with np.load(src/"evaluations.npz",allow_pickle=False) as z:
                        metrics=z["metrics"].copy();valid=z["cv"]==0
                        front=z["pareto_request_indices"].copy();xy=z["candidate_xy"].copy()
                        first=z["selected"][:100].copy()
                    assert metrics.shape==(10000,10) and np.isfinite(metrics).all()
                    assert xy.shape==(case["candidates"],2)
                    if candidate_xy is None:candidate_xy=xy
                    else:np.testing.assert_array_equal(candidate_xy,xy)
                    if job["seed"] in prefix:
                        np.testing.assert_array_equal(prefix[job["seed"]][0],first)
                        np.testing.assert_array_equal(prefix[job["seed"]][1],metrics[:100])
                    else:prefix[job["seed"]]=(first,metrics[:100].copy())
                    assert float(valid.mean())==done["post_feasible_rate"]
                    f=metrics[:,:3]*[-1,1,1]
                    exact=nondominated(f[valid])
                    assert valid[front].all()
                    assert {tuple(p) for p in exact}=={tuple(p) for p in f[front]}
                    data.append(dict(job=job,done=done,f=f,valid=valid,front=exact,metrics=metrics,
                        trace=read(src/"trace.json")))
                    audits.append(dict(id=job["id"],exact_front_matches=True,records=10000,feasible_records=int(valid.sum()),
                        evaluations_sha256=sha(src/"evaluations.npz"),done_sha256=sha(src/"DONE.json")))
                pool=np.vstack([d["f"][d["valid"]] for d in data])
                low,high=pool.min(0),pool.max(0)
                span=np.maximum(high-low,1e-12);ref=np.full(3,1.1)
                norms[case["id"]]=dict(ideal=low.tolist(),nadir=high.tolist(),width=span.tolist(),reference=ref.tolist(),
                    scope="Pooled 48 runs in this balanced interim snapshot; final 60-run bounds can differ")
                indicator=HV(ref_point=ref)
                def hv(points):return float(indicator((points-low)/span)) if len(points) else 0.
                index={}
                for d in data:
                    j,done=d["job"],d["done"]
                    value=hv(d["front"])
                    independent=volume3((d["front"]-low)/span,ref)
                    assert abs(value-independent)<1e-10
                    row=dict(case=case["id"],method=j["method"],seed=j["seed"],HV=value,
                        independent_HV_error=abs(value-independent),CPU_seconds=done["worker_cpu_seconds"],
                        wall_seconds=done["worker_wall_seconds"],feasible_rate=float(d["valid"].mean()),
                        duplicate_rate=done["duplicate_request_rate"],unique_layouts=done["unique_feasible_layouts"],
                        restored_seconds=done["restore_wall_seconds"])
                    all_rows.append(row);d["row"]=row;index[j["method"],j["seed"]]=d
                for family in ("CCGDE3","LMOCSO","DGEA"):
                    pairs=[]
                    for seed in seeds:
                        a,b=index[family,seed],index[family+"_problem",seed]
                        ar,br=a["row"],b["row"]
                        limit=ar["CPU_seconds"]
                        eligible=[p for p in b["trace"] if p["elapsed_cpu_seconds"]<=limit]
                        if br["CPU_seconds"]<=limit:
                            timed_hv=br["HV"];count=10000
                        elif eligible:
                            count=eligible[-1]["evaluations"]
                            timed_hv=hv(nondominated(b["f"][:count][b["valid"][:count]]))
                        else:timed_hv=0.;count=0
                        pair=dict(case=case["id"],family=family,seed=seed,baseline_HV=ar["HV"],improved_HV=br["HV"],
                            HV_change_percent=100*(br["HV"]/ar["HV"]-1),
                            baseline_CPU=ar["CPU_seconds"],improved_CPU=br["CPU_seconds"],
                            CPU_change_percent=100*(br["CPU_seconds"]/ar["CPU_seconds"]-1),
                            improved_HV_at_baseline_CPU=timed_hv,fixed_CPU_change_percent=100*(timed_hv/ar["HV"]-1),
                            timed_requests=count,baseline_duplicate_rate=ar["duplicate_rate"],improved_duplicate_rate=br["duplicate_rate"],
                            improved_dominates_baseline_fraction=cover(b["front"],a["front"]),
                            baseline_dominates_improved_fraction=cover(a["front"],b["front"]))
                        pairs.append(pair);all_pairs.append(pair)
                    mean=lambda key:float(np.mean([p[key] for p in pairs]))
                    summary=dict(case=case["id"],family=family,seeds=8,baseline_HV_mean=mean("baseline_HV"),
                        improved_HV_mean=mean("improved_HV"),HV_mean_change_percent=100*(mean("improved_HV")/mean("baseline_HV")-1),
                        HV_paired_wins=sum(p["improved_HV"]>p["baseline_HV"]+1e-10 for p in pairs),
                        HV_paired_losses=sum(p["improved_HV"]<p["baseline_HV"]-1e-10 for p in pairs),
                        minimum_paired_HV_change_percent=min(p["HV_change_percent"] for p in pairs),
                        maximum_paired_HV_change_percent=max(p["HV_change_percent"] for p in pairs),
                        baseline_CPU_mean=mean("baseline_CPU"),improved_CPU_mean=mean("improved_CPU"),
                        CPU_mean_change_percent=100*(mean("improved_CPU")/mean("baseline_CPU")-1),
                        fixed_CPU_HV_mean_change_percent=100*(mean("improved_HV_at_baseline_CPU")/mean("baseline_HV")-1),
                        fixed_CPU_paired_wins=sum(p["improved_HV_at_baseline_CPU"]>p["baseline_HV"]+1e-10 for p in pairs),
                        baseline_duplicate_rate_mean=mean("baseline_duplicate_rate"),
                        improved_duplicate_rate_mean=mean("improved_duplicate_rate"),
                        improved_dominates_baseline_fraction_mean=mean("improved_dominates_baseline_fraction"),
                        baseline_dominates_improved_fraction_mean=mean("baseline_dominates_improved_fraction"))
                    summaries.append(summary)
                    print(now(),json.dumps(summary,ensure_ascii=False),flush=True)
                print(now(),"CASE AUDIT PASS",case["id"],"48 source-matched runs, paired initialization, independent fronts and HV",flush=True)
            table(output/"summary.csv",summaries);table(output/"paired.csv",all_pairs);table(output/"runs.csv",all_rows)
            save_json(output/"normalization.json",norms)
            save_json(output/"audit.json",dict(status="passed",runs=144,records=1440000,source_fingerprints_matched=True,
                shared_initialization_verified=True,independent_front_HV_passed=True,rows=audits,
                scope="Saved front/HV, finite objectives, manifests, candidate equality and initial layouts checked. Final campaign performs all-layout geometry and full objective rechecks."))
            labels={"flat_800":"平地·800候选点","flat_1600":"平地·1600候选点","ridge_1600":"山脊·1600候选点"}
            lines=["阶段性实验结果：前8个完整配对种子",f"生成时间：{now()}",
                "覆盖3个算例、6种方法、每种8个种子，共144次运行。每次400台风机、10000次评价请求。",
                "这是截至当前的阶段描述；10个种子的正式统计、几何核查和目标重算尚未完成。未计算或宣称显著性。",
                "百分比比较均为改进版均值相对各自基线均值。HV越高越好，CPU耗时越低越好。",
                "每个算例的48次运行共享一个归一化边界；边界可能随最终60次结果变化，不能与旧单种子百分比直接比较。",""]
            for r in summaries:
                lines.append(f"{labels[r['case']]} / {r['family']}：HV {r['HV_mean_change_percent']:+.2f}%，质量胜出 {r['HV_paired_wins']}/8；CPU {r['CPU_mean_change_percent']:+.2f}%；相同CPU预算的HV {r['fixed_CPU_HV_mean_change_percent']:+.2f}%，胜出 {r['fixed_CPU_paired_wins']}/8；重复率 {100*r['baseline_duplicate_rate_mean']:.2f}% → {100*r['improved_duplicate_rate_mean']:.2f}%。")
            lines += ["",f"144次运行记录的可行率范围：{min(r['feasible_rate'] for r in all_rows):.2%}–{max(r['feasible_rate'] for r in all_rows):.2%}。",
                "阶段核查已通过：源代码/配置/环境匹配、相同配对初始化、重建非支配前沿和独立HV计算。",
                "已有每次运行结束时的独立Pareto布局几何检查；全部布局的最终几何核查由正式批次收尾执行。",
                "不修改任何算法、冻结方案或正式分析；不追加优化运行。",
                "CCGDE3为项目现有Python适配版；转子重叠Jensen、合成地形与经济参数为研究模型。"]
            (output/"阶段结果说明.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
            import matplotlib.pyplot as plt
            plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False})
            fig,axes=plt.subplots(1,3,figsize=(15,5),layout="constrained")
            families=("CCGDE3","LMOCSO","DGEA")
            colors=("#3366A8","#D58025","#3B8D62")
            for ax,case in zip(axes,plan["cases"]):
                rr=[next(r for r in summaries if r["case"]==case["id"] and r["family"]==family) for family in families]
                x=np.arange(3)
                for k,key in enumerate(("HV_mean_change_percent","fixed_CPU_HV_mean_change_percent","CPU_mean_change_percent")):
                    y=[r[key] for r in rr]
                    bars=ax.bar(x+(k-1)*.25,y,.24,color=colors[k],label=("同请求数 HV ↑","同 CPU 预算 HV ↑","总 CPU 耗时 ↓")[k])
                    ax.bar_label(bars,labels=[f"{v:+.1f}%" for v in y],fontsize=7,padding=3)
                ax.axhline(0,color="#666666",linewidth=.7)
                ax.set(xticks=x,xticklabels=families,title=labels[case["id"]],ylabel="改进版相对各自基线均值变化（%）")
                ax.margins(y=.2);ax.grid(axis="y",alpha=.15)
            axes[0].legend(fontsize=8)
            fig.suptitle("阶段结果：每组8个新种子，最终10种子结论待完成",fontsize=13)
            fig.savefig(output/"interim_comparison.png",dpi=170)
            plt.close(fig)
            save_json(output/"DONE.json",dict(status="complete",finished_at=now(),runs_summarized=144,seeds=seeds,
                feasible_rate_min=min(r["feasible_rate"] for r in all_rows),feasible_rate_max=max(r["feasible_rate"] for r in all_rows),
                summary=summaries,scope="Balanced interim descriptive summary, not the final 180-run conclusion"))
            print(now(),"DONE interim summary; main campaign unchanged",flush=True)
        except BaseException:
            save_json(output/"FAILED.json",dict(error=traceback.format_exc(),finished_at=now()))
            traceback.print_exc();raise


if __name__=="__main__":main()
