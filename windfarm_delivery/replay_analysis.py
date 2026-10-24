"""Describe verified mapping categories without changing optimization or ranking."""
from pathlib import Path
import csv
import json
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/"windfarm_research_v4"))
import bootstrap_v4
import numpy as np
import matplotlib.pyplot as plt
from campaign_figures import table, savefig


def run(src,out):
    src,out=Path(src),Path(out)
    done=json.loads((src/"DONE.json").read_text(encoding="utf-8"))
    assert done["replayed_runs"]==24 and done["replayed_requests"]==48000
    with (src/"aggregate_mapping_counts.csv").open(encoding="utf-8-sig") as stream:
        counts=list(csv.DictReader(stream))
    fields=["identical_keys","identical_full_order","feasible_subset_projection","repair_mapping_collision"]
    labels=["相同连续向量","不同向量、同完整排序","新排序、同可行子集","新排序经修复得到旧布局"]
    colors=["#425F7C","#8294BB","#3E9B8A","#D38B42"]
    fractions=[]
    for row in counts:
        n,duplicates=int(row["requests"]),int(row["original_duplicates"])
        assert n==6000 and sum(int(row[k]) for k in fields)==duplicates
        fractions.append(dict(M=int(row["M"]),variant=row["variant"],requests=n,duplicates=duplicates,
            duplicate_rate=duplicates/n,
            **{k+"_fraction_all_requests":int(row[k])/n for k in fields},
            **{k+"_fraction_duplicates":int(row[k])/duplicates if duplicates else "" for k in fields}))
    table(out/"mapping_fractions.csv",fractions)
    variants=["NSGA3_base","LM_base","DGEA_base","DGEA_no_writeback"]
    names=["NSGA-III","LMOCSO基础","DGEA基础","DGEA无回写"]
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":10.5})
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharex=True,layout="constrained")
    for ax,size in zip(axes,(800,1600)):
        selected=[next(r for r in fractions if r["M"]==size and r["variant"]==v) for v in variants]
        left=np.zeros(4)
        for field,label,color in zip(fields,labels,colors):
            value=np.array([r[field+"_fraction_all_requests"]*100 for r in selected])
            ax.barh(names,value,left=left,label=label,color=color,height=.62)
            left+=value
        ax.invert_yaxis()
        ax.set(xlim=(0,100),xlabel="占全部请求的比例（%）",title=f"M={size}，每法3种子共6000次请求")
        ax.grid(axis="x",alpha=.15)
        for i,v in enumerate(left):
            ax.text(v+1,i,f"{v:.1f}%",va="center",fontsize=9)
    handles,legend=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc="outside lower center",ncol=2,frameon=False,fontsize=9.5)
    savefig(fig,out,"verified_encoding_repair_decomposition")
    lm=[r for r in fractions if r["variant"]=="LM_base"]
    dg=[r for r in fractions if r["variant"]=="DGEA_base"]
    text=(
        "编码与修复回放：已核对的结论\n\n"
        "24条原轨迹、48000次请求均逐项匹配原布局、目标、修复状态和算法摘要。"
        "使用保存目标查取，没有新的物理布局评价，没有追加新的优化排名。\n\n"
        f"LMOCSO：M800、M1600的重复率分别为{lm[0]['duplicate_rate']*100:.2f}%、{lm[1]['duplicate_rate']*100:.2f}%。"
        f"在各自重复请求中，新排序仍选中同一可行子集分别占"
        f"{lm[0]['feasible_subset_projection_fraction_duplicates']*100:.2f}%、"
        f"{lm[1]['feasible_subset_projection_fraction_duplicates']*100:.2f}%。"
        "因此连续向量或完整排序发生变化，并不意味着固定N的离散选址发生变化。"
        "这提供了编码信息损失的直接轨迹证据，而不只是对高重复率的猜测。\n\n"
        f"DGEA基础版：M800、M1600的重复请求中，新排序经修复映射到旧布局分别占"
        f"{dg[0]['repair_mapping_collision_fraction_duplicates']*100:.2f}%、"
        f"{dg[1]['repair_mapping_collision_fraction_duplicates']*100:.2f}%。"
        "取消回写后总体重复明显减少，但既有筛查中六个配对HV均下降。"
        "修复及回写既可能收缩候选映射，也使后续搜索围绕实际可行布局展开；"
        "不能把减少重复自动等同于提高搜索质量。\n\n"
        "NSGA-III在所回放的六次筛查中没有重复最终布局。"
        "相同修复接口不会必然使所有搜索方法发生同样程度的映射合并。\n\n"
        "分类是有序且互斥的：先判相同连续向量，再判相同完整排序，"
        "然后区分新排序的原始子集是否已经可行。"
        "相同向量或排序的请求仍可能再次经过修复，不能把第四类视为全部修复相关影响。"
        "第四类表示当前新排序在修复映射中与历史最终布局重合，"
        "不是对历史修复造成的总重复或最终HV影响的因果估计。\n\n"
        "图中分母为全部6000次请求，柱长合计为重复率。"
        "文字中约一半或超过九成的分母是各自重复请求，两个比例不能混用。"
        "这些结果来自2000请求的筛查轨迹，不代表10000请求确认实验已完成。"
        "独立确认中的分组、回写、范围归一化与种群补足比较仍应按预定方案完成并统一解释。\n"
    )
    (out/"回放结论与比例口径.txt").write_text(text,encoding="utf-8")
    return dict(status="complete",replayed_runs=24,requests=48000,aggregate_rows=len(fractions),
        all_fraction_denominators_explicit=True,new_optimization_runs=0,
        visual_review_required=True,scope="Current mapping-stage collisions, not total historical causal effects.")
