"""Integration fixtures for the declared CC adaptation, not MATLAB equivalence."""
from pathlib import Path
import hashlib
import json
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "windfarm_research_v4"))
import bootstrap_v4
import numpy as np
import algorithms as A
from campaign_figures import check_complete, table


def run(src, out):
    check_complete(Path(src))
    out = Path(out)
    checks = []
    def check(name, condition):
        assert condition, name
        checks.append(dict(name=name, passed=True))

    # Each case specifies an independently interpretable constraint-dominance
    # outcome. These are artificial evaluator responses, not wind-farm runs.
    cases = [
        ("feasible_child_dominates", [2,2,2], 0, [1,1,1], 0, ["C"]),
        ("feasible_parent_dominates", [1,1,1], 0, [2,2,2], 0, ["P"]),
        ("feasible_incomparable", [1,2,3], 0, [2,1,3], 0, ["P","C"]),
        ("feasible_equal", [1,1,1], 0, [1,1,1], 0, ["C"]),
        ("parent_feasible", [9,9,9], 0, [0,0,0], 1, ["P"]),
        ("child_feasible", [0,0,0], 1, [9,9,9], 0, ["C"]),
        ("invalid_child_smaller", [0,0,0], 2, [9,9,9], 1, ["C"]),
        ("invalid_parent_smaller", [9,9,9], 1, [0,0,0], 2, ["P"]),
        ("invalid_equal", [1,1,1], 1, [2,2,2], 1, ["C"]),
    ]
    n = len(cases)
    class FixtureFarm:
        def __init__(self):
            self.records = []
            self.inputs = []
        def evaluate(self, keys):
            i = len(self.records)
            self.inputs.append(np.array(keys).copy())
            row = cases[i % n]
            f, cv = (row[1], row[2]) if i < n else (row[3], row[4])
            self.records.append((i, list(f), cv))
            return np.array(keys).copy(), np.array(f, dtype=float), float(cv)

    class RecordingRng:
        def __init__(self, seed):
            self.inner = np.random.default_rng(seed)
            self.donors = []
        def random(self, *args, **kwargs):
            return self.inner.random(*args, **kwargs)
        def integers(self, *args, **kwargs):
            return self.inner.integers(*args, **kwargs)
        def choice(self, values, size, replace):
            answer = self.inner.choice(values, size, replace=replace)
            self.donors.append((np.array(values), np.array(answer), replace))
            return answer

    x = np.linspace(.1, .9, n * 12).reshape(n, 12)
    pop = A.Pop(x, np.zeros((n,3)), np.zeros(n), np.zeros_like(x))
    engine = A.Engine.__new__(A.Engine)
    engine.farm = FixtureFarm()
    engine.budget = 2*n+1
    engine.rng = RecordingRng(20261020)
    engine.group_list = [np.arange(0,12,2), np.arange(1,12,2)]
    engine.group_index = 0
    engine.species = [pop.take(np.arange(n)), pop.take(np.arange(n))]
    captures = []
    original_select = A.gde_select
    def capture(pool, size):
        captures.append(pool)
        return original_select(pool, size)
    A.gde_select = capture
    try:
        engine.cc_step()
    finally:
        A.gde_select = original_select
    check("A full component step evaluates each contextual parent and each child", len(engine.farm.records)==2*n)
    inputs = np.array(engine.farm.inputs)
    active, other = engine.group_list
    check("Unrepaired contextual parents preserve the active species component",
          np.array_equal(inputs[:n,active], x[:,active]))
    check("All unrepaired trials share the same complement context",
          np.all(inputs[:,other] == inputs[0,other]))
    check("DE/PM offspring stay within random-key bounds", np.all((inputs[n:]>=0)&(inputs[n:]<=1)))
    check("Each adapted DE donor pair excludes its target and uses distinct donors",
          len(engine.rng.donors)==n and all(i not in donors and len(set(donors))==2 and not replace
          for i, (_,donors,replace) in enumerate(engine.rng.donors)))
    expected_f, expected_cv = [], []
    decision_rows = []
    for name, pf, pcv, cf, ccv, keep in cases:
        decision_rows.append(dict(case=name,parent_CV=pcv,child_CV=ccv,expected_retained="+".join(keep)))
        for label in keep:
            expected_f.append(pf if label=="P" else cf)
            expected_cv.append(pcv if label=="P" else ccv)
    check("Nine parent-child constraint-dominance cases match the source decision rules",
          np.array_equal(captures[0].f, expected_f) and np.array_equal(captures[0].cv, expected_cv))
    check("Species cardinality is retained after GDE3 truncation", len(engine.species[0])==n)
    check("Group scheduler advances cyclically", engine.group_index==1)
    engine.cc_step()
    check("Odd final request is consumed once without budget overrun", len(engine.farm.records)==2*n+1)
    check("Unaffected species survives an odd-budget terminal step",
          np.array_equal(engine.species[1].x, pop.x))

    f = np.array([[0,4],[1,3],[2,2],[3,1],[4,0]], float)
    for scale in (1., 1e6):
        p = A.Pop(np.arange(5)[:,None], f*[1,scale], np.zeros(5), np.zeros((5,1)))
        kept = A.gde_select(p,3).x[:,0]
        check(f"Iterative crowding on a five-point front preserves extrema and central spacing, scale={scale:g}",
              np.array_equal(kept,[0,2,4]))
    # These assertions freeze the reviewed reference, including an upstream
    # suspicious variable reference. They do not correct or execute MATLAB.
    ref = ROOT / "windfarm_study_v3/reference/CCGDE3"
    matlab = (ref/"CCGDE3.m").read_text(encoding="utf-8")
    assert "Dec2        = Population1.decs;" in matlab
    sources = [ROOT/"windfarm_study_v3/algorithms.py", ROOT/"windfarm_research_v4/variants.py",
               *sorted(ref.glob("*.m"))]
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    table(out/"pairwise_decision_cases.csv", decision_rows)
    explanation = (
        "这批检查覆盖当前Python协同差分进化适配的关键接口：共同上下文、活跃分量、"
        "约束支配、迭代拥挤度删除、子种群大小与奇数预算。评价器响应是人为构造的固定测试样例，"
        "不是新的风电场优化实验，也不是MATLAB逐代等价验证。\n\n"
        "与所保存PlatEMO参考的差异超过组数一项：\n"
        "1. 参考固定两组；当前主配置每组约100维。筛查已有两组40/80个体对照。\n"
        "2. 参考差分供体有放回抽样，可能重复或等于目标；当前供体互异且排除目标。\n"
        "3. 参考每个后代随机选择另一组的非支配协作者；当前一轮固定共同上下文。\n"
        "4. 参考保留原父代评价并评价新子代；当前在同一上下文重新评价父代，"
        "一对比较消耗两次请求。相同请求预算因此不等于相同差分生成次数。\n"
        "5. 当前全局修复和回写可能改变活跃组外的分量，所以子问题仍然耦合。\n"
        "6. 参考源代码第二组更新出现Dec2 = Population1.decs，静态看与前后变量不一致。"
        "未在MATLAB执行，不能认定该引用对所有运行的影响，也不把照搬可疑代码当成正确复现。\n\n"
        "因此本项目应使用“CCGDE3机制启发的约束随机键适配”这一准确范围。"
        "尾流/随机分组对照只识别当前适配内的分组效应。"
        "两组配置比较不能消除供体、协作者和父代重评差异，"
        "不能据此给原CCGDE3算法下普遍优劣结论。\n"
    )
    (out/"CC适配验证与参考差异.txt").write_text(explanation, encoding="utf-8")
    return dict(status="complete",checks=checks,artificial_evaluator_calls=2*n+1,
                new_optimization_runs=0,matlab_equivalence_claim=False,
                reviewed_reference_hashes=hashes,
                unresolved_scope="Reference-to-adaptation stochastic/configuration differences remain explicit.")
