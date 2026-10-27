# 考虑位置禁布约束的大规模多目标风电场布局优化

在场界、禁布区、坡度和最小机间距约束下，从候选位置中选择400台风机，联合优化年发电量、度电成本和有效开发包络面积。项目包含 **CCGDE3、LMOCSO、DGEA三条基线及对应改进版**，以及已完成的180次正式对比实验。

当前模型采用转子圆盘重叠Jensen尾流、Weibull风速分布与非负功率积分、完整图最小生成树连接成本，以及凸包与允许开发区域的交集面积。六种方法共享同一模型和精确评价后端；CCGDE3为保留协同差分进化框架的Python适配版，高程采用合成DEM。

## 报告与正式结果

- [课题研究汇报书 Word版](reports/课题研究汇报书.docx)
- [性能均值与标准差](reference_results/confirmation_20261024/analysis/summary.csv)
- [配对检验及Holm校正](reference_results/confirmation_20261024/analysis/primary_tests.csv)
- [全部180次运行统计](reference_results/confirmation_20261024/analysis/all_runs.csv)
- [原始评价记录、布局、轨迹、图表与核查材料](reference_results/confirmation_20261024/)

每组使用10个新随机种子，每次10000次评价请求，初始100个布局配对一致。

| 算例 | 风况与地形 | 候选点数 | 风机数 |
|---|---|---:|---:|
| A | obs_00，平地 | 800 | 400 |
| B | obs_00，平地 | 1600 | 400 |
| C | obs_01，合成山脊 | 1600 | 400 |

下表为改进版HV均值相对各自基线的变化。HV衡量三目标解集质量，其增幅不是发电量增幅。

| 算法 | A | B | C | 三个算例的配对胜出次数 |
|---|---:|---:|---:|---|
| CCGDE3改进版 | +15.71% | +4.75% | +4.02% | 9/10、8/10、6/10 |
| LMOCSO改进版 | +108.11% | +210.65% | +401.93% | 10/10、10/10、10/10 |
| DGEA改进版 | +93.37% | +169.93% | +282.35% | 10/10、10/10、10/10 |

LMOCSO与DGEA改进版在相同请求数、相同预算两种口径下均取得三个算例各10次配对全胜，对应Holm校正p值均为0.0352。CCGDE3改进版平均时间减少22.30%–31.15%；其质量提升在A通过校正后的检验，在B、C未达到0.05阈值。

跑满预算时，LMOCSO改进版时间增加80.88%–132.38%，DGEA增加5.56%–10.31%；二者在相同时间内仍取得更高质量。全部180万条评价记录通过几何约束核查，可行率100%。重复率方面，LMOCSO明显改善，CCGDE3增加，DGEA在A略降、在B和C增加。报告保留这些差异并分析原因。

## 代码与证据

| 路径 | 内容 |
|---|---|
| `windfarm_validation/` | 当前六方法统一入口、冻结计划、统计与独立核查 |
| `windfarm_research_v6/` | 重叠尾流、可行投影、CCGDE3最终改进与完整续跑 |
| `windfarm_research_v7/` | LMOCSO/DGEA最终改进、稀疏回写、物理局部交换、精确Prim |
| `windfarm_study_v3/` | 基线核心、功率积分、经济模型与基础约束修复 |
| `windfarm_terrain_v5/` | 合成高程、坡度过滤和地表连接长度 |
| `windflo_standard/data/Scenarios/` | WindFLO风况、场界与禁布区 |
| `reference_results/confirmation_20261024/` | 180份配置、完整评价记录、轨迹、前沿、统计图表与核查证据 |
| `reports/` | 最新汇报书及版本说明 |

当前统一入口使用 **V6的CC改进与V7的LM/DGEA改进**。V6早期LM/DGEA方案保留为历史代码，不用于本轮正式结果。早期120次地形与修复实验的入口见[历史复现说明](docs/legacy_reproduction.md)。

每个正式运行均公开`evaluations.npz`、`trace.json`、`DONE.json`、`manifest.json`和`case_metadata.json`。原始证据中的历史路径原样保留；复现时使用下面的准备命令生成当前机器的路径。

## 环境与文件核验

后台入口支持Windows 10/11、Python 3.12 x64。正式实验使用Python 3.12.4，依赖版本固定在`requirements.txt`

在PowerShell中执行，后续命令均以仓库根目录为工作目录：

```powershell
git clone https://github.com/l1160/windfarm-layout-optimization.git
Set-Location windfarm-layout-optimization
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe verify_publication.py
```

最后一条命令只核对公开文件指纹、180份结果及报告版本，不执行优化。

## 后台复现

先准备并后台执行有界实现检查：

```powershell
.\.venv\Scripts\python.exe confirmation.py prepare-check
.\Start-Validation.ps1 -Job .\verification\confirmation_public\check_job.json -Python .\.venv\Scripts\python.exe
Get-Content .\verification\confirmation_public\DONE.json
```

确认检查的`status`为`passed`后，准备并启动180次正式比较：

```powershell
.\.venv\Scripts\python.exe confirmation.py prepare
.\Start-Validation.ps1 -Job .\reproduced\confirmation_public\campaign_job.json -Python .\.venv\Scripts\python.exe
.\Read-Validation.ps1 -Output .\reproduced\confirmation_public -Follow
```

准备命令只生成配置。输入为仓库内的场景、经济参数与地形代码；配置位于`reproduced/confirmation_public/configs/`，评价记录与检查点位于`runs/`，统计与图表位于`analysis/`。

启动后应看到本次新日志，以及`progress.json`中的请求数、更新时间实际增加。完整完成要求根目录`DONE.json`包含`runs: 180`、`independent_audits_passed: true`，且`analysis/DONE.json`存在。

重复执行相同启动命令会防止重复进程、跳过完成项，并从未完成项的完整`last.pkl`续跑，保留原日志。源码、配置或环境改变时拒绝继续原目录。暂停时在批次根目录创建内容为`{}`的`PAUSE_REQUEST.json`；明确恢复时移除该标记，再执行同一启动命令。不要删除检查点后重新开始。

## 只重建保存结果的统计与图表

```powershell
.\Start-ConfirmationAnalysis.ps1 -Python .\.venv\Scripts\python.exe
Get-ChildItem .\reproduced\confirmation_reanalysis -Filter 'analysis_*.log' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 |
    Get-Content -Tail 20 -Wait
```

此任务读取`reference_results/confirmation_20261024/`，在`reproduced/confirmation_reanalysis/`输出统计、图表及原方案的目标一致性复核，不增加优化运行。新目录的`DONE.json`是完成标志。后台任务同样有重复启动保护，保留已有输出和日志。

## 模型和比较规则

位置使用随机键表示，约束和重复判定作用于真实机位集合。所有局部交换的精确评价均计入预算，缓存命中也计一次请求。最终前沿统一由全部已访问的可行布局重建。

每个算例的60次运行共享HV归一化范围，参考点为`[1.1,1.1,1.1]`，不跨算例直接比较HV绝对值。每组采用10个配对种子，执行精确双侧符号翻转检验，对18项主要比较统一Holm校正。

高程通过坡度禁布、轮毂高差和地表连接长度进入模型；环境风况按扇区给定。度电成本采用参考技术参数及连接长度缩放。模型公式、参数和实际布局的工程取舍见汇报书。

## 来源与引用

- WindFLO来自[d9w/WindFLO](https://github.com/d9w/WindFLO)，来源与MIT声明见`windflo_standard/data/SOURCE.json`和`windflo_standard/reference/LICENSE`。
- 算法参考[BIMK/PlatEMO](https://github.com/BIMK/PlatEMO)，固定提交和原声明见`windfarm_study_v3/reference/`。研究使用应引用PlatEMO与CCGDE3、LMOCSO、DGEA原始论文。
- 成本参数依据NREL《2022 Cost of Wind Energy Review》，具体参数和本项目的缩放规则见`windfarm_study_v3/economics.json`。
