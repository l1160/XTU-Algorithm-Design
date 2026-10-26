# LMOCSO与DGEA的直接优化版本

> 本页记录V7的单种子开发对比。最新三算例、十种子的180次统一验证见[仓库首页](../README.md)与`../reference_results/confirmation_20261024/`，汇报书采用最终正式结果。

本版本根据已完成V6结果，直接实现一套集成改进。只比较LMOCSO、DGEA各自的基线和新版本，共四次；没有自动参数网格、逐项消融或额外GPU任务。CCGDE3的既有版本及结果保留。

## 已落实的算法修改

- **LMOCSO**：继续使用竞争配对、原速度更新公式和APD选择；活动种群仅设置20个体的下限，依据原选择结果变化。初始100条共同布局仍全部计入预算。与父布局相同的子代可以保留新变量和速度，复用目标值，不再强制每一步换机位。
- **DGEA**：直接使用原预选择返回的个体和等级，不把淘汰的候选重新并回工作种群；原方向生成公式保留，合法的大幅机位变化不再被统一截成最多8个交换。
- **稀疏回写**：优先保留原算子输出，只修改实际改变入选状态的坐标。边界键值相等时，仅处理必要的边界并列项；物理布局不变时保留连续状态。
- **物理引导的局部改进**：原算子约占80%评价请求，其余预算用于尾流、包络面积或连接长度引导的可行单点交换。利用预先计算的风频加权尾流影响筛选候选，优先尝试减轻尾流、把边界机位移入包络、改善连接的变化。筛选是廉价启发式；所有提交的方案仍调用准确三目标评价器并计入预算。
- **质量选择**：优先保留原算法的质量选择结果和可行非支配方案，取消V6中固定一半权重追求机位集合差异的截断规则。布局去重后优先保留新子代的学习状态。

## 精确评价后端

尾流、功率积分、经济参数、禁布和面积定义保持V6模型。完整连接图的MST计算改为精确稠密Prim，没有删边、近似电缆长度或更改经济目标。已与独立稀疏MST核对至401个节点，并验证保存布局的全部目标一致至浮点误差范围。

两条基线也使用同一新评价后端，因此本轮的CPU差异不会把仅对改进版启用的通用后端优化混入算法对比。父布局的尾流缓存访问会更新最近使用顺序，减少仍在频繁使用的父布局被提前淘汰。

## 本轮固定对照

四个入口为`LMOCSO`、`LMOCSO_problem`、`DGEA`、`DGEA_problem`；后两类`_problem`在本目录指V7版本，不能与V6同名入口的数据混合。

条件：obs_00、flat、1600候选点、400台风机、初始100、10000次评价请求、种子901，单工作进程顺序执行。种子901是开发比较，不作独立多种子确认。

主HV尺度在运行前冻结为V6首轮的公共边界和参考点。比原经验理想点更好的目标允许负的归一化坐标；超出参考点的解不贡献该HV。同时报告纳入全部新旧数据范围的联合归一化作为敏感性检查。HV不是百分数，不能当作发电量增幅。

进程CPU轨迹按代记录，用来给出首次达到对应基线最终HV的记录时刻；该时间是代级记录的上界时刻，不宣称精确到单次评价。

## 运行、进度和恢复

当前环境已确认：Python为`D:\pycharm\python.exe`，依赖目录为`D:\suanfa\windfarm_proto\.deps`。输入沿用仓库中的`windflo_standard/data/Scenarios/obs_00.xml`、平坦合成地形及`windfarm_study_v3/economics.json`；无需神经网络权重。

本轮后台启动或续跑命令：

```powershell
Set-Location -LiteralPath 'D:\suanfa\github_public\windfarm-layout-optimization'
.\Start-V7Recovery.ps1 -Job 'D:\suanfa\github_public\windfarm-layout-optimization\reproduced\v7_direct_20261024_s901\controller_job.json' -Python 'D:\pycharm\python.exe'
```

任务计划程序负责脱离终端运行；相同任务防重复，已有完成项跳过，源码/配置改变时拒绝继续原目录，中断项使用完整`last.pkl`恢复。保留所有原日志与结果。

查看状态与日志：

```powershell
$v7Root='D:\suanfa\github_public\windfarm-layout-optimization\reproduced\v7_direct_20261024_s901'
Get-Content -LiteralPath (Join-Path $v7Root 'progress.json')
$v7State=Get-Content -LiteralPath (Join-Path $v7Root 'progress.json') -Raw | ConvertFrom-Json
if ($v7State.active_method) {
    $v7Run=Join-Path (Join-Path $v7Root 'runs') $v7State.active_method
    $v7Log=Get-ChildItem -LiteralPath $v7Run -Filter 'worker_*.log' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    Get-Content -LiteralPath $v7Log.FullName -Tail 25 -Wait
}
```

整体完成需要四个方法均有`DONE.json`，且根目录与`analysis`目录的`DONE.json`均完成。输出包括`summary.csv`、`paired_differences.csv`、`convergence.csv`、`comparison.png`，每个方法保留完整评价记录。

关闭日志终端或Codex不会停止已成功提交的任务；电脑仍需保持开机。控制器运行时请求抑制自动睡眠，退出时释放。手动睡眠、注销或关机仍可能中断。

暂停整轮：在本轮根目录创建`PAUSE_REQUEST.json`，当前方法会在完整检查点后退出。恢复时移除根目录的该标记并执行同一后台启动命令，控制器仅移除自己创建的子任务暂停标记。

## 正确性边界

七项必要检查已通过：精确MST、稀疏回写、物理目标一致性、合法远跳、DGEA预选择保留、四入口初始化/预算/续跑、集成运行器恢复。程序正确不等于性能已经更好，实际收益以本轮完整对照结果为准。

本轮运行维护记录：LM改进版完成数据导出后，最后一次进度JSON替换遇到Windows文件共享占用。结果与完整检查点已保存，状态恢复记录见本轮IO_RECOVERY.json。后续由experiment_tools/v7_io_recovery.py补充原子写入重试，数学模型、算法代码、配置及已完成结果保持原指纹；未重新运行已完成的LM实验。历史错误保存在FAILED_20261024_142031_preserved.json，当前运行状态以progress.json及最终DONE.json为准。
