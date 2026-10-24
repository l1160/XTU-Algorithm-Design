# 考虑位置禁布约束的大规模多目标风电场布局优化

课程研究项目的源码、输入数据和复现入口，供小组成员分工运行和复核实验。

研究对象是在场界、禁布区和风机最小间距约束下，从候选位置中选择固定数量的风机，联合优化年发电量、布局相关度电成本和有效开发包络面积。当前正式对照为 **CCGDE3-inspired 适配版、LMOCSO、DGEA**；新实验入口不会启动 NSGA-III。

## 包含什么

- `windflo_standard/data/Scenarios/`：WindFLO 的 20 个风况与场界场景，包括 10 个带矩形禁布区的场景；主实验使用 `obs_00` 和 `obs_01`。
- `windfarm_terrain_v5/data/`：flat、rolling、ridge 三种**合成高程**，分别有 25、50、100 m 网格及 50 m 网格上的候选点。数据使用局部米制坐标，没有真实场址坐标或 EPSG。
- `windfarm_study_v3/`：候选点、经济模型、约束修复和算法核心。
- `windfarm_research_v4/model_v4.py`：修正后的平面尾流模型；运行应使用根目录入口，不应单独把 V3 基类当作最终模型。
- `windfarm_terrain_v5/`：高程、坡度禁布、三维几何尾流和地表连线长度模型。
- `windfarm_delivery/`：两组冻结实验的运行与统计代码。
- `reference_results/`：已完成的 120 次正式实验的逐次/汇总统计、配对比较、收敛数据和代表布局。

公开仓库不包含个人账号凭据、进程日志、旧检查点、第三方依赖安装目录或报告/PPT。完整历史实验档案保留在项目负责人本机。旧兼容模块和参考 MATLAB 文件中的 NSGA 名称不代表当前研究基线；当前入口只接受下表两组计划。

## 环境与安装

当前后台运行入口支持 **Windows 10/11、Python 3.12 x64、CPU**。Linux/macOS 尚未适配 Windows 文件锁及任务计划程序，请勿直接照搬 Windows 运行命令。

在 PowerShell 中执行：

```powershell
git clone https://github.com/l1160/windfarm-layout-optimization.git
cd windfarm-layout-optimization
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe reproduce.py --campaign terrain --dry-run
.\.venv\Scripts\python.exe reproduce.py --campaign repair --dry-run
```

最后两条命令只核验代码、输入与计划，不启动优化。依赖安装需要联网；安装完成后实验不依赖网络或 Codex。首次地形实验会生成地表距离缓存，初始化可能比后续运行慢。四进程批量运行需要容纳多个 1600×1600×24 的尾流数组；内存有限时使用下面的单进程小组分工入口。

## 七人分工：每人只运行自己的一份

全组使用同一个仓库提交版本。成员编号固定为 1–7，每人顺序运行分配到的任务，同时仅占用一个工作进程。编号不同的任务不重叠；每组 60 次实验分为 8–9 次/人。

查看分配表，不运行实验：

```powershell
.\.venv\Scripts\python.exe team.py --campaign terrain --member 1 --members 7 --list
```

后台启动（成员 2–7 将 `-Member 1` 改为自己的编号）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start-TeamRun.ps1 -Campaign terrain -Member 1 -Members 7
```

这一组完成后，再运行修复消融：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start-TeamRun.ps1 -Campaign repair -Member 1 -Members 7
```

任务由 Windows 任务计划程序启动，使用隐藏的 Python 进程、独立日志和重复启动锁。成功运行后关闭查看日志的终端或 Codex、断网不会停止实验；**关机、睡眠或注销可能中断运行**，请保持电脑开机并避免睡眠。重新执行同一命令会跳过完成项并从自己的完整 `last.pkl` 继续，保留原日志；不要删除检查点后重新开始。不要加载他人提供的未知 pickle 文件。

查看状态与日志：

```powershell
Get-Content .\reproduced\terrain\team_1_of_7.json
Get-ChildItem .\reproduced\terrain -Recurse -Filter progress.json | ForEach-Object { Get-Content $_.FullName }
$latestLog = Get-ChildItem .\reproduced\terrain -Recurse -Filter 'worker_*.log' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content -LiteralPath $latestLog.FullName -Tail 20 -Wait
```

应看到**本次新日志**及 `progress.json` 中 `requests`、`updated_at` 更新。只有任务状态或 PID 不能说明优化正在推进。成员完成标志是 `team_编号_of_7.json` 的 `status: complete`；单次实验完成标志为该目录的 `DONE.json`。启动前失败的详细信息在 `.background/` 的新日志中。按 Ctrl+C 退出日志查看不会结束后台任务。

暂停：在相应 `reproduced/terrain/` 或 `reproduced/repair/` 创建内容为 `{}` 的 `PAUSE_REQUEST.json`，工作进程将在保存完整检查点后退出；恢复前移除此标记，再运行原启动命令。不要在运行中修改源码、预算、种子或输入。

## 汇总组员结果

成员提交自己 `reproduced/terrain/` 和 `reproduced/repair/` 下的实验子目录，其中至少保留 `DONE.json`、`evaluations.npz`、`trace.json`、`case_metadata.json`。已完成实验不需要传输检查点。请通过网盘或共享磁盘交付，不要把大检查点提交进 Git。

负责人将同一提交版本产生的实验子目录合并到自己相应的 `reproduced/` 目录，遇到同名任务先核对来源，不要覆盖。每组应有 60 个不同任务目录。然后运行完整入口：它跳过已完成任务，补齐缺项并生成整体统计；如果 60 次均已完成，只进行汇总。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start-Reproduction.ps1 -Python .\.venv\Scripts\python.exe -Campaign terrain
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start-Reproduction.ps1 -Python .\.venv\Scripts\python.exe -Campaign repair
```

完整入口默认每组最多四个并行 CPU 工作进程，内存不足时请先完成七人分工再汇总，避免在一台机器同时启动两组完整实验。完整成功标志为 `reproduced/<campaign>/DONE.json` 和 `analysis/` 统计表。比较结果前核对所有成员使用相同 Git 提交、依赖版本与配置；不要把不一致的模型结果混合计算 HV。

## 冻结的实验设置

| 实验 | 场景与规模 | 算法/处理 | 随机种子 | 预算 |
|---|---|---|---|---|
| terrain | obs_00/01，ridge，1600 候选点、400 台风机 | CCwake_base、LM_base、DGEA_base | 801–810 | 每次 10,000 请求，共 60 次 |
| repair | obs_01，平面，1600 候选点、400 台风机 | LMOCSO/DGEA × 无修复、贪心修复、有界回溯 | 601–610 | 每次 10,000 请求，共 60 次 |

初始种群为 100，配对方案共享初始布局，初始化计入预算。M 表示候选点/随机键维数，不是已安装风机数量。经济目标采用参考成本参数与布局相关连接长度，不能解释为真实场址报价。高程扩展只包含几何效应，未模拟复杂地形风速增益或完整流场；地表 MST 也不是施工可行的电缆路径设计。

评价包括可行率、AEP、LCOE、有效开发包络面积、Pareto 集、HV、收敛和重复布局诊断。HV 在同一案例内共同归一化，参考点为 `[1.1,1.1,1.1]`，可能大于 1，**不是百分数**，不可直接跨场景比较。统计使用配对符号翻转检验和预先指定比较族的 Holm 校正。

已完成结果中 ridge 两风况的平均 HV 排序为 CCwake > DGEA > LMOCSO；修复消融中贪心修复改善两算法结果，而本组测试的回溯分支未被触发。这些结论仅针对本模型、适配实现和预算。CCGDE3-inspired 与原始 MATLAB 实现存在分组、协作上下文及预算计数差异，不宣称严格复现原算法。

## 来源与使用说明

- WindFLO 数据及参考实现来自 [d9w/WindFLO](https://github.com/d9w/WindFLO)，具体来源记录在 `windflo_standard/data/SOURCE.json`，MIT 声明保留在 `windflo_standard/reference/LICENSE`。
- 算法参考 [BIMK/PlatEMO](https://github.com/BIMK/PlatEMO)，固定来源提交和引用见 `windfarm_study_v3/reference/SOURCE.json`、`ATTRIBUTION.txt` 及各 `.m` 文件原始声明。Python 为面向本约束随机键问题的适配实现，未做完整 MATLAB 运行等价验证。
- 使用 PlatEMO 相关代码的研究成果应按原声明致谢并引用 Tian、Cheng、Zhang、Jin (2017)，以及所用算法原始论文。第三方代码继续遵循各自声明，本仓库没有把它们重新授权为 MIT。
- 合成 DEM 的生成公式见 `windfarm_terrain_v5/terrain.py`；不能将这些数据称为实测高程。

代码出现问题时提交 Issue，并附 Python 版本、Git 提交号、实验名称以及错误日志片段；不要上传密码或访问令牌。
