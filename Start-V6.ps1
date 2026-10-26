param(
    [ValidateSet('Check','Run')][string]$Mode='Check',
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$Output,
    [string]$Config='',
    [string]$DependencyPath=''
)
$ErrorActionPreference='Stop'
$projectRoot=$PSScriptRoot
$pythonAbsolute=(Resolve-Path -LiteralPath $Python).Path
$hiddenPython=Join-Path (Split-Path -Parent $pythonAbsolute) 'pythonw.exe'
if (-not (Test-Path -LiteralPath $hiddenPython)) { throw 'pythonw.exe is required next to the chosen interpreter.' }
$outputAbsolute=[IO.Path]::GetFullPath($Output)
$configAbsolute=''
if ($Mode -eq 'Run') {
    if (-not $Config) { throw 'Run mode requires one explicit configuration file.' }
    $configAbsolute=(Resolve-Path -LiteralPath $Config).Path
}
$dependencyAbsolute=''
if ($DependencyPath) { $dependencyAbsolute=(Resolve-Path -LiteralPath $DependencyPath).Path }
New-Item -ItemType Directory -Force -Path $outputAbsolute | Out-Null
$job=[ordered]@{mode=$Mode;output=$outputAbsolute;config=$configAbsolute;dependency_path=$dependencyAbsolute}
$identity=([Security.Cryptography.SHA256]::Create()).ComputeHash([Text.Encoding]::UTF8.GetBytes($projectRoot+'|'+$outputAbsolute))
$suffix=([BitConverter]::ToString($identity)).Replace('-','').Substring(0,16)
$taskName='WindfarmV6_'+$suffix
$stateRoot=Join-Path $projectRoot '.background'
New-Item -ItemType Directory -Force -Path $stateRoot | Out-Null
$jobPath=Join-Path $stateRoot ($taskName+'.json')
$jobText=$job | ConvertTo-Json
if (Test-Path -LiteralPath $jobPath) {
    if ((Get-Content -LiteralPath $jobPath -Raw).Trim() -ne $jobText.Trim()) { throw 'Existing task configuration differs; choose a separate output.' }
} else { $jobText | Set-Content -LiteralPath $jobPath -Encoding utf8 }
$worker=Join-Path $projectRoot 'windfarm_research_v6\worker.py'
$arguments='"'+$worker+'" --job "'+$jobPath+'"'
$existing=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and $existing.State -eq 'Running') { Write-Output 'Already running; duplicate launch prevented.'; exit 0 }
if (Test-Path -LiteralPath (Join-Path $outputAbsolute 'DONE.json')) { Write-Output 'Already complete; existing results retained.'; exit 0 }
if (-not $existing) {
    $action=New-ScheduledTaskAction -Execute $hiddenPython -Argument $arguments -WorkingDirectory $projectRoot
    $principal=New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    $settings=New-ScheduledTaskSettingsSet -Hidden -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings -Description 'Explicit single windfarm V6 run or bounded functional checks; full-state checkpoints.' | Out-Null
} elseif ($existing.Actions.Execute -ne $hiddenPython -or $existing.Actions.Arguments -ne $arguments) { throw 'Existing scheduled task action differs.' }
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State
Write-Output (Join-Path $outputAbsolute 'progress.json')
Write-Output 'Verify a newly updated worker log and progress; task state alone is not proof of execution.'
