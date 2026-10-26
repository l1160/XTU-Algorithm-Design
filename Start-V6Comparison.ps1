param([Parameter(Mandatory=$true)][string]$Plan,[Parameter(Mandatory=$true)][string]$Python)
$ErrorActionPreference='Stop'
$projectRoot=$PSScriptRoot
$planAbsolute=(Resolve-Path -LiteralPath $Plan).Path
$frozenPlan=Get-Content -LiteralPath $planAbsolute -Raw | ConvertFrom-Json
$pythonAbsolute=(Resolve-Path -LiteralPath $Python).Path
$hiddenPython=Join-Path (Split-Path -Parent $pythonAbsolute) 'pythonw.exe'
if (-not (Test-Path -LiteralPath $hiddenPython)) { throw 'pythonw.exe is required.' }
$identity=([Security.Cryptography.SHA256]::Create()).ComputeHash([Text.Encoding]::UTF8.GetBytes($planAbsolute))
$suffix=([BitConverter]::ToString($identity)).Replace('-','').Substring(0,16)
$taskName='WindfarmV6Comparison_'+$suffix
$worker=Join-Path $projectRoot 'experiment_tools\v6_compare.py'
$arguments='"'+$worker+'" --plan "'+$planAbsolute+'" --stage campaign'
$existing=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and $existing.State -eq 'Running') { Write-Output 'Comparison already running; no duplicate launch.'; exit 0 }
if (Test-Path -LiteralPath (Join-Path $frozenPlan.output 'DONE.json')) { Write-Output 'Comparison complete; existing results preserved.'; exit 0 }
if (-not $existing) {
    $action=New-ScheduledTaskAction -Execute $hiddenPython -Argument $arguments -WorkingDirectory $projectRoot
    $principal=New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    $settings=New-ScheduledTaskSettingsSet -Hidden -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings -Description 'One frozen sequential comparison of CCGDE3/LMOCSO/DGEA and their problem-specific variants.' | Out-Null
} elseif ($existing.Actions.Execute -ne $hiddenPython -or $existing.Actions.Arguments -ne $arguments) { throw 'Existing scheduled action differs.' }
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State
Write-Output (Join-Path $frozenPlan.output 'progress.json')
