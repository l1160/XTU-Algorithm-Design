param([Parameter(Mandatory=$true)][string]$Job,[Parameter(Mandatory=$true)][string]$Python)
$ErrorActionPreference='Stop'
$jobAbsolute=(Resolve-Path -LiteralPath $Job).Path
$settingsFile=Get-Content -LiteralPath $jobAbsolute -Raw | ConvertFrom-Json
$pythonAbsolute=(Resolve-Path -LiteralPath $Python).Path
$hiddenPython=Join-Path (Split-Path -Parent $pythonAbsolute) 'pythonw.exe'
if (-not (Test-Path -LiteralPath $hiddenPython)) { throw 'pythonw.exe is required.' }
$digest=([Security.Cryptography.SHA256]::Create()).ComputeHash([Text.Encoding]::UTF8.GetBytes($jobAbsolute))
$taskName='WindfarmV7_'+([BitConverter]::ToString($digest)).Replace('-','').Substring(0,16)
$worker=Join-Path $PSScriptRoot 'windfarm_research_v7\worker.py'
$arguments='"'+$worker+'" --job "'+$jobAbsolute+'"'
$existing=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and $existing.State -eq 'Running') { Write-Output 'Already running; duplicate prevented.'; exit 0 }
if (Test-Path -LiteralPath (Join-Path $settingsFile.output 'DONE.json')) { Write-Output 'Already complete; results preserved.'; exit 0 }
if (-not $existing) {
    $action=New-ScheduledTaskAction -Execute $hiddenPython -Argument $arguments -WorkingDirectory $PSScriptRoot
    $principal=New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    $settings=New-ScheduledTaskSettingsSet -Hidden -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings -Description 'V7 integrated LM/DGEA optimization with full checkpoints, no parameter grid.' | Out-Null
} elseif ($existing.Actions.Execute -ne $hiddenPython -or $existing.Actions.Arguments -ne $arguments) { throw 'Existing task differs.' }
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State
Write-Output $settingsFile.output
