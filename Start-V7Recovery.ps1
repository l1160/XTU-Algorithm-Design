param([Parameter(Mandatory=$true)][string]$Job,[Parameter(Mandatory=$true)][string]$Python)
$ErrorActionPreference='Stop'
$jobAbsolute=(Resolve-Path -LiteralPath $Job).Path
$jobValue=Get-Content -LiteralPath $jobAbsolute -Raw | ConvertFrom-Json
$hidden=Join-Path (Split-Path -Parent (Resolve-Path -LiteralPath $Python).Path) 'pythonw.exe'
$digest=([Security.Cryptography.SHA256]::Create()).ComputeHash([Text.Encoding]::UTF8.GetBytes($jobAbsolute))
$taskName='WindfarmV7IO_'+([BitConverter]::ToString($digest)).Replace('-','').Substring(0,16)
$script=Join-Path $PSScriptRoot 'experiment_tools\v7_io_recovery.py'
$arguments='"'+$script+'" --job "'+$jobAbsolute+'"'
$existing=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and $existing.State -eq 'Running') { 'Already running; duplicate prevented.'; exit 0 }
if (Test-Path -LiteralPath (Join-Path $jobValue.output 'DONE.json')) { 'Already complete; existing results retained.'; exit 0 }
if (-not $existing) {
    $action=New-ScheduledTaskAction -Execute $hidden -Argument $arguments -WorkingDirectory $PSScriptRoot
    $principal=New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    $settings=New-ScheduledTaskSettingsSet -Hidden -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings -Description 'Continue the frozen V7 comparison with atomic status-write retry only.' | Out-Null
} elseif ($existing.Actions.Execute -ne $hidden -or $existing.Actions.Arguments -ne $arguments) { throw 'Existing task differs.' }
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State
