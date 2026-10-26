param([Parameter(Mandatory=$true)][string]$Python,[string]$Output)
$ErrorActionPreference='Stop'
$pythonAbsolute=(Resolve-Path -LiteralPath $Python).Path
$hiddenPython=Join-Path (Split-Path -Parent $pythonAbsolute) 'pythonw.exe'
if (-not (Test-Path -LiteralPath $hiddenPython)) { throw 'pythonw.exe is required.' }
if (-not $Output) { $Output=Join-Path $PSScriptRoot 'reproduced\confirmation_reanalysis' }
$outputAbsolute=[IO.Path]::GetFullPath($Output)
$digest=([Security.Cryptography.SHA256]::Create()).ComputeHash([Text.Encoding]::UTF8.GetBytes($outputAbsolute))
$taskName='WindfarmSavedAnalysis_'+([BitConverter]::ToString($digest)).Replace('-','').Substring(0,16)
$worker=Join-Path $PSScriptRoot 'publication_analysis_worker.py'
$arguments='"'+$worker+'" --output "'+$outputAbsolute+'"'
$existing=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and $existing.State -eq 'Running') { Write-Output 'Already running; duplicate prevented.'; exit 0 }
if (Test-Path -LiteralPath (Join-Path $outputAbsolute 'DONE.json')) { Write-Output 'Already complete; results preserved.'; exit 0 }
if (-not $existing) {
    $action=New-ScheduledTaskAction -Execute $hiddenPython -Argument $arguments -WorkingDirectory $PSScriptRoot
    $principal=New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    $settings=New-ScheduledTaskSettingsSet -Hidden -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings -Description 'Reanalyze saved windfarm results and figures; no optimization.' | Out-Null
} elseif ($existing.Actions.Execute -ne $hiddenPython -or $existing.Actions.Arguments -ne $arguments) { throw 'Existing task differs.' }
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State
Write-Output $outputAbsolute
