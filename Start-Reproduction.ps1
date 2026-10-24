param([Parameter(Mandatory=$true)][string]$Python,[ValidateSet('repair','terrain')][string]$Campaign='repair')
$ErrorActionPreference='Stop'
$packageRoot=$PSScriptRoot
$pythonAbsolute=(Resolve-Path -LiteralPath $Python).Path
$hiddenPython=Join-Path (Split-Path -Parent $pythonAbsolute) 'pythonw.exe'
if (-not (Test-Path -LiteralPath $hiddenPython)) { throw 'This Windows launcher requires pythonw.exe next to the selected python.exe.' }
$identity=([System.Security.Cryptography.SHA256]::Create()).ComputeHash([Text.Encoding]::UTF8.GetBytes($packageRoot))
$suffix=([BitConverter]::ToString($identity)).Replace('-','').Substring(0,12)
$taskName='WindfarmReproduce_'+$Campaign+'_'+$suffix
$wrapper=Join-Path $packageRoot ('background_'+$Campaign+'.py')
$wrapperText=@"
import os, runpy, sys
from pathlib import Path
root=Path(__file__).resolve().parent
os.environ['WINDFARM_BACKGROUND_WORKER']='1'
os.environ['WINDFARM_REPRO_CAMPAIGN']='$Campaign'
sys.argv=[str(root/'reproduce.py'),'--campaign','$Campaign']
runpy.run_path(str(root/'reproduce.py'),run_name='__main__')
"@
if (Test-Path -LiteralPath $wrapper) {
    if ((Get-Content -LiteralPath $wrapper -Raw).Trim() -ne $wrapperText.Trim()) { throw 'Existing background entry differs; inspect instead of overwriting.' }
} else { $wrapperText | Set-Content -LiteralPath $wrapper -Encoding utf8 }
$existing=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and $existing.State -eq 'Running') { Write-Output 'Already running; no duplicate launch.'; exit 0 }
if (Test-Path -LiteralPath (Join-Path $packageRoot ('reproduced\'+$Campaign+'\DONE.json'))) { Write-Output 'Already complete; previous results preserved.'; exit 0 }
if (-not $existing) {
    $action=New-ScheduledTaskAction -Execute $hiddenPython -Argument ('"'+$wrapper+'"') -WorkingDirectory $packageRoot
    $principal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    $settings=New-ScheduledTaskSettingsSet -Hidden -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 1)
    Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings -Description 'Reproduce one frozen research campaign under this package root; original evidence remains read-only.' | Out-Null
} elseif ($existing.Actions.Execute -ne $hiddenPython -or $existing.Actions.Arguments -ne ('"'+$wrapper+'"')) { throw 'Existing task differs.' }
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State
Write-Output (Join-Path $packageRoot ('reproduced\'+$Campaign+'\progress.json'))
