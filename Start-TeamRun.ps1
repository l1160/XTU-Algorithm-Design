param(
    [ValidateSet('repair','terrain')][string]$Campaign='terrain',
    [ValidateRange(1,100)][int]$Member=1,
    [ValidateRange(1,100)][int]$Members=7,
    [string]$Python=(Join-Path $PSScriptRoot '.venv\Scripts\python.exe')
)
$ErrorActionPreference='Stop'
if ($Member -gt $Members) { throw 'Member must not exceed Members.' }
$projectRoot=$PSScriptRoot
$pythonAbsolute=(Resolve-Path -LiteralPath $Python).Path
$hiddenPython=Join-Path (Split-Path -Parent $pythonAbsolute) 'pythonw.exe'
if (-not (Test-Path -LiteralPath $hiddenPython)) { throw 'pythonw.exe is missing.' }
$identity=([Security.Cryptography.SHA256]::Create()).ComputeHash([Text.Encoding]::UTF8.GetBytes($projectRoot))
$suffix=([BitConverter]::ToString($identity)).Replace('-','').Substring(0,12)
$taskName="WindfarmTeam_${Campaign}_${Member}_${Members}_$suffix"
$stateRoot=Join-Path $projectRoot '.background'
New-Item -ItemType Directory -Force -Path $stateRoot | Out-Null
$wrapper=Join-Path $stateRoot ($taskName+'.py')
$wrapperText=@"
import os,runpy,sys,traceback,datetime
from pathlib import Path
root=Path(__file__).resolve().parent.parent
os.chdir(root)
os.environ['WINDFARM_BACKGROUND_WORKER']='1'
sys.path.insert(0,str(root))
sys.stdin=open(os.devnull)
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
sys.stdout=sys.stderr=(root/'.background'/('$taskName'+'_'+stamp+'.log')).open('a',encoding='utf-8',buffering=1)
sys.argv=[str(root/'team.py'),'--campaign','$Campaign','--member','$Member','--members','$Members']
try:runpy.run_path(str(root/'team.py'),run_name='__main__')
except Exception:traceback.print_exc();raise
"@
if (Test-Path -LiteralPath $wrapper) {
    if ((Get-Content -LiteralPath $wrapper -Raw).Trim() -ne $wrapperText.Trim()) { throw 'Existing launcher differs; inspect first.' }
} else { $wrapperText | Set-Content -LiteralPath $wrapper -Encoding utf8 }
$existing=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and $existing.State -eq 'Running') { Write-Output 'Already running; no duplicate launch.'; exit 0 }
if (-not $existing) {
    $action=New-ScheduledTaskAction -Execute $hiddenPython -Argument ('"'+$wrapper+'"') -WorkingDirectory $projectRoot
    $principal=New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    $settings=New-ScheduledTaskSettingsSet -Hidden -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 1)
    Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings -Description 'Run one assigned shard of a frozen windfarm experiment campaign.' | Out-Null
} elseif ($existing.Actions.Execute -ne $hiddenPython -or $existing.Actions.Arguments -ne ('"'+$wrapper+'"')) { throw 'Existing task differs.' }
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State
Write-Output (Join-Path $projectRoot "reproduced\$Campaign\team_${Member}_of_${Members}.json")
Write-Output 'A task state alone does not prove progress. Check fresh worker logs and request counts.'
