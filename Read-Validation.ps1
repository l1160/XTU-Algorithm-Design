param([Parameter(Mandatory=$true)][string]$Output,[switch]$Follow)
$ErrorActionPreference='Stop'
$folder=(Resolve-Path -LiteralPath $Output).Path
if (Test-Path -LiteralPath (Join-Path $folder 'progress.json')) { Get-Content -LiteralPath (Join-Path $folder 'progress.json') }
if (Test-Path -LiteralPath (Join-Path $folder 'DONE.json')) { Write-Output 'Campaign complete: DONE.json exists; see analysis and audit outputs.' }
if ($Follow) {
    $log=Get-ChildItem -LiteralPath $folder -Filter 'worker_*.log' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($log) { Get-Content -LiteralPath $log.FullName -Tail 20 -Wait }
}
