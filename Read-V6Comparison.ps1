param([Parameter(Mandatory=$true)][string]$Plan)
$ErrorActionPreference='Stop'
$frozen=Get-Content -LiteralPath $Plan -Raw | ConvertFrom-Json
$overall=Join-Path $frozen.output 'progress.json'
if (Test-Path -LiteralPath $overall) {
    $state=Get-Content -LiteralPath $overall -Raw | ConvertFrom-Json
    [pscustomobject]@{Status=$state.status;ActiveMethod=$state.active_method;Completed=@($state.completed_methods).Count;Total=$state.total_runs;Updated=$state.updated_at} | Format-List
}
$rows=foreach ($job in $frozen.jobs) {
    $progress=Join-Path $job.output 'progress.json'
    $done=Join-Path $job.output 'DONE.json'
    if (Test-Path -LiteralPath $done) { $p=Get-Content -LiteralPath $done -Raw | ConvertFrom-Json }
    elseif (Test-Path -LiteralPath $progress) { $p=Get-Content -LiteralPath $progress -Raw | ConvertFrom-Json }
    else { $p=$null }
    [pscustomobject]@{Method=$job.method;Status=$(if ($p) {$p.status} else {'queued'});Requests=$(if ($p) {$p.requests} else {0});Budget=$frozen.requests;Unique=$(if ($p) {$p.unique_layouts} else {0});Updated=$(if ($p) {$p.updated_at} else {''})}
}
$rows | Format-Table -AutoSize
Write-Output (Join-Path $frozen.output 'analysis\summary.csv')
