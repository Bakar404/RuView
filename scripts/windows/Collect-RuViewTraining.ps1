<#
.SYNOPSIS
    Capture synchronized iPhone camera keypoints and raw ESP32 CSI.
#>

[CmdletBinding()]
param(
    [ValidateRange(10, 86400)][int]$Duration = 1800,
    [ValidateRange(0, 20)][int]$Camera = 0,
    [ValidateRange(1024, 65535)][int]$TapPort = 5007,
    [switch]$Preview
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $env:LOCALAPPDATA 'RuView\pose-env\Scripts\python.exe'
$nodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
if ($nodeCommand) {
    $node = $nodeCommand.Source
} else {
    $node = 'C:\Program Files\nodejs\node.exe'
}

if (-not (Test-Path $python)) {
    throw "Pose environment not found at $python"
}
if (-not (Test-Path $node)) {
    throw 'Node.js is not installed.'
}

$gtDir = Join-Path $repo 'data\ground-truth'
$csiDir = Join-Path $repo 'data\recordings'
$pairedDir = Join-Path $repo 'data\paired'
New-Item -ItemType Directory -Force -Path $gtDir, $csiDir, $pairedDir | Out-Null

$started = Get-Date
$recorderLog = Join-Path $env:TEMP 'ruview-training-recorder.log'
$recorderErr = "$recorderLog.err"
$stopFile = Join-Path $env:TEMP "ruview-training-stop-$PID.flag"
Remove-Item $stopFile -ErrorAction SilentlyContinue
$recorder = Start-Process -FilePath $python -ArgumentList @(
    (Join-Path $repo 'scripts\record-csi-udp.py'),
    '--port', $TapPort,
    '--duration', $Duration,
    '--output', $csiDir,
    '--stop-file', $stopFile
) -RedirectStandardOutput $recorderLog -RedirectStandardError $recorderErr -PassThru

Start-Sleep -Seconds 2
if ($recorder.HasExited) {
    $details = if (Test-Path $recorderErr) { Get-Content $recorderErr -Raw } else { '' }
    throw "Raw CSI recorder failed to start. $details"
}

$collectorArgs = @(
    (Join-Path $repo 'scripts\collect-ground-truth.py'),
    '--camera', $Camera,
    '--duration', $Duration,
    '--no-server-recording'
)
if ($Preview) {
    $collectorArgs += '--preview'
}

try {
    & $python @collectorArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Camera collector exited with code $LASTEXITCODE"
    }
} finally {
    if (-not $recorder.HasExited) {
        Set-Content $stopFile 'stop'
        $recorder.WaitForExit(5000) | Out-Null
    }
    if (-not $recorder.HasExited) {
        Stop-Process -Id $recorder.Id -Force
    }
    Remove-Item $stopFile -ErrorAction SilentlyContinue
}

$groundTruth = Get-ChildItem $gtDir -Filter 'keypoints_*.jsonl' |
    Where-Object LastWriteTime -ge $started |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
$csi = Get-ChildItem $csiDir -Filter '*.csi.jsonl' |
    Where-Object LastWriteTime -ge $started |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $groundTruth) {
    throw 'Camera collection completed without a ground-truth file.'
}
if (-not $csi -or $csi.Length -eq 0) {
    $details = if (Test-Path $recorderLog) { Get-Content $recorderLog -Raw } else { '' }
    throw "Raw CSI collection completed without data. Confirm the relay uses --tap-port $TapPort. $details"
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$paired = Join-Path $pairedDir "iphone-camo-$stamp.paired.jsonl"
& $node (Join-Path $repo 'scripts\align-ground-truth.js') `
    --gt $groundTruth.FullName --csi $csi.FullName --output $paired
if ($LASTEXITCODE -ne 0) {
    throw "Ground-truth alignment exited with code $LASTEXITCODE"
}

Write-Host ''
Write-Host "Paired dataset: $paired" -ForegroundColor Green
Write-Host "Train with:" -ForegroundColor White
Write-Host "  node scripts\train-wiflow-supervised.js --data `"$paired`" --output models\iphone-room" -ForegroundColor Gray
