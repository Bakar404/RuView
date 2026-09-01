# Re-align all captures with the node-axis fix (align-ground-truth.js).
# --window-frames now counts TICKS; 7 ticks ~= the old 20 interleaved frames.
$ErrorActionPreference = 'Continue'
$map = @(
  @{ gt = 'keypoints_20260830_212120'; csi = 'rec_1788150082' },
  @{ gt = 'keypoints_20260830_212216'; csi = 'rec_1788150138' },
  @{ gt = 'keypoints_20260830_212335'; csi = 'rec_1788150138' },
  @{ gt = 'keypoints_20260830_212928'; csi = 'rec_1788150570' },
  @{ gt = 'keypoints_20260830_212950'; csi = 'rec_1788150592' },
  @{ gt = 'keypoints_20260830_213206'; csi = 'rec_1788150728' }
)
New-Item -ItemType Directory -Force -Path data\paired-v2 | Out-Null
foreach ($m in $map) {
  Write-Host "`n=== $($m.gt)  <-  $($m.csi) ===" -ForegroundColor Cyan
  node scripts\align-ground-truth.js `
    --gt "data\ground-truth\$($m.gt).vis.jsonl" `
    --csi "data\recordings\$($m.csi).jsonl" `
    --window-frames 7 `
    --output "data\paired-v2\$($m.gt).paired.jsonl"
}
