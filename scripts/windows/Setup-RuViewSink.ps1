<#
.SYNOPSIS
    One-shot setup for a RuView sink/server host on Windows 11 (x86_64),
    tuned for a multi-node ESP32-S3 deployment.

.DESCRIPTION
    Brings up the RuView sensing server in Docker and works around the
    Docker Desktop for Windows multi-source UDP defect (issues #374 / #386)
    that silently drops CSI frames from all but one ESP32 node.

    What it does, in order:
      1. Preflight  - verifies architecture, Docker, Python, git, free ports.
      2. Repo       - clones or fast-forwards the RuView checkout.
      3. Patch      - remaps the compose UDP port 5005 -> 5006 (idempotent).
      4. Firewall   - adds an inbound allow rule for the node-facing UDP port.
      5. Relay      - starts scripts/udp-relay.py on the host.
      6. Stack      - docker compose up -d with the chosen CSI source.
      7. Verify     - waits for /health, then optionally replays synthetic
                      ADR-018 CSI frames end-to-end through the relay.

    WHY THE RELAY IS REQUIRED
    Docker Desktop runs its engine inside a WSL2 / Hyper-V VM. Inbound UDP
    arriving from several distinct source IPs is demultiplexed onto a single
    virtual socket, so the first ESP32 to transmit "wins" and every other
    node's frames are discarded at the VM boundary. The relay receives on the
    host and re-emits from one loopback socket, so every datagram reaches the
    container from an identical source and nothing is dropped.

    A single-node setup does not need this. Three nodes absolutely do.

.PARAMETER InstallDir
    Where the RuView checkout lives. Created if missing.

.PARAMETER ListenPort
    Host UDP port the ESP32 nodes transmit to. Must match the firmware's
    CONFIG_CSI_TARGET_PORT. Default 5005.

.PARAMETER ForwardPort
    Loopback UDP port the relay forwards to, which Docker maps into the
    container. Default 5006.

.PARAMETER CsiSource
    esp32     - ingest real CSI frames from hardware (normal operation)
    simulated - synthetic frames, for demoing with no hardware present
    wifi      - host Wi-Fi RSSI/scan data only (no true CSI)

    Note: 'auto' is deliberately not offered. Since issue #937 it aborts with
    exit code 78 when no source is detected instead of silently falling back.

.PARAMETER ApiToken
    Secret for RUVIEW_API_TOKEN, which enables authentication on /api/v1/*.
    Pass 'generate' to mint a random 32-byte hex token.

    Required if you ever republish the TCP ports on a LAN-facing address.
    Note that enabling auth may stop the bundled browser dashboard from
    loading data, since its fetches do not attach a bearer token.

.PARAMETER AllowUnauthenticated
    Sets RUVIEW_ALLOW_UNAUTHENTICATED=1, satisfying the issue #864 guard
    without a token. This is the default, and it is reasonable *for this
    compose file specifically* because it publishes the TCP ports as
    127.0.0.1:3000 and 127.0.0.1:3001 - host loopback only, unreachable from
    the LAN. Only the UDP CSI port is exposed to the network.

    If you widen those mappings, supply -ApiToken instead.

.PARAMETER SelfTest
    After the stack is healthy, replay synthetic CSI through the full relay
    path and confirm the server's readings actually advance. This validates
    the entire sink before any board is flashed.

.PARAMETER Down
    Tear down: stop the compose stack and the relay, then exit.

.EXAMPLE
    .\Setup-RuViewSink.ps1 -SelfTest
    Full install, then prove ingest works with no hardware attached.

.EXAMPLE
    .\Setup-RuViewSink.ps1 -CsiSource simulated -SelfTest
    Demo mode on a machine that will never see an ESP32.

.EXAMPLE
    .\Setup-RuViewSink.ps1 -Down
    Stop everything.

.NOTES
    Run from an elevated prompt if you want the firewall rule added
    automatically; otherwise the script prints the exact command to run.
#>

[CmdletBinding()]
param(
    [string]$RepoUrl = 'https://github.com/Bakar404/RuView.git',
    [string]$InstallDir = (Join-Path $env:USERPROFILE 'RuView'),
    [ValidateRange(1024, 65535)][int]$ListenPort = 5005,
    [ValidateRange(1024, 65535)][int]$ForwardPort = 5006,
    [ValidateRange(1, 65535)][int]$HttpPort = 3000,
    [ValidateSet('esp32', 'simulated', 'wifi')][string]$CsiSource = 'esp32',
    [string]$ApiToken,
    [switch]$AllowUnauthenticated,
    [string]$UdpBind = '0.0.0.0',
    [string]$UdpAllow = '172.16.0.0/12,192.168.65.0/24,10.0.0.0/8',
    [switch]$UdpInsecureLan,
    [switch]$SkipFirewall,
    [switch]$SelfTest,
    [switch]$Down
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:StepNumber = 0
$script:Warnings = @()
$RelayPidFile = Join-Path $env:TEMP 'ruview-udp-relay.pid'

function Write-Step {
    param([string]$Message)
    $script:StepNumber++
    Write-Host ''
    Write-Host ("[{0}] {1}" -f $script:StepNumber, $Message) -ForegroundColor Cyan
}

function Write-Ok    { param([string]$m) Write-Host "    OK    $m" -ForegroundColor Green }
function Write-Info  { param([string]$m) Write-Host "          $m" -ForegroundColor Gray }

function Write-Warn {
    param([string]$m)
    Write-Host "    WARN  $m" -ForegroundColor Yellow
    $script:Warnings += $m
}

function Write-Fail {
    param([string]$m, [string]$Fix)
    Write-Host "    FAIL  $m" -ForegroundColor Red
    if ($Fix) { Write-Host "          Fix: $Fix" -ForegroundColor Yellow }
    Write-Host ''
    # `exit` from a function terminates the whole script, which is what we want
    # here: a clean non-zero status with no PowerShell stack trace noise.
    exit 1
}

# git writes ordinary progress ("From https://github.com/...") to stderr. Under
# $ErrorActionPreference='Stop' Windows PowerShell 5.1 turns any native stderr
# into a terminating RemoteException, so a perfectly successful fetch aborts the
# script. Neither 2>$null nor 2>&1 suppresses that in 5.1 - the preference has
# to be relaxed inside a scope, which is what this function exists to do.
# Returns combined output; sets $script:GitExitCode for real failure checks.
# Same hazard, every other native tool. docker compose reports all of its
# progress ("Container docker-sensing-server-1 Recreate") on stderr, so a
# successful `up -d` is just as fatal under 5.1 as a successful git fetch.
# Any native command in this script must go through one of these two helpers.
function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$NativeArgs
    )
    $ErrorActionPreference = 'Continue'
    # 2>&1 wraps stderr lines in ErrorRecords, which Out-String renders as a
    # multi-line "NativeCommandError" dump. Flatten them back to plain text so
    # ordinary tool output stays readable.
    $lines = & $Exe @NativeArgs 2>&1 | ForEach-Object {
        if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { $_ }
    }
    $script:NativeExitCode = $LASTEXITCODE
    return (($lines) -join [Environment]::NewLine)
}

function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArgs)
    $ErrorActionPreference = 'Continue'
    $lines = & git @GitArgs 2>&1 | ForEach-Object {
        if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { $_ }
    }
    $script:GitExitCode = $LASTEXITCODE
    return (($lines) -join [Environment]::NewLine)
}

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-RelayProcess {
    if (-not (Test-Path $RelayPidFile)) { return $null }
    $relayPid = Get-Content $RelayPidFile -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $relayPid) { return $null }
    Get-Process -Id ([int]$relayPid) -ErrorAction SilentlyContinue
}

function Stop-Relay {
    $proc = Get-RelayProcess
    if ($proc) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Write-Ok "Stopped UDP relay (PID $($proc.Id))."
    }
    Remove-Item $RelayPidFile -ErrorAction SilentlyContinue
}

# Resolve a working Python launcher once, and reuse it everywhere.
function Resolve-Python {
    $ErrorActionPreference = 'Continue'
    foreach ($candidate in @('python', 'py')) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        try {
            $v = & $candidate -c 'import sys; print(sys.version_info[0])' 2>$null
            if ($LASTEXITCODE -eq 0 -and $v -eq '3') { return $cmd.Source }
        } catch { continue }
    }
    return $null
}

function Test-UdpPortFree {
    param([int]$Port)
    try {
        $probe = [System.Net.Sockets.UdpClient]::new($Port)
        $probe.Close()
        return $true
    } catch {
        return $false
    }
}

Write-Host ''
Write-Host '  RuView sink setup - Windows 11 / multi-node ESP32-S3' -ForegroundColor White
Write-Host '  ----------------------------------------------------' -ForegroundColor DarkGray

$ComposeFile = Join-Path $InstallDir 'docker\docker-compose.yml'

# ---------------------------------------------------------------- teardown --
if ($Down) {
    Write-Step 'Tearing down'
    Stop-Relay
    if (Test-Path $ComposeFile) {
        Push-Location $InstallDir
        try {
            Invoke-Native docker compose -f 'docker/docker-compose.yml' down | Write-Info
            Write-Ok 'Compose stack stopped.'
        } finally { Pop-Location }
    } else {
        Write-Info 'No compose file found; nothing to stop.'
    }
    Write-Host ''
    return
}

# --------------------------------------------------------------- preflight --
Write-Step 'Preflight checks'

if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
    Write-Warn 'This host is ARM64. The published image is multi-arch so it will run, but the ESP-IDF firmware image (espressif/idf) is amd64-only and will not build here.'
} else {
    Write-Ok "Architecture: $env:PROCESSOR_ARCHITECTURE"
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Fail 'Docker not found on PATH.' 'Install Docker Desktop: https://docs.docker.com/desktop/install/windows-install/'
}
# `docker info` is the only reliable liveness probe; the CLI exists even when
# the engine is stopped.
$null = Invoke-Native docker info
if ($script:NativeExitCode -ne 0) {
    Write-Fail 'Docker CLI is present but the engine is not responding.' 'Start Docker Desktop and wait for the whale icon to settle, then re-run.'
}
Write-Ok 'Docker engine is responding.'

$Python = Resolve-Python
if (-not $Python) {
    Write-Fail 'No working Python 3 interpreter found.' 'Install Python 3 and ensure it is on PATH: https://www.python.org/downloads/'
}
Write-Ok "Python 3: $Python"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Fail 'git not found on PATH.' 'Install Git for Windows: https://git-scm.com/download/win'
}
Write-Ok 'git is available.'

# The relay must own the node-facing port. A stale relay from a previous run
# is the usual culprit, so clear that before declaring a conflict.
Stop-Relay
if (-not (Test-UdpPortFree -Port $ListenPort)) {
    Write-Fail "UDP port $ListenPort is already bound by another process." "Find it with: Get-NetUDPEndpoint -LocalPort $ListenPort | Select-Object OwningProcess"
}
Write-Ok "UDP port $ListenPort is free."

# ------------------------------------------------------------------- repo ---
Write-Step 'Fetching the RuView repository'

if (Test-Path (Join-Path $InstallDir '.git')) {
    Push-Location $InstallDir
    try {
        # An existing checkout may be shallow from an earlier version of this
        # script. Deepen it, otherwise switching branches fails with "did not
        # match any file(s) known to git" and pulling fails with "refusing to
        # merge unrelated histories".
        if ((Invoke-Git rev-parse --is-shallow-repository).Trim() -eq 'true') {
            Write-Info 'Existing checkout is shallow - converting it to a full clone...'
            Invoke-Git fetch --unshallow | Write-Info
        }
        Invoke-Git remote set-branches origin '*' | Write-Info
        Invoke-Git fetch origin | Write-Info
        if ($script:GitExitCode -ne 0) {
            Write-Warn 'git fetch reported a failure; continuing with the existing checkout.'
        }
        Write-Ok "Existing checkout refreshed: $InstallDir"
        Write-Info 'Local edits preserved - not doing a hard reset.'
    } finally { Pop-Location }
} else {
    if (-not (Test-Path $InstallDir)) {
        New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    }
    # Deliberately NOT --depth 1. A shallow clone implies --single-branch, which
    # leaves the checkout unable to see other branches and unable to pull
    # without hitting unrelated-histories errors. The repo is small enough that
    # full history costs little and saves a lot of confusion later.
    Invoke-Git clone $RepoUrl $InstallDir | Write-Info
    if ($script:GitExitCode -ne 0) { Write-Fail "Clone failed from $RepoUrl" 'Check the URL and your network/proxy settings.' }
    Write-Ok "Cloned to $InstallDir"
}

if (-not (Test-Path $ComposeFile)) {
    Write-Fail "Compose file missing at $ComposeFile" 'The checkout looks incomplete. Delete the directory and re-run.'
}

$RelayScript = Join-Path $InstallDir 'scripts\udp-relay.py'
$SynthScript = Join-Path $InstallDir 'scripts\synth-csi-udp.py'
if (-not (Test-Path $RelayScript)) { Write-Fail "Missing $RelayScript" 'Upstream layout changed; check scripts/ in the repo.' }

# ------------------------------------------------------- compose UDP patch --
Write-Step 'Patching docker-compose.yml (UDP mapping + auth passthrough)'

# Compose MERGES list-valued keys such as `ports` across override files rather
# than replacing them. An override would therefore leave BOTH 5005 and 5006
# published, and Docker's 5005 bind would then collide with the relay. Editing
# the mapping in place is what docs/TROUBLESHOOTING.md section 9 prescribes.
$composeText = Get-Content $ComposeFile -Raw
$backup = "$ComposeFile.orig"
$composeDirty = $false

$desired = "- `"$ForwardPort`:5005/udp`""
$original = '- "5005:5005/udp"'

if ($composeText -match [regex]::Escape($desired)) {
    Write-Ok "UDP mapping already patched: host $ForwardPort -> container 5005/udp."
} elseif ($composeText -match [regex]::Escape($original)) {
    if (-not (Test-Path $backup)) { Copy-Item $ComposeFile $backup }
    $composeText = $composeText -replace [regex]::Escape($original), $desired
    $composeDirty = $true
    Write-Ok "UDP mapping patched: host $ForwardPort -> container 5005/udp."
} else {
    Write-Warn "Could not find the expected '$original' mapping. The upstream compose file may have changed - verify the UDP mapping by hand."
}

# Issue #864: docker-entrypoint.sh refuses to start (exit 64) when
# RUVIEW_API_TOKEN is unset and the in-container bind is 0.0.0.0, because
# /ws/sensing would otherwise stream live vitals to anyone who can reach the
# socket. Compose only forwards variables that appear in its `environment:`
# block, so exporting them in the parent shell is not enough - they have to be
# declared here or they never reach the container.
$authKeys = @(
    '      - RUVIEW_ALLOW_UNAUTHENTICATED=${RUVIEW_ALLOW_UNAUTHENTICATED:-}'
    '      - RUVIEW_API_TOKEN=${RUVIEW_API_TOKEN:-}'
    '      - RUVIEW_BIND_ADDR=${RUVIEW_BIND_ADDR:-0.0.0.0}'
)
if ($composeText -match 'RUVIEW_ALLOW_UNAUTHENTICATED') {
    Write-Ok 'Auth environment passthrough already declared.'
} else {
    $anchor = '      - RUST_LOG=info'
    if ($composeText.Contains($anchor)) {
        if (-not (Test-Path $backup)) { Copy-Item $ComposeFile $backup }
        $composeText = $composeText.Replace($anchor, ($anchor + "`n" + ($authKeys -join "`n")))
        $composeDirty = $true
        Write-Ok 'Added RUVIEW_API_TOKEN / RUVIEW_ALLOW_UNAUTHENTICATED / RUVIEW_BIND_ADDR passthrough.'
    } else {
        Write-Warn "Could not find the '$anchor' anchor line; add the RUVIEW_* variables to the environment block by hand."
    }
}

# ADR-296 (sensor data-plane bind hardening): the UDP CSI receiver binds
# 127.0.0.1 by default, *independently* of the HTTP --bind-addr. Inside a
# container that is fatal but silent - Docker delivers forwarded UDP to eth0,
# never to loopback, so every CSI frame is discarded and the server reports
# "no data yet" while looking completely healthy.
#
# A routable UDP bind is fail-closed: it is refused unless a source policy is
# supplied. An allowlist scoped to the Docker gateway ranges is the correct
# choice here, because the only sender is the host-side relay forwarding in
# through the published port. That is strictly safer than --udp-insecure-lan.
$udpKeys = @(
    '      - RUVIEW_UDP_BIND=${RUVIEW_UDP_BIND:-0.0.0.0}'
    '      - RUVIEW_UDP_ALLOW=${RUVIEW_UDP_ALLOW:-}'
    '      - RUVIEW_UDP_INSECURE_LAN=${RUVIEW_UDP_INSECURE_LAN:-false}'
)
if ($composeText -match 'RUVIEW_UDP_BIND') {
    Write-Ok 'UDP data-plane environment passthrough already declared.'
    # Repair an earlier revision of this script, which defaulted the flag to an
    # empty string. RUVIEW_UDP_INSECURE_LAN maps to a Rust bool argument, so an
    # empty value is present-but-unparseable and the server aborts at startup
    # with "a value is required for '--udp-insecure-lan'". Only true/false are
    # accepted, so the default has to be a literal.
    $brokenBool = '      - RUVIEW_UDP_INSECURE_LAN=${RUVIEW_UDP_INSECURE_LAN:-}'
    $fixedBool  = '      - RUVIEW_UDP_INSECURE_LAN=${RUVIEW_UDP_INSECURE_LAN:-false}'
    if ($composeText.Contains($brokenBool)) {
        if (-not (Test-Path $backup)) { Copy-Item $ComposeFile $backup }
        $composeText = $composeText.Replace($brokenBool, $fixedBool)
        $composeDirty = $true
        Write-Ok 'Repaired RUVIEW_UDP_INSECURE_LAN default (empty -> false).'
    }
} else {
    $anchor = '      - RUST_LOG=info'
    if ($composeText.Contains($anchor)) {
        if (-not (Test-Path $backup)) { Copy-Item $ComposeFile $backup }
        $composeText = $composeText.Replace($anchor, ($anchor + "`n" + ($udpKeys -join "`n")))
        $composeDirty = $true
        Write-Ok 'Added RUVIEW_UDP_BIND / RUVIEW_UDP_ALLOW / RUVIEW_UDP_INSECURE_LAN passthrough.'
    } else {
        Write-Warn "Could not find the '$anchor' anchor line; add the UDP variables to the environment block by hand."
    }
}

if ($composeDirty) {
    Set-Content $ComposeFile $composeText -NoNewline -Encoding UTF8
    Write-Info "Backup of the original saved to $backup"
}

# ------------------------------------------------------------------- auth ---
Write-Step 'Resolving the security posture (issue #864)'

if ($ApiToken) {
    if ($ApiToken -eq 'generate') {
        $ApiToken = -join ((1..32) | ForEach-Object { '{0:x2}' -f (Get-Random -Maximum 256) })
        Write-Ok 'Generated a random 32-byte API token.'
        Write-Host "          RUVIEW_API_TOKEN=$ApiToken" -ForegroundColor Yellow
        Write-Info 'Save this now - it is not persisted anywhere.'
    } else {
        Write-Ok 'Using the supplied API token.'
    }
    $env:RUVIEW_API_TOKEN = $ApiToken
    $env:RUVIEW_ALLOW_UNAUTHENTICATED = ''
    Write-Info 'Auth is enforced on /api/v1/*. The bundled dashboard does not send a bearer token, so it may show no data.'
} else {
    # Default. Safe here only because this compose file publishes the TCP
    # ports on 127.0.0.1, so they are unreachable from the LAN.
    $env:RUVIEW_ALLOW_UNAUTHENTICATED = '1'
    $env:RUVIEW_API_TOKEN = ''
    if (-not $AllowUnauthenticated) {
        Write-Info 'No -ApiToken given; defaulting to RUVIEW_ALLOW_UNAUTHENTICATED=1.'
    }
    Write-Ok 'Unauthenticated mode enabled.'
    Write-Info 'REST/WebSocket are published on 127.0.0.1 only, so they are not reachable from the LAN.'
    Write-Info 'If you widen those port mappings, re-run with -ApiToken generate.'
}

# The UDP receiver must bind a routable address inside the container, and that
# bind is refused without a source policy. Default to an allowlist covering the
# Docker gateway ranges, since the host relay is the only legitimate sender.
$env:RUVIEW_UDP_BIND = $UdpBind
if ($UdpInsecureLan) {
    $env:RUVIEW_UDP_ALLOW = ''
    $env:RUVIEW_UDP_INSECURE_LAN = 'true'
    Write-Warn 'UDP data plane accepts any source (-UdpInsecureLan).'
} else {
    $env:RUVIEW_UDP_ALLOW = $UdpAllow
    $env:RUVIEW_UDP_INSECURE_LAN = 'false'
    Write-Ok "UDP receiver bind $UdpBind, sources restricted to $UdpAllow"
}

# --------------------------------------------------------------- firewall ---
Write-Step 'Configuring the Windows firewall'

$ruleName = "RuView CSI ingest (UDP $ListenPort)"
if ($SkipFirewall) {
    Write-Info 'Skipped by request (-SkipFirewall).'
} else {
    $existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Ok 'Inbound allow rule already present.'
    } elseif (Test-Admin) {
        New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow `
            -Protocol UDP -LocalPort $ListenPort -Profile Private, Domain | Out-Null
        Write-Ok "Added inbound allow rule for UDP $ListenPort (Private/Domain profiles)."
    } else {
        # Not fatal: python.exe is frequently already permitted, so the relay
        # may well receive LAN traffic without an explicit rule.
        Write-Warn 'Not elevated, so the firewall rule was not added. If the nodes appear silent, run this from an admin prompt:'
        Write-Host "          New-NetFirewallRule -DisplayName '$ruleName' -Direction Inbound -Action Allow -Protocol UDP -LocalPort $ListenPort -Profile Private,Domain" -ForegroundColor Yellow
    }
}

# ------------------------------------------------------------------ relay ---
Write-Step 'Starting the host UDP relay'

$relayLog = Join-Path $env:TEMP 'ruview-udp-relay.log'
# -u is essential: Python block-buffers stdout when it is redirected to a
# file, so without it the relay's banner and its periodic stats never reach
# the log and the process looks dead even while it is forwarding fine.
# Note also that udp-relay.py prints its stats from inside the receive loop,
# so a relay that receives nothing legitimately writes nothing.
$relayArgs = @('-u', $RelayScript, '--listen-port', $ListenPort, '--forward-port', $ForwardPort)

$relay = Start-Process -FilePath $Python -ArgumentList $relayArgs `
    -RedirectStandardOutput $relayLog -RedirectStandardError "$relayLog.err" `
    -WindowStyle Hidden -PassThru

Start-Sleep -Seconds 2
if ($relay.HasExited) {
    $err = if (Test-Path "$relayLog.err") { Get-Content "$relayLog.err" -Raw } else { '(no output)' }
    Write-Fail "The relay exited immediately: $err" "Run it manually to see why: $Python `"$RelayScript`""
}

$relay.Id | Set-Content $RelayPidFile
Write-Ok "Relay running (PID $($relay.Id)): 0.0.0.0:$ListenPort -> 127.0.0.1:$ForwardPort"
Write-Info "Log: $relayLog"

# ------------------------------------------------------------------ stack ---
Write-Step 'Bringing up the sensing server'

Push-Location $InstallDir
try {
    $env:CSI_SOURCE = $CsiSource
    Write-Info "CSI_SOURCE=$CsiSource"

    # docker/docker-compose.yml declares BOTH `build:` and `image:`. When the
    # image is absent locally, Compose builds it from source rather than
    # pulling - which compiles the whole Rust workspace (ndarray-linalg with
    # openblas-static) and takes 20+ minutes or simply fails. Pulling first
    # puts the published multi-arch image in the local cache so Compose
    # reuses it and skips the build entirely.
    $image = 'ruvnet/wifi-densepose:latest'
    $cached = ((Invoke-Native docker images -q $image) -split "`n" | Where-Object { $_.Trim() } | Select-Object -First 1)
    if ($cached) {
        Write-Ok "Image already cached locally ($image)."
    } else {
        Write-Info "Pulling $image (avoids a from-source Rust build)..."
        Invoke-Native docker pull $image | Write-Info
        if ($script:NativeExitCode -ne 0) {
            Write-Warn 'Pull failed. Compose may now attempt a lengthy from-source build.'
            Write-Info 'A toomanyrequests error means Docker Hub anonymous rate limits - run: docker login'
        } else {
            Write-Ok 'Image pulled.'
        }
    }

    Invoke-Native docker compose -f 'docker/docker-compose.yml' up -d sensing-server |
        Out-String | Write-Info

    if ($script:NativeExitCode -ne 0) {
        Stop-Relay
        Write-Fail 'docker compose up failed.' 'Inspect the output above, then retry in the foreground: docker compose -f docker/docker-compose.yml up sensing-server'
    }
    Write-Ok 'Container started.'
} finally { Pop-Location }

# ----------------------------------------------------------------- health ---
Write-Step 'Waiting for the server to become healthy'

$healthUrl = "http://127.0.0.1:$HttpPort/health"
$healthy = $false
foreach ($attempt in 1..30) {
    try {
        $resp = Invoke-WebRequest -Uri $healthUrl -TimeoutSec 3 -UseBasicParsing
        if ($resp.StatusCode -eq 200) { $healthy = $true; break }
    } catch {
        Start-Sleep -Seconds 2
    }
}

if (-not $healthy) {
    Write-Warn "No 200 from $healthUrl after ~60s."
    Write-Info 'Container logs (last 40 lines):'
    Push-Location $InstallDir
    try { Invoke-Native docker compose -f 'docker/docker-compose.yml' logs --tail 40 sensing-server | Write-Info }
    finally { Pop-Location }
    Write-Info 'Common causes:'
    Write-Info '  exit 64 - issue #864 auth guard. Re-run with -ApiToken generate, or check that the'
    Write-Info '            RUVIEW_* passthrough lines really are in the compose environment block.'
    Write-Info '  exit 78 - the CSI source probe found nothing. Re-run with -CsiSource simulated to'
    Write-Info '            confirm the stack itself is sound before blaming the hardware.'
} else {
    Write-Ok "Healthy: $healthUrl"
}

# --------------------------------------------------------------- self test --
if ($SelfTest) {
    Write-Step 'End-to-end self-test with synthetic CSI'

    if (-not (Test-Path $SynthScript)) {
        Write-Warn "synth-csi-udp.py not found at $SynthScript - skipping."
    } else {
        # Deliberately aimed at the relay's listen port, not the container, so
        # this exercises the identical path the ESP32 nodes will use.
        Write-Info "Replaying ADR-018 frames (magic 0xC5110001) to 127.0.0.1:$ListenPort at 20 Hz for 20s..."

        $synthArgs = @('-u', $SynthScript, '--host', '127.0.0.1', '--port', $ListenPort,
                       '--rate-hz', '20', '--duration-s', '20', '--motion-after-s', '8')
        $synth = Start-Process -FilePath $Python -ArgumentList $synthArgs -WindowStyle Hidden -PassThru

        Start-Sleep -Seconds 12
        try {
            $latest = Invoke-WebRequest -Uri "http://127.0.0.1:$HttpPort/api/v1/sensing/latest" `
                -TimeoutSec 5 -UseBasicParsing
            Write-Ok 'Server responded to /api/v1/sensing/latest while frames were in flight:'
            Write-Info ($latest.Content.Substring(0, [Math]::Min(400, $latest.Content.Length)))
        } catch {
            Write-Warn "Could not read /api/v1/sensing/latest: $($_.Exception.Message)"
        }

        $synth.WaitForExit(20000) | Out-Null
        if (-not $synth.HasExited) { Stop-Process -Id $synth.Id -Force -ErrorAction SilentlyContinue }

        # The relay's own counters are the ground truth for whether datagrams
        # actually traversed the host boundary.
        if (Test-Path $relayLog) {
            $logText = Get-Content $relayLog -Raw -ErrorAction SilentlyContinue
            $forwarded = Select-String -Path $relayLog -Pattern 'forwarded (\d+) pkts' -ErrorAction SilentlyContinue
            if ($forwarded) {
                Write-Ok 'Relay confirmed packet flow:'
                $forwarded | Select-Object -Last 3 | ForEach-Object { Write-Info $_.Line.Trim() }
            } elseif ($logText -and $logText -match 'listening on') {
                # Banner present but no stats: the relay is healthy and simply
                # never received anything. udp-relay.py emits stats only from
                # inside its receive loop, so silence here means zero packets.
                Write-Warn 'Relay is alive but forwarded nothing - the synthetic frames never reached it.'
                Write-Info 'Check that no firewall rule is blocking loopback UDP, and that nothing else holds the port.'
            } else {
                Write-Warn 'Relay log is empty - the process may have died on startup.'
                if (Test-Path "$relayLog.err") {
                    Write-Info (Get-Content "$relayLog.err" -Raw)
                }
            }
        }
    }
}

# ---------------------------------------------------------------- summary ---
Write-Step 'Summary'

$lanIps = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' -and $_.PrefixOrigin -ne 'WellKnown' } |
    Select-Object -ExpandProperty IPAddress -Unique

Write-Host ''
Write-Host '  Dashboard      : ' -NoNewline; Write-Host "http://localhost:$HttpPort" -ForegroundColor Green
Write-Host '  CSI ingest     : ' -NoNewline; Write-Host "UDP $ListenPort (host) -> $ForwardPort (relay) -> 5005 (container)" -ForegroundColor Green
Write-Host '  Data source    : ' -NoNewline; Write-Host $CsiSource -ForegroundColor Green
Write-Host ''

if ($lanIps) {
    Write-Host '  Provision each ESP32-S3 node against this host:' -ForegroundColor White
    foreach ($ip in $lanIps) {
        Write-Host "    python firmware/esp32-csi-node/provision.py --port COM<n> --ssid `"<SSID>`" --password `"<PASS>`" --target-ip $ip" -ForegroundColor Gray
    }
    Write-Host ''
    Write-Host '  Assign this machine a STATIC IP (or a DHCP reservation) before' -ForegroundColor Yellow
    Write-Host '  provisioning - the nodes hard-code the target address in NVS.' -ForegroundColor Yellow
} else {
    Write-Warn 'No LAN IPv4 address detected; the nodes will have nothing to target.'
}

if ($script:Warnings.Count -gt 0) {
    Write-Host ''
    Write-Host "  $($script:Warnings.Count) warning(s):" -ForegroundColor Yellow
    $script:Warnings | ForEach-Object { Write-Host "    - $_" -ForegroundColor Yellow }
}

Write-Host ''
Write-Host '  Stop everything with: .\Setup-RuViewSink.ps1 -Down' -ForegroundColor DarkGray
Write-Host ''
