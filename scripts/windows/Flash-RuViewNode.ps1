<#
.SYNOPSIS
    Flash and provision one ESP32-S3 CSI node for a RuView mesh.

.DESCRIPTION
    Runs a single board end to end: verifies tooling, finds the serial port,
    writes the v0.6.7 prebuilt images at their partition offsets, provisions
    WiFi credentials and the sink address into NVS, then watches the host UDP
    relay to confirm the node is actually streaming.

    Run it once per board, changing only -NodeId.

    NODE IDs AND TDM SLOTS
    Each node needs a distinct identity and its own transmit slot, otherwise
    the nodes talk over each other. The TDM slot is derived automatically as
    NodeId - 1, so -NodeId 1/2/3 yields slots 0/1/2 across a 3-node mesh.

    WHY -Reset MATTERS ON A FIRST RUN
    provision.py is additive: it merges CLI flags over per-port state cached
    on this machine, keyed by COM port. Flashing several boards through the
    same port can therefore leak the previous board's settings into the next
    one. -Reset wipes both the cached state and the device's NVS, which is
    what you want the first time you touch any given board.

.PARAMETER NodeId
    Node identity, 1-6. Also determines the TDM slot (NodeId - 1).

.PARAMETER Ssid
    WiFi SSID the node joins. Must be a 2.4 GHz network - the ESP32-S3 has no
    5 GHz radio, and this is the single most common provisioning failure.

.PARAMETER Password
    WiFi password. Prompted for securely when omitted.

.PARAMETER TargetIp
    Sink address the node streams CSI to. Auto-detected from this host's LAN
    address when omitted. Pin it with a static lease first: the value is burned
    into the node's NVS and it will not follow a DHCP change.

.PARAMETER Port
    Serial port, e.g. COM7. Auto-detected when exactly one candidate exists.

.PARAMETER EraseFirst
    Full chip erase before writing. Use on a recycled board, or when a flashed
    board boot-loops.

.PARAMETER SkipFlash
    Provision only, leaving the existing firmware in place.

.PARAMETER SkipProvision
    Flash only, without writing WiFi credentials.

.EXAMPLE
    .\Flash-RuViewNode.ps1 -NodeId 1 -Ssid "MyWiFi"
    First board: flash, provision, verify.

.EXAMPLE
    .\Flash-RuViewNode.ps1 -NodeId 2 -Ssid "MyWiFi" -Port COM8
    Second board, explicit port.

.NOTES
    Requires the sink to be running (Setup-RuViewSink.ps1) for verification,
    since verification reads the relay's log.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateRange(1, 6)][int]$NodeId,
    [string]$Ssid,
    [securestring]$Password,
    [string]$TargetIp,
    [string]$Port,
    [ValidateRange(1, 6)][int]$TdmTotal = 3,
    [int]$Baud = 460800,
    [string]$InstallDir = (Join-Path $env:USERPROFILE 'RuView'),
    [int]$ListenPort = 5005,
    [switch]$EraseFirst,
    [switch]$SkipFlash,
    [switch]$SkipProvision,
    [switch]$NoReset
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:StepNumber = 0
function Write-Step {
    param([string]$m)
    $script:StepNumber++
    Write-Host ''
    Write-Host ("[{0}] {1}" -f $script:StepNumber, $m) -ForegroundColor Cyan
}
function Write-Ok   { param([string]$m) Write-Host "    OK    $m" -ForegroundColor Green }
function Write-Info { param([string]$m) Write-Host "          $m" -ForegroundColor Gray }
function Write-Warn { param([string]$m) Write-Host "    WARN  $m" -ForegroundColor Yellow }
function Write-Fail {
    param([string]$m, [string]$Fix)
    Write-Host "    FAIL  $m" -ForegroundColor Red
    if ($Fix) { Write-Host "          Fix: $Fix" -ForegroundColor Yellow }
    Write-Host ''
    exit 1
}

Write-Host ''
Write-Host "  RuView node flasher - node $NodeId of $TdmTotal" -ForegroundColor White
Write-Host '  -------------------------------------------' -ForegroundColor DarkGray

# Native stderr becomes a terminating NativeCommandError under
# $ErrorActionPreference='Stop', and neither 2>$null nor 2>&1 suppresses it in
# Windows PowerShell 5.1. The preference itself has to be relaxed. Assigning
# it inside a function creates a function-scoped copy, so the caller's
# strictness is untouched.
function Invoke-Quiet {
    param([string]$Exe, [string[]]$Arguments)
    $ErrorActionPreference = 'SilentlyContinue'
    & $Exe @Arguments 2>&1 | Out-Null
    return ($LASTEXITCODE -eq 0)
}

# Same guard, but keeps the combined output for logging.
function Invoke-Native {
    param([string]$Exe, [string[]]$Arguments)
    $ErrorActionPreference = 'SilentlyContinue'
    $out = & $Exe @Arguments 2>&1 | Out-String
    return [pscustomobject]@{ ExitCode = $LASTEXITCODE; Output = $out }
}

# --------------------------------------------------------------- preflight --
Write-Step 'Preflight'

# NodeId maps to TDM slot NodeId-1, so a node numbered above the slot count
# would be given a slot that no other node ever yields to. It would transmit
# into a gap that does not exist and its frames would collide.
if ($NodeId -gt $TdmTotal) {
    Write-Fail "Node $NodeId does not fit a $TdmTotal-slot schedule." "Either number this node 1..$TdmTotal, or raise -TdmTotal to the real number of boards in the mesh. Every node in the mesh must use the same -TdmTotal."
}

$Python = $null
foreach ($c in @('python', 'py')) {
    $cmd = Get-Command $c -ErrorAction SilentlyContinue
    if (-not $cmd) { continue }
    if (Invoke-Quiet $cmd.Source @('-c', 'import sys; sys.exit(0 if sys.version_info[0]==3 else 1)')) {
        $Python = $cmd.Source
        break
    }
}
if (-not $Python) { Write-Fail 'No Python 3 found.' 'Install Python 3 and put it on PATH.' }
Write-Ok "Python: $Python"

$binDir = Join-Path $InstallDir 'firmware\esp32-csi-node\release_bins'
$provision = Join-Path $InstallDir 'firmware\esp32-csi-node\provision.py'
if (-not (Test-Path $binDir))     { Write-Fail "Prebuilt binaries not found at $binDir" 'Check -InstallDir points at your RuView checkout.' }
if (-not (Test-Path $provision))  { Write-Fail "provision.py not found at $provision" 'Check -InstallDir points at your RuView checkout.' }

# 8 MB layout images. These suit N16R8 (16 MB) boards too - the partition
# table simply does not map the upper flash, and --flash_size detect keeps the
# bootloader header honest about the real chip size.
$images = [ordered]@{
    '0x0'     = 'bootloader.bin'
    '0x8000'  = 'partition-table.bin'
    '0xf000'  = 'ota_data_initial.bin'
    '0x20000' = 'esp32-csi-node.bin'
}
foreach ($f in $images.Values) {
    $p = Join-Path $binDir $f
    if (-not (Test-Path $p)) { Write-Fail "Missing image: $p" 'Re-pull the repository; release_bins is incomplete.' }
}
Write-Ok "Firmware images present ($($images.Count) files)."

# provision.py needs BOTH of these; nvs-partition-gen is easy to overlook and
# fails late, after the flash has already succeeded.
# provision.py accepts either NVS module name and its own error text points at
# esp-idf-nvs-partition-gen. The "nvs-partition-gen" name in that file's
# docstring is stale - no such package exists on PyPI.
$nvsModules = @('esp_idf_nvs_partition_gen', 'nvs_partition_gen')

$haveEsptool = Invoke-Quiet $Python @('-c', 'import esptool')
$haveNvs = $false
foreach ($mod in $nvsModules) {
    if (Invoke-Quiet $Python @('-c', "import $mod")) { $haveNvs = $true; break }
}

if (-not $haveEsptool -or -not $haveNvs) {
    Write-Info 'Installing esptool and the NVS partition generator...'
    $r = Invoke-Native $Python @('-m', 'pip', 'install', 'esptool>=5.0', 'esp-idf-nvs-partition-gen')
    if ($r.ExitCode -ne 0) {
        Write-Info $r.Output
        Write-Fail 'pip install failed.' "Run by hand: $Python -m pip install `"esptool>=5.0`" esp-idf-nvs-partition-gen"
    }

    $haveEsptool = Invoke-Quiet $Python @('-c', 'import esptool')
    $haveNvs = $false
    foreach ($mod in $nvsModules) {
        if (Invoke-Quiet $Python @('-c', "import $mod")) { $haveNvs = $true; break }
    }
    if (-not $haveEsptool) {
        Write-Fail 'esptool still not importable after install.' "Run by hand: $Python -m pip install `"esptool>=5.0`""
    }
    if (-not $haveNvs) {
        Write-Fail 'NVS partition generator still not importable after install.' "Run by hand: $Python -m pip install esp-idf-nvs-partition-gen"
    }
    Write-Ok 'Python packages installed.'
} else {
    Write-Ok 'esptool and NVS partition generator available.'
}

# ------------------------------------------------------------------- port ---
Write-Step 'Locating the serial port'

if ($Port) {
    Write-Ok "Using the supplied port: $Port"
} else {
    $ports = [System.IO.Ports.SerialPort]::GetPortNames() | Sort-Object
    if (-not $ports -or $ports.Count -eq 0) {
        Write-Fail 'No serial ports found.' 'Connect the board over USB. If nothing appears, the board may need a CP210x or CH340 driver, or you may be using a charge-only USB cable - a surprisingly common cause.'
    }

    # Desktops commonly expose a motherboard serial port, so "more than one
    # port" is normal and does not mean the board is ambiguous. Identify the
    # ESP32 by its USB bridge instead of making the user guess a number.
    $espPattern = 'CP210|CH34|CH91|FT232|Silicon Labs|USB.?SERIAL|JTAG/serial|USB Serial Device'
    $named = @()
    Get-CimInstance Win32_PnPEntity -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '\(COM\d+\)' } |
        ForEach-Object {
            $null = $_.Name -match '\((COM\d+)\)'
            $named += [pscustomobject]@{ Port = $Matches[1]; Name = $_.Name }
        }

    $likely = @($named | Where-Object { $_.Name -match $espPattern -and $ports -contains $_.Port })

    if ($ports.Count -eq 1) {
        $Port = $ports[0]
        Write-Ok "Auto-detected: $Port"
    } elseif ($likely.Count -eq 1) {
        $Port = $likely[0].Port
        Write-Ok "Auto-detected: $Port"
        Write-Info "Matched: $($likely[0].Name)"
        $others = @($ports | Where-Object { $_ -ne $Port })
        if ($others) { Write-Info "Ignored non-ESP32 ports: $($others -join ', ')" }
    } else {
        Write-Info 'Several ports present:'
        if ($named) {
            $named | ForEach-Object { Write-Info "  $($_.Name)" }
        } else {
            $ports | ForEach-Object { Write-Info "  $_" }
        }
        if ($likely.Count -gt 1) {
            Write-Info ''
            Write-Info "More than one looks like an ESP32: $(($likely | ForEach-Object { $_.Port }) -join ', ')"
            Write-Info 'If only one board is connected, this board exposes both a USB-serial bridge and'
            Write-Info 'native USB. Prefer the CP210x/CH340 bridge port for flashing.'
        }
        Write-Fail "Ambiguous port selection: $($ports -join ', ')" 'Re-run with -Port COMn to pick one. The ESP32 is the CP210x, CH340, or USB JTAG/serial entry above - not a plain "Communications Port".'
    }
}

# --------------------------------------------------------------- sink addr --
if (-not $SkipProvision) {
    Write-Step 'Resolving the sink address'

    if (-not $TargetIp) {
        # The ThinkCentre runs Docker, so it carries extra virtual adapters
        # (WSL, Hyper-V, bridges). Enumerating every IPv4 address would be
        # ambiguous at best and would silently pick a virtual subnet the ESP32s
        # cannot reach at worst. The interface owning the default route is the
        # one that actually faces the router, so resolve that first.
        $TargetIp = $null
        try {
            $route = Get-NetRoute -DestinationPrefix '0.0.0.0/0' -ErrorAction Stop |
                Sort-Object { $_.RouteMetric + (Get-NetIPInterface -InterfaceIndex $_.InterfaceIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue).InterfaceMetric } |
                Select-Object -First 1
            if ($route) {
                $TargetIp = Get-NetIPAddress -AddressFamily IPv4 -InterfaceIndex $route.InterfaceIndex -ErrorAction Stop |
                    Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' } |
                    Select-Object -ExpandProperty IPAddress -First 1
            }
        } catch { $TargetIp = $null }

        if ($TargetIp) {
            Write-Ok "Auto-detected sink address: $TargetIp (default-route interface)"
        } else {
            $cands = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
                Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' -and $_.PrefixOrigin -ne 'WellKnown' } |
                Select-Object -ExpandProperty IPAddress -Unique
            if (-not $cands) { Write-Fail 'Could not determine a LAN address for this host.' 'Pass -TargetIp explicitly.' }
            if (@($cands).Count -gt 1) {
                Write-Info "Multiple local addresses: $($cands -join ', ')"
                Write-Fail 'Ambiguous sink address.' 'Pass -TargetIp explicitly so the node targets the right interface.'
            }
            $TargetIp = @($cands)[0]
            Write-Ok "Auto-detected sink address: $TargetIp"
        }
    } else {
        Write-Ok "Sink address: $TargetIp"
    }
    Write-Info 'This value is written into NVS. Give the host a static IP or DHCP reservation, or ingest breaks on the next lease change.'

    if (-not $Ssid) { $Ssid = Read-Host 'WiFi SSID' }
    if (-not $Password) { $Password = Read-Host "Password for '$Ssid'" -AsSecureString }
    $plainPw = [Runtime.InteropServices.Marshal]::PtrToStringBSTR(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Password))
    if (-not $plainPw) { Write-Fail 'Empty WiFi password.' 'Open networks are not supported by the provisioning flow.' }
    Write-Ok "SSID: $Ssid"
    Write-Info 'The ESP32-S3 is 2.4 GHz only. A 5 GHz SSID will provision fine and then never associate.'
}

# ------------------------------------------------------------------ flash ---
if (-not $SkipFlash) {
    Write-Step "Flashing $Port"

    if ($EraseFirst) {
        Write-Info 'Erasing the chip first...'
        $r = Invoke-Native $Python @('-m', 'esptool', '--chip', 'esp32s3', '--port', $Port, '--baud', $Baud, 'erase_flash')
        Write-Info $r.Output
        if ($r.ExitCode -ne 0) { Write-Fail 'erase_flash failed.' 'Hold BOOT, tap RESET, release BOOT to force download mode, then retry.' }
        Write-Ok 'Chip erased.'
    }

    $flashArgs = @('-m', 'esptool', '--chip', 'esp32s3', '--port', $Port, '--baud', $Baud,
                   'write_flash', '--flash_mode', 'dio', '--flash_size', 'detect')
    foreach ($offset in $images.Keys) {
        $flashArgs += $offset
        $flashArgs += (Join-Path $binDir $images[$offset])
    }

    Write-Info "Offsets: $(($images.Keys | ForEach-Object { "$_=$($images[$_])" }) -join '  ')"
    $r = Invoke-Native $Python $flashArgs
    Write-Info $r.Output

    if ($r.ExitCode -ne 0) {
        Write-Fail 'Flashing failed.' 'Most often the board is not in download mode: hold BOOT, tap RESET, release BOOT, then re-run. A lower -Baud 115200 also helps on marginal cables.'
    }
    Write-Ok 'Firmware written.'
    Start-Sleep -Seconds 3
} else {
    Write-Step 'Flashing skipped (-SkipFlash)'
}

# -------------------------------------------------------------- provision ---
if (-not $SkipProvision) {
    $slot = $NodeId - 1
    Write-Step "Provisioning node $NodeId (TDM slot $slot of $TdmTotal)"

    $provArgs = @($provision, '--port', $Port, '--chip', 'esp32s3',
                  '--ssid', $Ssid, '--password', $plainPw,
                  '--target-ip', $TargetIp, '--target-port', $ListenPort,
                  '--node-id', $NodeId, '--tdm-slot', $slot, '--tdm-total', $TdmTotal)

    # Additive merge is keyed by COM port on this machine, so without a reset
    # the previous board's identity can bleed into this one.
    if (-not $NoReset) {
        $provArgs += '--reset'
        Write-Info 'Using --reset to clear cached per-port state and device NVS.'
    }

    $r = Invoke-Native $Python $provArgs
    Write-Info $r.Output
    if ($r.ExitCode -ne 0) {
        Write-Fail 'Provisioning failed.' 'Close any serial monitor holding the port, then retry. Add -EraseFirst if NVS looks corrupt.'
    }
    Write-Ok "Node $NodeId provisioned -> ${TargetIp}:${ListenPort}"
} else {
    Write-Step 'Provisioning skipped (-SkipProvision)'
}

# ------------------------------------------------------------------ verify --
Write-Step 'Verifying the node is streaming'

$relayLog = Join-Path $env:TEMP 'ruview-udp-relay.log'
if (-not (Test-Path $relayLog)) {
    Write-Warn 'No relay log found - the sink does not appear to be running.'
    Write-Info 'Start it with: .\scripts\windows\Setup-RuViewSink.ps1 -CsiSource esp32'
} else {
    $before = (Get-Item $relayLog).Length
    Write-Info 'Waiting up to 45s for frames (the node must boot, join WiFi, then start capture)...'

    $seen = $false
    foreach ($i in 1..15) {
        Start-Sleep -Seconds 3
        $after = (Get-Item $relayLog).Length
        if ($after -gt $before) {
            $tail = Get-Content $relayLog -Tail 6 | Where-Object { $_ -match 'forwarded' }
            if ($tail) {
                Write-Ok 'Frames are arriving at the relay:'
                $tail | ForEach-Object { Write-Info $_.Trim() }
                $seen = $true
                break
            }
        }
    }

    if (-not $seen) {
        Write-Warn 'No new frames observed within 45s.'
        Write-Info 'Work through these in order:'
        Write-Info '  1. Wrong band      - the SSID must be 2.4 GHz.'
        Write-Info '  2. Wrong sink IP   - confirm the host address has not changed since provisioning.'
        Write-Info '  3. Firewall        - inbound UDP on this host must be allowed.'
        Write-Info '  4. Zero CSI yield  - if the node joins but sends nothing, this is the display-less'
        Write-Info '                       probe defect (RuView#893). The prebuilts are only verified on'
        Write-Info '                       C6, so an S3 DevKitC clone may need a devkitc-overlay rebuild.'
        Write-Info ''
        Write-Info "  Watch the node's own console to tell these apart:"
        Write-Info "    $Python -m serial.tools.miniterm $Port 115200"
    }
}

# ---------------------------------------------------------------- summary ---
Write-Step 'Summary'
Write-Host ''
Write-Host "  Node       : $NodeId (TDM slot $($NodeId - 1) of $TdmTotal)" -ForegroundColor Green
Write-Host "  Port       : $Port" -ForegroundColor Green
if (-not $SkipProvision) { Write-Host "  Streaming  : ${TargetIp}:${ListenPort}" -ForegroundColor Green }
Write-Host ''
if ($NodeId -lt $TdmTotal) {
    $nextSsid = if ($Ssid) { " -Ssid `"$Ssid`"" } else { '' }
    Write-Host '  Next board - connect it, then run:' -ForegroundColor White
    Write-Host "    .\Flash-RuViewNode.ps1 -NodeId $($NodeId + 1)$nextSsid" -ForegroundColor Gray
} else {
    Write-Host '  All nodes flashed. Place them 2-4 m apart with the subject between them,' -ForegroundColor White
    Write-Host '  then open http://localhost:3000 to watch live sensing.' -ForegroundColor White
}
Write-Host ''
