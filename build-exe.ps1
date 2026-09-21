# build-exe.ps1 - package the Cline Gateway desktop client as a single .exe
#
#   .\build-exe.ps1              build
#   .\build-exe.ps1 -Clean       wipe build/ dist/ first
#   .\build-exe.ps1 -Console     keep a console window (useful for debugging)
#
# Output: .\dist\ClineGateway.exe   (~single file, no Python needed to run)

[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$Console
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

Write-Host "Cline Gateway - exe build" -ForegroundColor Cyan
Write-Host "  workdir: $Here"

if ($Clean) {
    foreach ($d in @("build", "dist")) {
        if (Test-Path $d) { Remove-Item $d -Recurse -Force; Write-Host "  removed $d" }
    }
}

# find a python (prefer the venv; a PATH python is used read-only, never pip-installed into)
$py = Join-Path $Here ".venv\Scripts\python.exe"
$usingVenv = Test-Path $py
if (-not $usingVenv) { $py = (Get-Command python -ErrorAction Stop).Source }
Write-Host "  python : $py$(if (-not $usingVenv) { '  (no .venv — will NOT pip install)' })"

# make sure the build deps are present (install only into the project venv —
# installing into a shared/system Python mutates the user's machine)
& $py -c "import PyInstaller" 2>$null
$needInstaller = $LASTEXITCODE -ne 0
& $py -c "import yaml, httpx, fastapi, uvicorn" 2>$null
$needRuntime = $LASTEXITCODE -ne 0
if (($needInstaller -or $needRuntime) -and -not $usingVenv) {
    throw "build deps missing and no .venv found at $Here\.venv - " +
          "create it first:  python -m venv .venv; .\.venv\Scripts\python -m pip install -r requirements.txt pyinstaller"
}
if ($needInstaller) {
    Write-Host "  installing pyinstaller (venv)..." -ForegroundColor Yellow
    & $py -m pip install --upgrade pyinstaller
}
if ($needRuntime) {
    Write-Host "  installing runtime deps (venv)..." -ForegroundColor Yellow
    & $py -m pip install -r (Join-Path $Here "requirements.txt")
}

$mode = if ($Console) { "--console" } else { "--windowed" }

# icon: official Cline logo, extracted to assets/ by tools/make_icon.py
$icon = Join-Path $Here "assets\cline.ico"
$iconArg = @()
if (Test-Path $icon) { $iconArg = @("--icon", $icon) }

# exe properties (right-click -> Details) with the branding line
$versionFile = Join-Path $Here "tools\exe_version.txt"
$versionArg = @()
if (Test-Path $versionFile) { $versionArg = @("--version-file", $versionFile) }

$args = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onefile",
    $mode,
    "--name", "ClineGateway",
    # uvicorn loads these dynamically, so PyInstaller cannot see them
    "--collect-submodules", "uvicorn",
    "--hidden-import", "uvicorn.logging",
    "--hidden-import", "uvicorn.loops",
    "--hidden-import", "uvicorn.loops.auto",
    "--hidden-import", "uvicorn.protocols",
    "--hidden-import", "uvicorn.protocols.http",
    "--hidden-import", "uvicorn.protocols.http.auto",
    "--hidden-import", "uvicorn.protocols.http.h11_impl",
    "--hidden-import", "uvicorn.protocols.websockets",
    "--hidden-import", "uvicorn.protocols.websockets.auto",
    "--hidden-import", "uvicorn.lifespan",
    "--hidden-import", "uvicorn.lifespan.on",
    "--hidden-import", "uvicorn.lifespan.off",
    "--hidden-import", "anyio._backends._asyncio",
    "--hidden-import", "httpcore",
    "--hidden-import", "h11",
    "--hidden-import", "yaml",
    # pywebview + its Windows WebView2 backend
    "--collect-submodules", "webview",
    "--hidden-import", "clr",
    "--hidden-import", "pythonnet",
    # ship the example config, the app icon assets, and the dashboard page
    "--add-data", "config.example.yaml;.",
    "--add-data", "assets;assets",
    "--add-data", "cline_gateway/dashboard;cline_gateway/dashboard",
    "--paths", $Here,
    "--specpath", $Here
) + $iconArg + $versionArg + @("gui_main.py")

Write-Host "`n  building (this takes a minute)...`n" -ForegroundColor Cyan
& $py @args
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

$exe = Join-Path $Here "dist\ClineGateway.exe"
if (-not (Test-Path $exe)) { throw "expected output missing: $exe" }

$size = [math]::Round((Get-Item $exe).Length / 1MB, 1)
Write-Host "`n  built: $exe  ($size MB)" -ForegroundColor Green
Write-Host "`nRun it, or copy it next to your accounts/ folder." -ForegroundColor Cyan
Write-Host "It writes config.yaml beside itself on first launch." -ForegroundColor Cyan
