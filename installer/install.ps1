[CmdletBinding()]
param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "Palmistry"),
    [switch]$WithModel,
    [switch]$SkipDependencies
)
$ErrorActionPreference = "Stop"
$sourceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$targetRoot = [IO.Path]::GetFullPath($InstallDir)
$sourcePrefix = $sourceRoot.TrimEnd("\") + "\"
$targetPrefix = $targetRoot.TrimEnd("\") + "\"
if ($targetPrefix.StartsWith($sourcePrefix, [StringComparison]::OrdinalIgnoreCase) -or
    $sourcePrefix.StartsWith($targetPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "InstallDir must not overlap the source repository (same directory, child or parent)."
}
New-Item -ItemType Directory -Force -Path $targetRoot | Out-Null
$excluded = @(".git", "frontend", ".venv", ".web_bundle", ".web_live_tmp", ".web_static", "artifacts", "dataset", ".codex", ".agents", "__pycache__")
Get-ChildItem -LiteralPath $sourceRoot -Force | ForEach-Object {
    if ($excluded -notcontains $_.Name) {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $targetRoot $_.Name) -Recurse -Force
    }
    if ($LASTEXITCODE -ne 0) { throw "Creating the Python virtual environment failed." }
}
$venvDir = Join-Path $targetRoot ".venv"
if (-not (Test-Path -LiteralPath (Join-Path $venvDir "Scripts\python.exe"))) {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $pyLauncher) {
        & $pyLauncher.Source -3 -m venv $venvDir
    } else {
        $python = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $python) {
            throw "Python 3.11+ was not found. Install Python and run the installer again."
        }
        & $python.Source -m venv $venvDir
    }
}
$venvPython = Join-Path $venvDir "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Could not create the virtual environment at $venvDir."
}
& $venvPython -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) { throw "Palmistry requires Python 3.11 or newer. Recreate the app virtual environment with a supported Python." }
if (-not $SkipDependencies) {
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed. Check internet or retry with an offline wheel cache." }
    & $venvPython -m pip install -r (Join-Path $targetRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "Dependency install failed. Retry the installer after fixing network access." }
    if ($WithModel) {
        Write-Host "-WithModel is kept for compatibility; this release uses the NumPy model and has no extra ML dependencies."
    }
}
New-Item -ItemType Directory -Force -Path (Join-Path $targetRoot "dataset\images") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $targetRoot "dataset\imports") | Out-Null
$launcherPath = Join-Path $targetRoot "run_palmistry.bat"
$launcherSource = Join-Path $sourceRoot "run_palmistry.bat"
if (-not (Test-Path -LiteralPath $launcherSource -PathType Leaf)) {
    throw "Launcher source was not found at $launcherSource."
}
Copy-Item -LiteralPath $launcherSource -Destination $launcherPath -Force
$startMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
New-Item -ItemType Directory -Force -Path $startMenu | Out-Null
$wsh = New-Object -ComObject WScript.Shell
foreach ($shortcutPath in @(
    (Join-Path $startMenu "Palmistry Live Camera.lnk"),
    (Join-Path ([Environment]::GetFolderPath("Desktop")) "Palmistry Live Camera.lnk")
)) {
    $shortcut = $wsh.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $launcherPath
    $shortcut.WorkingDirectory = $targetRoot
    $shortcut.WindowStyle = 7
    $shortcut.Description = "Palmistry - giao dien ung dung desktop"
    $shortcut.IconLocation = "$env:SystemRoot\System32\SHELL32.dll,137"
    $shortcut.Save()
}
Write-Host "Palmistry installed to $targetRoot"
Start-Process -FilePath $launcherPath -WorkingDirectory $targetRoot -WindowStyle Hidden
