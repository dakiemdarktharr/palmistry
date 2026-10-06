[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$AppRoot,
    [Parameter(Mandatory = $true)]
    [string]$PythonPath,
    [string]$Url = "http://127.0.0.1:8501/keypoints",
    [int]$WaitSeconds = 45
)

$ErrorActionPreference = "Stop"
$AppRoot = [IO.Path]::GetFullPath($AppRoot)
$PythonPath = [IO.Path]::GetFullPath($PythonPath)
$uiScript = Join-Path $AppRoot "giao_dien_ui.py"
$desktopScript = Join-Path $AppRoot "palmistry_desktop.py"
$logDirectory = Join-Path $AppRoot "artifacts"
$healthUrl = ([Uri]$Url).GetLeftPart([UriPartial]::Authority) + "/api/health"
$startupMutex = [Threading.Mutex]::new($false, "Local\PalmistryServer8501")
$mutexHeld = $false

try {
    foreach ($required in @($PythonPath, $uiScript, $desktopScript)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Palmistry file was not found: $required"
        }
    }
    & $PythonPath -B -c "from PySide6.QtWebEngineWidgets import QWebEngineView" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Desktop UI requires PySide6. Install requirements.txt with the app's Python and try again."
    }

    function Test-PalmistryReady {
        try {
            $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        } catch {
            return $false
        }
        if ($health.service -ne "palmistry" -or $health.project_root -ne $AppRoot) {
            throw "Port 8501 belongs to another app or Palmistry installation. Close that app before opening this installation."
        }
        return $true
    }

    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONUTF8 = "1"
    try {
        $mutexHeld = $startupMutex.WaitOne([Math]::Max(5, $WaitSeconds) * 1000)
    } catch [Threading.AbandonedMutexException] {
        $mutexHeld = $true
    }
    if (-not $mutexHeld) { throw "Another Palmistry launch is still starting. Try again shortly." }
    New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
    if (-not (Test-PalmistryReady)) {
        # An HTTP listener without our identity endpoint must not be reused.
        try { $existing = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2 } catch { $existing = $null }
        if ($existing) {
            throw "A previous Palmistry version or another app is using port 8501. Close its server and reopen Palmistry."
        }
        $backend = Start-Process -FilePath $PythonPath -ArgumentList @("-u", "`"$uiScript`"") -WorkingDirectory $AppRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDirectory "keypoints-ui.stdout.log") -RedirectStandardError (Join-Path $logDirectory "keypoints-ui.stderr.log")
    }

    $deadline = (Get-Date).AddSeconds([Math]::Max(5, $WaitSeconds))
    while ((Get-Date) -lt $deadline) {
        if (Test-PalmistryReady) { break }
        if ($backend -and $backend.HasExited) {
            throw "Palmistry server stopped. Check artifacts/keypoints-ui.stderr.log."
        }
        Start-Sleep -Milliseconds 500
    }
    if (-not (Test-PalmistryReady)) {
        throw "Palmistry did not respond within $WaitSeconds seconds. Check artifacts/keypoints-ui.stderr.log."
    }

    $windowPython = Join-Path (Split-Path -Parent $PythonPath) "pythonw.exe"
    if (-not (Test-Path -LiteralPath $windowPython -PathType Leaf)) {
        $windowPython = $PythonPath
    }
    Start-Process -FilePath $windowPython -ArgumentList @("`"$desktopScript`"") -WorkingDirectory $AppRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDirectory "desktop.$PID.stdout.log") -RedirectStandardError (Join-Path $logDirectory "desktop.$PID.stderr.log")
} catch {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, "Palmistry", "OK", "Error") | Out-Null
    exit 1
} finally {
    if ($mutexHeld) { $startupMutex.ReleaseMutex() }
    $startupMutex.Dispose()
}
