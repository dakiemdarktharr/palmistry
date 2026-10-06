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

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Palmistry Python was not found: $PythonPath"
}
if (-not (Test-Path -LiteralPath $uiScript -PathType Leaf)) {
    throw "giao_dien_ui.py was not found in: $AppRoot"
}

function Test-PalmistryReady {
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
        return ($response.StatusCode -eq 200)
    } catch {
        return $false
    }
}

if (-not (Test-PalmistryReady)) {
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONUTF8 = "1"
    $null = Start-Process -FilePath $PythonPath -ArgumentList @("-u", $uiScript) -WorkingDirectory $AppRoot -WindowStyle Hidden
}

$deadline = (Get-Date).AddSeconds([Math]::Max(5, $WaitSeconds))
while ((Get-Date) -lt $deadline) {
    if (Test-PalmistryReady) { break }
    Start-Sleep -Milliseconds 500
}
if (-not (Test-PalmistryReady)) {
    throw "Palmistry did not respond at $Url within $WaitSeconds seconds. Check artifacts/keypoints-ui.stderr.log."
}

$operaCandidates = @(
    (Join-Path $env:LOCALAPPDATA "Programs\Opera\opera.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Opera GX\opera.exe"),
    (Join-Path ${env:ProgramFiles} "Opera\opera.exe"),
    (Join-Path ${env:ProgramFiles} "Opera\launcher.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Opera\opera.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Opera\launcher.exe")
) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) }
$operaCandidates = @($operaCandidates)

if ($operaCandidates.Count -gt 0) {
    Start-Process -FilePath $operaCandidates[0] -ArgumentList @($Url)
    Write-Host "Opened Palmistry in Opera: $Url"
} else {
    Start-Process -FilePath $Url
    Write-Warning "Opera was not found; opened the default browser: $Url"
}
