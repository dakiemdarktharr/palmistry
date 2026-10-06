[CmdletBinding()]
param(
    [string]$OutputDir = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..")).Path "dist")
)
$ErrorActionPreference = "Stop"
$sourceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stage = Join-Path $env:TEMP "PalmistryInstaller_$stamp"
$archive = Join-Path $OutputDir "PalmistryInstaller_$stamp.zip"
$excluded = @(".git", "frontend", ".venv", ".web_bundle", ".web_live_tmp", ".web_static", "artifacts", "dataset", ".codex", ".agents", "__pycache__", "dist")
New-Item -ItemType Directory -Force -Path $stage, $OutputDir | Out-Null
Get-ChildItem -LiteralPath $sourceRoot -Force | ForEach-Object {
    if ($excluded -notcontains $_.Name) {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $stage $_.Name) -Recurse -Force
    }
}
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $archive -CompressionLevel Optimal
Remove-Item -LiteralPath $stage -Recurse -Force
Write-Host "Created $archive"
