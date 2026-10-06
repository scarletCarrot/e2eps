# Build the submission ZIP: source + Git history + executable.
# Usage (repo root, clean working tree):  .\make_submission.ps1
# Output: ..\e2eps-submission.zip

$ErrorActionPreference = "Stop"
$name = "e2eps-submission"

if (git status --porcelain) { throw "Commit or stash your changes first (git status not clean)." }

pytest -q
.\build_exe.ps1

$staging = Join-Path $env:TEMP $name
if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
New-Item -ItemType Directory $staging | Out-Null

$exclude = @(".venv", "venv", "build", "output", ".pytest_cache", "e2eps.egg-info")
Get-ChildItem -Force | Where-Object { $exclude -notcontains $_.Name } |
    Copy-Item -Destination $staging -Recurse -Force
Get-ChildItem $staging -Recurse -Force -Directory -Filter __pycache__ | Remove-Item -Recurse -Force

$zip = Join-Path (Resolve-Path "..") "$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($staging, $zip)   # keeps hidden .git
Write-Host "Created $zip"
