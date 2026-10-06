# Build the standalone E2EPS executable (Windows).
# Usage:  .\build_exe.ps1
# Output: dist\e2eps\e2eps.exe  (+ scenarios\ copied next to it)

$ErrorActionPreference = "Stop"
pip install -e ".[dev]"
pyinstaller --noconfirm --clean --name e2eps --onedir `
    --collect-submodules e2eps `
    --exclude-module tkinter `
    run_e2eps.py
Copy-Item -Recurse -Force scenarios dist\e2eps\scenarios
& dist\e2eps\e2eps.exe --version
Write-Host "Built dist\e2eps\e2eps.exe"
