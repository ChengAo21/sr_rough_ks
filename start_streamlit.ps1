param(
    [string]$PythonPath = "",
    [ValidateRange(1024,65535)][int]$Port = 8507,
    [switch]$PySR,
    [switch]$SkipInstall
)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskBundled = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$taskLocalPackages = Join-Path $taskRoot '.runtime\python'
$taskSharedPackages = [IO.Path]::GetFullPath((Join-Path $taskRoot '..\RE4_code\.runtime\python'))
$taskPrepared = (Test-Path -LiteralPath (Join-Path $taskSharedPackages 'streamlit\__init__.py')) -and (Test-Path -LiteralPath $taskBundled)
$taskPreviousPath = $env:PYTHONPATH
$taskPreviousEncoding = $env:PYTHONIOENCODING
Push-Location -LiteralPath $taskRoot
try {
    $env:PYTHONIOENCODING = 'utf-8'
    if ($taskPrepared -and [string]::IsNullOrWhiteSpace($PythonPath)) {
        $taskPython = $taskBundled
        $env:PYTHONPATH = "$taskLocalPackages;$taskSharedPackages"
        if ($PySR -and -not (Test-Path -LiteralPath (Join-Path $taskLocalPackages 'pysr\__init__.py'))) {
            & $taskPython -m pip install --target $taskLocalPackages -r requirements-pysr.txt --upgrade
            if ($LASTEXITCODE -ne 0) { throw 'PySR dependency installation failed.' }
        }
    } else {
        $taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
        if (-not (Test-Path -LiteralPath $taskPython)) {
            if ([string]::IsNullOrWhiteSpace($PythonPath)) {
                if (Get-Command python -ErrorAction SilentlyContinue) { $PythonPath = (Get-Command python).Source }
                elseif (Get-Command py -ErrorAction SilentlyContinue) { $PythonPath = (& py -3 -c 'import sys; print(sys.executable)').Trim() }
                else { throw 'Install Python 3.11+, or supply -PythonPath C:\path\to\python.exe.' }
            }
            & $PythonPath -m venv (Join-Path $taskRoot '.venv')
            if ($LASTEXITCODE -ne 0) { throw 'Failed to create the virtual environment.' }
        }
        if (-not $SkipInstall) {
            $taskRequirements = if ($PySR) { 'requirements-pysr.txt' } else { 'requirements.txt' }
            & $taskPython -m pip install -r $taskRequirements
            if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
        }
    }
    # Workspace-local Julia state. Reuse an already installed Julia if present.
    $env:JULIA_DEPOT_PATH = (Join-Path $taskRoot '.runtime\julia-depot') + ';' + (Join-Path $env:USERPROFILE '.julia')
    $env:PYTHON_JULIAPKG_PROJECT = Join-Path $taskRoot '.runtime\julia-project'
    $taskKnownJulia = Join-Path $env:USERPROFILE '.julia\juliaup\julia-1.10.5+0.x64.w64.mingw32\bin\julia.exe'
    if ((Test-Path -LiteralPath $taskKnownJulia) -and -not $env:PYTHON_JULIAPKG_EXE) { $env:PYTHON_JULIAPKG_EXE = $taskKnownJulia }
    $env:PYTHON_JULIACALL_THREADS = '2'
    $env:JULIA_NUM_THREADS = '2'
    $env:JULIA_NUM_PRECOMPILE_TASKS = '2'
    $env:SSH_DIR = Join-Path $taskRoot '.runtime\ssh'
    New-Item -ItemType Directory -Force $env:SSH_DIR | Out-Null
    Write-Host "Roughness Lab -> http://127.0.0.1:$Port" -ForegroundColor Cyan
    & $taskPython -m streamlit run (Join-Path $taskRoot 'streamlit_app.py') --server.address 127.0.0.1 --server.port $Port --server.headless true --browser.gatherUsageStats false
} finally {
    $env:PYTHONPATH = $taskPreviousPath
    $env:PYTHONIOENCODING = $taskPreviousEncoding
    Pop-Location
}
