$ErrorActionPreference = "Stop"

Write-Host "Setting up Agent AI Local Testing Environment..." -ForegroundColor Cyan

# Check if .venv exists, if not create it
if (-not (Test-Path -Path ".venv")) {
    Write-Host "Creating virtual environment in .venv..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
$activateScript = ".\.venv\Scripts\Activate.ps1"
if (Test-Path -Path $activateScript) {
    . $activateScript
} else {
    Write-Host "Failed to find activation script at $activateScript" -ForegroundColor Red
    exit 1
}

# Install dependencies if requirements.txt is newer than the environment or just install quietly
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet
Write-Host "Dependencies installed successfully!" -ForegroundColor Green

# Run all tests individually
Write-Host "`nRunning Python Tests..." -ForegroundColor Cyan
$testFailed = $false
foreach ($file in Get-ChildItem -Filter "test_*.py") {
    Write-Host "`n--- Running $($file.Name) ---" -ForegroundColor Yellow
    $process = Start-Process -FilePath "python" -ArgumentList "`"$($file.FullName)`"" -NoNewWindow -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        $testFailed = $true
        Write-Host "$($file.Name) FAILED!" -ForegroundColor Red
    }
}

if ($testFailed) {
    Write-Host "`nTesting phase complete. Some tests FAILED." -ForegroundColor Red
    exit 1
} else {
    Write-Host "`nTesting phase complete. All tests PASSED." -ForegroundColor Green
}
