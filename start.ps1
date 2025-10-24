# Driver Behavior Distraction Detection - Start Script
# This script starts the Driver Behavior Detection application.

param(
    [switch]$NoVenv = $false
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Driver Behavior Detection - Start Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment if it exists and not skipped.
if (-not $NoVenv -and (Test-Path "driver_safety")) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & ".\driver_safety\Scripts\Activate.ps1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Failed to activate virtual environment." -ForegroundColor Yellow
    } else {
        Write-Host "Virtual environment activated." -ForegroundColor Green
    }
    Write-Host ""
}

# Set Qt plugin path for PySide6
Write-Host "Configuring Qt environment..." -ForegroundColor Yellow
$pythonPath = python -c "import site; print(site.getsitepackages()[0])"
$qtPluginPath = "$pythonPath\PySide6\plugins"

if (Test-Path $qtPluginPath) {
    $env:QT_PLUGIN_PATH = $qtPluginPath
    Write-Host "Qt plugin path set: $qtPluginPath" -ForegroundColor Green
} else {
    Write-Host "WARNING: Qt plugin path not found at $qtPluginPath" -ForegroundColor Yellow
    Write-Host "The application may not start correctly." -ForegroundColor Yellow
}
Write-Host ""

# Verify model files exist
Write-Host "Verifying model files..." -ForegroundColor Yellow
$modelFiles = @(
    "models\yolo\yolov8_driver_face_detector.pt",
    "models\xgboost\resnet_features_behavior_classifier.json"
)

$allModelsFound = $true
foreach ($modelFile in $modelFiles) {
    if (Test-Path $modelFile) {
        Write-Host "✓ Found: $modelFile" -ForegroundColor Green
    } else {
        Write-Host "✗ Missing: $modelFile" -ForegroundColor Red
        $allModelsFound = $false
    }
}

if (-not $allModelsFound) {
    Write-Host ""
    Write-Host "ERROR: Some required model files are missing." -ForegroundColor Red
    Write-Host "Please run setup.ps1 first or ensure model files are in the models/ directory." -ForegroundColor Red
    exit 1
}
Write-Host ""

# Start the application
Write-Host "Starting Driver Behavior Detection application..." -ForegroundColor Yellow
Write-Host ""

try {
    python main.py
} catch {
    Write-Host "ERROR: Failed to start the application." -ForegroundColor Red
    Write-Host "Error details: $_" -ForegroundColor Red
    exit 1
}

