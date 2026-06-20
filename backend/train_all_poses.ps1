# PowerShell script to train on all yoga poses
# Run this from the classification model directory

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Yoga Pose Training Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Activate virtual environment
if (Test-Path .venv\Scripts\activate.ps1) {
    Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
    .venv\Scripts\activate.ps1
} else {
    Write-Host "Warning: Virtual environment not found. Make sure you've created one." -ForegroundColor Red
}

# Step 1: Preprocess dataset
Write-Host "`nStep 1: Preprocessing dataset..." -ForegroundColor Green
python preprocess_new_dataset.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Preprocessing failed! Check errors above." -ForegroundColor Red
    exit 1
}

# Step 2: Train model
Write-Host "`nStep 2: Training model..." -ForegroundColor Green
python training.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Training failed! Check errors above." -ForegroundColor Red
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Training completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan




