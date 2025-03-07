# Rename notebook to notebooks
if (Test-Path "notebook") {
    Rename-Item -Path "notebook" -NewName "notebooks"
}

# Move logs to airflow/logs
if (-not (Test-Path "airflow\logs")) {
    New-Item -ItemType Directory -Force -Path "airflow\logs"
}
if (Test-Path "logs\*") {
    Move-Item -Force "logs\*" "airflow\logs\"
}
Remove-Item -Force "logs" -ErrorAction SilentlyContinue

# Move metrics to infrastructure/monitoring
if (-not (Test-Path "infrastructure\monitoring")) {
    New-Item -ItemType Directory -Force -Path "infrastructure\monitoring"
}
if (Test-Path "metrics\*") {
    Move-Item -Force "metrics\*" "infrastructure\monitoring\"
}
Remove-Item -Force "metrics" -ErrorAction SilentlyContinue

# Move changelog content to docs
if (Test-Path "changelog.txt") {
    Move-Item -Force "changelog.txt" "docs\CHANGELOG.md"
}

# Clean up temporary files
Remove-Item -Force "cleanup.ps1" -ErrorAction SilentlyContinue
Remove-Item -Force "cleanup_remaining.ps1" -ErrorAction SilentlyContinue
Remove-Item -Force "project_description.txt" -ErrorAction SilentlyContinue

# Create data subdirectories as per project memory
New-Item -ItemType Directory -Force -Path "data\raw"
New-Item -ItemType Directory -Force -Path "data\processed"

# Create cache directory for feature storage
New-Item -ItemType Directory -Force -Path "cache"

# Create src subdirectories as per project memory
$srcDirs = @(
    "data_processing",
    "training",
    "prediction",
    "deployment",
    "monitoring",
    "utils"
)
foreach ($dir in $srcDirs) {
    New-Item -ItemType Directory -Force -Path "src\$dir"
}

Write-Host "Final project structure cleanup completed!"
