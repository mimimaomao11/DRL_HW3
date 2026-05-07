$scripts = @(
    "training\train_dqn.py",
    "training\train_double.py",
    "training\train_dueling.py",
    "training\train_keras.py",
    "training\train_lightning.py",
    "training\train_rainbow.py"
)

foreach ($script in $scripts) {
    Write-Host "Running $script..."
    venv\Scripts\python.exe $script
}

Write-Host "Copying plots to docs/plots..."
Copy-Item -Path "results\plots\*.png" -Destination "docs\plots\" -Force
Write-Host "Done!"
