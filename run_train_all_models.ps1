<#
run_train_all_models.ps1

Usage: run from project root (where train_features.py & predict_features.py live)
.\.venv\Scripts\Activate.ps1
.\run_train_all_models.ps1

This script trains multiple models and runs predictions for each model.
Outputs:
  - models saved to ans_model\
  - training logs to logs\
  - predictions to predictions_by_model\
#>

# ===== Editable config =====
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$features = "data/results_features.jsonl"
$modelsDir = "ans_model"
$logDir = "logs"
$predDir = "predictions_by_model"
$label_mode = "best_clip"
$epochs_nn = 30
$batch_size = 64
$val_split = 0.2
$seed = 42

# models to attempt
$model_list = @("xgboost","random_forest","logistic","tiny_mlp","deep_mlp","cnn","transformer")

# Ensure dirs
if (-not (Test-Path $modelsDir)) { New-Item -ItemType Directory -Path $modelsDir | Out-Null }
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
if (-not (Test-Path $predDir)) { New-Item -ItemType Directory -Path $predDir | Out-Null }

Write-Host "Using features file:" $features
if (-not (Test-Path $features)) {
    Write-Error "Features file not found: $features. Aborting."
    exit 1
}

# determine feature dim using python -c (PowerShell-friendly)
$py_cmd = "import json,sys; rec=json.loads(open(r'$features').readline()); steps=[25,35,45]; feats = len(steps)*2; feats += sum(1 for i in range(len(steps)) for j in range(i+1,len(steps))); feats += 6; print(feats)"
try {
    $dim_raw = & python -c $py_cmd 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Python call to determine feature dim failed. Output:`n$dim_raw`nFalling back to default dim=15"
        $dim = 15
    } else {
        $dim = [int]$dim_raw.Trim()
    }
} catch {
    Write-Warning "Exception while detecting feature dim. Falling back to default dim=15"
    $dim = 15
}

Write-Host "Detected feature vector dimension (approx):" $dim

foreach ($m in $model_list) {
    Write-Host "`n=========="
    Write-Host "Attempting model:" $m

    # INFO messages about cnn/transformer capabilities
    if ($m -eq "cnn") {
        $sqrt = [math]::Sqrt([int]$dim)
        if ($sqrt -ne [math]::Floor($sqrt)) {
            Write-Warning "Input dim ($dim) is not a perfect square. The CNN model will pad features to the next square automatically (model handles this)."
        }
    }

    if ($m -eq "transformer") {
        if (([int]$dim % 4) -ne 0) {
            Write-Warning "Input dim ($dim) not divisible by 4. The transformer implementation pads internally so training will continue."
        }
    }

    # prepare output paths
    if ($m -in @("xgboost","random_forest","logistic")) {
        $out_model = Join-Path $modelsDir ("ans_steps_features_$m.joblib")
    } else {
        $out_model = Join-Path $modelsDir ("ans_steps_features_$m.pth")
    }
    $log_file = Join-Path $logDir ("train_$m.log")

    # build command with safe args array
    $cmd = @(
        "python", "train_features.py",
        "--features", $features,
        "--out", $out_model,
        "--model_type", $m,
        "--label_mode", $label_mode,
        "--val_split", $val_split,
        "--seed", $seed
    )

    if ($m -in @("tiny_mlp","deep_mlp","cnn","transformer")) {
        $cmd += @("--epochs", $epochs_nn, "--batch_size", $batch_size)
    }

    # run training and capture output (call exe + args safely)
    Write-Host "Training command:" ($cmd -join " ")
    Write-Host "Logging to $log_file"
    & $cmd[0] @($cmd[1..($cmd.Length-1)]) > $log_file 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Training for $m FAILED. See $log_file"
        # continue to next model instead of aborting
        continue
    } else {
        Write-Host "Training finished for $m. Saved model: $out_model"
    }

    # run prediction using predict_features.py
    $pred_file = Join-Path $predDir ("predictions_$m.jsonl")
    $pred_log = Join-Path $logDir ("predict_$m.log")
    $pred_cmd = @("python", "predict_features.py", "--features", $features, "--model", $out_model, "--out", $pred_file)
    Write-Host "Running predict:" ($pred_cmd -join " ")
    & $pred_cmd[0] @($pred_cmd[1..($pred_cmd.Length-1)]) > $pred_log 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Prediction for $m FAILED. See $pred_log"
        continue
    } else {
        Write-Host "Predictions saved to $pred_file"
    }
}

Write-Host "`nAll done. Check $logDir and $predDir for logs and prediction outputs."
