param(
    [string]$Python = 'python',
    [string]$Data = (Join-Path $PSScriptRoot 'data\dataset_split'),
    [int]$Epochs = 50,
    [int]$BatchSize = 32,
    [int[]]$Seeds = @(42,43,44),
    [string]$Device = 'cuda',
    [int]$Workers = 0
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Get-Command -Name $Python -ErrorAction SilentlyContinue)) { throw 'Python executable not found. Activate an environment or set -Python.' }
function Run-Python {
    param([string[]]$Arguments)
    & $Python -B @Arguments
    if ($LASTEXITCODE -ne 0) { throw ('Python command failed: '+($Arguments -join ' ')) }
}
$manifest = Join-Path $PSScriptRoot 'prepared\manifest.jsonl'
if (-not (Test-Path -LiteralPath $manifest)) {
    Run-Python -Arguments @('prepare_data.py','--data',$Data,'--clean-duplicates')
}
function Check-ExistingRun {
    param([string]$Run)
    $configFile = Join-Path $Run 'config.json'
    if (Test-Path -LiteralPath $configFile) {
        $config = Get-Content -LiteralPath $configFile -Raw | ConvertFrom-Json
        $hash = (Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($config.epochs -ne $Epochs -or $config.batch_size -ne $BatchSize -or $config.size -ne 64 -or -not $config.gray -or $config.pretrained -or $config.class_weighted -or $config.amp -or $config.lr -ne 0.001 -or $config.weight_decay -ne 0 -or $config.manifest_sha256 -ne $hash) {
            throw ('Existing run uses different settings: '+$Run+'. Use fresh output paths with train.py.')
        }
    }
}
foreach ($seed in $Seeds) {
    foreach ($model in @('plain_cnn','resnext50','resnext50_dense')) {
        $run = Join-Path $PSScriptRoot ('runs\{0}_raw_seed{1}' -f $model,$seed)
        Check-ExistingRun -Run $run
        if (-not (Test-Path -LiteralPath (Join-Path $run 'test_metrics.json'))) {
            Run-Python -Arguments @('train.py','--model',$model,'--view','raw','--epochs',"$Epochs",'--batch-size',"$BatchSize",'--seed',"$seed",'--device',$Device,'--workers',"$Workers",'--output',$run)
        }
    }
    $cropRun = Join-Path $PSScriptRoot ('runs\resnext50_dense_mp_seed{0}' -f $seed)
    Check-ExistingRun -Run $cropRun
    if (-not (Test-Path -LiteralPath (Join-Path $cropRun 'test_metrics.json'))) {
        Run-Python -Arguments @('train.py','--model','resnext50_dense','--view','mp','--epochs',"$Epochs",'--batch-size',"$BatchSize",'--seed',"$seed",'--device',$Device,'--workers',"$Workers",'--output',$cropRun)
    }
    $raw = Join-Path $PSScriptRoot ('runs\resnext50_dense_raw_seed{0}\best.pt' -f $seed)
    $report = Join-Path $PSScriptRoot ('reports\pipeline_seed{0}' -f $seed)
    if (-not (Test-Path -LiteralPath (Join-Path $report 'metrics.json'))) {
        Run-Python -Arguments @('evaluate_pipeline.py','--checkpoint',(Join-Path $cropRun 'best.pt'),'--raw-checkpoint',$raw,'--output',$report,'--batch-size',"$BatchSize",'--device',$Device)
    }
}
Run-Python -Arguments @('summarize.py')
Write-Host 'Completed. Read reports\comparison and reports\pipeline_seed*.'
