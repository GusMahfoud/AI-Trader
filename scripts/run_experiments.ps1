<#
.SYNOPSIS
    Run the dueling-champion experiment battery in tiers.

.DESCRIPTION
    Tier 0  Walk-forward gate (dueling vs MLP) — run and evaluate this FIRST.
    Tier 1  Reward-shaping sweep (a/b/c/d) — single-window train + compare.
    Tier 2  Feature ablations (volume/vol/price/all) — single-window train + compare.
    Tier 3  N-step=4 sensitivity — single-window train + compare.
    Tier 4  Walk-forward confirmation of the winners (you pass the configs).

.EXAMPLE
    ./scripts/run_experiments.ps1 -Tier 0
    ./scripts/run_experiments.ps1 -Tier all
    ./scripts/run_experiments.ps1 -Tier 4 -Winners dueling_per_n2,reward_tune_c
#>
param(
    [ValidateSet("0", "1", "2", "3", "4", "all")]
    [string]$Tier = "all",

    [string]$CondaEnv = "ai-trader",
    [string]$Seeds = "42,43,44",
    [int]$Episodes = 10,

    # Only used by the Tier-4 confirmation block.
    [string[]]$Winners = @()
)

$ErrorActionPreference = "Stop"

function Invoke-AiTrader {
    param([string[]]$CliArgs)
    Write-Host ">>> python -m ai_trader $($CliArgs -join ' ')" -ForegroundColor Cyan
    conda run --no-capture-output -n $CondaEnv python -m ai_trader @CliArgs
    if ($LASTEXITCODE -ne 0) { throw "ai_trader exited with code $LASTEXITCODE" }
}

function Start-Train {
    param([string]$RunId, [string]$Yaml)
    Invoke-AiTrader @("--mode", "train", "--run-id", $RunId, "--override", "experiments/$Yaml")
}

function Start-Compare {
    param([string]$EvalId, [string]$TrainRunId, [string]$Yaml)
    # Must pass the same --override as training so compare builds the matching
    # network + feature set; otherwise the checkpoint fails to load (state_dict mismatch).
    $ckpt = "results/$TrainRunId/checkpoints/double_dqn_best.pt"
    Invoke-AiTrader @("--mode", "compare", "--checkpoint", $ckpt, "--run-id", $EvalId,
        "--override", "experiments/$Yaml",
        "--split", "test", "--episodes", "$Episodes", "--seeds", $Seeds)
}

function Start-WalkForward {
    param([string]$RunId, [string]$Yaml)
    Invoke-AiTrader @("--mode", "walk_forward", "--run-id", $RunId, "--override", "experiments/$Yaml")
}

# Tier definitions: each row is @(train-run-id, config, eval-run-id).
$Tier1 = @(
    @("reward_tune_a_v3", "reward_tune_a.yaml", "eval_reward_a_v3"),
    @("reward_tune_b_v3", "reward_tune_b.yaml", "eval_reward_b_v3"),
    @("reward_tune_c_v3", "reward_tune_c.yaml", "eval_reward_c_v3"),
    @("reward_tune_d_v3", "reward_tune_d.yaml", "eval_reward_d_v3")
)
$Tier2 = @(
    @("feat_volume_v3", "feat_volume.yaml", "eval_feat_volume_v3"),
    @("feat_volregime_v3", "feat_volregime.yaml", "eval_feat_volregime_v3"),
    @("feat_priceposition_v3", "feat_priceposition.yaml", "eval_feat_priceposition_v3"),
    @("feat_all_v3", "feat_all.yaml", "eval_feat_all_v3")
)
# Leading comma forces a single-element array-of-array; @( @(...) ) would collapse
# into a flat string array and the loop would index into characters.
$Tier3 = , @("dueling_per_n4_v3", "dueling_per_n4.yaml", "eval_n4_v3")

function Invoke-TrainCompareTier {
    param([string]$Name, [array]$Specs)
    Write-Host "`n=== $Name ===" -ForegroundColor Yellow
    foreach ($s in $Specs) {
        if ($s -isnot [array]) { throw "Tier spec is not an array (PowerShell array collapse?): $s" }
        Start-Train   -RunId $s[0] -Yaml $s[1]
        Start-Compare -EvalId $s[2] -TrainRunId $s[0] -Yaml $s[1]
    }
}

if ($Tier -in @("0", "all")) {
    Write-Host "`n=== TIER 0: Walk-forward gate ===" -ForegroundColor Yellow
    Start-WalkForward -RunId "wf_dueling_v3" -Yaml "dueling_per_n2.yaml"
    Start-WalkForward -RunId "wf_mlp_v3"     -Yaml "champion_v1.yaml"
    Write-Host "Gate: compare results/wf_dueling_v3/walk_forward_test.csv vs wf_mlp_v3 + buy_and_hold." -ForegroundColor Green
}

if ($Tier -in @("1", "all")) { Invoke-TrainCompareTier -Name "TIER 1: Reward shaping sweep" -Specs $Tier1 }
if ($Tier -in @("2", "all")) { Invoke-TrainCompareTier -Name "TIER 2: Feature ablations" -Specs $Tier2 }
if ($Tier -in @("3", "all")) { Invoke-TrainCompareTier -Name "TIER 3: N-step sensitivity" -Specs $Tier3 }

# Tier 4 is opt-in: select it with -Tier 4 (or -Tier all) and pass the winning
# config stems via -Winners (no .yaml suffix).
if ($Tier -in @("4", "all")) {
    if ($Winners.Count -gt 0) {
        Write-Host "`n=== TIER 4: Walk-forward confirmation of winners ===" -ForegroundColor Yellow
        foreach ($w in $Winners) { Start-WalkForward -RunId "wf_${w}_v3" -Yaml "$w.yaml" }
    }
    elseif ($Tier -eq "4") {
        Write-Host "Tier 4 selected but no -Winners passed; nothing to confirm." -ForegroundColor Yellow
    }
}

Write-Host "`nDone. Metrics: results/<eval-id>/compare_metrics_test.csv and results/wf_*/walk_forward_test.csv" -ForegroundColor Green
