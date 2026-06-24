<#
.SYNOPSIS
    Fraction-era model battery for the AI-Trader champion. One command, many models.

.DESCRIPTION
    Funnel design: a CHEAP single-window screen across many configs (Tiers 1-5), then
    EXPENSIVE walk-forward confirmation (Tier 6) of only the few survivors you pick.
    Walk-forward is ~4x the cost of a train, so it is never run on everything.

    Every experiment file is a one-variable delta from the champion base (config.yaml),
    which is now Dueling + PER + N2 + risk_penalty 0.002 + capital-fraction sizing.

    Tier 0  Walk-forward baselines: champion vs champion_shares (sizing ablation)
            vs champion_mlp (architecture ablation). The trustworthy reference.
    Tier 1  Sizing sweep        (screen): max_exposure / trade_fraction.
    Tier 2  Reward sweep        (screen): risk_penalty / reward_scale / penalties.
    Tier 3  Algorithm sweep     (screen): n_steps / replay.
    Tier 4  Feature ablations   (screen): regime features.
    Tier 5  Risk controls       (screen): stop-loss thresholds.
    Tier 6  Walk-forward confirmation. By default this AUTO-SELECTS the top -TopN
            configs from the screen (those whose test Sharpe beats both buy-and-hold
            AND the champion) and walk-forwards only those. Pass -Winners to override.

    Any screen tier first trains+compares the champion itself as the in-window
    reference bar (run id: champion / eval_champion).

.EXAMPLE
    ./scripts/run_experiments.ps1 -Tier 0          # baselines only (sanity first)
    ./scripts/run_experiments.ps1 -Tier all        # baselines + screen + auto-confirm winners
    ./scripts/run_experiments.ps1 -Tier 6          # auto-pick winners from an existing screen
    ./scripts/run_experiments.ps1 -Tier 6 -Winners rew_risk_004,size_exposure_50   # manual override
#>
param(
    [ValidateSet("0", "1", "2", "3", "4", "5", "6", "all")]
    [string]$Tier = "all",

    [string]$CondaEnv = "ai-trader",
    [string]$Seeds = "42,43,44",
    [int]$Episodes = 10,

    # Tier 6: how many top screen survivors to walk-forward when auto-selecting.
    [int]$TopN = 5,

    # Tier 6: manual override (config stems, no .yaml). Empty = auto-select from the screen.
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
    # Same --override as training so the network + feature set match the checkpoint.
    $ckpt = "results/$TrainRunId/checkpoints/double_dqn_best.pt"
    Invoke-AiTrader @("--mode", "compare", "--checkpoint", $ckpt, "--run-id", $EvalId,
        "--override", "experiments/$Yaml",
        "--split", "test", "--episodes", "$Episodes", "--seeds", $Seeds)
}

function Start-WalkForward {
    param([string]$RunId, [string]$Yaml)
    Invoke-AiTrader @("--mode", "walk_forward", "--run-id", $RunId, "--override", "experiments/$Yaml")
}

# Screen tiers: each row is @(train-run-id, config, eval-run-id).
$Tier1 = @(
    @("size_exposure_50", "size_exposure_50.yaml", "eval_size_exposure_50"),
    @("size_tradefrac_10", "size_tradefrac_10.yaml", "eval_size_tradefrac_10"),
    @("size_tradefrac_50", "size_tradefrac_50.yaml", "eval_size_tradefrac_50")
)
$Tier2 = @(
    @("rew_risk_001", "rew_risk_001.yaml", "eval_rew_risk_001"),
    @("rew_risk_004", "rew_risk_004.yaml", "eval_rew_risk_004"),
    @("rew_risk_008", "rew_risk_008.yaml", "eval_rew_risk_008"),
    @("rew_scale_50", "rew_scale_50.yaml", "eval_rew_scale_50"),
    @("rew_scale_200", "rew_scale_200.yaml", "eval_rew_scale_200"),
    @("rew_pos_0", "rew_pos_0.yaml", "eval_rew_pos_0"),
    @("rew_inact_0", "rew_inact_0.yaml", "eval_rew_inact_0")
)
$Tier3 = @(
    @("algo_n1", "algo_n1.yaml", "eval_algo_n1"),
    @("algo_n3", "algo_n3.yaml", "eval_algo_n3"),
    @("algo_n4", "algo_n4.yaml", "eval_algo_n4"),
    @("algo_uniform", "algo_uniform.yaml", "eval_algo_uniform")
)
$Tier4 = @(
    @("feat_volume", "feat_volume.yaml", "eval_feat_volume"),
    @("feat_volregime", "feat_volregime.yaml", "eval_feat_volregime"),
    @("feat_priceposition", "feat_priceposition.yaml", "eval_feat_priceposition"),
    @("feat_all", "feat_all.yaml", "eval_feat_all")
)
$Tier5 = @(
    @("risk_sl_10", "risk_sl_10.yaml", "eval_risk_sl_10"),
    @("risk_sl_15", "risk_sl_15.yaml", "eval_risk_sl_15")
)

function Invoke-TrainCompareTier {
    param([string]$Name, [array]$Specs)
    Write-Host "`n=== $Name ===" -ForegroundColor Yellow
    foreach ($s in $Specs) {
        if ($s -isnot [array]) { throw "Tier spec collapsed to non-array: $s" }
        Start-Train   -RunId $s[0] -Yaml $s[1]
        Start-Compare -EvalId $s[2] -TrainRunId $s[0] -Yaml $s[1]
    }
}

# Read a single config's averaged test Sharpe for one agent from its compare CSV.
function Get-Sharpe {
    param([string]$Csv, [string]$Agent)
    if (-not (Test-Path $Csv)) { return $null }
    $row = Import-Csv $Csv | Where-Object { $_.agent -eq $Agent } | Select-Object -First 1
    if ($null -eq $row) { return $null }
    return [double]$row.avg_sharpe
}

# Rank screen survivors: keep configs whose Sharpe beats both buy-and-hold and the
# champion, sort by Sharpe desc, return the top N stems. This is the auto-funnel.
function Select-Winners {
    param([int]$TopN)
    $champSharpe = Get-Sharpe -Csv "results/eval_champion/compare_metrics_test.csv" -Agent "double_dqn"
    if ($null -eq $champSharpe) {
        throw "No champion screen found (results/eval_champion/). Run the screen (Tier 1-5 or all) first."
    }

    $cands = @()
    foreach ($d in Get-ChildItem -Path "results" -Directory -Filter "eval_*") {
        $stem = $d.Name -replace '^eval_', ''
        if ($stem -eq "champion") { continue }
        $csv = Join-Path $d.FullName "compare_metrics_test.csv"
        $dd = Get-Sharpe -Csv $csv -Agent "double_dqn"
        $bh = Get-Sharpe -Csv $csv -Agent "buy_and_hold"
        if ($null -eq $dd -or $null -eq $bh) { continue }
        if ($dd -gt $bh -and $dd -gt $champSharpe) {
            $cands += [pscustomobject]@{ Stem = $stem; Sharpe = $dd; BH = $bh }
        }
    }

    $top = $cands | Sort-Object Sharpe -Descending | Select-Object -First $TopN
    Write-Host ("Champion in-window Sharpe: {0:N3}" -f $champSharpe) -ForegroundColor Green
    if ($top.Count -eq 0) {
        Write-Host "No screen config beat both buy-and-hold and the champion." -ForegroundColor Yellow
    }
    else {
        $top | Format-Table @{L = "config"; E = { $_.Stem } },
        @{L = "sharpe"; E = { "{0:N3}" -f $_.Sharpe } },
        @{L = "b&h"; E = { "{0:N3}" -f $_.BH } } | Out-Host
    }
    return @($top | ForEach-Object { $_.Stem })
}

$screenTiers = @("1", "2", "3", "4", "5")
$runScreen = ($Tier -eq "all") -or ($screenTiers -contains $Tier)

# Tier 0 - trustworthy walk-forward baselines + ablations.
if ($Tier -in @("0", "all")) {
    Write-Host "`n=== TIER 0: Walk-forward baselines ===" -ForegroundColor Yellow
    Start-WalkForward -RunId "wf_champion"        -Yaml "champion.yaml"
    Start-WalkForward -RunId "wf_champion_shares" -Yaml "champion_shares.yaml"
    Start-WalkForward -RunId "wf_champion_mlp"    -Yaml "champion_mlp.yaml"
    Write-Host "Gate: compare results/wf_champion/walk_forward_summary.csv vs shares/mlp + buy_and_hold." -ForegroundColor Green
}

# Screen reference - the champion in-window bar every screen tier ranks against.
if ($runScreen) {
    Write-Host "`n=== SCREEN REFERENCE: champion (single-window) ===" -ForegroundColor Yellow
    Start-Train   -RunId "champion" -Yaml "champion.yaml"
    Start-Compare -EvalId "eval_champion" -TrainRunId "champion" -Yaml "champion.yaml"
}

if ($Tier -in @("1", "all")) { Invoke-TrainCompareTier -Name "TIER 1: Sizing sweep" -Specs $Tier1 }
if ($Tier -in @("2", "all")) { Invoke-TrainCompareTier -Name "TIER 2: Reward sweep" -Specs $Tier2 }
if ($Tier -in @("3", "all")) { Invoke-TrainCompareTier -Name "TIER 3: Algorithm sweep" -Specs $Tier3 }
if ($Tier -in @("4", "all")) { Invoke-TrainCompareTier -Name "TIER 4: Feature ablations" -Specs $Tier4 }
if ($Tier -in @("5", "all")) { Invoke-TrainCompareTier -Name "TIER 5: Risk controls" -Specs $Tier5 }

# Tier 6 - walk-forward confirmation. Auto-selects the top screen survivors unless
# -Winners is given. Walk-forward is expensive, so only the few that beat the bar run.
if ($Tier -in @("6", "all")) {
    Write-Host "`n=== TIER 6: Walk-forward confirmation ===" -ForegroundColor Yellow
    $picked = if ($Winners.Count -gt 0) { $Winners } else { Select-Winners -TopN $TopN }

    if ($picked.Count -gt 0) {
        # Ensure the champion walk-forward reference exists (don't re-run if already done).
        if (-not (Test-Path "results/wf_champion/walk_forward_summary.csv")) {
            Start-WalkForward -RunId "wf_champion" -Yaml "champion.yaml"
        }
        foreach ($w in $picked) { Start-WalkForward -RunId "wf_$w" -Yaml "$w.yaml" }
    }
    else {
        Write-Host "Nothing to confirm - champion remains best in-window." -ForegroundColor Yellow
    }
}

Write-Host "`nDone." -ForegroundColor Green
Write-Host "Screen rankings : results/eval_*/compare_metrics_test.csv (vs eval_champion + buy_and_hold)" -ForegroundColor Green
Write-Host "Walk-forward    : results/wf_*/walk_forward_summary.csv" -ForegroundColor Green
