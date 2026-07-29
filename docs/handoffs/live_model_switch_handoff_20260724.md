# CBOND ON Live Model Switch Handoff - 2026-07-24

## 1. Current Runtime State

- Real repo root: `C:\Users\BaiYang\CBOND_ON\cbond_on`.
- Current live scheduler:
  - `pid=31416`
  - `started_at=2026-07-24T15:47:17`
  - `status=idle_after_run`
  - `today=2026-07-24`
  - `target=2026-07-27`
  - heartbeat checked at `2026-07-24T16:04:49`.
- Today's failed scheduled run was manually repaired before this handoff:
  - Manual repair model: `lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708`
  - Local output: `D:\cbond_on\results\live\2026-07-27\trade_list.csv`
  - DB write date follows existing live semantics: `db_trade_day=2026-07-23`
  - Do not rewrite today's DB unless the user explicitly confirms.

## 2. Current Live Chain

Live config: `cbond_on/config/live/live_config.json5`.

Main live flow:

```text
live_scheduler
  -> DataHub clean ready gate
  -> local derived data build
  -> live factor build
  -> champion score
  -> challenger scores
  -> shadow return update
  -> T1430 state feature update
  -> model switch
  -> clean + score merge
  -> o_0005 allowlist
  -> select_signals
  -> trade_list.csv
  -> DB replace_date
```

Current model switch config:

- Mode: `scoreopt_t1430_fusion_gate`
- Feature set: `path_full_t1430`
- Metric / score mode: `trim20_lcb10`
- Base selector:
  - `lookback_days=60`
  - `nearest_k=40`
  - `min_periods=40`
  - `margin=0.0005`
- Fusion:
  - policy: `base_low_confidence_only`
  - `champion_third_veto.enabled=true`
  - robust:
    - `lookback_days=360`
    - `min_periods=120`
    - `alpha=100.0`
    - `target_clip=0.0075`
    - `margin=0.0005`
- Hard data policies:
  - `stale_history_policy=fail`
  - `shadow_return_update.fail_on_error=true`
  - `allowlist no-filter` is forbidden by code.

Models:

| Role | Name | Model ID |
|---|---|---|
| Champion | Regsim w10_s15_d05 | `lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708` |
| Challenger 1 | Ensemble rankavg + labeltop20 | `ensemble_rankavg_baseline_hl20_labeltop20_20260626` |
| Challenger 2 | HL20 live baseline | `lgbm_screened_no_winsor_neutral_tminus1_weight_recent_hl20_20260625` |

Ensemble challenger sources:

- `baseline`: `lgbm_screened_no_winsor_neutral_tminus1_refit1_202401_rerun_20260623`
- `hl20`: `lgbm_screened_no_winsor_neutral_tminus1_weight_recent_hl20_20260625`
- `labeltop20`: `lgbm_screened_no_winsor_neutral_tminus1_weight_labeltop20_20260625`

## 3. Incident Summary

Today, 2026-07-24, the scheduled live run failed after all scores and shadow returns had been produced.

Failure step:

```text
model switch selected Regsim
  -> reason=fusion_base_robust_not_confident
  -> fallback_reason=margin_default
  -> old strict guard treated this as forbidden fallback
  -> live stopped before trade_list / DB write
```

Root issue:

- The old strict policy treated model-selection low confidence as a hard error.
- The user clarified the intended policy:
  - Data/score/shadow failures should be hard errors.
  - Model-selection uncertainty should not kill live after all model scores are available.
  - It should choose a deterministic model, continue writing, and leave a warning / dashboard note.

## 4. Changes Made In This Handoff

Changed files:

- `cbond_on/infra/live/model_switch.py`
- `cbond_on/app/usecases/live_runtime.py`
- `tests/test_live_model_switch.py`

Important: The worktree already had many unrelated changes before this handoff. Do not revert unrelated modified/deleted/untracked files.

### 4.1 Model Switch Logic

For the current T1430 selector:

- If base selector has a clear first place:
  - select base first place.
- If base selector first and second are too close:
  - keep the base current best model as the selected base result.
  - do not force champion at this stage.
  - continue into `champion_third_veto`.
- If `champion_third_veto` triggers:
  - select best challenger.
- If veto does not trigger:
  - run robust.
- If robust has a clear first place:
  - select robust first place.
- If robust is not confident:
  - select the base current best model.
  - record this as soft degrade, not hard failure.

This means the old behavior:

```text
base margin_default -> champion
```

is no longer the live T1430 behavior. The new behavior is:

```text
base margin_default -> base best score model
```

### 4.2 Live Runtime Failure Policy

Model-selection uncertainty no longer raises `RuntimeError`.

Now it:

- prints `live model switch soft degrade`
- appends `warnings` into `model_switch_decision.json`
- writes a front-end day note to:
  - `D:\cbond_on\results\live\scheduler\dashboard_notes.json`
- continues into selected score / allowlist / select_signals / DB write.

Hard errors remain for:

- factor/data build failure
- required model score missing
- shadow return update failure
- return history stale when `stale_history_policy=fail`
- clean daily empty
- no score matched to clean data
- missing o_0005 allowlist / no-filter fallback
- universe empty after allowlist
- empty strategy picks
- DB write failure, except the existing `FileNotFoundError` skip path.

## 5. Read-Only Validation After Change

Command run:

```powershell
py -m pytest tests/test_live_model_switch.py -q
```

Result:

```text
17 passed
```

Command run:

```powershell
py -m pytest tests/test_live_dashboard_model_compare.py -q
```

Result:

```text
3 passed
```

Read-only current-config decision check for `score_day=2026-07-24`:

```text
selected = Ensemble rankavg + labeltop20
model_id = ensemble_rankavg_baseline_hl20_labeltop20_20260626
reason = fusion_base_robust_not_confident
fallback_reason = margin_default
warning = 模型选择提示：base与robust均未形成高置信切换信号，已按规则选择 Ensemble rankavg + labeltop20，实盘继续写库。
```

This was a read-only selector check. It did not rewrite today's DB.

## 6. Scheduler Restart

The scheduler was restarted so the next run uses the new Python code.

Current state:

```text
pid = 31416
started_at = 2026-07-24T15:47:17
status = idle_after_run
target = 2026-07-27
```

## 7. Critical Operating Rules For Next Agent

Do not change live scope silently.

Before changing any of these, report the exact final口径 to the user and wait for confirmation:

- live model
- champion/challenger composition
- model switch mode
- neutralization
- factor set
- allowlist / universe
- DB table / write date semantics
- scheduler entrypoint
- run directory files
- output TWAP columns

For live incidents:

1. First check scheduler state/log.
2. Then verify config actually loaded by the running process.
3. Remember Python scheduler does not hot-reload code; restart after live-code changes.
4. Do not assume memory is current; inspect `live_config.json5`, `pid.json`, `state.json`, and day logs.
5. Do not rewrite live DB unless explicitly requested.

For fallback language:

- Avoid using "fallback" as a generic word for ordinary low-confidence decision.
- User's intended口径:
  - data/score missing: hard error
  - model choice low confidence: soft degrade + warning + continue
  - no silent no-filter universe expansion

## 8. Current Known Dirty Worktree

Relevant files changed by this handoff:

- `cbond_on/app/usecases/live_runtime.py`
- `cbond_on/infra/live/model_switch.py`
- `tests/test_live_model_switch.py`

Other dirty files existed and should not be reverted without user confirmation:

- `cbond_on/common/repo_hygiene_guard.py`
- `cbond_on/config/live/live_config.json5`
- `cbond_on/infra/benchmark/service.py`
- `cbond_on/infra/model/runners/train_lgbm.py`
- `liveLaunch/web/static/app.js`
- `liveLaunch/web/static/style.css`
- `tests/test_strict_cycle_returns.py`
- deleted old regsim grid configs under outer `config/...`
- untracked dynamic feature contribution configs/tests

## 9. Immediate Next-Step Recommendation

For the next trading day, if data, scores, shadow returns, allowlist, and DB are normal, live should write even if model selection is low-confidence.

Before the next live window, the next agent should only do read-only checks unless the user explicitly asks for changes:

```powershell
Get-Content D:\cbond_on\results\live\scheduler\pid.json -Raw
Get-Content D:\cbond_on\results\live\scheduler\state.json -Raw
Get-Content D:\cbond_on\results\live\2026-07-24\logs\live_scheduler_2026-07-24.log -Tail 80
py -m pytest tests/test_live_model_switch.py -q
```

If the next model reruns today's live path for validation, it must use `db_write=false` or an explicit dry-run path unless the user requests DB overwrite.
