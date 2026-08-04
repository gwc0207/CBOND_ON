# Causal LSTM Model-Switch Arbitration — Frozen Holdout (2026-07-30)

## Outcome

The first small causal LSTM grid is not promotable.  The validation-selected
LSTM made zero direct-LCB overrides on the final chronological holdout, so its
selector return and Sharpe exactly equal the strict Base fallback and remain
below frozen current routing.

This is a negative result for this specific, predeclared representation:
`path44_state_only` (44 frozen T1430 state features, no return input), a
shallow LSTM, and the existing direct candidate-versus-Base uncertainty rule.
It does not show an actionable temporal return-prediction edge.

## Scope and safety boundary

- Research-only, offline replay.
- No change to `cbond_on/config/live`, `cbond_on/run`, `liveLaunch`, the
  production DB, scheduler, Champion list, model state,
  `D:/cbond_on/results/live`, or `D:/cbond_on/results/analysis`.
- Input is the pre-existing byte-copied snapshot:
  `D:/cbond_on/results/experiments/model_switch_temporal_arbitration_20260729/input_snapshot_20260729/`.
- Canonical output:
  `D:/cbond_on/results/experiments/model_switch_lstm_20260730/run_20260730T_lstm_path44_v1/`.
- Independent deterministic reproduction:
  `D:/cbond_on/results/experiments/model_switch_lstm_20260730/run_20260730T_lstm_path44_v1_reprocheck/`.
- Both manifests record `database_writes=false`, `live_runtime_called=false`,
  and `scheduler_called=false`.

## Frozen comparison contract

- Return/current-routing panel: 538 score days, 2024-05-08 through
  2026-07-27.
- State/return complete pairs: 532.  The six missing state rows occur in the
  final holdout and break a sequence; no interpolation or zero filling is
  used.
- Chronological split, set before training:

  | Period | Dates | Return days | Complete state/return days |
  | --- | --- | ---: | ---: |
  | Train | 2024-05-08..2025-10-15 | 350 | 350 |
  | Validation | 2025-10-16..2026-02-09 | 81 | 81 |
  | Final holdout | 2026-02-10..2026-07-27 | 107 | 101 |

- Input at score day `t`: the 44 T1430 state features over
  `t-L+1..t`; state is available at the decision time.  Same-day return is a
  target only and is never an input.
- Target: two Helmert contrasts of jointly clipped (75bp maximum pairwise
  spread) Regsim/Ensemble/HL20 standalone `day_return` values.  The output is
  transformed back to coherent sum-zero three-model utilities.
- The Base decision was rebuilt through read-only
  `decide_scoreopt_t1430_dispersion()` and matched frozen
  `base_model_id`, `base_reason`, and `base_score_gap` on `538/538` days.
- Intervention is limited to the previous frozen envelope:
  `base_reason=margin_default` with existing Robust reason in
  `score_best`/`margin_default`.  All routes outside it copy frozen current
  routing.
- Action rule remains direct `r-b` cross LCB > 0.  Neither the LSTM nor
  Robust top1-top2 gap is an action gate.

## Small hyperparameter grid and selection

Eight fixed CPU trials were run once:

```text
sequence length {20, 40}
hidden size     {8, 16}
head dropout    {0.0, 0.1}
```

Each trial uses a one-layer unidirectional LSTM and linear two-contrast head,
a single fixed seed (`20260730`), deterministic PyTorch CPU operations,
AdamW, gradient clipping, and chronological early stopping within the train
period.  A fixed target scale is numerical conditioning only; no future or
holdout-dependent target normalizer is fitted.

The grid selection key was validation **two-contrast OOS RMSE** only; ties are
broken by validation direction hit rate and trial id.  Validation selector
return/Sharpe are reported but never select a trial.  Final holdout return,
Sharpe, epoch count, and seed are excluded from selection.

| Rank | Trial | Validation contrast RMSE | Validation direction hit |
| ---: | --- | ---: | ---: |
| 1 | `lstm_seq40_hidden8_drop0p1` | 20.7287bp | 50.00% |
| 2 | `lstm_seq40_hidden8_drop0` | 20.7309bp | 51.23% |
| 3 | `lstm_seq20_hidden8_drop0p1` | 20.7423bp | 51.23% |
| 4 | `lstm_seq20_hidden8_drop0` | 20.7424bp | 51.85% |
| 5..8 | hidden size 16 variants | 20.7684..20.7743bp | 45.06..49.38% |

The first-ranked trial used 26 epochs selected without seeing the external
validation or final-holdout labels.  It was then refit once on train plus
validation only; no final-holdout state or label entered the scaler or fit.

## Final chronological holdout result

| Strategy | Total return | Sharpe | Max drawdown | LCB overrides | Delta total return vs current |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen current routing | +17.0029% | 2.6313 | -5.2825% | 7 existing Robust overrides | 0.0000pp |
| Strict Base fallback in envelope | +16.5700% | 2.6269 | -5.1530% | 0 | -0.4329pp |
| Selected LSTM | +16.5700% | 2.6269 | -5.1530% | 0 | -0.4329pp |

- The selected 40-day LSTM had valid sequence forecasts on `55/107` final
  score days.  Multiple missing state rows begin on 2026-05-12, and the
  required 40-day state sequence never recovers before the final date.
- There were 51 in-envelope days; 27 forecast-versus-Base candidate
  disagreements had a complete direct-LCB evaluation.  All 27 kept Base.
- On those 27 direct cases, realised candidate-minus-Base return averaged
  `-7.6949bp`, candidate win rate was `40.74%`, and mean LCB was `-8.3520bp`.
  Thus the zero-override result reflects both conservative calibration and
  unfavourable realised direct outcomes, not merely lack of a forecast.

Final pairwise forecasting diagnostics (55 ready days):

| Pair | Correlation | OOS R2 | Direction hit | RMSE |
| --- | ---: | ---: | ---: | ---: |
| Regsim - Ensemble | 0.0746 | -0.0431 | 38.18% | 34.1258bp |
| Regsim - HL20 | 0.1699 | 0.0046 | 47.27% | 33.5523bp |
| Ensemble - HL20 | 0.2108 | -0.0369 | 60.00% | 29.5608bp |

The modest individual correlations are not a validated aggregate signal:
two of three OOS R2 values are negative, and no candidate clears the
predeclared direct-LCB action rule.

## Verification

- `py -m pytest -q tests/test_model_switch_lstm_arbitration.py tests/test_model_switch_temporal_arbitration.py tests/test_live_model_switch.py`
  -> `37 passed`.
- Focused LSTM tests cover state-only input construction, missing-state
  sequence breaks, train-only scaling, strict target-before-prediction
  assertions, deterministic CPU fitting, two-contrast coherence, fixed
  eight-trial grid, validation-only selection, and output-root refusal.
- A second complete replay had byte-identical validation grid, daily
  forecasts, daily cross-arbitration decisions, selector metrics, diagnostic
  files, resolved plan, and all nine checkpoints.

## Limitations and next boundary

- The frozen state source remains `14:30_not_strict_1429_certified`; this
  experiment does not repair or prove strict 14:29 PIT compliance.
- The final holdout is historical, not prospective shadow evidence.  Do not
  tune its LCB, threshold, seed, sequence length, or architecture after seeing
  this outcome.
- Do not alter the live model selection, Champion artifact, scheduler, or DB
  from this result.
- A possible next research branch is a separately pre-registered representation
  with lower-dimensional state and lagged model-relative information, followed
  by a new untouched evaluation or prospective shadow period.  It is not part
  of this result.
