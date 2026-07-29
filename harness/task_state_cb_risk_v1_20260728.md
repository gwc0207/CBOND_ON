# Task State: CB-Risk v1

## Objective

- Build an independent, shadow-only CB-Risk v1 foundation: PIT input contract, convertible-bond risk exposure construction, factor-risk estimation, portfolio risk/attribution, reporting, and verification. Do not alter the current live selection or trading chain.

## Risk Level

- High: the repository has a production live chain, although this implementation is explicitly isolated from it.

## Current Verified Facts

- The live cutoff is 14:29; production currently writes an `o_0001` trade list to PostgreSQL.
- The current live chain uses the Regsim champion with model switching enabled.
- Risk inputs must consume DataHub-local artifacts. Same-day `daily_base` is not a valid 14:29 input without a PIT publication contract.
- Existing live preprocessing neutralizes alpha factors; it is not a Barra risk model.

## Files Read

- `AGENTS.md`
- `harness/README.md`
- `harness/workflows/live_change.md`
- `harness/skills/cbond-live-safety-gate/SKILL.md`
- `harness/context/source_of_truth.md`
- `harness/policies/agent_operating_policy.md`
- `cbond_on/config/live/live_config.json5`
- `cbond_on/config/live/live_models_config.json5`
- `cbond_on/config/live/live_factors_config.json5`
- `cbond_on/config/benchmark/benchmark_config.json5`

## Files Changed

- `cbond_on/config/risk/cb_risk_v1_config.json5`
- `cbond_on/domain/risk/*`
- `cbond_on/infra/risk/*`
- `cbond_on/schemas/config/risk.py`
- `cbond_on/bootstrap/risk.py`
- `cbond_on/cli/risk.py`
- `cbond_on/workflows/research/cb_risk.py`
- `cbond_on/app/pipelines/risk_pipeline.py`
- `cbond_on/app/usecases/risk_runtime.py`
- `docs/cb_risk_v1_data_contract.md`
- `tests/test_cb_risk_*.py`

## Commands Run

- `py harness/tools/agent_preflight.py --mode live-change`
- `py -m pytest tests/test_cb_risk_*.py -q -p no:cacheprovider` (15 passed)
- `py -m cbond_on.common.architecture_guard` (passed)
- `py -m cbond_on.common.repo_hygiene_guard` (passed)
- `py -m cbond_on.cli.risk --config risk/cb_risk_v1 --start 2026-07-14 --end 2026-07-24 --dry-run` (9 factor-return days, no writes)
- `py -m cbond_on.cli.risk --config risk/cb_risk_v1 --start 2026-01-01 --end 2026-07-27 --positions D:/cbond_on/results/live/2026-07-28/trade_list.csv --output-root D:/cbond_on/results/risk/cb_risk_v1/yesterday_signal_20260727` (135 requested days, 134 factor-return days, isolated report written)

## Artifacts

- `docs/cb_risk_v1_data_contract.md` specifies the upstream PIT schema and publication gates.
- `python -m cbond_on.cli.risk --config risk/cb_risk_v1 ...` is the isolated offline/shadow entrypoint. It writes only to `results/risk` unless `--dry-run` is used.
- Yesterday's report: `D:/cbond_on/results/risk/cb_risk_v1/yesterday_signal_20260727/run_20260728_145300/risk_report.html`.

## Open Risks

- DataHub does not currently publish the PIT industry, float-market-cap, and financial history needed for a full industry/stock-style model.
- Current daily-twap coverage is not sufficient to claim a fully validated execution-horizon risk model until its missing-data contract is accepted.
- No live DB/scheduler/model-state operation is authorized or planned.
- The implemented core is deliberately a bond-style v0: industry, float-market-cap, and fundamentals remain disabled until the PIT contract is delivered.

## Next Action

- Obtain/DataHub-publish PIT input and dated benchmark holdings; then run full walk-forward calibration and a separately approved shadow-live sidecar.

## Handoff Summary

- CB-Risk core is implemented and test-verified. It remains shadow-only and must remain opt-in until a separately approved live integration phase.
