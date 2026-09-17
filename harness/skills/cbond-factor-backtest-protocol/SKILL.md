# cbond-factor-backtest-protocol

Use this skill for factor design, factor factory output, validation, screening,
or factor-to-model admission. For identity, registration, operator source,
retirement, storage boundaries, or live release, first read
`harness/skills/cbond-factor-governance/SKILL.md`.

## Required Reads

1. `harness/workflows/factor_backtest_protocol.md`
2. `docs/因子工程治理规则.md`
3. `docs/ai_factor_factory_dify_prompt.md` when Dify/AI factory is involved.

## Procedure

1. Run `py harness/tools/agent_preflight.py --mode factor-backtest`.
2. Resolve the exact canonical `factor_table` and use its manifest-bound reader.
3. Identify source fields and time visibility; reject future/label/PnL leakage.
4. Check NaN, Inf, constants, duplicates, and correlation risk.
5. Use only registered factor IDs and the approved feature/release manifest.
6. Run the required validation/backtest via existing governed entrypoints.
7. Keep research, release, and live-admission status explicit.

## Hard Stops

- Direct `factor_data_root`, old `factor_data`, and direct parquet are not normal factor inputs.
- A passing backtest does not imply live eligibility.
- Live admission requires owner confirmation, an immutable release, and no-DB replay.
