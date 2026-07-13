# cbond-factor-backtest-protocol

Use this skill for factor design, factor factory output, factor validation,
factor screening, or factor-to-model admission.

## Required Reads

1. `harness/workflows/factor_backtest_protocol.md`
2. `docs/开发规则.md`
3. `docs/ai_factor_factory_dify_prompt.md` if Dify or AI factor factory is
   involved.

## Procedure

1. Run `py harness/tools/agent_preflight.py --mode factor-backtest`.
2. Identify source fields and time visibility.
3. Validate field whitelist and forbidden future/label/PnL usage.
4. Check NaN, inf, constants, duplicate/high-correlation risk.
5. Run factor build/backtest through existing entrypoints.
6. Keep promotion status explicit.

## Hard Stops

- Passing backtest does not imply live eligibility.
- Old Dify `candidate_json` drafts are not trusted final output.
- Live admission requires owner confirmation and live workflow.
