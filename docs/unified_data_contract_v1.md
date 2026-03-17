# Unified Data Contract V1 (ON Landing)

This document defines the V1 data contract currently implemented in `CBOND_ON`.

## Ownership (Decoupled Mode)

- `raw/clean` production is owned by `CBOND_DATA_HUB`.
- `CBOND_ON` only consumes data contract outputs (panel/label/factor/model/strategy).
- Live pipeline in ON calls Data Hub CLI for:
  - `raw sync-history`
  - `raw sync-intraday`
  - `clean build`

## 1) Time Semantics

- `run_day`: day used for today's incremental snapshot pull and same-day score output.
- `target_day`: day used for final live trade list output.
- `prev_trade_day`: previous trading day before `run_day`.
- `lookback_start_day`: trading-day lookback anchor for rebuild.

Scheduler policy:

- `schedule.target_policy = "today"`: `target_day = today`.
- `schedule.target_policy = "next_trading_day_after_cutoff"`: `target_day = next trading day`.

## 2) Trading Day Service

Function: `cbond_on.core.trading_days.list_trading_days_from_raw`

- Primary source: `metadata.trading_calendar`.
- Fallback: raw file existence when calendar is missing/incomplete.

## 3) Raw Layer Contract

- History sync window: `[lookback_start_day, prev_trade_day]`.
- Intraday sync window: `run_day` only.
- History sources: `NFS` and optional `DB`.
- Intraday source: `Redis snapshot`.

Redis watermark default location:

- `{raw_data_root}/state/watermarks/{asset}.json`

## 4) Clean Layer Contract

Snapshot output (canonical):

- `{cleaned_data_root}/snapshot/cbond/{YYYY-MM}/{YYYYMMDD}.parquet`

Kline output (canonical):

- `{cleaned_data_root}/kline/cbond/{YYYY-MM}/{YYYY-MM-DD}.parquet`

Compatibility reads are kept for legacy paths:

- Snapshot legacy: `{cleaned_data_root}/snapshot/{YYYY-MM}/{YYYYMMDD}.parquet`
- Kline legacy: `{cleaned_data_root}/kline/{YYYY-MM}/{YYYY-MM-DD}.parquet`

## 5) Rebuild Rules

- `clean/panel/factor`: rebuild on `[lookback_start_day, run_day]`.
- `label`: rebuild only to `prev_trade_day`.
- `model score`: score `run_day` only; `label_cutoff = prev_trade_day`.

## 6) Config Compatibility

V1-compatible fields were added while preserving old keys:

- `live.schedule.target_policy`
- `live.source.history.*`
- `live.source.intraday.*`
- `cleaned_data.kline_enabled`
- `raw_data.assets`, `raw_data.source.*`

Existing keys continue to work during migration.
