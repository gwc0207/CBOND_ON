# Data Hub Integration V2 (ON)

This file tracks ON-side adoption of the canonical contract:
- Canonical spec: `C:/Users/BaiYang/CBOND_DATA_HUB/cbond_data_hub/docs/unified_data_contract_v2.md`

## ON Current Status
- Live pipeline is consumer-only for shared `raw/clean` and checks DataHub `manifest + done` readiness.
- ON consumes shared `raw/clean`, and still produces project-local:
  - panel
  - label
  - factor
  - model scores
  - live trade list

## ON Next Mandatory Changes
1. Keep ON as a pure consumer (no ON-side raw/clean producer entrypoints).
2. Move WC/SP to the same shared roots and readiness contract.
3. Decide whether panel should remain ON-local or be standardized by DataHub.

## Runtime Reminder
- ON must not become a second writer for shared `raw/clean` in V2.
- Data Hub remains single writer.
