# Data Hub Integration V2 (ON)

This file tracks ON-side adoption of the canonical contract:
- Canonical spec: `C:/Users/BaiYang/CBOND_DATA_HUB/cbond_data_hub/docs/unified_data_contract_v2.md`

## ON Current Status
- Live pipeline raw/clean is routed through Data Hub CLI (`cbond_on.services.data_hub.gateway`).
- ON consumes shared `raw/clean`, and still produces project-local:
  - panel
  - label
  - factor
  - model scores
  - live trade list

## ON Next Mandatory Changes
1. Add `manifest + done` readiness check before panel/label/factor/model in live run.
2. Move ON paths config to shared roots for raw/clean once WC/SP are aligned.
3. Remove residual direct raw/clean producer modules after stability window.

## Runtime Reminder
- ON must not become a second writer for shared `raw/clean` in V2.
- Data Hub remains single writer.
