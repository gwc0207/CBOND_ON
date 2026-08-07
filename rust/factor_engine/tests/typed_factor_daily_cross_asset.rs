#[path = "../src/typed_factor_daily_cross_asset.rs"]
mod typed_factor_daily_cross_asset;

use chrono::{Duration, NaiveDate};
use typed_factor_daily_cross_asset::{
    bsfst_stock_return_bond_flow_mutual_information60, bssrc_bond_stock_rank_correlation60,
    TypedFactorCrossAssetBaseRow, TypedFactorCrossAssetPriceRow, TypedFactorDailyCrossAssetContext,
    TypedFactorDailyCrossAssetError,
};

const TARGET: &str = "110001.SH";

const BONDS: [(&str, &str, &str, f64, usize); 5] = [
    ("110001", "XSHG", "600001", 0.15, 0),
    ("110002", "SHSE", "600001", 0.75, 0),
    ("127001", "XSHE", "000001", 1.45, 1),
    ("123001", "SZSE", "000002", 2.10, 2),
    ("113001", "XSHG", "600003", 2.75, 3),
];

fn day(offset: i64) -> NaiveDate {
    NaiveDate::from_ymd_opt(2026, 1, 1).expect("valid date") + Duration::days(offset)
}

fn stock_return(offset: i64, group: usize) -> f64 {
    // The exact shared tie on group 1/2 every ninth session audits Pandas'
    // average-percentile stock ranks rather than an arbitrary ordinal rank.
    if offset % 9 == 0 && group == 2 {
        return stock_return(offset, 1);
    }
    let phase = [0.15, 1.55, 3.0, 4.4][group];
    0.021 * (0.37 * offset as f64 + phase).sin()
        + 0.004 * (0.11 * offset as f64 + group as f64).cos()
}

fn bond_return(offset: i64, phase: f64, index: usize) -> f64 {
    // This exact target/second-convertible tie audits Pandas average bond ranks.
    if offset % 7 == 0 && index == 1 {
        return bond_return(offset, 0.15, 0);
    }
    0.018 * (0.29 * offset as f64 + phase).sin()
        + 0.006 * (0.17 * offset as f64 + 0.3 * index as f64).cos()
}

fn baseline_context() -> TypedFactorDailyCrossAssetContext {
    let mut ctx = TypedFactorDailyCrossAssetContext::new(day(65));
    for offset in 0_i64..65 {
        for (index, (code, exchange, stock_code, phase, group)) in BONDS.iter().enumerate() {
            let bond_return = bond_return(offset, *phase, index);
            let underlying_return = stock_return(offset, *group);
            let amount = 1_000_000.0
                * (0.49 * (0.47 * offset as f64 + 0.61 * index as f64).sin()
                    + 0.13 * (0.19 * offset as f64 + index as f64).cos())
                .exp();
            let bond_previous = 100.0 + index as f64;
            let stock_previous = 50.0 + *group as f64;
            ctx.price_rows.push(TypedFactorCrossAssetPriceRow {
                trade_date: day(offset),
                code: (*code).to_string(),
                exchange_code: (*exchange).to_string(),
                prev_close_price: bond_previous,
                close_price: bond_previous * bond_return.exp(),
                amount,
            });
            ctx.base_rows.push(TypedFactorCrossAssetBaseRow {
                trade_date: day(offset),
                code: (*code).to_string(),
                exchange_code: (*exchange).to_string(),
                stock_code: (*stock_code).to_string(),
                stk_prev_close_price: stock_previous,
                stk_close_price: stock_previous * underlying_return.exp(),
            });
        }
    }
    ctx
}

fn assert_close(actual: f64, expected: f64) {
    assert!(actual.is_finite(), "expected finite value, got {actual:?}");
    assert!(
        (actual - expected).abs() <= 1e-12,
        "actual={actual:.17}, expected={expected:.17}"
    );
}

#[test]
fn two_cross_asset_outputs_match_python_goldens_with_average_rank_ties() {
    let ctx = baseline_context();

    // Generated from the unmodified Python kernels with this precise source
    // generator.  The path includes exact tied bond and distinct-stock returns,
    // making these values sensitive to `rank(method="average", pct=True)`.
    assert_close(
        bsfst_stock_return_bond_flow_mutual_information60(&ctx, TARGET).unwrap(),
        0.014842791178936348,
    );
    assert_close(
        bssrc_bond_stock_rank_correlation60(&ctx, TARGET).unwrap(),
        -0.20237608815565833,
    );
}

#[test]
fn bssrc_excludes_unmapped_history_before_the_tail_observation_gate() {
    let mut ctx = baseline_context();
    for row in ctx
        .base_rows
        .iter_mut()
        .filter(|row| row.code == "110001" && row.trade_date >= day(5) && row.trade_date <= day(20))
    {
        row.stock_code.clear();
    }

    // Python drops the sixteen unmapped bond/days in its final inner join.
    // Keeping them as NaN physical rows would leave only 44 finite pairs in
    // the latest 60 records; the inner-join path retains 49 and is finite.
    assert!(bssrc_bond_stock_rank_correlation60(&ctx, TARGET)
        .unwrap()
        .is_finite());
}

#[test]
fn score_day_rows_are_strictly_excluded() {
    let baseline = baseline_context();
    let mut contaminated = baseline.clone();
    let mut price = contaminated.price_rows[0].clone();
    price.trade_date = contaminated.score_date;
    price.close_price = 9_999_999.0;
    price.amount = 9_999_999_999.0;
    contaminated.price_rows.push(price);
    let mut base = contaminated.base_rows[0].clone();
    base.trade_date = contaminated.score_date;
    base.stk_close_price = 9_999_999.0;
    contaminated.base_rows.push(base);

    let expected_bsfst =
        bsfst_stock_return_bond_flow_mutual_information60(&baseline, TARGET).unwrap();
    let expected_bssrc = bssrc_bond_stock_rank_correlation60(&baseline, TARGET).unwrap();
    assert_eq!(
        bsfst_stock_return_bond_flow_mutual_information60(&contaminated, TARGET)
            .unwrap()
            .to_bits(),
        expected_bsfst.to_bits()
    );
    assert_eq!(
        bssrc_bond_stock_rank_correlation60(&contaminated, TARGET)
            .unwrap()
            .to_bits(),
        expected_bssrc.to_bits()
    );
}

#[test]
fn any_global_source_duplicate_fails_closed_before_target_selection() {
    let mut price_duplicate = baseline_context();
    let duplicate = price_duplicate
        .price_rows
        .iter()
        .find(|row| row.code == "127001" && row.trade_date == day(11))
        .expect("seeded unrelated row")
        .clone();
    price_duplicate.price_rows.push(duplicate);
    assert!(matches!(
        bsfst_stock_return_bond_flow_mutual_information60(&price_duplicate, TARGET),
        Err(TypedFactorDailyCrossAssetError::DuplicateStrictPriorDate {
            source: "market_cbond.daily_price",
            ..
        })
    ));

    let mut base_duplicate = baseline_context();
    let duplicate = base_duplicate
        .base_rows
        .iter()
        .find(|row| row.code == "123001" && row.trade_date == day(12))
        .expect("seeded unrelated row")
        .clone();
    base_duplicate.base_rows.push(duplicate);
    assert!(matches!(
        bssrc_bond_stock_rank_correlation60(&base_duplicate, TARGET),
        Err(TypedFactorDailyCrossAssetError::DuplicateStrictPriorDate {
            source: "market_cbond.daily_base",
            ..
        })
    ));
}

#[test]
fn mismatched_source_anchor_and_stale_terminal_code_return_nan() {
    let mut anchor_mismatch = baseline_context();
    anchor_mismatch
        .base_rows
        .retain(|row| row.trade_date != day(64));
    assert!(
        bsfst_stock_return_bond_flow_mutual_information60(&anchor_mismatch, TARGET)
            .unwrap()
            .is_nan()
    );
    assert!(
        bssrc_bond_stock_rank_correlation60(&anchor_mismatch, TARGET)
            .unwrap()
            .is_nan()
    );

    let mut stale = baseline_context();
    stale
        .base_rows
        .retain(|row| !(row.code == "110001" && row.trade_date == day(64)));
    assert!(
        bsfst_stock_return_bond_flow_mutual_information60(&stale, TARGET)
            .unwrap()
            .is_nan()
    );
    assert!(bssrc_bond_stock_rank_correlation60(&stale, TARGET)
        .unwrap()
        .is_nan());
}

#[test]
fn inconsistent_shared_underlying_and_terminal_invalid_values_fail_closed() {
    let mut inconsistent = baseline_context();
    inconsistent
        .base_rows
        .iter_mut()
        .find(|row| row.code == "110002" && row.trade_date == day(10))
        .expect("seeded shared-underlying row")
        .stk_close_price *= 1.05;
    assert!(matches!(
        bssrc_bond_stock_rank_correlation60(&inconsistent, TARGET),
        Err(TypedFactorDailyCrossAssetError::InconsistentUnderlyingReturn { .. })
    ));

    let mut terminal_invalid = baseline_context();
    terminal_invalid
        .base_rows
        .iter_mut()
        .find(|row| row.code == "110001" && row.trade_date == day(64))
        .expect("seeded target terminal row")
        .stk_close_price = 0.0;
    assert!(
        bsfst_stock_return_bond_flow_mutual_information60(&terminal_invalid, TARGET)
            .unwrap()
            .is_nan()
    );
    assert!(
        bssrc_bond_stock_rank_correlation60(&terminal_invalid, TARGET)
            .unwrap()
            .is_nan()
    );
}

#[test]
fn bs_fst_preserves_the_60_session_calendar_gap_instead_of_compressing_it() {
    let mut gapped = baseline_context();
    gapped
        .price_rows
        .retain(|row| !(row.code == "110001" && row.trade_date == day(30)));
    // Python's strict source-calendar reindex leaves day 30 and its following
    // amount change missing.  A compressed 59-row implementation has a
    // different categorical MI; this is the Python golden for the gapped path.
    assert_close(
        bsfst_stock_return_bond_flow_mutual_information60(&gapped, TARGET).unwrap(),
        0.015125942340523333,
    );
}
