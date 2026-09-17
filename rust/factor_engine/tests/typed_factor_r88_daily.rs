#[path = "../src/typed_factor_r88_daily.rs"]
mod typed_factor_r88_daily;

use chrono::{Duration, NaiveDate};
use typed_factor_r88_daily::{
    bsab_downside_beta60, bsab_upside_beta60, bsct_upper_tail_dependence60,
    compute_r88_daily_signal, drrc_return_trade_size_rank_spearman60,
    drrq_return_amount_opposite_tail_excess60, dtwm_session_afternoon_late_log_slope,
    osa_terminal_amount_streak60, prepare_r88_daily_values, TypedFactorR88DailyBaseRow,
    TypedFactorR88DailyContext, TypedFactorR88DailyError, TypedFactorR88DailyPriceRow,
    TypedFactorR88DailySignal, TypedFactorR88DailyTwapRow,
};

const TARGET: &str = "110001.SH";

fn day(offset: i64) -> NaiveDate {
    NaiveDate::from_ymd_opt(2026, 1, 1).expect("valid date") + Duration::days(offset)
}

fn push_price(
    ctx: &mut TypedFactorR88DailyContext,
    trade_date: NaiveDate,
    code: &str,
    exchange: &str,
    bond_return: f64,
    amount: f64,
    deal: f64,
) {
    let previous = 100.0;
    ctx.price_rows.push(TypedFactorR88DailyPriceRow {
        trade_date,
        code: code.to_string(),
        exchange_code: exchange.to_string(),
        prev_close_price: previous,
        close_price: previous * bond_return.exp(),
        amount,
        deal,
    });
}

fn push_base(ctx: &mut TypedFactorR88DailyContext, trade_date: NaiveDate, stock_return: f64) {
    let previous = 50.0;
    ctx.base_rows.push(TypedFactorR88DailyBaseRow {
        trade_date,
        code: "110001".to_string(),
        exchange_code: "XSHG".to_string(),
        stk_prev_close_price: previous,
        stk_close_price: previous * stock_return.exp(),
    });
}

fn assert_close(actual: f64, expected: f64) {
    assert!(actual.is_finite(), "expected finite result, got {actual:?}");
    assert!(
        (actual - expected).abs() <= 1e-11,
        "actual={actual:.17}, expected={expected:.17}"
    );
}

fn assert_same_bits_or_nan(actual: f64, expected: f64) {
    assert!(
        (actual.is_nan() && expected.is_nan()) || actual.to_bits() == expected.to_bits(),
        "actual={actual:?} ({:#x}), expected={expected:?} ({:#x})",
        actual.to_bits(),
        expected.to_bits(),
    );
}

fn beta_and_copula_context() -> TypedFactorR88DailyContext {
    let mut ctx = TypedFactorR88DailyContext::new(day(60));
    for offset in 0_i64..60 {
        // 30 strictly negative and 30 strictly positive values, while the
        // bond is an exact two-beta transformation of the stock log return.
        let stock_return = (offset as f64 - 29.5) * 0.001;
        push_price(
            &mut ctx,
            day(offset),
            "110001",
            "XSHG",
            2.0 * stock_return,
            1_000.0 + offset as f64,
            10.0,
        );
        push_base(&mut ctx, day(offset), stock_return);
    }
    ctx
}

#[test]
fn asymmetric_betas_and_empirical_upper_tail_match_their_formulae() {
    let ctx = beta_and_copula_context();
    assert_close(bsab_upside_beta60(&ctx, TARGET).unwrap(), 2.0);
    assert_close(bsab_downside_beta60(&ctx, TARGET).unwrap(), 2.0);
    // Bond and stock return order is identical.  The upper empirical-copula
    // conditional probability is therefore one; the separate lower-tail gate
    // also has the required fifteen observations.
    assert_close(bsct_upper_tail_dependence60(&ctx, TARGET).unwrap(), 1.0);
}

fn osa_context() -> TypedFactorR88DailyContext {
    let mut ctx = TypedFactorR88DailyContext::new(day(60));
    for offset in 0_i64..60 {
        push_price(&mut ctx, day(offset), "110001", "XSHG", 0.001, 100.0, 2.0);
        // Retain an all-market calendar independently of the target's source
        // rows, as the Python seasoning factor does.
        push_price(&mut ctx, day(offset), "110002", "XSHG", 0.002, 200.0, 4.0);
    }
    ctx
}

#[test]
fn observable_amount_streak_distinguishes_zero_from_missing() {
    let baseline = osa_context();
    assert_close(
        osa_terminal_amount_streak60(&baseline, TARGET).unwrap(),
        60.0,
    );

    let mut terminal_zero = baseline.clone();
    terminal_zero
        .price_rows
        .iter_mut()
        .find(|row| row.code == "110001" && row.trade_date == day(59))
        .expect("seeded terminal target row")
        .amount = 0.0;
    // A visible completed terminal price and zero amount is a formula zero,
    // not a missing return or an implicit fill.
    assert_close(
        osa_terminal_amount_streak60(&terminal_zero, TARGET).unwrap(),
        0.0,
    );
}

const RANK_CODES: [&str; 5] = ["110001", "110002", "110003", "110004", "110005"];

fn rank_coupling_context() -> TypedFactorR88DailyContext {
    let mut ctx = TypedFactorR88DailyContext::new(day(60));
    for offset in 0_i64..60 {
        for (index, code) in RANK_CODES.iter().enumerate() {
            let rank = ((offset as usize + index) % RANK_CODES.len()) as f64;
            // Return and trade-size cross-sectional ordering are identical on
            // each date, but the target's rank varies over time so Pearson is
            // well-defined and should be exactly one.
            push_price(
                &mut ctx,
                day(offset),
                code,
                "XSHG",
                (rank - 2.0) * 0.001,
                1_000.0 * (rank + 1.0),
                1.0,
            );
        }
    }
    ctx
}

#[test]
fn return_trade_size_rank_coupling_uses_full_market_average_percentile_ranks() {
    let ctx = rank_coupling_context();
    assert_close(
        drrc_return_trade_size_rank_spearman60(&ctx, TARGET).unwrap(),
        1.0,
    );
}

fn tail_contradiction_context() -> TypedFactorR88DailyContext {
    let mut ctx = TypedFactorR88DailyContext::new(day(60));
    for offset in 0_i64..60 {
        // Target is always rank 1/5 for return and rank 5/5 for amount.
        push_price(
            &mut ctx,
            day(offset),
            "110001",
            "XSHG",
            -0.003,
            5_000.0,
            10.0,
        );
        for (index, code) in RANK_CODES.iter().skip(1).enumerate() {
            push_price(
                &mut ctx,
                day(offset),
                code,
                "XSHG",
                (index as f64 - 0.5) * 0.001,
                1_000.0 + index as f64 * 500.0,
                10.0,
            );
        }
    }
    ctx
}

#[test]
fn rank_tail_contradiction_subtracts_the_explicit_independence_baseline() {
    let ctx = tail_contradiction_context();
    assert_close(
        drrq_return_amount_opposite_tail_excess60(&ctx, TARGET).unwrap(),
        1.0 - 0.20 * (1.0 - 0.80),
    );
}

fn twap_context() -> TypedFactorR88DailyContext {
    let mut ctx = TypedFactorR88DailyContext::new(day(60));
    for offset in 0_i64..60 {
        push_price(
            &mut ctx,
            day(offset),
            "110001",
            "XSHG",
            0.001,
            1_000.0,
            10.0,
        );
        ctx.twap_rows.push(TypedFactorR88DailyTwapRow {
            trade_date: day(offset),
            code: "110001".to_string(),
            exchange_code: "XSHG".to_string(),
            twap_1300_1330: 100.0,
            twap_1400_1430: 105.0,
        });
    }
    ctx
}

#[test]
fn completed_session_twap_slope_uses_the_latest_strict_prior_join() {
    let ctx = twap_context();
    assert_close(
        dtwm_session_afternoon_late_log_slope(&ctx, TARGET).unwrap(),
        (105.0_f64 / 100.0).ln(),
    );
}

#[test]
fn every_r88_signal_is_strict_t_minus_one_and_dispatches_by_exact_name() {
    let baseline = beta_and_copula_context();
    let mut contaminated = baseline.clone();
    contaminated.price_rows.push(TypedFactorR88DailyPriceRow {
        trade_date: contaminated.score_date,
        code: "110001".to_string(),
        exchange_code: "XSHG".to_string(),
        prev_close_price: 100.0,
        close_price: 9_999_999.0,
        amount: 9_999_999.0,
        deal: 1.0,
    });
    contaminated.base_rows.push(TypedFactorR88DailyBaseRow {
        trade_date: contaminated.score_date,
        code: "110001".to_string(),
        exchange_code: "XSHG".to_string(),
        stk_prev_close_price: 50.0,
        stk_close_price: 9_999_999.0,
    });
    for signal in [
        "bsab_upside_beta60",
        "bsab_downside_beta60",
        "bsct_upper_tail_dependence60",
    ] {
        assert_eq!(
            compute_r88_daily_signal(&baseline, TARGET, signal)
                .unwrap()
                .to_bits(),
            compute_r88_daily_signal(&contaminated, TARGET, signal)
                .unwrap()
                .to_bits(),
            "score-day data affected {signal}"
        );
    }

    for (clean, mut score_day_rows, signals) in [
        (
            osa_context(),
            Vec::<TypedFactorR88DailyPriceRow>::new(),
            vec!["osa_terminal_amount_streak60"],
        ),
        (
            rank_coupling_context(),
            Vec::<TypedFactorR88DailyPriceRow>::new(),
            vec!["drrc_return_trade_size_rank_spearman60"],
        ),
        (
            tail_contradiction_context(),
            Vec::<TypedFactorR88DailyPriceRow>::new(),
            vec!["drrq_return_amount_opposite_tail_excess60"],
        ),
    ] {
        let mut dirty = clean.clone();
        score_day_rows.push(TypedFactorR88DailyPriceRow {
            trade_date: dirty.score_date,
            code: "110001".to_string(),
            exchange_code: "XSHG".to_string(),
            prev_close_price: 100.0,
            close_price: 9_999_999.0,
            amount: 9_999_999.0,
            deal: 1.0,
        });
        dirty.price_rows.extend(score_day_rows);
        for signal in signals {
            assert_eq!(
                compute_r88_daily_signal(&clean, TARGET, signal)
                    .unwrap()
                    .to_bits(),
                compute_r88_daily_signal(&dirty, TARGET, signal)
                    .unwrap()
                    .to_bits(),
                "score-day data affected {signal}"
            );
        }
    }

    let clean_twap = twap_context();
    let mut dirty_twap = clean_twap.clone();
    dirty_twap.price_rows.push(TypedFactorR88DailyPriceRow {
        trade_date: dirty_twap.score_date,
        code: "110001".to_string(),
        exchange_code: "XSHG".to_string(),
        prev_close_price: 100.0,
        close_price: 9_999_999.0,
        amount: 1.0,
        deal: 1.0,
    });
    dirty_twap.twap_rows.push(TypedFactorR88DailyTwapRow {
        trade_date: dirty_twap.score_date,
        code: "110001".to_string(),
        exchange_code: "XSHG".to_string(),
        twap_1300_1330: 1.0,
        twap_1400_1430: 9_999_999.0,
    });
    assert_eq!(
        compute_r88_daily_signal(&clean_twap, TARGET, "dtwm_session_afternoon_late_log_slope")
            .unwrap()
            .to_bits(),
        compute_r88_daily_signal(&dirty_twap, TARGET, "dtwm_session_afternoon_late_log_slope")
            .unwrap()
            .to_bits()
    );
}

#[test]
fn duplicate_strict_prior_source_fails_before_target_selection_and_signal_metadata_is_exact() {
    let mut ctx = rank_coupling_context();
    let duplicate = ctx
        .price_rows
        .iter()
        .find(|row| row.code == "110005" && row.trade_date == day(10))
        .expect("seeded unrelated strict-prior row")
        .clone();
    ctx.price_rows.push(duplicate);
    assert!(matches!(
        drrc_return_trade_size_rank_spearman60(&ctx, TARGET),
        Err(TypedFactorR88DailyError::DuplicateStrictPriorDate {
            source: "market_cbond.daily_price",
            ..
        })
    ));
    let signal = TypedFactorR88DailySignal::parse("bsct_upper_tail_dependence60")
        .expect("registered isolated signal");
    assert_eq!(signal.signal(), "bsct_upper_tail_dependence60");
    assert_eq!(signal.family(), "prior_bond_stock_copula_tail_dependence");
    assert!(signal.source_contract().contains("daily_base"));
    assert!(matches!(
        compute_r88_daily_signal(&beta_and_copula_context(), TARGET, "not_a_signal"),
        Err(TypedFactorR88DailyError::UnknownSignal(_))
    ));
}

const CACHE_CODES: [&str; 5] = [
    "110001.SH",
    "110002.SH",
    "110003.SH",
    "110004.SH",
    "110005.SH",
];

fn cache_parity_context() -> TypedFactorR88DailyContext {
    let mut ctx = TypedFactorR88DailyContext::new(day(60));
    for offset in 0_i64..60 {
        for (code_index, code) in CACHE_CODES.iter().enumerate() {
            let stock_return = (offset as f64 - 29.5) * 0.001 * (1.0 + code_index as f64 * 0.01);
            let rank_shift = ((offset as usize + code_index) % CACHE_CODES.len()) as f64 - 2.0;
            let bond_return = 1.3 * stock_return + rank_shift * 0.0003;
            let bare = code.trim_end_matches(".SH");
            push_price(
                &mut ctx,
                day(offset),
                bare,
                "XSHG",
                bond_return,
                1_000.0 + 100.0 * (code_index as f64 + 1.0) + offset as f64,
                1.0 + code_index as f64,
            );
            let stock_previous = 50.0;
            ctx.base_rows.push(TypedFactorR88DailyBaseRow {
                trade_date: day(offset),
                code: bare.to_string(),
                exchange_code: "XSHG".to_string(),
                stk_prev_close_price: stock_previous,
                stk_close_price: stock_previous * stock_return.exp(),
            });
            ctx.twap_rows.push(TypedFactorR88DailyTwapRow {
                trade_date: day(offset),
                code: bare.to_string(),
                exchange_code: "XSHG".to_string(),
                twap_1300_1330: 100.0 + code_index as f64,
                twap_1400_1430: 100.5 + code_index as f64 + offset as f64 * 0.001,
            });
        }
    }
    ctx
}

/// The dispatcher cache must only remove duplicate source preparation.  It
/// may never alter the direct formula's float operation order or fail-closed
/// NaN behavior for any signal/code output pair.
#[test]
fn prepared_daily_cache_matches_every_direct_signal_across_codes_bitwise_or_nan() {
    let ctx = cache_parity_context();
    let signals = [
        TypedFactorR88DailySignal::BsabUpsideBeta60,
        TypedFactorR88DailySignal::BsabDownsideBeta60,
        TypedFactorR88DailySignal::BsctUpperTailDependence60,
        TypedFactorR88DailySignal::OsaTerminalAmountStreak60,
        TypedFactorR88DailySignal::DrrcReturnTradeSizeRankSpearman60,
        TypedFactorR88DailySignal::DrrqReturnAmountOppositeTailExcess60,
        TypedFactorR88DailySignal::DtwmSessionAfternoonLateLogSlope,
    ];
    let cache = prepare_r88_daily_values(
        &ctx,
        signals,
        CACHE_CODES.iter().map(|code| (*code).to_string()),
    )
    .expect("cache source preparation succeeds");
    for signal in signals {
        for code in CACHE_CODES {
            let direct = compute_r88_daily_signal(&ctx, code, signal.signal())
                .expect("direct signal succeeds");
            let cached = cache
                .lookup(signal, code)
                .expect("requested signal map is present");
            assert_same_bits_or_nan(cached, direct);
        }
    }
}
