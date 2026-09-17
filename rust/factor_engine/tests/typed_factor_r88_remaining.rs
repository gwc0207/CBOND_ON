#[path = "../src/typed_factor_r88_remaining.rs"]
mod typed_factor_r88_remaining;

use chrono::{Duration, NaiveDate};
use typed_factor_r88_remaining::{
    clock_ns, compute_r88_remaining_signal, csn_pql_churn_neighbor_gap,
    hybrid_current_flow_vs_hist_overnight_response, hybrid_current_range_vs_hist_twap_curve,
    isgm_stockvol_trade_quote_clock_center_gap, itr_stock_shock_same_bin_directional_agreement,
    joint_tail_range_coexpansion, joint_tail_signed_cojump, joint_tail_terminal_location_coshock,
    prepare_r88_remaining_values, qgeo_micro_last_next_return_sign_alignment,
    sng_peer_return_dispersion1, strict_1429_visible, ucd_peer_stock_return_dispersion1,
    R88RemainingBondStockMapRow, R88RemainingDailyBaseRow, R88RemainingDailyPriceRow,
    R88RemainingDailyTwapRow, R88RemainingIntradayRow, TypedFactorR88RemainingContext,
    TypedFactorR88RemainingError, TypedFactorR88RemainingSignal,
};

const TARGET: &str = "110001.SH";

fn day(offset: i64) -> NaiveDate {
    NaiveDate::from_ymd_opt(2026, 1, 1).expect("valid test epoch") + Duration::days(offset)
}

fn intraday_row(
    trade_date: NaiveDate,
    code: &str,
    time_ns: i64,
    seq: i64,
    last: f64,
    volume: f64,
    amount: f64,
    num_trades: f64,
) -> R88RemainingIntradayRow {
    R88RemainingIntradayRow {
        trade_date,
        code: code.to_string(),
        exchange_code: "XSHG".to_string(),
        time_ns,
        seq,
        last,
        volume,
        amount,
        num_trades,
        ask_price: [last + 1.0, last + 1.1, last + 1.2, last + 1.3, last + 1.4],
        bid_price: [last - 1.0, last - 1.1, last - 1.2, last - 1.3, last - 1.4],
        ask_volume: [10.0, 11.0, 12.0, 13.0, 14.0],
        bid_volume: [20.0, 21.0, 22.0, 23.0, 24.0],
    }
}

fn price_row(
    trade_date: NaiveDate,
    code: &str,
    prev_close: f64,
    close: f64,
    amount: f64,
) -> R88RemainingDailyPriceRow {
    R88RemainingDailyPriceRow {
        trade_date,
        code: code.to_string(),
        exchange_code: "XSHG".to_string(),
        prev_close_price: prev_close,
        close_price: close,
        amount,
    }
}

fn base_row(
    trade_date: NaiveDate,
    code: &str,
    cb_amount: f64,
    stock_code: &str,
    stock_close: f64,
    stock_volatility: f64,
    stk_amount: f64,
    index: usize,
) -> R88RemainingDailyBaseRow {
    R88RemainingDailyBaseRow {
        trade_date,
        code: code.to_string(),
        exchange_code: "XSHG".to_string(),
        cb_amount,
        stock_code: stock_code.to_string(),
        stock_close_price: stock_close,
        stock_volatility,
        stk_amount,
        year_to_mat: 1.0 + index as f64 * 0.02,
        duration: 0.5 + index as f64 * 0.015,
        bond_prem_ratio: 0.05 + index as f64 * 0.003,
        ytm: 0.01 + index as f64 * 0.0007,
        remain_size: 100.0 + index as f64 * 2.0,
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(actual.is_finite(), "expected finite value, got {actual:?}");
    assert!(
        (actual - expected).abs() < 1e-10,
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

#[test]
fn metadata_and_strict_pit_cutoff_are_explicit() {
    assert!(strict_1429_visible(clock_ns(14, 29, 0, 999)));
    assert!(!strict_1429_visible(clock_ns(14, 29, 0, 1_000)));
    assert!(!strict_1429_visible(clock_ns(14, 29, 30, 0)));
    assert!(!strict_1429_visible(clock_ns(14, 30, 0, 0)));

    let signal = TypedFactorR88RemainingSignal::parse("csn_pql_churn_neighbor_gap")
        .expect("isolated signal exists");
    assert_eq!(signal.family(), "csn_passive_queue_local_dislocation");
    assert!(signal.source_contract().contains("num_trades"));
    assert_eq!(
        TypedFactorR88RemainingSignal::parse("hybrid_current_range_vs_hist_twap_curve")
            .expect("hybrid signal")
            .signal(),
        "hybrid_current_range_vs_hist_twap_curve"
    );
}

fn hybrid_context() -> TypedFactorR88RemainingContext {
    let mut ctx = TypedFactorR88RemainingContext::new(day(30));
    for offset in 0..20 {
        ctx.daily_base_rows.push(base_row(
            day(offset),
            TARGET,
            100.0,
            "600000",
            100.0 + offset as f64,
            0.2,
            1_000.0 + offset as f64,
            1,
        ));
        ctx.daily_twap_rows.push(R88RemainingDailyTwapRow {
            trade_date: day(offset),
            code: TARGET.to_string(),
            exchange_code: "XSHG".to_string(),
            // From the second completed day forward, opening is one percent
            // above the immediately preceding late reference.
            twap_0930_0935: 101.0,
            twap_0935_1000: 102.01,
            twap_1300_1330: 100.0,
            twap_1400_1430: 102.0,
            twap_1430_1442: 100.0,
        });
    }
    for (index, (last, volume, amount, trades)) in [
        (100.0, 0.0, 0.0, 0.0),
        (101.0, 1.0, 10.0, 1.0),
        (103.0, 2.0, 20.0, 2.0),
        (102.0, 3.0, 35.0, 3.0),
        (105.0, 4.0, 50.0, 4.0),
    ]
    .into_iter()
    .enumerate()
    {
        ctx.intraday_rows.push(intraday_row(
            ctx.score_date,
            TARGET,
            clock_ns(9, 30, index as i64, 0),
            index as i64,
            last,
            volume,
            amount,
            trades,
        ));
    }
    ctx
}

#[test]
fn hybrid_formulae_use_completed_history_and_pre_filter_old_python_1430_rows() {
    let baseline = hybrid_context();
    let curve = (102.01_f64 / 101.0 - 1.0) + (102.0 / 100.0 - 1.0) - 2.0 * (100.0 / 102.0 - 1.0);
    assert_close(
        hybrid_current_range_vs_hist_twap_curve(&baseline, TARGET).unwrap(),
        0.05 * curve,
    );
    assert_close(
        hybrid_current_flow_vs_hist_overnight_response(&baseline, TARGET).unwrap(),
        0.5 * 0.01,
    );

    // The old Python source accepted <=14:30.  Parity must prefilter that old
    // reference to 14:29; these values prove the Rust side cannot see them.
    let mut after_cutoff = baseline.clone();
    after_cutoff.intraday_rows.push(intraday_row(
        after_cutoff.score_date,
        TARGET,
        clock_ns(14, 30, 0, 0),
        999,
        10_000.0,
        1e12,
        1e12,
        1e12,
    ));
    assert_eq!(
        hybrid_current_range_vs_hist_twap_curve(&after_cutoff, TARGET)
            .unwrap()
            .to_bits(),
        hybrid_current_range_vs_hist_twap_curve(&baseline, TARGET)
            .unwrap()
            .to_bits()
    );
    assert_eq!(
        hybrid_current_flow_vs_hist_overnight_response(&after_cutoff, TARGET)
            .unwrap()
            .to_bits(),
        hybrid_current_flow_vs_hist_overnight_response(&baseline, TARGET)
            .unwrap()
            .to_bits()
    );
}

fn joint_context() -> TypedFactorR88RemainingContext {
    let mut ctx = TypedFactorR88RemainingContext::new(day(10));
    ctx.daily_price_rows
        .push(price_row(day(9), TARGET, 100.0, 101.0, 1_000.0));
    ctx.daily_base_rows.push(base_row(
        day(9),
        TARGET,
        100.0,
        "600000",
        100.0,
        0.2,
        1_000.0,
        1,
    ));
    for (index, (bond, stock)) in [(100.0, 50.0), (105.0, 45.0), (102.0, 48.0)]
        .into_iter()
        .enumerate()
    {
        let time = clock_ns(13, 30 + index as i64 * 10, 0, 0);
        ctx.intraday_rows.push(intraday_row(
            ctx.score_date,
            TARGET,
            time,
            index as i64,
            bond,
            index as f64,
            index as f64,
            index as f64,
        ));
        ctx.intraday_rows.push(intraday_row(
            ctx.score_date,
            "600000.SH",
            time,
            index as i64,
            stock,
            index as f64,
            index as f64,
            index as f64,
        ));
    }
    ctx
}

#[test]
fn joint_tail_products_use_exact_tminus1_mapping_and_strict_cutoff() {
    let baseline = joint_context();
    assert_close(
        joint_tail_signed_cojump(&baseline, TARGET).unwrap(),
        -0.0008,
    );
    assert_close(
        joint_tail_range_coexpansion(&baseline, TARGET).unwrap(),
        0.05 * (5.0 / 50.0),
    );
    assert_close(
        joint_tail_terminal_location_coshock(&baseline, TARGET).unwrap(),
        (0.4 - 0.6) * -0.0008,
    );

    let mut contaminated = baseline.clone();
    contaminated.intraday_rows.push(intraday_row(
        contaminated.score_date,
        TARGET,
        clock_ns(14, 30, 0, 0),
        88,
        1.0,
        1.0,
        1.0,
        1.0,
    ));
    assert_eq!(
        joint_tail_signed_cojump(&baseline, TARGET)
            .unwrap()
            .to_bits(),
        joint_tail_signed_cojump(&contaminated, TARGET)
            .unwrap()
            .to_bits()
    );
}

#[test]
fn itr_same_bin_shock_agreement_uses_only_contiguous_five_minute_endpoints() {
    let mut ctx = TypedFactorR88RemainingContext::new(day(1));
    ctx.bond_stock_map.push(R88RemainingBondStockMapRow {
        code: TARGET.to_string(),
        stock_code: "600000.SH".to_string(),
        as_of_date: ctx.score_date,
    });
    let returns: [f64; 13] = [
        0.001, -0.002, 0.003, -0.004, 0.005, -0.006, 0.007, -0.008, 0.009, -0.010, 0.011, -0.012,
        0.013,
    ];
    let mut bond = 100.0_f64;
    let mut stock = 50.0_f64;
    ctx.intraday_rows.push(intraday_row(
        ctx.score_date,
        TARGET,
        clock_ns(9, 30, 0, 0),
        0,
        bond,
        0.0,
        0.0,
        0.0,
    ));
    ctx.intraday_rows.push(intraday_row(
        ctx.score_date,
        "600000.SH",
        clock_ns(9, 30, 0, 0),
        0,
        stock,
        0.0,
        0.0,
        0.0,
    ));
    for (index, value) in returns.into_iter().enumerate() {
        bond *= value.exp();
        stock *= value.exp();
        let time = clock_ns(9, 35 + index as i64 * 5, 0, 0);
        ctx.intraday_rows.push(intraday_row(
            ctx.score_date,
            TARGET,
            time,
            index as i64 + 1,
            bond,
            0.0,
            0.0,
            0.0,
        ));
        ctx.intraday_rows.push(intraday_row(
            ctx.score_date,
            "600000.SH",
            time,
            index as i64 + 1,
            stock,
            0.0,
            0.0,
            0.0,
        ));
    }
    assert_close(
        itr_stock_shock_same_bin_directional_agreement(&ctx, TARGET).unwrap(),
        1.0,
    );
}

fn wide_daily_context() -> TypedFactorR88RemainingContext {
    let mut ctx = TypedFactorR88RemainingContext::new(day(30));
    for index in 0..31_usize {
        let code = format!("11{index:04}.SH");
        ctx.panel_codes.push(code.clone());
        for offset in 0..21 {
            ctx.daily_price_rows.push(price_row(
                day(offset),
                &code,
                100.0,
                100.0 * (1.0 + 0.001 * (index + 1) as f64),
                1_000.0 + offset as f64 * 10.0 + index as f64,
            ));
            ctx.daily_base_rows.push(base_row(
                day(offset),
                &code,
                1_000.0 + index as f64,
                "600000",
                100.0 + offset as f64 * 0.1 * (index + 1) as f64,
                0.1 + index as f64 * 0.01,
                500.0 + (offset * offset) as f64 * (index + 1) as f64,
                index,
            ));
        }
    }
    ctx
}

#[test]
fn tminus1_underlying_and_structural_peer_dispersion_are_finite_on_31_name_snapshot() {
    let ctx = wide_daily_context();
    let target = "110015.SH";
    assert!(ucd_peer_stock_return_dispersion1(&ctx, target)
        .unwrap()
        .is_finite());
    assert!(sng_peer_return_dispersion1(&ctx, target)
        .unwrap()
        .is_finite());
}

#[test]
fn isgm_stockvol_gate_multiplies_trade_quote_clock_center_gap() {
    let mut ctx = wide_daily_context();
    ctx.score_date = day(30);
    let target = "110030.SH";
    // Make the target the maximum stock-vol observation, hence smoothstep rank
    // one.  It will then expose the raw trade-versus-quote clock gap directly.
    let last = ctx
        .daily_base_rows
        .last_mut()
        .expect("wide context has rows");
    // The last vector item already belongs to 110030 at the latest anchor.
    last.stock_volatility = 10.0;
    for index in 0..12_i64 {
        let mut row = intraday_row(
            ctx.score_date,
            target,
            clock_ns(9, 30 + index * 5, 0, 0),
            index,
            100.0 + index as f64,
            index as f64,
            index as f64,
            if index == 0 {
                0.0
            } else if index == 1 {
                20.0
            } else {
                20.0 + (index - 1) as f64
            },
        );
        row.ask_price[0] += index as f64 * 0.01;
        row.bid_price[0] += index as f64 * 0.01;
        ctx.intraday_rows.push(row);
    }
    let value = isgm_stockvol_trade_quote_clock_center_gap(&ctx, target).unwrap();
    assert!(value.is_finite());
    assert!(
        value < 0.0,
        "early trade mass should precede uniformly revised quotes: {value}"
    );
}

#[test]
fn qgeo_alignment_rejects_after_cutoff_and_keeps_same_session_next_return_sign() {
    let mut ctx = TypedFactorR88RemainingContext::new(day(2));
    for index in 0..8_i64 {
        ctx.intraday_rows.push(intraday_row(
            ctx.score_date,
            TARGET,
            clock_ns(9, 30, index, 0),
            index,
            100.0 + index as f64,
            index as f64,
            index as f64,
            index as f64,
        ));
    }
    let baseline = qgeo_micro_last_next_return_sign_alignment(&ctx, TARGET);
    assert_close(baseline, 1.0);
    ctx.intraday_rows.push(intraday_row(
        ctx.score_date,
        TARGET,
        clock_ns(14, 30, 0, 0),
        99,
        1.0,
        1.0,
        1.0,
        1.0,
    ));
    assert_eq!(
        qgeo_micro_last_next_return_sign_alignment(&ctx, TARGET).to_bits(),
        baseline.to_bits()
    );
}

#[test]
fn csn_churn_neighbor_gap_uses_same_day_ranked_microstructure_cross_section() {
    let mut ctx = TypedFactorR88RemainingContext::new(day(5));
    for code_index in 0..30_i64 {
        let code = format!("11{code_index:04}.SH");
        ctx.panel_codes.push(code.clone());
        let mut cumulative_trades = 0.0;
        for step in 0..12_i64 {
            if step > 0 {
                let period = 3 + code_index % 3;
                if (step + code_index) % period != 0 {
                    cumulative_trades += 1.0 + ((step + code_index) % 3) as f64;
                }
            }
            let mut row = intraday_row(
                ctx.score_date,
                &code,
                clock_ns(9, 30, step, 0),
                step,
                100.0,
                0.0,
                0.0,
                cumulative_trades,
            );
            row.ask_volume[0] = 10.0
                + code_index as f64 * 0.11
                + step as f64 * (0.10 + (code_index % 5) as f64 * 0.02);
            row.bid_volume[0] = 12.0
                + code_index as f64 * 0.07
                + step as f64 * (0.06 + (code_index % 4) as f64 * 0.015);
            ctx.intraday_rows.push(row);
        }
    }
    let value = csn_pql_churn_neighbor_gap(&ctx, "110000.SH");
    assert!(
        value.is_finite(),
        "expected finite local rank gap, got {value:?}"
    );
}

/// Structural performance guard: the expensive cross-sectional state is
/// prepared once per requested signal, then every output code is read through
/// the immutable cache.  The direct functions remain the exact reference.
#[test]
fn prepared_cross_section_cache_matches_direct_values_bitwise_or_nan() {
    let daily = wide_daily_context();
    let daily_cache = prepare_r88_remaining_values(
        &daily,
        [
            TypedFactorR88RemainingSignal::UcdPeerStockReturnDispersion1,
            TypedFactorR88RemainingSignal::SngPeerReturnDispersion1,
        ],
    )
    .expect("daily cache builds");
    for code in &daily.panel_codes {
        let ucd_direct = ucd_peer_stock_return_dispersion1(&daily, code).unwrap();
        let ucd_cached = daily_cache
            .lookup(
                TypedFactorR88RemainingSignal::UcdPeerStockReturnDispersion1,
                code,
            )
            .expect("requested UCD signal is cached");
        assert_same_bits_or_nan(ucd_cached, ucd_direct);

        let sng_direct = sng_peer_return_dispersion1(&daily, code).unwrap();
        let sng_cached = daily_cache
            .lookup(
                TypedFactorR88RemainingSignal::SngPeerReturnDispersion1,
                code,
            )
            .expect("requested SNG signal is cached");
        assert_same_bits_or_nan(sng_cached, sng_direct);
    }

    let mut csn = TypedFactorR88RemainingContext::new(day(5));
    for code_index in 0..30_i64 {
        let code = format!("11{code_index:04}.SH");
        csn.panel_codes.push(code.clone());
        let mut cumulative_trades = 0.0;
        for step in 0..12_i64 {
            if step > 0 {
                let period = 3 + code_index % 3;
                if (step + code_index) % period != 0 {
                    cumulative_trades += 1.0 + ((step + code_index) % 3) as f64;
                }
            }
            let mut row = intraday_row(
                csn.score_date,
                &code,
                clock_ns(9, 30, step, 0),
                step,
                100.0,
                0.0,
                0.0,
                cumulative_trades,
            );
            row.ask_volume[0] = 10.0
                + code_index as f64 * 0.11
                + step as f64 * (0.10 + (code_index % 5) as f64 * 0.02);
            row.bid_volume[0] = 12.0
                + code_index as f64 * 0.07
                + step as f64 * (0.06 + (code_index % 4) as f64 * 0.015);
            csn.intraday_rows.push(row);
        }
    }
    let csn_cache = prepare_r88_remaining_values(
        &csn,
        [TypedFactorR88RemainingSignal::CsnPqlChurnNeighborGap],
    )
    .expect("CSN cache builds");
    for code in &csn.panel_codes {
        let direct = csn_pql_churn_neighbor_gap(&csn, code);
        let cached = csn_cache
            .lookup(TypedFactorR88RemainingSignal::CsnPqlChurnNeighborGap, code)
            .expect("requested CSN signal is cached");
        assert_same_bits_or_nan(cached, direct);
    }
}

#[test]
fn prepared_joint_and_isgm_caches_match_direct_values_bitwise_or_nan() {
    let mut joint = joint_context();
    joint.panel_codes.push(TARGET.to_string());
    let joint_cache = prepare_r88_remaining_values(
        &joint,
        [
            TypedFactorR88RemainingSignal::JointTailSignedCojump,
            TypedFactorR88RemainingSignal::JointTailRangeCoexpansion,
            TypedFactorR88RemainingSignal::JointTailTerminalLocationCoshock,
        ],
    )
    .expect("joint cache builds");
    for (signal, direct) in [
        (
            TypedFactorR88RemainingSignal::JointTailSignedCojump,
            joint_tail_signed_cojump(&joint, TARGET).unwrap(),
        ),
        (
            TypedFactorR88RemainingSignal::JointTailRangeCoexpansion,
            joint_tail_range_coexpansion(&joint, TARGET).unwrap(),
        ),
        (
            TypedFactorR88RemainingSignal::JointTailTerminalLocationCoshock,
            joint_tail_terminal_location_coshock(&joint, TARGET).unwrap(),
        ),
    ] {
        let cached = joint_cache
            .lookup(signal, TARGET)
            .expect("requested joint signal is cached");
        assert_same_bits_or_nan(cached, direct);
    }

    let mut isgm = wide_daily_context();
    isgm.score_date = day(30);
    let target = "110030.SH";
    let last = isgm
        .daily_base_rows
        .last_mut()
        .expect("wide context has rows");
    last.stock_volatility = 10.0;
    for index in 0..12_i64 {
        let mut row = intraday_row(
            isgm.score_date,
            target,
            clock_ns(9, 30 + index * 5, 0, 0),
            index,
            100.0 + index as f64,
            index as f64,
            index as f64,
            if index == 0 {
                0.0
            } else if index == 1 {
                20.0
            } else {
                20.0 + (index - 1) as f64
            },
        );
        row.ask_price[0] += index as f64 * 0.01;
        row.bid_price[0] += index as f64 * 0.01;
        isgm.intraday_rows.push(row);
    }
    let isgm_cache = prepare_r88_remaining_values(
        &isgm,
        [TypedFactorR88RemainingSignal::IsgmStockvolTradeQuoteClockCenterGap],
    )
    .expect("ISGM cache builds");
    let cached = isgm_cache
        .lookup(
            TypedFactorR88RemainingSignal::IsgmStockvolTradeQuoteClockCenterGap,
            target,
        )
        .expect("requested ISGM signal is cached");
    assert_same_bits_or_nan(
        cached,
        isgm_stockvol_trade_quote_clock_center_gap(&isgm, target).unwrap(),
    );
}

#[test]
fn prepared_hybrid_itr_and_qgeo_caches_match_direct_values_bitwise_or_nan() {
    let mut hybrid = hybrid_context();
    hybrid.panel_codes.push(TARGET.to_string());
    let hybrid_cache = prepare_r88_remaining_values(
        &hybrid,
        [
            TypedFactorR88RemainingSignal::HybridCurrentRangeVsHistTwapCurve,
            TypedFactorR88RemainingSignal::HybridCurrentFlowVsHistOvernightResponse,
        ],
    )
    .expect("hybrid cache builds");
    for (signal, direct) in [
        (
            TypedFactorR88RemainingSignal::HybridCurrentRangeVsHistTwapCurve,
            hybrid_current_range_vs_hist_twap_curve(&hybrid, TARGET).unwrap(),
        ),
        (
            TypedFactorR88RemainingSignal::HybridCurrentFlowVsHistOvernightResponse,
            hybrid_current_flow_vs_hist_overnight_response(&hybrid, TARGET).unwrap(),
        ),
    ] {
        let cached = hybrid_cache
            .lookup(signal, TARGET)
            .expect("requested hybrid signal is cached");
        assert_same_bits_or_nan(cached, direct);
    }

    let mut itr = TypedFactorR88RemainingContext::new(day(1));
    itr.panel_codes.push(TARGET.to_string());
    itr.bond_stock_map.push(R88RemainingBondStockMapRow {
        code: TARGET.to_string(),
        stock_code: "600000.SH".to_string(),
        as_of_date: itr.score_date,
    });
    let mut bond = 100.0_f64;
    let mut stock = 50.0_f64;
    for index in 0..14_i64 {
        if index > 0 {
            let delta: f64 = if index % 2 == 0 { 0.003 } else { -0.002 };
            bond *= delta.exp();
            stock *= delta.exp();
        }
        let time = clock_ns(9, 30 + index * 5, 0, 0);
        itr.intraday_rows.push(intraday_row(
            itr.score_date,
            TARGET,
            time,
            index,
            bond,
            0.0,
            0.0,
            0.0,
        ));
        itr.intraday_rows.push(intraday_row(
            itr.score_date,
            "600000.SH",
            time,
            index,
            stock,
            0.0,
            0.0,
            0.0,
        ));
    }
    let itr_cache = prepare_r88_remaining_values(
        &itr,
        [TypedFactorR88RemainingSignal::ItrStockShockSameBinDirectionalAgreement],
    )
    .expect("ITR cache builds");
    assert_same_bits_or_nan(
        itr_cache
            .lookup(
                TypedFactorR88RemainingSignal::ItrStockShockSameBinDirectionalAgreement,
                TARGET,
            )
            .expect("requested ITR signal is cached"),
        itr_stock_shock_same_bin_directional_agreement(&itr, TARGET).unwrap(),
    );

    let mut qgeo = TypedFactorR88RemainingContext::new(day(2));
    qgeo.panel_codes.push(TARGET.to_string());
    for index in 0..8_i64 {
        qgeo.intraday_rows.push(intraday_row(
            qgeo.score_date,
            TARGET,
            clock_ns(9, 30, index, 0),
            index,
            100.0 + index as f64,
            index as f64,
            index as f64,
            index as f64,
        ));
    }
    let qgeo_cache = prepare_r88_remaining_values(
        &qgeo,
        [TypedFactorR88RemainingSignal::QgeoMicroLastNextReturnSignAlignment],
    )
    .expect("QGEO cache builds");
    assert_same_bits_or_nan(
        qgeo_cache
            .lookup(
                TypedFactorR88RemainingSignal::QgeoMicroLastNextReturnSignAlignment,
                TARGET,
            )
            .expect("requested QGEO signal is cached"),
        qgeo_micro_last_next_return_sign_alignment(&qgeo, TARGET),
    );
}

#[test]
fn signal_dispatch_and_fail_closed_errors_are_preserved() {
    let ctx = hybrid_context();
    assert_close(
        compute_r88_remaining_signal(&ctx, TARGET, "hybrid_current_range_vs_hist_twap_curve")
            .unwrap(),
        hybrid_current_range_vs_hist_twap_curve(&ctx, TARGET).unwrap(),
    );
    assert!(matches!(
        compute_r88_remaining_signal(&ctx, TARGET, "not_a_signal"),
        Err(TypedFactorR88RemainingError::UnknownSignal(_))
    ));
    let mut duplicated = ctx.clone();
    duplicated
        .daily_twap_rows
        .push(duplicated.daily_twap_rows[0].clone());
    assert!(matches!(
        hybrid_current_range_vs_hist_twap_curve(&duplicated, TARGET),
        Err(TypedFactorR88RemainingError::DuplicateStrictPriorDate { .. })
    ));
}
