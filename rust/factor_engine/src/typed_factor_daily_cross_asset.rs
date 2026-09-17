//! Strict-prior cross-asset daily kernels for typed factor contracts.
//!
//! This module is reached internally from the one public Rust factor API. Its
//! typed inputs retain raw daily code/exchange fields so source normalization
//! stays exact before deriving either factor. `NaiveDate` inputs represent
//! dates already parsed by the caller; both kernels enforce
//! `trade_date < score_date`.

use chrono::NaiveDate;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const TYPED_FACTOR_CROSS_ASSET_WINDOW: usize = 60;
pub const TYPED_FACTOR_CROSS_ASSET_MIN_OBSERVATIONS: usize = 45;
pub const TYPED_FACTOR_CROSS_ASSET_MIN_TAIL_OBSERVATIONS: usize = 8;
pub const TYPED_FACTOR_CROSS_ASSET_EPS: f64 = 1e-12;
const JEFFREYS_PSEUDOCOUNT: f64 = 0.5;
const LOWER_TAIL: f64 = 0.25;
const UPPER_TAIL: f64 = 0.75;

/// One raw `market_cbond.daily_price` source row needed by both kernels.
#[derive(Clone, Debug)]
pub struct TypedFactorCrossAssetPriceRow {
    pub trade_date: NaiveDate,
    pub code: String,
    pub exchange_code: String,
    pub prev_close_price: f64,
    pub close_price: f64,
    pub amount: f64,
}

/// One raw `market_cbond.daily_base` source row needed by both kernels.
#[derive(Clone, Debug)]
pub struct TypedFactorCrossAssetBaseRow {
    pub trade_date: NaiveDate,
    pub code: String,
    pub exchange_code: String,
    pub stock_code: String,
    pub stk_prev_close_price: f64,
    pub stk_close_price: f64,
}

/// Strict-prior source material for one score date.
#[derive(Clone, Debug)]
pub struct TypedFactorDailyCrossAssetContext {
    pub score_date: NaiveDate,
    pub price_rows: Vec<TypedFactorCrossAssetPriceRow>,
    pub base_rows: Vec<TypedFactorCrossAssetBaseRow>,
}

impl TypedFactorDailyCrossAssetContext {
    pub fn new(score_date: NaiveDate) -> Self {
        Self {
            score_date,
            price_rows: Vec::new(),
            base_rows: Vec::new(),
        }
    }
}

/// Fail-closed source-contract errors for the isolated kernel implementation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypedFactorDailyCrossAssetError {
    DuplicateStrictPriorDate {
        source: &'static str,
        code: String,
        trade_date: NaiveDate,
    },
    InconsistentUnderlyingReturn {
        stock_code: String,
        trade_date: NaiveDate,
    },
}

impl fmt::Display for TypedFactorDailyCrossAssetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateStrictPriorDate {
                source,
                code,
                trade_date,
            } => write!(
                f,
                "typed_factor cross-asset {source} has duplicate strict-prior row for {code} on {trade_date}"
            ),
            Self::InconsistentUnderlyingReturn {
                stock_code,
                trade_date,
            } => write!(
                f,
                "typed_factor cross-asset has inconsistent strict-prior underlying return for {stock_code} on {trade_date}"
            ),
        }
    }
}

impl Error for TypedFactorDailyCrossAssetError {}

#[derive(Clone, Debug)]
struct PriceRow {
    trade_date: NaiveDate,
    code: String,
    prev_close_price: f64,
    close_price: f64,
    amount: f64,
}

#[derive(Clone, Debug)]
struct BaseRow {
    trade_date: NaiveDate,
    code: String,
    stock_code: String,
    stk_prev_close_price: f64,
    stk_close_price: f64,
}

#[derive(Clone, Debug)]
struct PreparedSources {
    price: Vec<PriceRow>,
    base: Vec<BaseRow>,
    sessions: Vec<NaiveDate>,
    anchor: NaiveDate,
}

#[derive(Clone, Copy, Debug)]
struct FlowRow {
    trade_date: NaiveDate,
    stock_previous_close: f64,
    stock_close: f64,
    amount: f64,
}

fn canonical_exchange(value: &str) -> String {
    let raw = value.trim().to_ascii_uppercase();
    match raw.as_str() {
        "XSHG" | "SHSE" => "SH".to_string(),
        "XSHE" | "SZSE" => "SZ".to_string(),
        "BSE" | "BJSE" => "BJ".to_string(),
        _ => raw,
    }
}

fn market_exchange(value: &str) -> bool {
    matches!(value, "SH" | "SZ" | "BJ")
}

/// Match Python `_canonical_market_code` for a raw daily field.
fn canonical_market_code(value: &str, exchange: &str) -> String {
    let mut text = value.trim().to_ascii_uppercase();
    if text.is_empty() || text == "NAN" {
        return String::new();
    }
    if text.ends_with(".0") {
        text.truncate(text.len() - 2);
    }
    if let Some((bare, raw_suffix)) = text.rsplit_once('.') {
        let suffix = canonical_exchange(raw_suffix);
        if !bare.is_empty() && market_exchange(&suffix) {
            return format!("{bare}.{suffix}");
        }
    }
    let suffix = canonical_exchange(exchange);
    if market_exchange(&suffix) {
        format!("{text}.{suffix}")
    } else {
        String::new()
    }
}

fn strict_price_rows(
    ctx: &TypedFactorDailyCrossAssetContext,
) -> Result<Vec<PriceRow>, TypedFactorDailyCrossAssetError> {
    let mut rows: Vec<PriceRow> = ctx
        .price_rows
        .iter()
        .filter(|row| row.trade_date < ctx.score_date)
        .filter_map(|row| {
            let code = canonical_market_code(&row.code, &row.exchange_code);
            (!code.is_empty()).then(|| PriceRow {
                trade_date: row.trade_date,
                code,
                prev_close_price: row.prev_close_price,
                close_price: row.close_price,
                amount: row.amount,
            })
        })
        .collect();
    rows.sort_by(|left, right| {
        left.trade_date
            .cmp(&right.trade_date)
            .then_with(|| left.code.cmp(&right.code))
    });
    for pair in rows.windows(2) {
        if pair[0].trade_date == pair[1].trade_date && pair[0].code == pair[1].code {
            return Err(TypedFactorDailyCrossAssetError::DuplicateStrictPriorDate {
                source: "market_cbond.daily_price",
                code: pair[0].code.clone(),
                trade_date: pair[0].trade_date,
            });
        }
    }
    Ok(rows)
}

fn strict_base_rows(
    ctx: &TypedFactorDailyCrossAssetContext,
) -> Result<Vec<BaseRow>, TypedFactorDailyCrossAssetError> {
    let mut rows: Vec<BaseRow> = ctx
        .base_rows
        .iter()
        .filter(|row| row.trade_date < ctx.score_date)
        .filter_map(|row| {
            let code = canonical_market_code(&row.code, &row.exchange_code);
            (!code.is_empty()).then(|| BaseRow {
                trade_date: row.trade_date,
                code,
                stock_code: canonical_market_code(&row.stock_code, &row.exchange_code),
                stk_prev_close_price: row.stk_prev_close_price,
                stk_close_price: row.stk_close_price,
            })
        })
        .collect();
    rows.sort_by(|left, right| {
        left.trade_date
            .cmp(&right.trade_date)
            .then_with(|| left.code.cmp(&right.code))
    });
    for pair in rows.windows(2) {
        if pair[0].trade_date == pair[1].trade_date && pair[0].code == pair[1].code {
            return Err(TypedFactorDailyCrossAssetError::DuplicateStrictPriorDate {
                source: "market_cbond.daily_base",
                code: pair[0].code.clone(),
                trade_date: pair[0].trade_date,
            });
        }
    }
    Ok(rows)
}

/// Normalize both complete sources before checking the requested output code.
/// This preserves Python's global duplicate error surface.
fn prepared_sources(
    ctx: &TypedFactorDailyCrossAssetContext,
) -> Result<Option<PreparedSources>, TypedFactorDailyCrossAssetError> {
    let price = strict_price_rows(ctx)?;
    let base = strict_base_rows(ctx)?;
    if price.is_empty() || base.is_empty() {
        return Ok(None);
    }
    let price_anchor = price
        .iter()
        .map(|row| row.trade_date)
        .max()
        .expect("nonempty price source has an anchor");
    let base_anchor = base
        .iter()
        .map(|row| row.trade_date)
        .max()
        .expect("nonempty base source has an anchor");
    if price_anchor != base_anchor {
        return Ok(None);
    }
    let sessions: BTreeSet<NaiveDate> = price
        .iter()
        .map(|row| row.trade_date)
        .chain(base.iter().map(|row| row.trade_date))
        .collect();
    Ok(Some(PreparedSources {
        price,
        base,
        sessions: sessions.into_iter().collect(),
        anchor: price_anchor,
    }))
}

#[inline]
fn finite_log_return(close: f64, previous_close: f64) -> f64 {
    if close.is_finite()
        && previous_close.is_finite()
        && close > TYPED_FACTOR_CROSS_ASSET_EPS
        && previous_close > TYPED_FACTOR_CROSS_ASSET_EPS
    {
        (close / previous_close).ln()
    } else {
        f64::NAN
    }
}

#[inline]
fn log_positive(value: f64) -> f64 {
    if value.is_finite() && value > TYPED_FACTOR_CROSS_ASSET_EPS {
        value.ln()
    } else {
        f64::NAN
    }
}

fn adjacent_change(values: &[f64]) -> Vec<f64> {
    let mut output = vec![f64::NAN; values.len()];
    for index in 1..values.len() {
        if values[index].is_finite() && values[index - 1].is_finite() {
            output[index] = values[index] - values[index - 1];
        }
    }
    output
}

fn sign3(value: f64) -> Option<usize> {
    if !value.is_finite() {
        None
    } else if value < -TYPED_FACTOR_CROSS_ASSET_EPS {
        Some(0)
    } else if value.abs() <= TYPED_FACTOR_CROSS_ASSET_EPS {
        Some(1)
    } else {
        Some(2)
    }
}

fn normalized_mutual_information_3state(stock_return: &[f64], amount_change: &[f64]) -> f64 {
    if stock_return.len() != amount_change.len() || stock_return.is_empty() {
        return f64::NAN;
    }
    if sign3(*stock_return.last().expect("nonempty checked above")).is_none()
        || sign3(*amount_change.last().expect("nonempty checked above")).is_none()
    {
        return f64::NAN;
    }

    let mut counts = [[JEFFREYS_PSEUDOCOUNT; 3]; 3];
    let mut observed = 0usize;
    for (stock, amount) in stock_return.iter().zip(amount_change.iter()) {
        if let (Some(stock_state), Some(amount_state)) = (sign3(*stock), sign3(*amount)) {
            counts[stock_state][amount_state] += 1.0;
            observed += 1;
        }
    }
    if observed < TYPED_FACTOR_CROSS_ASSET_MIN_OBSERVATIONS {
        return f64::NAN;
    }

    let total = counts.iter().flatten().sum::<f64>();
    let mut row_probability = [0.0; 3];
    let mut column_probability = [0.0; 3];
    for row in 0..3 {
        for column in 0..3 {
            let probability = counts[row][column] / total;
            row_probability[row] += probability;
            column_probability[column] += probability;
        }
    }
    let mut information = 0.0;
    for row in 0..3 {
        for column in 0..3 {
            let probability = counts[row][column] / total;
            information += probability
                * (probability / (row_probability[row] * column_probability[column])).ln();
        }
    }
    let result = information / 3.0_f64.ln();
    if result.is_finite() {
        result
    } else {
        f64::NAN
    }
}

fn flow_history_for_code(sources: &PreparedSources, code: &str) -> Option<Vec<FlowRow>> {
    let base_by_key: BTreeMap<(NaiveDate, String), &BaseRow> = sources
        .base
        .iter()
        .map(|row| ((row.trade_date, row.code.clone()), row))
        .collect();
    let mut history = Vec::new();
    for price in sources.price.iter().filter(|row| row.code == code) {
        if let Some(base) = base_by_key.get(&(price.trade_date, code.to_string())) {
            history.push(FlowRow {
                trade_date: price.trade_date,
                stock_previous_close: base.stk_prev_close_price,
                stock_close: base.stk_close_price,
                amount: price.amount,
            });
        }
    }
    history.sort_by_key(|row| row.trade_date);
    (history.last().map(|row| row.trade_date) == Some(sources.anchor)).then_some(history)
}

/// `bsfst_stock_return_bond_flow_mutual_information60`.
///
/// The requested code is interpreted exactly like Python's panel key: it must
/// already contain a valid exchange suffix (for example `110001.SH`).  Source
/// fields are normalized with their own `exchange_code` first.
pub fn bsfst_stock_return_bond_flow_mutual_information60(
    ctx: &TypedFactorDailyCrossAssetContext,
    raw_panel_code: &str,
) -> Result<f64, TypedFactorDailyCrossAssetError> {
    let Some(sources) = prepared_sources(ctx)? else {
        return Ok(f64::NAN);
    };
    let code = canonical_market_code(raw_panel_code, "");
    if code.is_empty() {
        return Ok(f64::NAN);
    }
    let Some(history) = flow_history_for_code(&sources, &code) else {
        return Ok(f64::NAN);
    };

    let by_date: BTreeMap<NaiveDate, &FlowRow> =
        history.iter().map(|row| (row.trade_date, row)).collect();
    let first = sources
        .sessions
        .len()
        .saturating_sub(TYPED_FACTOR_CROSS_ASSET_WINDOW);
    let mut stock_return = Vec::with_capacity(sources.sessions.len() - first);
    let mut amount_log = Vec::with_capacity(sources.sessions.len() - first);
    for session in sources.sessions.iter().skip(first) {
        if let Some(row) = by_date.get(session) {
            stock_return.push(finite_log_return(row.stock_close, row.stock_previous_close));
            amount_log.push(log_positive(row.amount));
        } else {
            stock_return.push(f64::NAN);
            amount_log.push(f64::NAN);
        }
    }
    Ok(normalized_mutual_information_3state(
        &stock_return,
        &adjacent_change(&amount_log),
    ))
}

/// Python uses `rank(method="average", pct=True)`: infinite values still rank,
/// while only `NaN` remains missing.
fn average_percentile_rank(values: Vec<(String, f64)>) -> BTreeMap<String, f64> {
    let mut ranked: Vec<(String, f64)> = values
        .into_iter()
        .filter(|(_, value)| !value.is_nan())
        .collect();
    ranked.sort_by(|left, right| {
        left.1
            .partial_cmp(&right.1)
            .expect("NaN values were excluded before rank ordering")
    });
    let denominator = ranked.len() as f64;
    let mut output = BTreeMap::new();
    let mut start = 0usize;
    while start < ranked.len() {
        let mut end = start + 1;
        while end < ranked.len() && ranked[end].1 == ranked[start].1 {
            end += 1;
        }
        // One-based ranks are `start + 1 ..= end`; their average divided by N
        // is Pandas' `pct=True, method="average"` result.
        let average_rank = (start + 1 + end) as f64 / 2.0;
        let percentile = average_rank / denominator;
        for (code, _) in &ranked[start..end] {
            output.insert(code.clone(), percentile);
        }
        start = end;
    }
    output
}

/// Unlike bsFST's finite-return guard, Python's rank kernel permits +/-Inf in
/// the bond-return ranking and leaves only NaN unranked.
fn rank_return(close: f64, previous_close: f64) -> f64 {
    if close > TYPED_FACTOR_CROSS_ASSET_EPS && previous_close > TYPED_FACTOR_CROSS_ASSET_EPS {
        (close / previous_close).ln()
    } else {
        f64::NAN
    }
}

fn bond_ranks(price: &[PriceRow]) -> BTreeMap<(NaiveDate, String), f64> {
    let mut by_date: BTreeMap<NaiveDate, Vec<(String, f64)>> = BTreeMap::new();
    for row in price {
        by_date.entry(row.trade_date).or_default().push((
            row.code.clone(),
            rank_return(row.close_price, row.prev_close_price),
        ));
    }
    let mut output = BTreeMap::new();
    for (trade_date, values) in by_date {
        for (code, rank) in average_percentile_rank(values) {
            output.insert((trade_date, code), rank);
        }
    }
    output
}

fn stock_ranks_by_bond(
    base: &[BaseRow],
) -> Result<BTreeMap<(NaiveDate, String), f64>, TypedFactorDailyCrossAssetError> {
    let mut by_stock: BTreeMap<(NaiveDate, String), Vec<&BaseRow>> = BTreeMap::new();
    for row in base.iter().filter(|row| !row.stock_code.is_empty()) {
        by_stock
            .entry((row.trade_date, row.stock_code.clone()))
            .or_default()
            .push(row);
    }

    let mut stock_values: BTreeMap<NaiveDate, Vec<(String, f64)>> = BTreeMap::new();
    for ((trade_date, stock_code), rows) in by_stock {
        let returns: Vec<f64> = rows
            .iter()
            .map(|row| rank_return(row.stk_close_price, row.stk_prev_close_price))
            .collect();
        let finite: Vec<f64> = returns
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        let stock_return = if finite.len() != returns.len() || finite.is_empty() {
            f64::NAN
        } else {
            let minimum = finite.iter().copied().fold(f64::INFINITY, f64::min);
            let maximum = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if maximum - minimum <= TYPED_FACTOR_CROSS_ASSET_EPS {
                finite[0]
            } else {
                return Err(
                    TypedFactorDailyCrossAssetError::InconsistentUnderlyingReturn {
                        stock_code,
                        trade_date,
                    },
                );
            }
        };
        stock_values
            .entry(trade_date)
            .or_default()
            .push((stock_code, stock_return));
    }

    let mut stock_rank: BTreeMap<(NaiveDate, String), f64> = BTreeMap::new();
    for (trade_date, values) in stock_values {
        for (stock_code, rank) in average_percentile_rank(values) {
            stock_rank.insert((trade_date, stock_code), rank);
        }
    }

    let mut output = BTreeMap::new();
    for row in base.iter().filter(|row| !row.stock_code.is_empty()) {
        if let Some(rank) = stock_rank.get(&(row.trade_date, row.stock_code.clone())) {
            output.insert((row.trade_date, row.code.clone()), *rank);
        }
    }
    Ok(output)
}

/// Compute the complete three-column BSSRC metric record in one pass.  Python
/// calculates every family member together, so the common denominator and both
/// tail-observation gates apply even when a caller requests only one tail.
fn rank_metrics(path: &[(NaiveDate, f64, f64)]) -> (f64, f64, f64) {
    let first = path.len().saturating_sub(TYPED_FACTOR_CROSS_ASSET_WINDOW);
    let recent = &path[first..];
    let Some((_, terminal_bond, terminal_stock)) = recent.last() else {
        return (f64::NAN, f64::NAN, f64::NAN);
    };
    if !terminal_bond.is_finite() || !terminal_stock.is_finite() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let pairs: Vec<(f64, f64)> = recent
        .iter()
        .filter_map(|(_, bond, stock)| {
            (bond.is_finite() && stock.is_finite()).then_some((*bond, *stock))
        })
        .collect();
    if pairs.len() < TYPED_FACTOR_CROSS_ASSET_MIN_OBSERVATIONS {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mean_bond = pairs.iter().map(|(bond, _)| bond).sum::<f64>() / pairs.len() as f64;
    let mean_stock = pairs.iter().map(|(_, stock)| stock).sum::<f64>() / pairs.len() as f64;
    let mut bond_square = 0.0;
    let mut stock_square = 0.0;
    let mut product = 0.0;
    let mut upper_stock = 0usize;
    let mut lower_stock = 0usize;
    for (bond, stock) in &pairs {
        let centered_bond = *bond - mean_bond;
        let centered_stock = *stock - mean_stock;
        bond_square += centered_bond * centered_bond;
        stock_square += centered_stock * centered_stock;
        product += centered_bond * centered_stock;
        if *stock >= UPPER_TAIL {
            upper_stock += 1;
        }
        if *stock <= LOWER_TAIL {
            lower_stock += 1;
        }
    }
    let denominator = (bond_square * stock_square).sqrt();
    if !denominator.is_finite()
        || denominator <= TYPED_FACTOR_CROSS_ASSET_EPS
        || upper_stock < TYPED_FACTOR_CROSS_ASSET_MIN_TAIL_OBSERVATIONS
        || lower_stock < TYPED_FACTOR_CROSS_ASSET_MIN_TAIL_OBSERVATIONS
    {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let correlation = product / denominator;
    let upper_alignment = pairs
        .iter()
        .filter(|(_, stock)| *stock >= UPPER_TAIL)
        .filter(|(bond, _)| *bond >= UPPER_TAIL)
        .count() as f64
        / upper_stock as f64;
    let lower_alignment = pairs
        .iter()
        .filter(|(_, stock)| *stock <= LOWER_TAIL)
        .filter(|(bond, _)| *bond <= LOWER_TAIL)
        .count() as f64
        / lower_stock as f64;
    if correlation.is_finite() && upper_alignment.is_finite() && lower_alignment.is_finite() {
        (correlation, upper_alignment, lower_alignment)
    } else {
        (f64::NAN, f64::NAN, f64::NAN)
    }
}

fn rank_correlation(path: &[(NaiveDate, f64, f64)]) -> f64 {
    rank_metrics(path).0
}

fn bssrc_path(
    ctx: &TypedFactorDailyCrossAssetContext,
    raw_panel_code: &str,
) -> Result<Option<Vec<(NaiveDate, f64, f64)>>, TypedFactorDailyCrossAssetError> {
    let Some(sources) = prepared_sources(ctx)? else {
        return Ok(None);
    };
    // Build global stock ranks before inspecting the requested panel code: the
    // Python reference raises inconsistent shared-underlying input globally.
    let bond_rank = bond_ranks(&sources.price);
    let stock_rank = stock_ranks_by_bond(&sources.base)?;
    let code = canonical_market_code(raw_panel_code, "");
    if code.is_empty() {
        return Ok(None);
    }

    let mut path = Vec::new();
    for price in sources.price.iter().filter(|row| row.code == code) {
        let key = (price.trade_date, code.clone());
        // Match Python's inner join of bond and mapped-stock ranks: an empty
        // underlying mapping removes the bond/day before the 60-row tail.
        if let Some(stock_rank) = stock_rank.get(&key) {
            path.push((
                price.trade_date,
                bond_rank.get(&key).copied().unwrap_or(f64::NAN),
                *stock_rank,
            ));
        }
    }
    path.sort_by_key(|(trade_date, _, _)| *trade_date);
    if path.last().map(|(trade_date, _, _)| *trade_date) != Some(sources.anchor) {
        return Ok(None);
    }
    Ok(Some(path))
}

/// `bssrc_bond_stock_rank_correlation60`.
///
/// The pre-selected correlation retains the Python family gate: the latest
/// 60-row path must also contain at least eight finite stock ranks in *both*
/// the lower and upper tails, even though the two tail-alignment outputs are
/// not returned here.
pub fn bssrc_bond_stock_rank_correlation60(
    ctx: &TypedFactorDailyCrossAssetContext,
    raw_panel_code: &str,
) -> Result<f64, TypedFactorDailyCrossAssetError> {
    Ok(bssrc_path(ctx, raw_panel_code)
        .map(|path| path.map_or(f64::NAN, |path| rank_correlation(&path)))?)
}

/// `bssrc_upper_rank_tail_alignment60`: conditional probability that a bond
/// reaches its upper cross-sectional return tail when its mapped stock does.
pub fn bssrc_upper_rank_tail_alignment60(
    ctx: &TypedFactorDailyCrossAssetContext,
    raw_panel_code: &str,
) -> Result<f64, TypedFactorDailyCrossAssetError> {
    Ok(bssrc_path(ctx, raw_panel_code)
        .map(|path| path.map_or(f64::NAN, |path| rank_metrics(&path).1))?)
}

/// `bssrc_lower_rank_tail_alignment60`: conditional probability that a bond
/// reaches its lower cross-sectional return tail when its mapped stock does.
pub fn bssrc_lower_rank_tail_alignment60(
    ctx: &TypedFactorDailyCrossAssetContext,
    raw_panel_code: &str,
) -> Result<f64, TypedFactorDailyCrossAssetError> {
    Ok(bssrc_path(ctx, raw_panel_code)
        .map(|path| path.map_or(f64::NAN, |path| rank_metrics(&path).2))?)
}
