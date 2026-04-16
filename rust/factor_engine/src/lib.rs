use chrono::NaiveDate;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{BTreeSet, HashMap};
use std::time::Instant;

const EPS: f64 = 1e-8;

#[derive(Clone, Debug)]
struct Group {
    start: usize,
    end: usize,
    dt: String,
    code: String,
}

#[derive(Clone)]
struct PanelData {
    dt: Vec<String>,
    code: Vec<String>,
    seq: Vec<i64>,
    trade_time_ns: Vec<i64>,
    cols: HashMap<String, Vec<f64>>,
    groups: Vec<Group>,
}

impl PanelData {
    fn len(&self) -> usize {
        self.dt.len()
    }

    fn col(&self, name: &str) -> PyResult<&Vec<f64>> {
        self.cols.get(name).ok_or_else(|| {
            PyErr::new::<PyKeyError, _>(format!("panel missing numeric column: {}", name))
        })
    }
}

#[derive(Clone)]
struct FactorSpec {
    name: String,
    factor: String,
    output_col: String,
    params: HashMap<String, ParamValue>,
}

#[derive(Clone, Debug)]
struct PlanLimits {
    max_windows: usize,
    max_levels: usize,
    max_time_ranges: usize,
    log_summary: bool,
}

impl Default for PlanLimits {
    fn default() -> Self {
        Self {
            max_windows: 8,
            max_levels: 8,
            max_time_ranges: 8,
            log_summary: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct FactorPlan {
    windows: Vec<i64>,
    levels: Vec<usize>,
    time_ranges: Vec<(String, String)>,
}

#[derive(Clone, Debug, Default)]
struct WindowStartCache {
    starts_by_window: HashMap<i64, Vec<usize>>,
}

#[derive(Clone, Debug)]
struct LatestQuote {
    last: f64,
    open: f64,
    ask_price1: f64,
    bid_price1: f64,
    prev_bar_close: f64,
    pre_close: f64,
}

#[derive(Clone, Default)]
struct DailySourceData {
    by_code: HashMap<String, Vec<DailyPoint>>,
}

#[derive(Clone)]
struct DailyPoint {
    date: String,
    values: HashMap<String, f64>,
}

#[derive(Clone, Default)]
struct AuxData {
    stock_latest: HashMap<(String, String), LatestQuote>,
    bond_stock_map: HashMap<String, String>,
    daily_sources: HashMap<String, DailySourceData>,
}

#[derive(Clone, Debug)]
enum ParamValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    ListF64(Vec<f64>),
}

impl FactorSpec {
    fn get_i64(&self, key: &str) -> Option<i64> {
        match self.params.get(key) {
            Some(ParamValue::Int(v)) => Some(*v),
            Some(ParamValue::Float(v)) => Some(*v as i64),
            Some(ParamValue::Str(v)) => v.parse::<i64>().ok(),
            _ => None,
        }
    }

    fn get_str(&self, key: &str) -> Option<String> {
        match self.params.get(key) {
            Some(ParamValue::Str(v)) => Some(v.clone()),
            Some(ParamValue::Int(v)) => Some(v.to_string()),
            Some(ParamValue::Float(v)) => Some(v.to_string()),
            Some(ParamValue::Bool(v)) => Some(v.to_string()),
            _ => None,
        }
    }

    fn param_i64(&self, key: &str, default: i64) -> i64 {
        match self.params.get(key) {
            Some(ParamValue::Int(v)) => *v,
            Some(ParamValue::Float(v)) => *v as i64,
            Some(ParamValue::Str(v)) => v.parse::<i64>().unwrap_or(default),
            _ => default,
        }
    }

    fn param_f64(&self, key: &str, default: f64) -> f64 {
        match self.params.get(key) {
            Some(ParamValue::Float(v)) => *v,
            Some(ParamValue::Int(v)) => *v as f64,
            Some(ParamValue::Str(v)) => v.parse::<f64>().unwrap_or(default),
            _ => default,
        }
    }

    fn param_bool(&self, key: &str, default: bool) -> bool {
        match self.params.get(key) {
            Some(ParamValue::Bool(v)) => *v,
            Some(ParamValue::Int(v)) => *v != 0,
            Some(ParamValue::Str(v)) => matches!(v.as_str(), "1" | "true" | "True" | "TRUE"),
            _ => default,
        }
    }

    fn param_str(&self, key: &str, default: &str) -> String {
        match self.params.get(key) {
            Some(ParamValue::Str(v)) => v.clone(),
            Some(ParamValue::Int(v)) => v.to_string(),
            Some(ParamValue::Float(v)) => v.to_string(),
            Some(ParamValue::Bool(v)) => v.to_string(),
            _ => default.to_string(),
        }
    }

    fn param_list_f64(&self, key: &str, default: &[f64]) -> Vec<f64> {
        match self.params.get(key) {
            Some(ParamValue::ListF64(v)) if !v.is_empty() => v.clone(),
            _ => default.to_vec(),
        }
    }
}

#[pymodule]
fn cbond_on_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_factor_frame, m)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (panel_df, specs_payload, stock_df=None, map_df=None, daily_data=None, _compute_params=None))]
fn compute_factor_frame(
    py: Python<'_>,
    panel_df: &Bound<'_, PyAny>,
    specs_payload: &Bound<'_, PyAny>,
    stock_df: Option<&Bound<'_, PyAny>>,
    map_df: Option<&Bound<'_, PyAny>>,
    daily_data: Option<&Bound<'_, PyAny>>,
    _compute_params: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let mut panel = parse_panel(py, panel_df)?;
    let specs = parse_specs(specs_payload)?;
    let plan_limits = parse_plan_limits(_compute_params)?;
    let plan = extract_factor_plan(&specs, &plan_limits)?;
    let window_cache = build_window_start_cache(&panel, &plan.windows);
    let aux = parse_aux_data(py, stock_df, map_df, daily_data)?;
    if panel.groups.is_empty() {
        return empty_df(py);
    }

    if plan_limits.log_summary {
        println!(
            "rust factor plan: windows={} levels={} ranges={} max_windows={} max_levels={} max_ranges={} window_set={:?} level_set={:?} range_set={:?}",
            plan.windows.len(),
            plan.levels.len(),
            plan.time_ranges.len(),
            plan_limits.max_windows,
            plan_limits.max_levels,
            plan_limits.max_time_ranges,
            plan.windows,
            plan.levels,
            plan.time_ranges
        );
    }

    let mut out_cols: HashMap<String, Vec<f64>> = HashMap::new();
    let mut factor_timings: Vec<(String, f64)> = Vec::with_capacity(specs.len());
    let t_all_factors = Instant::now();
    for spec in &specs {
        let t_factor = Instant::now();
        let values = compute_factor_values(py, &mut panel, panel_df, &aux, &window_cache, spec)?;
        let elapsed = t_factor.elapsed().as_secs_f64();
        factor_timings.push((spec.output_col.clone(), elapsed));
        out_cols.insert(spec.output_col.clone(), values);
    }
    if plan_limits.log_summary && !factor_timings.is_empty() {
        let mut values: Vec<f64> = factor_timings.iter().map(|(_, v)| *v).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = values.len();
        let idx50 = n / 2;
        let idx90 = (((n - 1) as f64) * 0.9).round() as usize;
        let min_v = values[0];
        let p50 = values[idx50];
        let p90 = values[idx90.min(n - 1)];
        let max_v = values[n - 1];
        let mut slow = factor_timings.clone();
        slow.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let topn = slow
            .iter()
            .take(5)
            .map(|(name, sec)| format!("{}:{:.3}s", name, sec))
            .collect::<Vec<_>>();
        println!(
            "rust factor timing dist: n={} total={:.2}s min={:.3}s p50={:.3}s p90={:.3}s max={:.3}s top={:?}",
            n,
            t_all_factors.elapsed().as_secs_f64(),
            min_v,
            p50,
            p90,
            max_v,
            topn
        );
    }

    build_output_df(py, &panel.groups, &out_cols)
}

fn norm_code(code: &str) -> String {
    code.trim().to_ascii_uppercase()
}

fn norm_dt(dt: &str) -> String {
    if dt.len() >= 10 {
        dt[0..10].to_string()
    } else {
        dt.to_string()
    }
}

fn parse_aux_data(
    py: Python<'_>,
    stock_df: Option<&Bound<'_, PyAny>>,
    map_df: Option<&Bound<'_, PyAny>>,
    daily_data: Option<&Bound<'_, PyAny>>,
) -> PyResult<AuxData> {
    let mut aux = AuxData::default();

    if let Some(df) = stock_df {
        let dt = col_to_str_vec(py, df, "dt")?;
        let code = col_to_str_vec(py, df, "code")?;
        let tns = if has_col(py, df, "__trade_time_ns__")? {
            col_to_i64_vec(py, df, "__trade_time_ns__")?
        } else {
            col_to_i64_vec(py, df, "trade_time")?
        };
        let n = dt.len();
        if code.len() == n && tns.len() == n && n > 0 {
            let last = if has_col(py, df, "last")? {
                col_to_f64_vec(py, df, "last")?
            } else {
                vec![f64::NAN; n]
            };
            let open = if has_col(py, df, "open")? {
                col_to_f64_vec(py, df, "open")?
            } else {
                vec![f64::NAN; n]
            };
            let ask1 = if has_col(py, df, "ask_price1")? {
                col_to_f64_vec(py, df, "ask_price1")?
            } else {
                vec![f64::NAN; n]
            };
            let bid1 = if has_col(py, df, "bid_price1")? {
                col_to_f64_vec(py, df, "bid_price1")?
            } else {
                vec![f64::NAN; n]
            };
            let prev = if has_col(py, df, "prev_bar_close")? {
                col_to_f64_vec(py, df, "prev_bar_close")?
            } else {
                vec![f64::NAN; n]
            };
            let pre_close = if has_col(py, df, "pre_close")? {
                col_to_f64_vec(py, df, "pre_close")?
            } else {
                vec![f64::NAN; n]
            };

            let mut s = 0usize;
            for i in 1..=n {
                let boundary = i == n || dt[i] != dt[s] || code[i] != code[s];
                if boundary {
                    let key = (norm_dt(&dt[s]), norm_code(&code[s]));
                    let r = i - 1;
                    aux.stock_latest.insert(
                        key,
                        LatestQuote {
                            last: last[r],
                            open: open[r],
                            ask_price1: ask1[r],
                            bid_price1: bid1[r],
                            prev_bar_close: prev[r],
                            pre_close: pre_close[r],
                        },
                    );
                    s = i;
                }
            }
        }
    }

    if let Some(df) = map_df {
        if has_col(py, df, "code")? && has_col(py, df, "stock_code")? {
            let code = col_to_str_vec(py, df, "code")?;
            let stock_code = col_to_str_vec(py, df, "stock_code")?;
            let n = code.len().min(stock_code.len());
            for i in 0..n {
                let c = norm_code(&code[i]);
                let s = norm_code(&stock_code[i]);
                if !c.is_empty() && !s.is_empty() {
                    aux.bond_stock_map.insert(c, s);
                }
            }
        }
    }

    if let Some(raw_daily) = daily_data {
        if !raw_daily.is_none() {
            let daily_dict = raw_daily.downcast::<PyDict>()?;
            for (raw_source, raw_df) in daily_dict.iter() {
                let source = raw_source.extract::<String>()?;
                let df = raw_df;
                if !has_col(py, &df, "trade_date")? || !has_col(py, &df, "code")? {
                    continue;
                }
                let dates = col_to_str_vec(py, &df, "trade_date")?;
                let codes = col_to_str_vec(py, &df, "code")?;
                let n = dates.len().min(codes.len());
                if n == 0 {
                    continue;
                }

                let cols_obj = df.getattr("columns")?;
                let cols_list_any = cols_obj.call_method0("tolist")?;
                let cols_list = cols_list_any.extract::<Vec<String>>()?;

                let mut numeric_cols: Vec<String> = Vec::new();
                let mut numeric_data: HashMap<String, Vec<f64>> = HashMap::new();
                for col in cols_list {
                    if col == "trade_date" || col == "code" {
                        continue;
                    }
                    let vec = match col_to_f64_vec(py, &df, &col) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    if vec.len() < n {
                        continue;
                    }
                    numeric_cols.push(col.clone());
                    numeric_data.insert(col, vec);
                }
                if numeric_cols.is_empty() {
                    continue;
                }

                let mut by_code: HashMap<String, Vec<DailyPoint>> = HashMap::new();
                for i in 0..n {
                    let code = norm_code(&codes[i]);
                    if code.is_empty() {
                        continue;
                    }
                    let trade_date = norm_dt(&dates[i]);
                    if trade_date.is_empty() {
                        continue;
                    }
                    let mut values = HashMap::<String, f64>::new();
                    for col in &numeric_cols {
                        if let Some(vec) = numeric_data.get(col) {
                            let v = vec[i];
                            if v.is_finite() {
                                values.insert(col.clone(), v);
                            }
                        }
                    }
                    if values.is_empty() {
                        continue;
                    }
                    by_code.entry(code).or_default().push(DailyPoint {
                        date: trade_date,
                        values,
                    });
                }
                for rows in by_code.values_mut() {
                    rows.sort_by(|a, b| a.date.cmp(&b.date));
                    let mut dedup: Vec<DailyPoint> = Vec::with_capacity(rows.len());
                    for row in rows.iter().cloned() {
                        if let Some(last) = dedup.last_mut() {
                            if last.date == row.date {
                                *last = row;
                                continue;
                            }
                        }
                        dedup.push(row);
                    }
                    *rows = dedup;
                }
                if !by_code.is_empty() {
                    aux.daily_sources
                        .insert(source, DailySourceData { by_code });
                }
            }
        }
    }

    Ok(aux)
}

fn empty_df(py: Python<'_>) -> PyResult<PyObject> {
    let pd = py.import_bound("pandas")?;
    let kwargs = PyDict::new_bound(py);
    let df = pd.call_method("DataFrame", (), Some(&kwargs))?;
    Ok(df.into_py(py))
}

fn parse_panel(py: Python<'_>, panel_df: &Bound<'_, PyAny>) -> PyResult<PanelData> {
    let dt = col_to_str_vec(py, panel_df, "dt")?;
    let code = col_to_str_vec(py, panel_df, "code")?;
    let seq = col_to_i64_vec(py, panel_df, "seq")?;
    let trade_time_ns = if has_col(py, panel_df, "__trade_time_ns__")? {
        col_to_i64_vec(py, panel_df, "__trade_time_ns__")?
    } else {
        let raw = col_to_i64_vec(py, panel_df, "trade_time")?;
        raw
    };
    let n = dt.len();
    if code.len() != n || seq.len() != n || trade_time_ns.len() != n {
        return Err(PyErr::new::<PyValueError, _>(
            "panel columns length mismatch",
        ));
    }

    let mut groups: Vec<Group> = Vec::new();
    if n > 0 {
        let mut s = 0usize;
        for i in 1..n {
            if dt[i] != dt[s] || code[i] != code[s] {
                groups.push(Group {
                    start: s,
                    end: i,
                    dt: dt[s].clone(),
                    code: code[s].clone(),
                });
                s = i;
            }
        }
        groups.push(Group {
            start: s,
            end: n,
            dt: dt[s].clone(),
            code: code[s].clone(),
        });
    }

    Ok(PanelData {
        dt,
        code,
        seq,
        trade_time_ns,
        cols: HashMap::new(),
        groups,
    })
}

fn parse_specs(specs_payload: &Bound<'_, PyAny>) -> PyResult<Vec<FactorSpec>> {
    let specs_list = specs_payload.downcast::<PyList>()?;
    let mut out = Vec::with_capacity(specs_list.len());
    for item in specs_list.iter() {
        let d = item.downcast::<PyDict>()?;
        let name = dict_get_str(d, "name")?;
        let factor = dict_get_str(d, "factor")?;
        let output_col = match d.get_item("output_col")? {
            Some(v) if !v.is_none() => v.extract::<String>()?,
            _ => name.clone(),
        };

        let mut params = HashMap::<String, ParamValue>::new();
        if let Some(p) = d.get_item("params")? {
            if !p.is_none() {
                let pdict = p.downcast::<PyDict>()?;
                for (k, v) in pdict.iter() {
                    let key = k.extract::<String>()?;
                    if let Ok(x) = v.extract::<i64>() {
                        params.insert(key, ParamValue::Int(x));
                    } else if let Ok(x) = v.extract::<f64>() {
                        params.insert(key, ParamValue::Float(x));
                    } else if let Ok(x) = v.extract::<bool>() {
                        params.insert(key, ParamValue::Bool(x));
                    } else if let Ok(x) = v.extract::<String>() {
                        params.insert(key, ParamValue::Str(x));
                    } else if let Ok(lst) = v.extract::<Vec<f64>>() {
                        params.insert(key, ParamValue::ListF64(lst));
                    }
                }
            }
        }

        out.push(FactorSpec {
            name,
            factor,
            output_col,
            params,
        });
    }
    Ok(out)
}

fn dict_get_i64_opt(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<i64>> {
    let Some(v) = d.get_item(key)? else {
        return Ok(None);
    };
    if v.is_none() {
        return Ok(None);
    }
    if let Ok(x) = v.extract::<i64>() {
        return Ok(Some(x));
    }
    if let Ok(x) = v.extract::<f64>() {
        return Ok(Some(x as i64));
    }
    if let Ok(x) = v.extract::<String>() {
        return Ok(x.parse::<i64>().ok());
    }
    Ok(None)
}

fn dict_get_bool_opt(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<bool>> {
    let Some(v) = d.get_item(key)? else {
        return Ok(None);
    };
    if v.is_none() {
        return Ok(None);
    }
    if let Ok(x) = v.extract::<bool>() {
        return Ok(Some(x));
    }
    if let Ok(x) = v.extract::<i64>() {
        return Ok(Some(x != 0));
    }
    if let Ok(x) = v.extract::<String>() {
        let t = x.trim().to_ascii_lowercase();
        if matches!(t.as_str(), "1" | "true" | "yes" | "y" | "on") {
            return Ok(Some(true));
        }
        if matches!(t.as_str(), "0" | "false" | "no" | "n" | "off") {
            return Ok(Some(false));
        }
    }
    Ok(None)
}

fn parse_plan_limits(compute_params: Option<&Bound<'_, PyAny>>) -> PyResult<PlanLimits> {
    let mut out = PlanLimits::default();
    let Some(any_params) = compute_params else {
        return Ok(out);
    };
    let Ok(root_dict) = any_params.downcast::<PyDict>() else {
        return Ok(out);
    };
    let Some(raw_backend) = root_dict.get_item("__compute_backend__")? else {
        return Ok(out);
    };
    let Ok(backend_dict) = raw_backend.downcast::<PyDict>() else {
        return Ok(out);
    };

    if let Some(v) = dict_get_i64_opt(&backend_dict, "plan_max_windows")? {
        if v > 0 {
            out.max_windows = v as usize;
        }
    }
    if let Some(v) = dict_get_i64_opt(&backend_dict, "plan_max_levels")? {
        if v > 0 {
            out.max_levels = v as usize;
        }
    }
    if let Some(v) = dict_get_i64_opt(&backend_dict, "plan_max_time_ranges")? {
        if v > 0 {
            out.max_time_ranges = v as usize;
        }
    }
    if let Some(v) = dict_get_bool_opt(&backend_dict, "plan_log_summary")? {
        out.log_summary = v;
    }
    Ok(out)
}

fn extract_factor_plan(specs: &[FactorSpec], limits: &PlanLimits) -> PyResult<FactorPlan> {
    let mut windows = BTreeSet::<i64>::new();
    let mut levels = BTreeSet::<usize>::new();
    let mut ranges = BTreeSet::<(String, String)>::new();

    for spec in specs {
        if let Some(w) = spec.get_i64("window_minutes") {
            if w > 0 {
                windows.insert(w);
            }
        }

        for key in ["levels", "depth_levels"] {
            if let Some(level) = spec.get_i64(key) {
                if level > 0 {
                    levels.insert(level as usize);
                }
            }
        }

        if spec.factor == "ret_open_to_time" {
            let st = spec
                .get_str("start_time")
                .unwrap_or_else(|| "09:30".to_string());
            let et = spec
                .get_str("end_time")
                .unwrap_or_else(|| "14:30".to_string());
            let _ = parse_hhmm(&st);
            let _ = parse_hhmm(&et);
            ranges.insert((st, et));
        }
    }

    if windows.len() > limits.max_windows {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "factor plan windows exceed limit: {} > {}",
            windows.len(),
            limits.max_windows
        )));
    }
    if levels.len() > limits.max_levels {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "factor plan levels exceed limit: {} > {}",
            levels.len(),
            limits.max_levels
        )));
    }
    if ranges.len() > limits.max_time_ranges {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "factor plan time_ranges exceed limit: {} > {}",
            ranges.len(),
            limits.max_time_ranges
        )));
    }

    Ok(FactorPlan {
        windows: windows.into_iter().collect(),
        levels: levels.into_iter().collect(),
        time_ranges: ranges.into_iter().collect(),
    })
}

fn build_window_start_cache(panel: &PanelData, windows: &[i64]) -> WindowStartCache {
    let mut starts_by_window = HashMap::<i64, Vec<usize>>::new();
    for &w in windows {
        let mut starts = Vec::<usize>::with_capacity(panel.groups.len());
        for g in &panel.groups {
            starts.push(window_start_ns(&panel.trade_time_ns, g.start, g.end, w));
        }
        starts_by_window.insert(w, starts);
    }
    WindowStartCache { starts_by_window }
}

fn cached_window_start(
    cache: &WindowStartCache,
    panel: &PanelData,
    group_idx: usize,
    g: &Group,
    window_minutes: i64,
) -> usize {
    if let Some(starts) = cache.starts_by_window.get(&window_minutes) {
        if group_idx < starts.len() {
            return starts[group_idx];
        }
    }
    window_start_ns(&panel.trade_time_ns, g.start, g.end, window_minutes)
}

fn dict_get_str(d: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    let v = d
        .get_item(key)?
        .ok_or_else(|| PyErr::new::<PyKeyError, _>(format!("spec missing key: {}", key)))?;
    v.extract::<String>()
}

fn has_col(_py: Python<'_>, df: &Bound<'_, PyAny>, col: &str) -> PyResult<bool> {
    let cols = df.getattr("columns")?;
    let contains = cols.call_method1("__contains__", (col,))?;
    contains.extract::<bool>().map_err(|e| e.into())
}

fn col_to_str_vec(_py: Python<'_>, df: &Bound<'_, PyAny>, col: &str) -> PyResult<Vec<String>> {
    let s = df.call_method1("__getitem__", (col,))?;
    let s2 = s.call_method1("astype", ("str",))?;
    let lst = s2.call_method0("tolist")?;
    lst.extract::<Vec<String>>().map_err(|e| e.into())
}

fn col_to_i64_vec(py: Python<'_>, df: &Bound<'_, PyAny>, col: &str) -> PyResult<Vec<i64>> {
    let s = df.call_method1("__getitem__", (col,))?;
    let pd = py.import_bound("pandas")?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("errors", "coerce")?;
    let num = pd.call_method("to_numeric", (s,), Some(&kwargs))?;
    let num = num.call_method1("fillna", (0i64,))?;
    let num = num.call_method1("astype", ("int64",))?;
    let arr_obj = num.call_method0("to_numpy")?;
    let arr = arr_obj.downcast::<PyArray1<i64>>()?;
    Ok(arr.readonly().as_slice()?.to_vec())
}

fn col_to_f64_vec(py: Python<'_>, df: &Bound<'_, PyAny>, col: &str) -> PyResult<Vec<f64>> {
    let s = df.call_method1("__getitem__", (col,))?;
    let pd = py.import_bound("pandas")?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("errors", "coerce")?;
    let num = pd.call_method("to_numeric", (s,), Some(&kwargs))?;
    let num = num.call_method1("astype", ("float64",))?;
    let arr_obj = num.call_method0("to_numpy")?;
    let arr = arr_obj.downcast::<PyArray1<f64>>()?;
    Ok(arr.readonly().as_slice()?.to_vec())
}

fn ensure_col(
    py: Python<'_>,
    panel: &mut PanelData,
    panel_df: &Bound<'_, PyAny>,
    col: &str,
) -> PyResult<()> {
    if !panel.cols.contains_key(col) {
        let values = col_to_f64_vec(py, panel_df, col)?;
        if values.len() != panel.len() {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "column {} length mismatch",
                col
            )));
        }
        panel.cols.insert(col.to_string(), values);
    }
    Ok(())
}

fn build_output_df(
    py: Python<'_>,
    groups: &[Group],
    out_cols: &HashMap<String, Vec<f64>>,
) -> PyResult<PyObject> {
    let pd = py.import_bound("pandas")?;
    let data = PyDict::new_bound(py);

    let dt: Vec<String> = groups.iter().map(|g| g.dt.clone()).collect();
    let code: Vec<String> = groups.iter().map(|g| g.code.clone()).collect();
    data.set_item("dt", dt)?;
    data.set_item("code", code)?;
    for (k, v) in out_cols {
        data.set_item(k, v)?;
    }
    let df = pd.call_method1("DataFrame", (data,))?;
    Ok(df.into_py(py))
}

fn window_start_ns(trade_ns: &[i64], start: usize, end: usize, window_minutes: i64) -> usize {
    if window_minutes <= 0 || end <= start {
        return start;
    }
    let end_ns = trade_ns[end - 1];
    let lookback = window_minutes
        .saturating_mul(60)
        .saturating_mul(1_000_000_000i64);
    let cutoff = end_ns.saturating_sub(lookback);
    let mut lo = start;
    let mut hi = end;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if trade_ns[mid] < cutoff {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn last_valid(slice: &[f64]) -> f64 {
    for &v in slice.iter().rev() {
        if v.is_finite() {
            return v;
        }
    }
    f64::NAN
}

fn first_valid(slice: &[f64]) -> f64 {
    for &v in slice {
        if v.is_finite() {
            return v;
        }
    }
    f64::NAN
}

fn sum_valid(slice: &[f64]) -> f64 {
    let mut acc = 0.0;
    for &v in slice {
        if v.is_finite() {
            acc += v;
        }
    }
    acc
}

fn mean_valid(slice: &[f64]) -> f64 {
    let mut acc = 0.0;
    let mut n = 0usize;
    for &v in slice {
        if v.is_finite() {
            acc += v;
            n += 1;
        }
    }
    if n == 0 {
        f64::NAN
    } else {
        acc / n as f64
    }
}

fn min_valid(slice: &[f64]) -> f64 {
    let mut out = f64::INFINITY;
    let mut has = false;
    for &v in slice {
        if v.is_finite() {
            out = out.min(v);
            has = true;
        }
    }
    if has {
        out
    } else {
        f64::NAN
    }
}

fn max_valid(slice: &[f64]) -> f64 {
    let mut out = f64::NEG_INFINITY;
    let mut has = false;
    for &v in slice {
        if v.is_finite() {
            out = out.max(v);
            has = true;
        }
    }
    if has {
        out
    } else {
        f64::NAN
    }
}

fn pct_rank(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![f64::NAN; n];
    let mut idx: Vec<usize> = (0..n).filter(|&i| values[i].is_finite()).collect();
    let m = idx.len();
    if m == 0 {
        return out;
    }
    idx.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut i = 0usize;
    while i < m {
        let mut j = i + 1;
        while j < m && values[idx[j]] == values[idx[i]] {
            j += 1;
        }
        let avg_rank = ((i + 1 + j) as f64) * 0.5;
        let pct = avg_rank / (m as f64);
        for k in i..j {
            out[idx[k]] = pct;
        }
        i = j;
    }
    out
}

fn std_sample(values: &[f64]) -> f64 {
    let vals: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let n = vals.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = vals.iter().sum::<f64>() / n as f64;
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / ((n - 1) as f64);
    var.sqrt()
}

fn corr_slice(x: &[f64], y: &[f64]) -> f64 {
    let mut a: Vec<f64> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    for i in 0..x.len().min(y.len()) {
        let xv = x[i];
        let yv = y[i];
        if xv.is_finite() && yv.is_finite() {
            a.push(xv);
            b.push(yv);
        }
    }
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let ma = a.iter().sum::<f64>() / n as f64;
    let mb = b.iter().sum::<f64>() / n as f64;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        sxy += da * db;
        sxx += da * da;
        syy += db * db;
    }
    if sxx <= EPS || syy <= EPS {
        return 0.0;
    }
    let r = sxy / (sxx.sqrt() * syy.sqrt());
    if r.is_finite() {
        r
    } else {
        0.0
    }
}

fn cov_slice(x: &[f64], y: &[f64]) -> f64 {
    let mut a: Vec<f64> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    for i in 0..x.len().min(y.len()) {
        let xv = x[i];
        let yv = y[i];
        if xv.is_finite() && yv.is_finite() {
            a.push(xv);
            b.push(yv);
        }
    }
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let ma = a.iter().sum::<f64>() / n as f64;
    let mb = b.iter().sum::<f64>() / n as f64;
    let mut s = 0.0;
    for i in 0..n {
        s += (a[i] - ma) * (b[i] - mb);
    }
    s / ((n - 1) as f64)
}

fn open_like_at(open: f64, ask1: f64, bid1: f64) -> f64 {
    if ask1.is_finite() && bid1.is_finite() {
        (ask1 + bid1) * 0.5
    } else {
        open
    }
}

fn parse_hhmm(value: &str) -> (u32, u32) {
    let p: Vec<&str> = value.split(':').collect();
    if p.len() >= 2 {
        let h = p[0].parse::<u32>().unwrap_or(0).min(23);
        let m = p[1].parse::<u32>().unwrap_or(0).min(59);
        (h, m)
    } else {
        (0, 0)
    }
}

fn date_start_ns(dt_value: &str) -> Option<i64> {
    let s = if dt_value.len() >= 10 {
        &dt_value[0..10]
    } else {
        dt_value
    };
    let d = NaiveDate::parse_from_str(s, "%Y-%m-%d").ok()?;
    let t = d.and_hms_opt(0, 0, 0)?;
    Some(t.and_utc().timestamp_nanos_opt()?)
}

fn cs_rank_by_dt(groups: &[Group], raw: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; raw.len()];
    let mut dt_to_idx: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, g) in groups.iter().enumerate() {
        dt_to_idx.entry(g.dt.as_str()).or_default().push(i);
    }
    for idxs in dt_to_idx.values() {
        let vals: Vec<f64> = idxs.iter().map(|&i| raw[i]).collect();
        let ranks = pct_rank(&vals);
        for (k, &i) in idxs.iter().enumerate() {
            let v = ranks[k];
            out[i] = if v.is_finite() { v } else { 0.0 };
        }
    }
    out
}

fn skew_unbiased(values: &[f64]) -> f64 {
    let vals: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let n = vals.len();
    if n < 3 {
        return 0.0;
    }
    let mean = vals.iter().sum::<f64>() / n as f64;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    for v in &vals {
        let d = *v - mean;
        sum2 += d * d;
        sum3 += d * d * d;
    }

    // Match pandas.Series.skew() (bias-corrected sample skewness):
    // n*sqrt(n-1)/(n-2) * (sum((x-mean)^3) / (sum((x-mean)^2)^(3/2)))
    if sum2 <= 0.0 {
        return 0.0;
    }
    let denom = sum2.powf(1.5);
    if denom <= 0.0 {
        return 0.0;
    }
    let adj = (n as f64) * ((n - 1) as f64).sqrt() / ((n - 2) as f64);
    let out = adj * (sum3 / denom);
    if out.is_finite() {
        out
    } else {
        0.0
    }
}

fn ts_rank_last(values: &[f64], window: usize) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let w = window.max(1).min(values.len());
    let tail = &values[values.len() - w..];
    let filtered: Vec<f64> = tail.iter().copied().filter(|v| v.is_finite()).collect();
    if filtered.is_empty() {
        return 0.0;
    }
    let last_v = *filtered.last().unwrap_or(&f64::NAN);
    if !last_v.is_finite() {
        return 0.0;
    }
    let ranks = pct_rank(&filtered);
    let last_rank = *ranks.last().unwrap_or(&0.0);
    if last_rank.is_finite() {
        last_rank
    } else {
        0.0
    }
}

fn rolling_last_rank_pct_series(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    let w = window.max(1);
    let mut out = vec![f64::NAN; n];
    if n == 0 {
        return out;
    }
    for i in 0..n {
        let s = i + 1usize - (i + 1).min(w);
        let last = values[i];
        if !last.is_finite() {
            continue;
        }
        let mut lt = 0usize;
        let mut eq = 0usize;
        let mut cnt = 0usize;
        for &v in &values[s..=i] {
            if v.is_finite() {
                cnt += 1;
                if v < last {
                    lt += 1;
                } else if v == last {
                    eq += 1;
                }
            }
        }
        if cnt > 0 {
            out[i] = (lt as f64 + (eq as f64 + 1.0) / 2.0) / cnt as f64;
        }
    }
    out
}

fn rolling_linear_decay_series(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    let w = window.max(1);
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        let s = i + 1usize - (i + 1).min(w);
        let mut num = 0.0;
        let mut den = 0.0;
        let mut k = 1usize;
        for &v in &values[s..=i] {
            let wt = k as f64;
            den += wt;
            if v.is_finite() {
                num += v * wt;
            }
            k += 1;
        }
        if den > 0.0 {
            out[i] = num / (den + EPS);
        }
    }
    out
}

fn argmax_pos_last(values: &[f64], window: usize) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let w = window.max(1).min(n);
    let s = n - w;
    let mut best_idx = 0usize;
    let mut best = f64::NEG_INFINITY;
    let mut has = false;
    for i in 0..w {
        let v = values[s + i];
        if v.is_finite() {
            if !has || v > best {
                has = true;
                best = v;
                best_idx = i;
            }
        }
    }
    if has {
        (best_idx + 1) as f64
    } else {
        0.0
    }
}

fn rolling_argmax_position_series(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    let w = window.max(1);
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        let s = i + 1usize - (i + 1).min(w);
        let mut best_idx = 0usize;
        let mut best = f64::NEG_INFINITY;
        let mut has = false;
        for (k, &v) in values[s..=i].iter().enumerate() {
            if v.is_finite() {
                if !has || v > best {
                    has = true;
                    best = v;
                    best_idx = k;
                }
            }
        }
        if has {
            out[i] = (best_idx + 1) as f64;
        }
    }
    out
}

fn cs_scale_by_dt(groups: &[Group], raw: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; raw.len()];
    let mut dt_to_idx: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, g) in groups.iter().enumerate() {
        dt_to_idx.entry(g.dt.as_str()).or_default().push(i);
    }
    for idxs in dt_to_idx.values() {
        let mut denom = 0.0;
        for &i in idxs {
            let v = raw[i];
            if v.is_finite() {
                denom += v.abs();
            }
        }
        if denom <= EPS {
            for &i in idxs {
                out[i] = 0.0;
            }
        } else {
            for &i in idxs {
                let v = raw[i];
                out[i] = if v.is_finite() { v / denom } else { 0.0 };
            }
        }
    }
    out
}

fn delta_last_from_slice(values: &[f64], periods: usize) -> f64 {
    let p = periods.max(1);
    if values.len() <= p {
        return 0.0;
    }
    values[values.len() - 1] - values[values.len() - 1 - p]
}

fn delay_last_from_slice(values: &[f64], periods: usize) -> f64 {
    let p = periods.max(1);
    if values.is_empty() {
        return 0.0;
    }
    if values.len() <= p {
        return values[0];
    }
    values[values.len() - 1 - p]
}

fn delay_value_or_nan(values: &[f64], periods: usize) -> f64 {
    let p = periods.max(1);
    if values.len() <= p {
        f64::NAN
    } else {
        values[values.len() - 1 - p]
    }
}

fn diff(values: &[f64], lag: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    if lag == 0 {
        return out;
    }
    for i in lag..values.len() {
        let a = values[i];
        let b = values[i - lag];
        out[i] = if a.is_finite() && b.is_finite() {
            a - b
        } else {
            f64::NAN
        };
    }
    out
}

fn pct_change(values: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    for i in 1..values.len() {
        let cur = values[i];
        let prev = values[i - 1];
        out[i] = if cur.is_finite() && prev.is_finite() && prev.abs() > EPS {
            (cur - prev) / prev
        } else {
            f64::NAN
        };
    }
    out
}

fn log_diff(values: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    for i in 1..values.len() {
        let cur = values[i];
        let prev = values[i - 1];
        out[i] = if cur.is_finite() && prev.is_finite() && cur > 0.0 && prev > 0.0 {
            cur.ln() - prev.ln()
        } else {
            f64::NAN
        };
    }
    out
}

fn rolling_mean_last(values: &[f64], window: usize) -> f64 {
    let w = window.max(1);
    if values.is_empty() {
        return f64::NAN;
    }
    let s = values.len().saturating_sub(w);
    mean_valid(&values[s..])
}

fn rolling_min_last(values: &[f64], window: usize) -> f64 {
    let w = window.max(1);
    if values.is_empty() {
        return f64::NAN;
    }
    let s = values.len().saturating_sub(w);
    min_valid(&values[s..])
}

fn rolling_max_last(values: &[f64], window: usize) -> f64 {
    let w = window.max(1);
    if values.is_empty() {
        return f64::NAN;
    }
    let s = values.len().saturating_sub(w);
    max_valid(&values[s..])
}

fn corr_last_window(x: &[f64], y: &[f64], window: usize) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }
    let n = x.len().min(y.len());
    let w = window.max(2).min(n);
    corr_slice(&x[n - w..n], &y[n - w..n])
}

fn cov_last_window(x: &[f64], y: &[f64], window: usize) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }
    let n = x.len().min(y.len());
    let w = window.max(2).min(n);
    cov_slice(&x[n - w..n], &y[n - w..n])
}

fn daily_values_asof(
    aux: &AuxData,
    source: &str,
    code: &str,
    asof_dt: &str,
    col: &str,
    lookback_days: usize,
) -> Vec<f64> {
    let Some(source_data) = aux.daily_sources.get(source) else {
        return Vec::new();
    };
    let Some(rows) = source_data.by_code.get(code) else {
        return Vec::new();
    };
    let mut out_rev: Vec<f64> = Vec::with_capacity(lookback_days.max(1));
    let limit = lookback_days.max(1);
    for row in rows.iter().rev() {
        if row.date.as_str() > asof_dt {
            continue;
        }
        if let Some(v) = row.values.get(col) {
            if v.is_finite() {
                out_rev.push(*v);
                if out_rev.len() >= limit {
                    break;
                }
            }
        }
    }
    out_rev.reverse();
    out_rev
}

fn rolling_sharpe_last_mean(
    values: &[f64],
    lookback_days: usize,
    smooth_days: usize,
    min_periods: usize,
    annualize: bool,
) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let returns = pct_change(values);
    let window = lookback_days.max(2);
    let min_count = min_periods.max(2);
    let ann = if annualize { 252.0f64.sqrt() } else { 1.0 };
    let mut sharpe_hist: Vec<f64> = Vec::new();
    for i in 0..returns.len() {
        let end = i + 1;
        let start = end.saturating_sub(window);
        let mut buf: Vec<f64> = Vec::with_capacity(window);
        for &v in &returns[start..end] {
            if v.is_finite() {
                buf.push(v);
            }
        }
        if buf.len() < min_count {
            continue;
        }
        let mean = mean_valid(&buf);
        let std = std_sample(&buf);
        if mean.is_finite() && std.is_finite() && std > EPS {
            sharpe_hist.push((mean / std) * ann);
        }
    }
    if sharpe_hist.is_empty() {
        return 0.0;
    }
    let smooth = smooth_days.max(1).min(sharpe_hist.len());
    let s = sharpe_hist.len() - smooth;
    let out = mean_valid(&sharpe_hist[s..]);
    if out.is_finite() {
        out
    } else {
        0.0
    }
}

fn compute_factor_values(
    py: Python<'_>,
    panel: &mut PanelData,
    panel_df: &Bound<'_, PyAny>,
    aux: &AuxData,
    window_cache: &WindowStartCache,
    spec: &FactorSpec,
) -> PyResult<Vec<f64>> {
    let mut out = vec![0.0; panel.groups.len()];
    match spec.factor.as_str() {
        "aacb" => {
            let levels = spec.param_i64("levels", 3).max(1) as usize;
            for i in 1..=levels {
                ensure_col(py, panel, panel_df, &format!("ask_price{}", i))?;
                ensure_col(py, panel, panel_df, &format!("bid_price{}", i))?;
            }
            ensure_col(py, panel, panel_df, "ask_price1")?;
            ensure_col(py, panel, panel_df, "bid_price1")?;
            let mut ask_cols: Vec<&Vec<f64>> = Vec::with_capacity(levels);
            let mut bid_cols: Vec<&Vec<f64>> = Vec::with_capacity(levels);
            for i in 1..=levels {
                let ask_name = format!("ask_price{}", i);
                let bid_name = format!("bid_price{}", i);
                ask_cols.push(panel.col(&ask_name)?);
                bid_cols.push(panel.col(&bid_name)?);
            }
            let ask1_col = panel.col("ask_price1")?;
            let bid1_col = panel.col("bid_price1")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let mut acc = 0.0f64;
                let mut cnt = 0usize;
                for r in g.start..g.end {
                    let mut ask_sum = 0.0;
                    let mut bid_sum = 0.0;
                    let mut ok = true;
                    for i in 0..levels {
                        let ap = ask_cols[i][r];
                        let bp = bid_cols[i][r];
                        if !ap.is_finite() || !bp.is_finite() {
                            ok = false;
                            break;
                        }
                        ask_sum += ap;
                        bid_sum += bp;
                    }
                    if !ok {
                        continue;
                    }
                    let ask_avg = ask_sum / levels as f64;
                    let bid_avg = bid_sum / levels as f64;
                    let ask1 = ask1_col[r];
                    let bid1 = bid1_col[r];
                    let mid = (ask1 + bid1) * 0.5;
                    if mid.abs() > EPS {
                        let v = (ask_avg - bid_avg) / mid;
                        if v.is_finite() {
                            acc += v;
                            cnt += 1;
                        }
                    }
                }
                out[gi] = if cnt > 0 { acc / cnt as f64 } else { 0.0 };
            }
        }
        "volen" => {
            let levels = spec.param_i64("levels", 5).max(1) as usize;
            let fast = spec.param_i64("fast", 60).max(1) as usize;
            let slow = spec.param_i64("slow", 10).max(1) as usize;
            for i in 1..=levels {
                ensure_col(py, panel, panel_df, &format!("ask_volume{}", i))?;
                ensure_col(py, panel, panel_df, &format!("bid_volume{}", i))?;
            }
            let mut ask_cols: Vec<&Vec<f64>> = Vec::with_capacity(levels);
            let mut bid_cols: Vec<&Vec<f64>> = Vec::with_capacity(levels);
            for i in 1..=levels {
                let ask_name = format!("ask_volume{}", i);
                let bid_name = format!("bid_volume{}", i);
                ask_cols.push(panel.col(&ask_name)?);
                bid_cols.push(panel.col(&bid_name)?);
            }
            for (gi, g) in panel.groups.iter().enumerate() {
                let n = g.end - g.start;
                if n == 0 || n < fast || n < slow {
                    out[gi] = 0.0;
                    continue;
                }
                let max_w = fast.max(slow);
                let start_k = n - max_w;
                let fast_start = n - fast;
                let slow_start = n - slow;
                let mut fast_sum = 0.0f64;
                let mut slow_sum = 0.0f64;
                let mut fast_ok = true;
                let mut slow_ok = true;
                for k in start_k..n {
                    let r = g.start + k;
                    let mut s = 0.0;
                    let mut ok = true;
                    for i in 0..levels {
                        let a = ask_cols[i][r];
                        let b = bid_cols[i][r];
                        if !a.is_finite() || !b.is_finite() {
                            ok = false;
                            break;
                        }
                        s += a + b;
                    }
                    let v = if ok { s } else { f64::NAN };
                    if k >= fast_start {
                        if v.is_finite() {
                            fast_sum += v;
                        } else {
                            fast_ok = false;
                        }
                    }
                    if k >= slow_start {
                        if v.is_finite() {
                            slow_sum += v;
                        } else {
                            slow_ok = false;
                        }
                    }
                }
                if fast_ok && slow_ok {
                    let f = fast_sum / fast as f64;
                    let s = slow_sum / slow as f64;
                    let v = if s.abs() > EPS { f / s } else { 0.0 };
                    out[gi] = if v.is_finite() { v } else { 0.0 };
                } else {
                    out[gi] = 0.0;
                }
            }
        }
        "ret_window" => {
            let window_minutes = spec.param_i64("window_minutes", 30);
            let price_col = spec.param_str("price_col", "last");
            ensure_col(py, panel, panel_df, &price_col)?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let ws = cached_window_start(window_cache, panel, gi, g, window_minutes);
                let p = &panel.col(&price_col)?[ws..g.end];
                let first = first_valid(p);
                let last = last_valid(p);
                out[gi] = if first.is_finite() && first.abs() > EPS && last.is_finite() {
                    (last - first) / first
                } else {
                    0.0
                };
            }
        }
        "ret_open_to_time" => {
            let price_col = spec.param_str("price_col", "last");
            ensure_col(py, panel, panel_df, &price_col)?;
            let st = spec.param_str("start_time", "09:30");
            let et = spec.param_str("end_time", "14:30");
            let (sh, sm) = parse_hhmm(&st);
            let (eh, em) = parse_hhmm(&et);
            for (gi, g) in panel.groups.iter().enumerate() {
                let base_ns = date_start_ns(&g.dt).unwrap_or(0);
                let start_ns = base_ns + (sh as i64 * 3600 + sm as i64 * 60) * 1_000_000_000i64;
                let end_ns = base_ns + (eh as i64 * 3600 + em as i64 * 60) * 1_000_000_000i64;
                let col = panel.col(&price_col)?;
                let mut first = f64::NAN;
                let mut last = f64::NAN;
                for r in g.start..g.end {
                    let t = panel.trade_time_ns[r];
                    if t >= start_ns && t <= end_ns {
                        let v = col[r];
                        if v.is_finite() {
                            if !first.is_finite() {
                                first = v;
                            }
                            last = v;
                        }
                    }
                }
                out[gi] = if first.is_finite() && first.abs() > EPS && last.is_finite() {
                    (last - first) / first
                } else {
                    0.0
                };
            }
        }
        "mom_slope" => {
            let window_minutes = spec.param_i64("window_minutes", 30);
            let price_col = spec.param_str("price_col", "last");
            ensure_col(py, panel, panel_df, &price_col)?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let ws = cached_window_start(window_cache, panel, gi, g, window_minutes);
                let px = &panel.col(&price_col)?[ws..g.end];
                let ts = &panel.trade_time_ns[ws..g.end];
                if px.len() < 2 {
                    out[gi] = 0.0;
                    continue;
                }
                let t0 = ts[0] as f64 / 1e9;
                let mut x: Vec<f64> = Vec::new();
                let mut y: Vec<f64> = Vec::new();
                for k in 0..px.len() {
                    if px[k].is_finite() {
                        x.push(ts[k] as f64 / 1e9 - t0);
                        y.push(px[k]);
                    }
                }
                if y.len() < 2 {
                    out[gi] = 0.0;
                    continue;
                }
                let mx = x.iter().sum::<f64>() / x.len() as f64;
                let my = y.iter().sum::<f64>() / y.len() as f64;
                let mut sxy = 0.0;
                let mut sxx = 0.0;
                for k in 0..y.len() {
                    sxy += (x[k] - mx) * (y[k] - my);
                    sxx += (x[k] - mx) * (x[k] - mx);
                }
                out[gi] = if sxx > EPS { sxy / sxx } else { 0.0 };
            }
        }
        "volatility" => {
            let window_minutes = spec.param_i64("window_minutes", 30);
            let price_col = spec.param_str("price_col", "last");
            let use_log = spec.param_bool("use_log_return", true);
            ensure_col(py, panel, panel_df, &price_col)?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let ws = cached_window_start(window_cache, panel, gi, g, window_minutes);
                let p = &panel.col(&price_col)?[ws..g.end];
                let ret = if use_log { log_diff(p) } else { pct_change(p) };
                let s = std_sample(&ret);
                out[gi] = if s.is_finite() { s } else { 0.0 };
            }
        }
        "range_ratio" => {
            let window_minutes = spec.param_i64("window_minutes", 30);
            let price_col = spec.param_str("price_col", "last");
            ensure_col(py, panel, panel_df, &price_col)?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let ws = cached_window_start(window_cache, panel, gi, g, window_minutes);
                let p = &panel.col(&price_col)?[ws..g.end];
                let lo = min_valid(p);
                let hi = max_valid(p);
                let last = last_valid(p);
                out[gi] =
                    if lo.is_finite() && hi.is_finite() && last.is_finite() && last.abs() > EPS {
                        (hi - lo) / last
                    } else {
                        0.0
                    };
            }
        }
        "price_position" => {
            let window_minutes = spec.param_i64("window_minutes", 30);
            let price_col = spec.param_str("price_col", "last");
            ensure_col(py, panel, panel_df, &price_col)?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let ws = cached_window_start(window_cache, panel, gi, g, window_minutes);
                let p = &panel.col(&price_col)?[ws..g.end];
                let lo = min_valid(p);
                let hi = max_valid(p);
                let last = last_valid(p);
                let denom = hi - lo;
                out[gi] = if lo.is_finite() && hi.is_finite() && last.is_finite() && denom > EPS {
                    (last - lo) / denom
                } else {
                    0.0
                };
            }
        }
        "volume_sum" => {
            let window_minutes = spec.param_i64("window_minutes", 30);
            let volume_col = spec.param_str("volume_col", "volume");
            ensure_col(py, panel, panel_df, &volume_col)?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let ws = cached_window_start(window_cache, panel, gi, g, window_minutes);
                out[gi] = sum_valid(&panel.col(&volume_col)?[ws..g.end]);
            }
        }
        "amount_sum" => {
            let window_minutes = spec.param_i64("window_minutes", 30);
            let amount_col = spec.param_str("amount_col", "amount");
            ensure_col(py, panel, panel_df, &amount_col)?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let ws = cached_window_start(window_cache, panel, gi, g, window_minutes);
                out[gi] = sum_valid(&panel.col(&amount_col)?[ws..g.end]);
            }
        }
        "vwap" => {
            let window_minutes = spec.param_i64("window_minutes", 30);
            let volume_col = spec.param_str("volume_col", "volume");
            let amount_col = spec.param_str("amount_col", "amount");
            ensure_col(py, panel, panel_df, &volume_col)?;
            ensure_col(py, panel, panel_df, &amount_col)?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let ws = cached_window_start(window_cache, panel, gi, g, window_minutes);
                let vol = sum_valid(&panel.col(&volume_col)?[ws..g.end]);
                let amt = sum_valid(&panel.col(&amount_col)?[ws..g.end]);
                out[gi] = if vol > EPS { amt / vol } else { 0.0 };
            }
        }
        "volume_imbalance" => {
            let window_minutes = spec.param_i64("window_minutes", 30);
            let levels = spec.param_i64("levels", 3).max(1) as usize;
            for i in 1..=levels {
                ensure_col(py, panel, panel_df, &format!("bid_volume{}", i))?;
                ensure_col(py, panel, panel_df, &format!("ask_volume{}", i))?;
            }
            for (gi, g) in panel.groups.iter().enumerate() {
                let ws = cached_window_start(window_cache, panel, gi, g, window_minutes);
                let mut bid = 0.0;
                let mut ask = 0.0;
                for r in ws..g.end {
                    for i in 1..=levels {
                        let b = panel.col(&format!("bid_volume{}", i))?[r];
                        let a = panel.col(&format!("ask_volume{}", i))?[r];
                        if b.is_finite() {
                            bid += b;
                        }
                        if a.is_finite() {
                            ask += a;
                        }
                    }
                }
                let denom = bid + ask;
                out[gi] = if denom > EPS {
                    (bid - ask) / denom
                } else {
                    0.0
                };
            }
        }
        "spread" => {
            let bid_col = spec.param_str("bid_col", "bid_price1");
            let ask_col = spec.param_str("ask_col", "ask_price1");
            ensure_col(py, panel, panel_df, &bid_col)?;
            ensure_col(py, panel, panel_df, &ask_col)?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let r = g.end - 1;
                let bid = panel.col(&bid_col)?[r];
                let ask = panel.col(&ask_col)?[r];
                let mid = (ask + bid) * 0.5;
                out[gi] = if mid.abs() > EPS && bid.is_finite() && ask.is_finite() {
                    (ask - bid) / mid
                } else {
                    0.0
                };
            }
        }
        "depth_imbalance" => {
            let levels = spec.param_i64("levels", 3).max(1) as usize;
            for i in 1..=levels {
                ensure_col(py, panel, panel_df, &format!("bid_volume{}", i))?;
                ensure_col(py, panel, panel_df, &format!("ask_volume{}", i))?;
            }
            for (gi, g) in panel.groups.iter().enumerate() {
                let r = g.end - 1;
                let mut bid = 0.0;
                let mut ask = 0.0;
                for i in 1..=levels {
                    let b = panel.col(&format!("bid_volume{}", i))?[r];
                    let a = panel.col(&format!("ask_volume{}", i))?[r];
                    if b.is_finite() {
                        bid += b;
                    }
                    if a.is_finite() {
                        ask += a;
                    }
                }
                let denom = bid + ask;
                out[gi] = if denom > EPS {
                    (bid - ask) / denom
                } else {
                    0.0
                };
            }
        }
        "midprice_move" => {
            let window_minutes = spec.param_i64("window_minutes", 30);
            let bid_col = spec.param_str("bid_col", "bid_price1");
            let ask_col = spec.param_str("ask_col", "ask_price1");
            ensure_col(py, panel, panel_df, &bid_col)?;
            ensure_col(py, panel, panel_df, &ask_col)?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let ws = cached_window_start(window_cache, panel, gi, g, window_minutes);
                let mut mids: Vec<f64> = Vec::new();
                for r in ws..g.end {
                    let b = panel.col(&bid_col)?[r];
                    let a = panel.col(&ask_col)?[r];
                    let m = (a + b) * 0.5;
                    if m.is_finite() {
                        mids.push(m);
                    }
                }
                let first = first_valid(&mids);
                let last = last_valid(&mids);
                out[gi] = if first.is_finite() && first.abs() > EPS && last.is_finite() {
                    (last - first) / first
                } else {
                    0.0
                };
            }
        }
        "turnover_rate" => {
            let window_minutes = spec.param_i64("window_minutes", 30);
            let volume_col = spec.param_str("volume_col", "volume");
            let shares_col = spec.param_str("shares_col", "num_trades");
            ensure_col(py, panel, panel_df, &volume_col)?;
            ensure_col(py, panel, panel_df, &shares_col)?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let ws = cached_window_start(window_cache, panel, gi, g, window_minutes);
                let vol = sum_valid(&panel.col(&volume_col)?[ws..g.end]);
                let shares = panel.col(&shares_col)?[g.end - 1];
                out[gi] = if shares.is_finite() && shares > EPS {
                    vol / shares
                } else {
                    0.0
                };
            }
        }
        "amihud_illiq" => {
            let window_minutes = spec.param_i64("window_minutes", 30);
            let price_col = spec.param_str("price_col", "last");
            let amount_col = spec.param_str("amount_col", "amount");
            ensure_col(py, panel, panel_df, &price_col)?;
            ensure_col(py, panel, panel_df, &amount_col)?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let ws = cached_window_start(window_cache, panel, gi, g, window_minutes);
                let p = &panel.col(&price_col)?[ws..g.end];
                let first = first_valid(p);
                let last = last_valid(p);
                let amt = sum_valid(&panel.col(&amount_col)?[ws..g.end]);
                out[gi] = if first.is_finite() && first > EPS && last.is_finite() && amt > EPS {
                    ((last - first) / first).abs() / amt
                } else {
                    0.0
                };
            }
        }
        "microprice_bias" => {
            let bid_col = spec.param_str("bid_col", "bid_price1");
            let ask_col = spec.param_str("ask_col", "ask_price1");
            let bid_vol_col = spec.param_str("bid_vol_col", "bid_volume1");
            let ask_vol_col = spec.param_str("ask_vol_col", "ask_volume1");
            ensure_col(py, panel, panel_df, &bid_col)?;
            ensure_col(py, panel, panel_df, &ask_col)?;
            ensure_col(py, panel, panel_df, &bid_vol_col)?;
            ensure_col(py, panel, panel_df, &ask_vol_col)?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let r = g.end - 1;
                let bid = panel.col(&bid_col)?[r];
                let ask = panel.col(&ask_col)?[r];
                let bv = panel.col(&bid_vol_col)?[r];
                let av = panel.col(&ask_vol_col)?[r];
                let denom = bv + av;
                let mid = (ask + bid) * 0.5;
                out[gi] = if bid > 0.0 && ask > 0.0 && denom > EPS && mid > EPS {
                    let micro = (ask * bv + bid * av) / denom;
                    (micro - mid) / mid
                } else {
                    0.0
                };
            }
        }
        "depth_slope" => {
            let levels = spec.param_i64("levels", 5).max(1) as usize;
            for i in 1..=levels {
                ensure_col(py, panel, panel_df, &format!("bid_price{}", i))?;
                ensure_col(py, panel, panel_df, &format!("ask_price{}", i))?;
                ensure_col(py, panel, panel_df, &format!("bid_volume{}", i))?;
                ensure_col(py, panel, panel_df, &format!("ask_volume{}", i))?;
            }
            ensure_col(py, panel, panel_df, "bid_price1")?;
            ensure_col(py, panel, panel_df, "ask_price1")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let r = g.end - 1;
                let mut bp = vec![0.0; levels];
                let mut ap = vec![0.0; levels];
                let mut bv = vec![0.0; levels];
                let mut av = vec![0.0; levels];
                for i in 0..levels {
                    bp[i] = panel.col(&format!("bid_price{}", i + 1))?[r];
                    ap[i] = panel.col(&format!("ask_price{}", i + 1))?[r];
                    bv[i] = panel.col(&format!("bid_volume{}", i + 1))?[r];
                    av[i] = panel.col(&format!("ask_volume{}", i + 1))?[r];
                }
                let calc_side = |px: &[f64], vol: &[f64]| -> f64 {
                    if px.len() < 2 {
                        return 0.0;
                    }
                    let mut wsum = 0.0;
                    let mut s = 0.0;
                    for i in 0..(px.len() - 1) {
                        let dp = (px[i + 1] - px[i]).abs();
                        let w = (vol[i] + vol[i + 1]) * 0.5;
                        if dp.is_finite() && w.is_finite() && w > 0.0 {
                            s += dp * w;
                            wsum += w;
                        }
                    }
                    if wsum > EPS {
                        s / wsum
                    } else {
                        0.0
                    }
                };
                let bid_slope = calc_side(&bp, &bv);
                let ask_slope = calc_side(&ap, &av);
                let mid = (panel.col("ask_price1")?[r] + panel.col("bid_price1")?[r]) * 0.5;
                out[gi] = if mid > EPS {
                    (ask_slope - bid_slope) / mid
                } else {
                    0.0
                };
            }
        }
        "return_skew" => {
            let window_minutes = spec.param_i64("window_minutes", 30);
            let price_col = spec.param_str("price_col", "last");
            let use_log = spec.param_bool("use_log_return", true);
            ensure_col(py, panel, panel_df, &price_col)?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let ws = cached_window_start(window_cache, panel, gi, g, window_minutes);
                let p = &panel.col(&price_col)?[ws..g.end];
                let ret = if use_log { log_diff(p) } else { pct_change(p) };
                out[gi] = skew_unbiased(&ret);
            }
        }
        "vwap_gap" => {
            let window_minutes = spec.param_i64("window_minutes", 30);
            let price_col = spec.param_str("price_col", "last");
            let volume_col = spec.param_str("volume_col", "volume");
            let amount_col = spec.param_str("amount_col", "amount");
            ensure_col(py, panel, panel_df, &price_col)?;
            ensure_col(py, panel, panel_df, &volume_col)?;
            ensure_col(py, panel, panel_df, &amount_col)?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let ws = cached_window_start(window_cache, panel, gi, g, window_minutes);
                let last = last_valid(&panel.col(&price_col)?[ws..g.end]);
                let vol = sum_valid(&panel.col(&volume_col)?[ws..g.end]);
                let amt = sum_valid(&panel.col(&amount_col)?[ws..g.end]);
                out[gi] = if last.is_finite() && vol > EPS {
                    let vwap = amt / vol;
                    if vwap > EPS {
                        (last - vwap) / vwap
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
            }
        }
        "order_flow_imbalance_v1" => {
            ensure_col(py, panel, panel_df, "bid_volume1")?;
            ensure_col(py, panel, panel_df, "ask_volume1")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let r = g.end - 1;
                let bid = panel.col("bid_volume1")?[r];
                let ask = panel.col("ask_volume1")?[r];
                let denom = bid + ask + EPS;
                out[gi] = if bid.is_finite() && ask.is_finite() {
                    (bid - ask) / denom
                } else {
                    0.0
                };
            }
        }
        "depth_weighted_imbalance_v1" => {
            let weights = spec.param_list_f64("weights", &[5.0, 4.0, 3.0, 2.0, 1.0]);
            let levels = weights.len().max(1);
            for i in 1..=levels {
                ensure_col(py, panel, panel_df, &format!("bid_volume{}", i))?;
                ensure_col(py, panel, panel_df, &format!("ask_volume{}", i))?;
            }
            for (gi, g) in panel.groups.iter().enumerate() {
                let r = g.end - 1;
                let mut bid = 0.0;
                let mut ask = 0.0;
                for i in 0..levels {
                    let w = weights[i];
                    bid += panel.col(&format!("bid_volume{}", i + 1))?[r] * w;
                    ask += panel.col(&format!("ask_volume{}", i + 1))?[r] * w;
                }
                let denom = bid + ask + EPS;
                out[gi] = if bid.is_finite() && ask.is_finite() {
                    (bid - ask) / denom
                } else {
                    0.0
                };
            }
        }
        "intraday_momentum_v1" => {
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "open")?;
            ensure_col(py, panel, panel_df, "ask_price1")?;
            ensure_col(py, panel, panel_df, "bid_price1")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let r = g.end - 1;
                let last = panel.col("last")?[r];
                let open = panel.col("open")?[r];
                let ask1 = panel.col("ask_price1")?[r];
                let bid1 = panel.col("bid_price1")?[r];
                let open_like = open_like_at(open, ask1, bid1);
                out[gi] = if last.is_finite() && open_like.abs() > EPS {
                    (last - open_like) / (open_like + EPS)
                } else {
                    0.0
                };
            }
        }
        "bid_ask_spread_v1" => {
            ensure_col(py, panel, panel_df, "bid_price1")?;
            ensure_col(py, panel, panel_df, "ask_price1")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let r = g.end - 1;
                let bid = panel.col("bid_price1")?[r];
                let ask = panel.col("ask_price1")?[r];
                let mid = (ask + bid) * 0.5;
                out[gi] = if mid.is_finite() {
                    (ask - bid) / (mid + EPS)
                } else {
                    0.0
                };
            }
        }
        "price_level_position_v1" => {
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "high")?;
            ensure_col(py, panel, panel_df, "low")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let r = g.end - 1;
                let last = panel.col("last")?[r];
                let high = panel.col("high")?[r];
                let low = panel.col("low")?[r];
                let spread = high - low;
                out[gi] = if spread > EPS {
                    (last - low) / (spread + EPS)
                } else {
                    0.0
                };
            }
        }
        "volume_price_trend_v1" => {
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "amount")?;
            let has_prev = has_col(py, panel_df, "prev_bar_close")?;
            let has_pre_close = has_col(py, panel_df, "pre_close")?;
            if has_prev {
                ensure_col(py, panel, panel_df, "prev_bar_close")?;
            } else if has_pre_close {
                ensure_col(py, panel, panel_df, "pre_close")?;
            }
            for (gi, g) in panel.groups.iter().enumerate() {
                let r = g.end - 1;
                let last = panel.col("last")?[r];
                let pre = if has_prev {
                    panel.col("prev_bar_close")?[r]
                } else if has_pre_close {
                    panel.col("pre_close")?[r]
                } else {
                    if g.end > g.start + 1 {
                        panel.col("last")?[g.end - 2]
                    } else {
                        f64::NAN
                    }
                };
                let amount = panel.col("amount")?[r];
                out[gi] = if pre.is_finite() && pre.abs() > EPS {
                    ((last - pre) / (pre + EPS)) * amount
                } else {
                    0.0
                };
            }
        }
        "trade_intensity_v1" => {
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "num_trades")?;
            let has_prev = has_col(py, panel_df, "prev_bar_close")?;
            let has_pre_close = has_col(py, panel_df, "pre_close")?;
            if has_prev {
                ensure_col(py, panel, panel_df, "prev_bar_close")?;
            } else if has_pre_close {
                ensure_col(py, panel, panel_df, "pre_close")?;
            }
            for (gi, g) in panel.groups.iter().enumerate() {
                let r = g.end - 1;
                let amount = panel.col("amount")?[r];
                let trades = panel.col("num_trades")?[r];
                let pre = if has_prev {
                    panel.col("prev_bar_close")?[r]
                } else if has_pre_close {
                    panel.col("pre_close")?[r]
                } else {
                    if g.end > g.start + 1 {
                        panel.col("last")?[g.end - 2]
                    } else {
                        f64::NAN
                    }
                };
                out[gi] = if trades > EPS && pre.abs() > EPS {
                    (amount / (trades + EPS)) / (pre + EPS)
                } else {
                    0.0
                };
            }
        }
        "stock_bond_momentum_gap_v1" => {
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "open")?;
            ensure_col(py, panel, panel_df, "ask_price1")?;
            ensure_col(py, panel, panel_df, "bid_price1")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let r = g.end - 1;
                let bond_last = panel.col("last")?[r];
                let bond_open = open_like_at(
                    panel.col("open")?[r],
                    panel.col("ask_price1")?[r],
                    panel.col("bid_price1")?[r],
                );
                let bond_ret = if bond_open.abs() > EPS {
                    (bond_last - bond_open) / (bond_open + EPS)
                } else {
                    0.0
                };

                let bcode = norm_code(&g.code);
                let dtk = norm_dt(&g.dt);
                let Some(stock_code) = aux.bond_stock_map.get(&bcode) else {
                    out[gi] = 0.0;
                    continue;
                };
                let Some(sq) = aux.stock_latest.get(&(dtk, stock_code.clone())) else {
                    out[gi] = 0.0;
                    continue;
                };
                let stock_open = open_like_at(sq.open, sq.ask_price1, sq.bid_price1);
                let stock_ret = if stock_open.abs() > EPS {
                    (sq.last - stock_open) / (stock_open + EPS)
                } else {
                    0.0
                };
                out[gi] = bond_ret - stock_ret;
            }
        }
        "premium_momentum_proxy_v1" => {
            ensure_col(py, panel, panel_df, "last")?;
            let has_prev = has_col(py, panel_df, "prev_bar_close")?;
            let has_pre_close = has_col(py, panel_df, "pre_close")?;
            if has_prev {
                ensure_col(py, panel, panel_df, "prev_bar_close")?;
            } else if has_pre_close {
                ensure_col(py, panel, panel_df, "pre_close")?;
            }
            for (gi, g) in panel.groups.iter().enumerate() {
                let r = g.end - 1;
                let bond_last = panel.col("last")?[r];
                let bond_prev = if has_prev {
                    panel.col("prev_bar_close")?[r]
                } else if has_pre_close {
                    panel.col("pre_close")?[r]
                } else if g.end > g.start + 1 {
                    panel.col("last")?[g.end - 2]
                } else {
                    f64::NAN
                };
                let bond_strength = if bond_prev.abs() > EPS {
                    (bond_last - bond_prev) / (bond_prev + EPS)
                } else {
                    0.0
                };

                let bcode = norm_code(&g.code);
                let dtk = norm_dt(&g.dt);
                let Some(stock_code) = aux.bond_stock_map.get(&bcode) else {
                    out[gi] = 0.0;
                    continue;
                };
                let Some(sq) = aux.stock_latest.get(&(dtk, stock_code.clone())) else {
                    out[gi] = 0.0;
                    continue;
                };
                let stock_prev = if sq.prev_bar_close.is_finite() {
                    sq.prev_bar_close
                } else {
                    sq.pre_close
                };
                let stock_strength = if stock_prev.abs() > EPS {
                    (sq.last - stock_prev) / (stock_prev + EPS)
                } else {
                    0.0
                };
                out[gi] = bond_strength - stock_strength;
            }
        }
        "volatility_scaled_return_v1" => {
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "open")?;
            ensure_col(py, panel, panel_df, "ask_price1")?;
            ensure_col(py, panel, panel_df, "bid_price1")?;
            ensure_col(py, panel, panel_df, "high")?;
            ensure_col(py, panel, panel_df, "low")?;
            let has_prev = has_col(py, panel_df, "prev_bar_close")?;
            let has_pre_close = has_col(py, panel_df, "pre_close")?;
            if has_prev {
                ensure_col(py, panel, panel_df, "prev_bar_close")?;
            } else if has_pre_close {
                ensure_col(py, panel, panel_df, "pre_close")?;
            }
            for (gi, g) in panel.groups.iter().enumerate() {
                let r = g.end - 1;
                let last = panel.col("last")?[r];
                let open = panel.col("open")?[r];
                let ask1 = panel.col("ask_price1")?[r];
                let bid1 = panel.col("bid_price1")?[r];
                let open_like = open_like_at(open, ask1, bid1);
                let high = panel.col("high")?[r];
                let low = panel.col("low")?[r];
                let pre = if has_prev {
                    panel.col("prev_bar_close")?[r]
                } else if has_pre_close {
                    panel.col("pre_close")?[r]
                } else {
                    if g.end > g.start + 1 {
                        panel.col("last")?[g.end - 2]
                    } else {
                        f64::NAN
                    }
                };
                let range = (high - low) / (pre + EPS);
                if range > EPS && open_like.abs() > EPS {
                    let ret = (last - open_like) / (open_like + EPS);
                    out[gi] = ret / (range + EPS);
                } else {
                    out[gi] = 0.0;
                }
            }
        }
        "alpha001_signed_power_v1" => {
            let stddev_window = spec.param_i64("stddev_window", 20).max(2) as usize;
            let ts_max_window = spec.param_i64("ts_max_window", 5).max(1) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            let has_prev = has_col(py, panel_df, "prev_bar_close")?;
            let has_pre_close = has_col(py, panel_df, "pre_close")?;
            if has_prev {
                ensure_col(py, panel, panel_df, "prev_bar_close")?;
            } else if has_pre_close {
                ensure_col(py, panel, panel_df, "pre_close")?;
            }
            let mut raw = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &panel.col("last")?[g.start..g.end];
                let mut pre: Vec<f64> = vec![f64::NAN; last.len()];
                if has_prev {
                    let p = &panel.col("prev_bar_close")?[g.start..g.end];
                    pre.copy_from_slice(p);
                } else if has_pre_close {
                    let p = &panel.col("pre_close")?[g.start..g.end];
                    pre.copy_from_slice(p);
                } else {
                    for i in 0..last.len() {
                        pre[i] = if i > 0 { last[i - 1] } else { f64::NAN };
                    }
                }
                let mut ret = vec![0.0; last.len()];
                for i in 0..last.len() {
                    let l = last[i];
                    let p = pre[i];
                    ret[i] = if l.is_finite() && p.is_finite() && p.abs() > EPS {
                        (l - p) / (p + EPS)
                    } else {
                        0.0
                    };
                }
                let mut std_ret = vec![0.0; ret.len()];
                for i in 0..ret.len() {
                    let s = i + 1usize - (i + 1).min(stddev_window);
                    let seg = &ret[s..=i];
                    let v = std_sample(seg);
                    std_ret[i] = if v.is_finite() { v } else { 0.0 };
                }
                let mut sp = vec![0.0; ret.len()];
                for i in 0..ret.len() {
                    let base = if ret[i] < 0.0 { std_ret[i] } else { last[i] };
                    sp[i] = base.signum() * base.abs() * base.abs();
                }
                raw[gi] = rolling_max_last(&sp, ts_max_window);
                if !raw[gi].is_finite() {
                    raw[gi] = 0.0;
                }
            }
            out = cs_rank_by_dt(&panel.groups, &raw)
                .into_iter()
                .map(|x| x - 0.5)
                .collect();
        }
        "alpha002_corr_volume_return_v1" => {
            let corr_window = spec.param_i64("corr_window", 6).max(2) as usize;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "open")?;
            ensure_col(py, panel, panel_df, "ask_price1")?;
            ensure_col(py, panel, panel_df, "bid_price1")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let volume = &panel.col("volume")?[g.start..g.end];
                let last = &panel.col("last")?[g.start..g.end];
                let open = &panel.col("open")?[g.start..g.end];
                let ask = &panel.col("ask_price1")?[g.start..g.end];
                let bid = &panel.col("bid_price1")?[g.start..g.end];
                let mut logv = vec![f64::NAN; volume.len()];
                for i in 0..volume.len() {
                    let v = volume[i].max(0.0);
                    logv[i] = (v + EPS).ln();
                }
                let dlog = diff(&logv, 2);
                let mut ret = vec![f64::NAN; last.len()];
                for i in 0..last.len() {
                    let op = open_like_at(open[i], ask[i], bid[i]);
                    ret[i] = if op.abs() > EPS && last[i].is_finite() {
                        (last[i] - op) / (op + EPS)
                    } else {
                        f64::NAN
                    };
                }
                let x = pct_rank(&dlog);
                let y = pct_rank(&ret);
                out[gi] = -corr_last_window(&x, &y, corr_window);
            }
        }
        "alpha003_corr_open_volume_v1" => {
            let corr_window = spec.param_i64("corr_window", 10).max(2) as usize;
            ensure_col(py, panel, panel_df, "open")?;
            ensure_col(py, panel, panel_df, "ask_price1")?;
            ensure_col(py, panel, panel_df, "bid_price1")?;
            ensure_col(py, panel, panel_df, "volume")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let open = &panel.col("open")?[g.start..g.end];
                let ask = &panel.col("ask_price1")?[g.start..g.end];
                let bid = &panel.col("bid_price1")?[g.start..g.end];
                let vol = &panel.col("volume")?[g.start..g.end];
                let mut o = vec![f64::NAN; open.len()];
                for i in 0..open.len() {
                    o[i] = open_like_at(open[i], ask[i], bid[i]);
                }
                let open_rank = pct_rank(&o);
                let vol_rank = pct_rank(vol);
                out[gi] = -corr_last_window(&open_rank, &vol_rank, corr_window);
            }
        }
        "alpha004_ts_rank_low_v1" => {
            let ts_rank_window = spec.param_i64("ts_rank_window", 9).max(1) as usize;
            ensure_col(py, panel, panel_df, "low")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let low = &panel.col("low")?[g.start..g.end];
                let low_rank = pct_rank(low);
                out[gi] = -ts_rank_last(&low_rank, ts_rank_window);
            }
        }
        "alpha005_vwap_gap_v1" => {
            let vwap_window = spec.param_i64("vwap_window", 10).max(1) as usize;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "open")?;
            ensure_col(py, panel, panel_df, "ask_price1")?;
            ensure_col(py, panel, panel_df, "bid_price1")?;
            ensure_col(py, panel, panel_df, "last")?;
            let mut gap1 = vec![0.0; panel.groups.len()];
            let mut gap2 = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let amount = &panel.col("amount")?[g.start..g.end];
                let volume = &panel.col("volume")?[g.start..g.end];
                let open = &panel.col("open")?[g.start..g.end];
                let ask = &panel.col("ask_price1")?[g.start..g.end];
                let bid = &panel.col("bid_price1")?[g.start..g.end];
                let last = &panel.col("last")?[g.start..g.end];
                let mut vwap = vec![f64::NAN; amount.len()];
                for i in 0..amount.len() {
                    vwap[i] = if volume[i].abs() > EPS {
                        amount[i] / (volume[i] + EPS)
                    } else {
                        f64::NAN
                    };
                }
                let mut open_like = vec![f64::NAN; open.len()];
                for i in 0..open.len() {
                    open_like[i] = open_like_at(open[i], ask[i], bid[i]);
                }
                let avg_vwap_last = rolling_mean_last(&vwap, vwap_window);
                let open_last = last_valid(&open_like);
                let vwap_last = last_valid(&vwap);
                let last_px = last_valid(last);
                gap1[gi] = if open_last.is_finite() && avg_vwap_last.is_finite() {
                    open_last - avg_vwap_last
                } else {
                    0.0
                };
                gap2[gi] = if last_px.is_finite() && vwap_last.is_finite() {
                    last_px - vwap_last
                } else {
                    0.0
                };
            }
            let rank1 = cs_rank_by_dt(&panel.groups, &gap1);
            let rank2 = cs_rank_by_dt(&panel.groups, &gap2);
            for i in 0..out.len() {
                out[i] = rank1[i] * (-rank2[i].abs());
            }
        }
        "alpha006_corr_open_volume_neg_v1" => {
            let corr_window = spec.param_i64("corr_window", 10).max(2) as usize;
            ensure_col(py, panel, panel_df, "open")?;
            ensure_col(py, panel, panel_df, "ask_price1")?;
            ensure_col(py, panel, panel_df, "bid_price1")?;
            ensure_col(py, panel, panel_df, "volume")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let open = &panel.col("open")?[g.start..g.end];
                let ask = &panel.col("ask_price1")?[g.start..g.end];
                let bid = &panel.col("bid_price1")?[g.start..g.end];
                let vol = &panel.col("volume")?[g.start..g.end];
                let mut o = vec![f64::NAN; open.len()];
                for i in 0..open.len() {
                    o[i] = open_like_at(open[i], ask[i], bid[i]);
                }
                out[gi] = -corr_last_window(&o, vol, corr_window);
            }
        }
        "alpha007_volume_breakout_v1" => {
            let adv_window = spec.param_i64("adv_window", 20).max(1) as usize;
            let delta_window = spec.param_i64("delta_window", 7).max(1) as usize;
            let ts_rank_window = spec.param_i64("ts_rank_window", 60).max(1) as usize;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "last")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let amount = &panel.col("amount")?[g.start..g.end];
                let volume = &panel.col("volume")?[g.start..g.end];
                let last = &panel.col("last")?[g.start..g.end];
                let adv = rolling_mean_last(amount, adv_window);
                let d = diff(last, delta_window);
                let d_last = last_valid(&d);
                if adv >= last_valid(volume) {
                    out[gi] = -1.0;
                } else {
                    let sign = if d_last.is_finite() {
                        d_last.signum()
                    } else {
                        0.0
                    };
                    let mut abs_d = Vec::with_capacity(d.len());
                    for v in d {
                        abs_d.push(v.abs());
                    }
                    let ts = ts_rank_last(&abs_d, ts_rank_window);
                    out[gi] = (-1.0 * ts) * sign;
                }
            }
        }
        "alpha008_open_return_momentum_v1" => {
            let sum_window = spec.param_i64("sum_window", 5).max(1) as usize;
            let delay_window = spec.param_i64("delay_window", 10).max(1) as usize;
            ensure_col(py, panel, panel_df, "open")?;
            ensure_col(py, panel, panel_df, "ask_price1")?;
            ensure_col(py, panel, panel_df, "bid_price1")?;
            ensure_col(py, panel, panel_df, "last")?;
            let mut raw = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let open = &panel.col("open")?[g.start..g.end];
                let ask = &panel.col("ask_price1")?[g.start..g.end];
                let bid = &panel.col("bid_price1")?[g.start..g.end];
                let last = &panel.col("last")?[g.start..g.end];
                let n = open.len();
                if n == 0 {
                    raw[gi] = 0.0;
                    continue;
                }
                let mut o = vec![f64::NAN; n];
                let mut ret = vec![f64::NAN; n];
                for i in 0..n {
                    o[i] = open_like_at(open[i], ask[i], bid[i]);
                    ret[i] = if o[i].abs() > EPS {
                        (last[i] - o[i]) / (o[i] + EPS)
                    } else {
                        f64::NAN
                    };
                }
                let mut sum_o = vec![f64::NAN; n];
                let mut sum_r = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(sum_window);
                    sum_o[i] = sum_valid(&o[s..=i]);
                    sum_r[i] = sum_valid(&ret[s..=i]);
                }
                let mut prod = vec![f64::NAN; n];
                for i in 0..n {
                    prod[i] = sum_o[i] * sum_r[i];
                }
                let delay_idx = n.saturating_sub(1 + delay_window);
                let delayed = if n > delay_window {
                    prod[delay_idx]
                } else {
                    f64::NAN
                };
                let delay_val = if delayed.is_finite() {
                    delayed
                } else {
                    prod[0]
                };
                raw[gi] = prod[n - 1] - delay_val;
            }
            out = cs_rank_by_dt(&panel.groups, &raw)
                .into_iter()
                .map(|x| -x)
                .collect();
        }
        "alpha009_close_change_filter_v1" => {
            let ts_window = spec.param_i64("ts_window", 5).max(1) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let p = &panel.col("last")?[g.start..g.end];
                let d = diff(p, 1);
                let last_d = last_valid(&d);
                let ts_min = rolling_min_last(&d, ts_window);
                let ts_max = rolling_max_last(&d, ts_window);
                out[gi] = if ts_min > 0.0 {
                    if last_d.is_finite() {
                        last_d
                    } else {
                        0.0
                    }
                } else if ts_max < 0.0 {
                    if last_d.is_finite() {
                        last_d
                    } else {
                        0.0
                    }
                } else {
                    if last_d.is_finite() {
                        -last_d
                    } else {
                        0.0
                    }
                };
            }
        }
        "alpha010_close_change_rank_v1" => {
            let ts_window = spec.param_i64("ts_window", 4).max(1) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            let mut raw = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let p = &panel.col("last")?[g.start..g.end];
                let d = diff(p, 1);
                let last_d = last_valid(&d);
                let ts_min = rolling_min_last(&d, ts_window);
                let ts_max = rolling_max_last(&d, ts_window);
                raw[gi] = if ts_min > 0.0 {
                    if last_d.is_finite() {
                        last_d
                    } else {
                        0.0
                    }
                } else if ts_max < 0.0 {
                    if last_d.is_finite() {
                        last_d
                    } else {
                        0.0
                    }
                } else {
                    if last_d.is_finite() {
                        -last_d
                    } else {
                        0.0
                    }
                };
            }
            out = cs_rank_by_dt(&panel.groups, &raw);
        }
        "alpha011_vwap_close_volume_v1" => {
            let ts_window = spec.param_i64("ts_window", 3).max(1) as usize;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "last")?;
            let amount_col = panel.col("amount")?;
            let volume_col = panel.col("volume")?;
            let last_col = panel.col("last")?;
            let mut max_raw = vec![f64::NAN; panel.groups.len()];
            let mut min_raw = vec![f64::NAN; panel.groups.len()];
            let mut delta_raw = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let amount = &amount_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let last = &last_col[g.start..g.end];
                let n = amount.len();
                if n == 0 {
                    max_raw[gi] = f64::NAN;
                    min_raw[gi] = f64::NAN;
                    delta_raw[gi] = 0.0;
                    continue;
                }
                let mut diff = vec![f64::NAN; n];
                for i in 0..n {
                    let amt = amount[i];
                    let vol = volume[i];
                    let px = last[i];
                    if amt.is_finite() && vol.is_finite() && px.is_finite() {
                        let vwap = amt / (vol + EPS);
                        diff[i] = vwap - px;
                    }
                }
                max_raw[gi] = rolling_max_last(&diff, ts_window);
                min_raw[gi] = rolling_min_last(&diff, ts_window);
                delta_raw[gi] = delta_last_from_slice(volume, ts_window);
            }
            let rank_max = cs_rank_by_dt(&panel.groups, &max_raw);
            let rank_min = cs_rank_by_dt(&panel.groups, &min_raw);
            let rank_delta = cs_rank_by_dt(&panel.groups, &delta_raw);
            for i in 0..out.len() {
                out[i] = (rank_max[i] + rank_min[i]) * rank_delta[i];
            }
        }
        "alpha012_volume_close_reversal_v1" => {
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "last")?;
            let volume_col = panel.col("volume")?;
            let last_col = panel.col("last")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let volume = &volume_col[g.start..g.end];
                let last = &last_col[g.start..g.end];
                let d_vol = delta_last_from_slice(volume, 1);
                let d_last = delta_last_from_slice(last, 1);
                out[gi] = if d_vol.is_finite() && d_last.is_finite() {
                    d_vol.signum() * (-d_last)
                } else {
                    0.0
                };
            }
        }
        "alpha013_cov_close_volume_v1" => {
            let cov_window = spec.param_i64("cov_window", 5).max(2) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "volume")?;
            let close_col = panel.col("last")?;
            let volume_col = panel.col("volume")?;
            let mut raw = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let close_rank = pct_rank(&close_col[g.start..g.end]);
                let vol_rank = pct_rank(&volume_col[g.start..g.end]);
                raw[gi] = cov_last_window(&close_rank, &vol_rank, cov_window);
            }
            let ranks = cs_rank_by_dt(&panel.groups, &raw);
            for i in 0..out.len() {
                out[i] = -ranks[i];
            }
        }
        "alpha014_return_open_volume_v1" => {
            let delta_window = spec.param_i64("delta_window", 3).max(1) as usize;
            let corr_window = spec.param_i64("corr_window", 10).max(2) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "open")?;
            ensure_col(py, panel, panel_df, "ask_price1")?;
            ensure_col(py, panel, panel_df, "bid_price1")?;
            ensure_col(py, panel, panel_df, "volume")?;
            let has_prev = has_col(py, panel_df, "prev_bar_close")?;
            let has_pre_close = has_col(py, panel_df, "pre_close")?;
            if has_prev {
                ensure_col(py, panel, panel_df, "prev_bar_close")?;
            } else if has_pre_close {
                ensure_col(py, panel, panel_df, "pre_close")?;
            }
            let last_col = panel.col("last")?;
            let open_col = panel.col("open")?;
            let ask_col = panel.col("ask_price1")?;
            let bid_col = panel.col("bid_price1")?;
            let volume_col = panel.col("volume")?;
            let mut delta_raw = vec![0.0; panel.groups.len()];
            let mut corr_raw = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let open = &open_col[g.start..g.end];
                let ask = &ask_col[g.start..g.end];
                let bid = &bid_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let n = g.end - g.start;
                let mut ret = vec![f64::NAN; n];
                let mut open_like = vec![f64::NAN; n];
                for i in 0..n {
                    let pre = if has_prev {
                        panel.col("prev_bar_close")?[g.start + i]
                    } else if has_pre_close {
                        panel.col("pre_close")?[g.start + i]
                    } else if i > 0 {
                        last[i - 1]
                    } else {
                        f64::NAN
                    };
                    let l = last[i];
                    ret[i] = if l.is_finite() && pre.is_finite() && pre.abs() > EPS {
                        (l - pre) / (pre + EPS)
                    } else {
                        f64::NAN
                    };
                    open_like[i] = open_like_at(open[i], ask[i], bid[i]);
                }
                delta_raw[gi] = delta_last_from_slice(&ret, delta_window);
                corr_raw[gi] = corr_last_window(&open_like, volume, corr_window);
            }
            let delta_rank = cs_rank_by_dt(&panel.groups, &delta_raw);
            for i in 0..out.len() {
                out[i] = (-delta_rank[i]) * corr_raw[i];
            }
        }
        "alpha015_high_volume_corr_v1" => {
            let corr_window = spec.param_i64("corr_window", 3).max(2) as usize;
            let sum_window = spec.param_i64("sum_window", 3).max(1) as usize;
            ensure_col(py, panel, panel_df, "high")?;
            ensure_col(py, panel, panel_df, "volume")?;
            let high_col = panel.col("high")?;
            let volume_col = panel.col("volume")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let high_rank = pct_rank(&high_col[g.start..g.end]);
                let vol_rank = pct_rank(&volume_col[g.start..g.end]);
                let n = high_rank.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let mut corr_series = vec![0.0; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(corr_window);
                    let c = corr_slice(&high_rank[s..=i], &vol_rank[s..=i]);
                    corr_series[i] = if c.is_finite() { c } else { 0.0 };
                }
                let ranked_corr = pct_rank(&corr_series);
                let start = n.saturating_sub(sum_window);
                let mut acc = 0.0;
                for v in &ranked_corr[start..n] {
                    if v.is_finite() {
                        acc += *v;
                    }
                }
                out[gi] = -acc;
            }
        }
        "alpha016_cov_high_volume_v1" => {
            let cov_window = spec.param_i64("cov_window", 5).max(2) as usize;
            ensure_col(py, panel, panel_df, "high")?;
            ensure_col(py, panel, panel_df, "volume")?;
            let high_col = panel.col("high")?;
            let volume_col = panel.col("volume")?;
            let mut raw = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let high_rank = pct_rank(&high_col[g.start..g.end]);
                let vol_rank = pct_rank(&volume_col[g.start..g.end]);
                raw[gi] = cov_last_window(&high_rank, &vol_rank, cov_window);
            }
            let ranks = cs_rank_by_dt(&panel.groups, &raw);
            for i in 0..out.len() {
                out[i] = -ranks[i];
            }
        }
        "alpha017_close_rank_volume_v1" => {
            let adv_window = spec.param_i64("adv_window", 20).max(1) as usize;
            let ts_rank_close_window = spec.param_i64("ts_rank_close_window", 10).max(1) as usize;
            let ts_rank_vol_window = spec.param_i64("ts_rank_vol_window", 5).max(1) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            let last_col = panel.col("last")?;
            let amount_col = panel.col("amount")?;
            let volume_col = panel.col("volume")?;
            let mut t1_raw = vec![0.0; panel.groups.len()];
            let mut t2_raw = vec![0.0; panel.groups.len()];
            let mut t3_raw = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let amount = &amount_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let n = last.len();
                t1_raw[gi] = ts_rank_last(last, ts_rank_close_window);
                if n >= 3 {
                    let a = last[n - 1];
                    let b = last[n - 2];
                    let c = last[n - 3];
                    t2_raw[gi] = if a.is_finite() && b.is_finite() && c.is_finite() {
                        a - 2.0 * b + c
                    } else {
                        0.0
                    };
                } else {
                    t2_raw[gi] = 0.0;
                }
                let mut ratio = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(adv_window);
                    let adv = mean_valid(&amount[s..=i]);
                    ratio[i] = if adv.is_finite() && volume[i].is_finite() {
                        volume[i] / (adv + EPS)
                    } else {
                        f64::NAN
                    };
                }
                t3_raw[gi] = ts_rank_last(&ratio, ts_rank_vol_window);
            }
            let t1_rank = cs_rank_by_dt(&panel.groups, &t1_raw);
            let t2_rank = cs_rank_by_dt(&panel.groups, &t2_raw);
            let t3_rank = cs_rank_by_dt(&panel.groups, &t3_raw);
            for i in 0..out.len() {
                out[i] = (-t1_rank[i]) * t2_rank[i] * t3_rank[i];
            }
        }
        "alpha018_close_open_vol_v1" => {
            let stddev_window = spec.param_i64("stddev_window", 5).max(2) as usize;
            let corr_window = spec.param_i64("corr_window", 10).max(2) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "open")?;
            ensure_col(py, panel, panel_df, "ask_price1")?;
            ensure_col(py, panel, panel_df, "bid_price1")?;
            let last_col = panel.col("last")?;
            let open_col = panel.col("open")?;
            let ask_col = panel.col("ask_price1")?;
            let bid_col = panel.col("bid_price1")?;
            let mut raw = vec![f64::NAN; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let open = &open_col[g.start..g.end];
                let ask = &ask_col[g.start..g.end];
                let bid = &bid_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    raw[gi] = f64::NAN;
                    continue;
                }
                let mut open_like = vec![f64::NAN; n];
                let mut diff = vec![f64::NAN; n];
                for i in 0..n {
                    open_like[i] = open_like_at(open[i], ask[i], bid[i]);
                    diff[i] = if last[i].is_finite() && open_like[i].is_finite() {
                        last[i] - open_like[i]
                    } else {
                        f64::NAN
                    };
                }
                let s = n.saturating_sub(stddev_window);
                let mut abs_tail = Vec::with_capacity(n - s);
                for &v in &diff[s..n] {
                    if v.is_finite() {
                        abs_tail.push(v.abs());
                    }
                }
                let std_diff = if abs_tail.len() >= 2 {
                    let v = std_sample(&abs_tail);
                    if v.is_finite() {
                        v
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                let corr_co = corr_last_window(last, &open_like, corr_window);
                let diff_last = diff[n - 1];
                raw[gi] = if diff_last.is_finite() {
                    std_diff + diff_last + corr_co
                } else {
                    f64::NAN
                };
            }
            let ranks = cs_rank_by_dt(&panel.groups, &raw);
            for i in 0..out.len() {
                out[i] = -ranks[i];
            }
        }
        "alpha019_close_momentum_sign_v1" => {
            let delta_window = spec.param_i64("delta_window", 7).max(1) as usize;
            let sum_window = spec.param_i64("sum_window", 250).max(1) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            let has_prev = has_col(py, panel_df, "prev_bar_close")?;
            let has_pre_close = has_col(py, panel_df, "pre_close")?;
            if has_prev {
                ensure_col(py, panel, panel_df, "prev_bar_close")?;
            } else if has_pre_close {
                ensure_col(py, panel, panel_df, "pre_close")?;
            }
            let last_col = panel.col("last")?;
            let mut sign_raw = vec![0.0; panel.groups.len()];
            let mut sum_raw = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    sign_raw[gi] = 0.0;
                    sum_raw[gi] = 0.0;
                    continue;
                }
                let delayed = delay_last_from_slice(last, delta_window);
                let last_change = last[n - 1] - delayed;
                let delta_last = delta_last_from_slice(last, delta_window);
                let delta_value = if delta_last.is_finite() {
                    delta_last
                } else {
                    0.0
                };
                let sign_term = (last_change + delta_value).signum();
                sign_raw[gi] = if sign_term.is_finite() {
                    sign_term
                } else {
                    0.0
                };

                let mut ret = vec![f64::NAN; n];
                for i in 0..n {
                    let pre = if has_prev {
                        panel.col("prev_bar_close")?[g.start + i]
                    } else if has_pre_close {
                        panel.col("pre_close")?[g.start + i]
                    } else if i > 0 {
                        last[i - 1]
                    } else {
                        f64::NAN
                    };
                    ret[i] = if last[i].is_finite() && pre.is_finite() && pre.abs() > EPS {
                        (last[i] - pre) / (pre + EPS)
                    } else {
                        f64::NAN
                    };
                }
                let start = n.saturating_sub(sum_window);
                sum_raw[gi] = sum_valid(&ret[start..n]);
            }
            let one_plus_sum: Vec<f64> = sum_raw.iter().map(|v| 1.0 + *v).collect();
            let sum_rank = cs_rank_by_dt(&panel.groups, &one_plus_sum);
            for i in 0..out.len() {
                out[i] = (-sign_raw[i]) * (1.0 + sum_rank[i]);
            }
        }
        "alpha020_open_delay_range_v1" => {
            let delay_window = spec.param_i64("delay_window", 1).max(1) as usize;
            ensure_col(py, panel, panel_df, "open")?;
            ensure_col(py, panel, panel_df, "ask_price1")?;
            ensure_col(py, panel, panel_df, "bid_price1")?;
            ensure_col(py, panel, panel_df, "high")?;
            ensure_col(py, panel, panel_df, "low")?;
            ensure_col(py, panel, panel_df, "last")?;
            let open_col = panel.col("open")?;
            let ask_col = panel.col("ask_price1")?;
            let bid_col = panel.col("bid_price1")?;
            let high_col = panel.col("high")?;
            let low_col = panel.col("low")?;
            let last_col = panel.col("last")?;
            let mut d1_raw = vec![0.0; panel.groups.len()];
            let mut d2_raw = vec![0.0; panel.groups.len()];
            let mut d3_raw = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let open = &open_col[g.start..g.end];
                let ask = &ask_col[g.start..g.end];
                let bid = &bid_col[g.start..g.end];
                let high = &high_col[g.start..g.end];
                let low = &low_col[g.start..g.end];
                let last = &last_col[g.start..g.end];
                let n = open.len();
                if n == 0 {
                    d1_raw[gi] = 0.0;
                    d2_raw[gi] = 0.0;
                    d3_raw[gi] = 0.0;
                    continue;
                }
                let open_last = open_like_at(open[n - 1], ask[n - 1], bid[n - 1]);
                d1_raw[gi] = open_last - delay_last_from_slice(high, delay_window);
                d2_raw[gi] = open_last - delay_last_from_slice(last, delay_window);
                d3_raw[gi] = open_last - delay_last_from_slice(low, delay_window);
            }
            let d1_rank = cs_rank_by_dt(&panel.groups, &d1_raw);
            let d2_rank = cs_rank_by_dt(&panel.groups, &d2_raw);
            let d3_rank = cs_rank_by_dt(&panel.groups, &d3_raw);
            for i in 0..out.len() {
                out[i] = (-d1_rank[i]) * d2_rank[i] * d3_rank[i];
            }
        }
        "alpha021_close_volatility_breakout_v1" => {
            let sum_window_long = spec.param_i64("sum_window_long", 5).max(1) as usize;
            let sum_window_short = spec.param_i64("sum_window_short", 2).max(1) as usize;
            let adv_window = spec.param_i64("adv_window", 10).max(1) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "amount")?;
            let last_col = panel.col("last")?;
            let volume_col = panel.col("volume")?;
            let amount_col = panel.col("amount")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let amount = &amount_col[g.start..g.end];
                if last.is_empty() {
                    out[gi] = 0.0;
                    continue;
                }
                let avg_long = rolling_mean_last(last, sum_window_long);
                let s_long = last.len().saturating_sub(sum_window_long);
                let std_long = {
                    let v = std_sample(&last[s_long..]);
                    if v.is_finite() {
                        v
                    } else {
                        0.0
                    }
                };
                let avg_short = rolling_mean_last(last, sum_window_short);
                let upper = avg_long + std_long;
                let lower = avg_long - std_long;
                let adv = rolling_mean_last(amount, adv_window);
                let vol_ratio = volume[volume.len() - 1] / (adv + EPS);
                out[gi] = if upper < avg_short {
                    -1.0
                } else if avg_short < lower {
                    1.0
                } else if vol_ratio >= 1.0 {
                    1.0
                } else {
                    -1.0
                };
            }
        }
        "alpha022_high_volume_corr_change_v1" => {
            let corr_window = spec.param_i64("corr_window", 5).max(2) as usize;
            let delta_window = spec.param_i64("delta_window", 5).max(1) as usize;
            let stddev_window = spec.param_i64("stddev_window", 10).max(2) as usize;
            ensure_col(py, panel, panel_df, "high")?;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "last")?;
            let high_col = panel.col("high")?;
            let volume_col = panel.col("volume")?;
            let last_col = panel.col("last")?;
            let mut delta_corr_raw = vec![0.0; panel.groups.len()];
            let mut std_close_raw = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let high = &high_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let last = &last_col[g.start..g.end];
                let n = high.len();
                if n == 0 {
                    delta_corr_raw[gi] = 0.0;
                    std_close_raw[gi] = 0.0;
                    continue;
                }
                let mut corr_series = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(corr_window);
                    let c = corr_slice(&high[s..=i], &volume[s..=i]);
                    corr_series[i] = if c.is_finite() { c } else { f64::NAN };
                }
                let c_last = corr_series[n - 1];
                let c_prev = if n > delta_window {
                    corr_series[n - 1 - delta_window]
                } else {
                    f64::NAN
                };
                delta_corr_raw[gi] = if c_last.is_finite() && c_prev.is_finite() {
                    c_last - c_prev
                } else {
                    0.0
                };
                let s = n.saturating_sub(stddev_window);
                let stdc = std_sample(&last[s..n]);
                std_close_raw[gi] = if stdc.is_finite() { stdc } else { 0.0 };
            }
            let rank_std = cs_rank_by_dt(&panel.groups, &std_close_raw);
            for i in 0..out.len() {
                out[i] = -(delta_corr_raw[i] * rank_std[i]);
            }
        }
        "alpha023_high_momentum_v1" => {
            let sum_window = spec.param_i64("sum_window", 10).max(1) as usize;
            let delta_window = spec.param_i64("delta_window", 2).max(1) as usize;
            ensure_col(py, panel, panel_df, "high")?;
            let high_col = panel.col("high")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let high = &high_col[g.start..g.end];
                if high.is_empty() {
                    out[gi] = 0.0;
                    continue;
                }
                let avg_high = rolling_mean_last(high, sum_window);
                if avg_high < high[high.len() - 1] {
                    out[gi] = -delta_last_from_slice(high, delta_window);
                } else {
                    out[gi] = 0.0;
                }
            }
        }
        "alpha024_close_trend_filter_v1" => {
            let sum_window = spec.param_i64("sum_window", 20).max(1) as usize;
            let delta_window = spec.param_i64("delta_window", 20).max(1) as usize;
            let ts_min_window = spec.param_i64("ts_min_window", 20).max(1) as usize;
            let short_delta_window = spec.param_i64("short_delta_window", 3).max(1) as usize;
            let trend_threshold = spec.param_f64("trend_threshold", 0.05);
            ensure_col(py, panel, panel_df, "last")?;
            let last_col = panel.col("last")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let mut avg_close = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(sum_window);
                    avg_close[i] = mean_valid(&last[s..=i]);
                }
                let rate = if n > delta_window {
                    let cur = avg_close[n - 1];
                    let base = avg_close[n - 1 - delta_window];
                    if cur.is_finite() && base.is_finite() {
                        (cur - base) / (base + EPS)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                if rate <= trend_threshold {
                    let ts_min = rolling_min_last(last, ts_min_window);
                    out[gi] = -(last[n - 1] - ts_min);
                } else {
                    out[gi] = -delta_last_from_slice(last, short_delta_window);
                }
            }
        }
        "alpha025_return_volume_vwap_range_v1" => {
            let adv_window = spec.param_i64("adv_window", 10).max(1) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "high")?;
            let has_prev = has_col(py, panel_df, "prev_bar_close")?;
            let has_pre_close = has_col(py, panel_df, "pre_close")?;
            if has_prev {
                ensure_col(py, panel, panel_df, "prev_bar_close")?;
            } else if has_pre_close {
                ensure_col(py, panel, panel_df, "pre_close")?;
            }
            let last_col = panel.col("last")?;
            let amount_col = panel.col("amount")?;
            let volume_col = panel.col("volume")?;
            let high_col = panel.col("high")?;
            let mut raw = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let n = g.end - g.start;
                if n == 0 {
                    raw[gi] = 0.0;
                    continue;
                }
                let last = last_col[g.end - 1];
                let pre = if has_prev {
                    panel.col("prev_bar_close")?[g.end - 1]
                } else if has_pre_close {
                    panel.col("pre_close")?[g.end - 1]
                } else if n > 1 {
                    last_col[g.end - 2]
                } else {
                    f64::NAN
                };
                let returns = if last.is_finite() && pre.is_finite() && pre.abs() > EPS {
                    (last - pre) / (pre + EPS)
                } else {
                    0.0
                };
                let amount = &amount_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let adv = rolling_mean_last(amount, adv_window);
                let vwap = amount[n - 1] / (volume[n - 1] + EPS);
                let high = high_col[g.end - 1];
                let price_range = high - last;
                raw[gi] = ((-returns) * adv) * vwap * price_range;
            }
            out = cs_rank_by_dt(&panel.groups, &raw);
        }
        "alpha026_volume_high_rank_corr_v1" => {
            let ts_rank_window = spec.param_i64("ts_rank_window", 5).max(1) as usize;
            let corr_window = spec.param_i64("corr_window", 5).max(2) as usize;
            let ts_max_window = spec.param_i64("ts_max_window", 3).max(1) as usize;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "high")?;
            let volume_col = panel.col("volume")?;
            let high_col = panel.col("high")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let volume = &volume_col[g.start..g.end];
                let high = &high_col[g.start..g.end];
                let ts_rank_vol = rolling_last_rank_pct_series(volume, ts_rank_window);
                let ts_rank_high = rolling_last_rank_pct_series(high, ts_rank_window);
                let n = volume.len();
                let mut corr_series = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(corr_window);
                    let c = corr_slice(&ts_rank_vol[s..=i], &ts_rank_high[s..=i]);
                    corr_series[i] = if c.is_finite() { c } else { f64::NAN };
                }
                let ts_max = rolling_max_last(&corr_series, ts_max_window);
                out[gi] = if ts_max.is_finite() { -ts_max } else { 0.0 };
            }
        }
        "alpha027_volume_vwap_corr_signal_v1" => {
            let corr_window = spec.param_i64("corr_window", 6).max(2) as usize;
            let sum_window = spec.param_i64("sum_window", 2).max(1) as usize;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            let amount_col = panel.col("amount")?;
            let volume_col = panel.col("volume")?;
            let mut avg_corr = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let amount = &amount_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let n = volume.len();
                if n == 0 {
                    avg_corr[gi] = 0.0;
                    continue;
                }
                let mut vwap = vec![f64::NAN; n];
                for i in 0..n {
                    vwap[i] = amount[i] / (volume[i] + EPS);
                }
                let r1 = pct_rank(volume);
                let r2 = pct_rank(&vwap);
                let mut corr = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(corr_window);
                    let c = corr_slice(&r1[s..=i], &r2[s..=i]);
                    corr[i] = if c.is_finite() { c } else { f64::NAN };
                }
                let m = rolling_mean_last(&corr, sum_window);
                avg_corr[gi] = if m.is_finite() { m } else { 0.0 };
            }
            let rank_avg = cs_rank_by_dt(&panel.groups, &avg_corr);
            for i in 0..out.len() {
                out[i] = if rank_avg[i] > 0.5 { -1.0 } else { 1.0 };
            }
        }
        "alpha028_adv_low_close_signal_v1" => {
            let adv_window = spec.param_i64("adv_window", 10).max(1) as usize;
            let corr_window = spec.param_i64("corr_window", 5).max(2) as usize;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "high")?;
            ensure_col(py, panel, panel_df, "low")?;
            ensure_col(py, panel, panel_df, "last")?;
            let amount_col = panel.col("amount")?;
            let high_col = panel.col("high")?;
            let low_col = panel.col("low")?;
            let last_col = panel.col("last")?;
            let mut raw = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let amount = &amount_col[g.start..g.end];
                let low = &low_col[g.start..g.end];
                let high = &high_col[g.start..g.end];
                let last = &last_col[g.start..g.end];
                let n = amount.len();
                if n == 0 {
                    raw[gi] = 0.0;
                    continue;
                }
                let mut adv = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(adv_window);
                    adv[i] = mean_valid(&amount[s..=i]);
                }
                let corr = corr_last_window(&adv, low, corr_window);
                let mid_price = (high[n - 1] + low[n - 1]) / 2.0;
                raw[gi] = (corr + mid_price) - last[n - 1];
            }
            out = cs_rank_by_dt(&panel.groups, &raw);
        }
        "alpha029_complex_rank_signal_v1" => {
            let ts_min_window = spec.param_i64("ts_min_window", 2).max(1) as usize;
            let ts_rank_window = spec.param_i64("ts_rank_window", 5).max(1) as usize;
            let delay_window = spec.param_i64("delay_window", 3).max(1) as usize;
            let min_window = spec.param_i64("min_window", 5).max(1) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            let has_prev = has_col(py, panel_df, "prev_bar_close")?;
            let has_pre_close = has_col(py, panel_df, "pre_close")?;
            if has_prev {
                ensure_col(py, panel, panel_df, "prev_bar_close")?;
            } else if has_pre_close {
                ensure_col(py, panel, panel_df, "pre_close")?;
            }
            let last_col = panel.col("last")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let mut pre = vec![f64::NAN; n];
                for i in 0..n {
                    pre[i] = if has_prev {
                        panel.col("prev_bar_close")?[g.start + i]
                    } else if has_pre_close {
                        panel.col("pre_close")?[g.start + i]
                    } else if i > 0 {
                        last[i - 1]
                    } else {
                        f64::NAN
                    };
                }
                let mut returns = vec![f64::NAN; n];
                for i in 0..n {
                    returns[i] = if last[i].is_finite() && pre[i].is_finite() && pre[i].abs() > EPS
                    {
                        (last[i] - pre[i]) / (pre[i] + EPS)
                    } else {
                        f64::NAN
                    };
                }
                let delta_close = diff(last, 1);
                let mut neg_delta = vec![f64::NAN; n];
                for i in 0..n {
                    neg_delta[i] = if delta_close[i].is_finite() {
                        -delta_close[i]
                    } else {
                        f64::NAN
                    };
                }
                let rank_delta = pct_rank(&neg_delta);
                let mut ts_min_rank = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(ts_min_window);
                    ts_min_rank[i] = rolling_min_last(&rank_delta[s..=i], ts_min_window);
                }
                let mut log_sum = vec![f64::NAN; n];
                for i in 0..n {
                    if ts_min_rank[i].is_finite() {
                        log_sum[i] = (ts_min_rank[i].abs() + EPS).ln();
                    }
                }
                let finite: Vec<f64> = log_sum.iter().copied().filter(|v| v.is_finite()).collect();
                let mean_log = if finite.is_empty() {
                    0.0
                } else {
                    finite.iter().sum::<f64>() / finite.len() as f64
                };
                let denom = finite.iter().map(|v| (v - mean_log).abs()).sum::<f64>() + EPS;
                let mut scaled = vec![f64::NAN; n];
                for i in 0..n {
                    if log_sum[i].is_finite() {
                        scaled[i] = (log_sum[i] - mean_log) / denom;
                    }
                }
                let rank_scaled = pct_rank(&scaled);
                let min_rank = rolling_min_last(&rank_scaled, min_window);
                let mut delay_ret = vec![f64::NAN; n];
                for i in 0..n {
                    let x = if returns[i].is_finite() {
                        -returns[i]
                    } else {
                        f64::NAN
                    };
                    delay_ret[i] = x;
                }
                if delay_window > 0 {
                    for i in (0..n).rev() {
                        delay_ret[i] = if i >= delay_window {
                            delay_ret[i - delay_window]
                        } else {
                            f64::NAN
                        };
                    }
                }
                let ts_rank_ret = ts_rank_last(&delay_ret, ts_rank_window);
                out[gi] = min_rank + ts_rank_ret;
            }
        }
        "alpha030_close_sign_volume_v1" => {
            let delay1 = spec.param_i64("delay1", 1).max(1) as usize;
            let delay2 = spec.param_i64("delay2", 2).max(1) as usize;
            let delay3 = spec.param_i64("delay3", 3).max(1) as usize;
            let sum_window_short = spec.param_i64("sum_window_short", 5).max(1) as usize;
            let sum_window_long = spec.param_i64("sum_window_long", 10).max(1) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "volume")?;
            let last_col = panel.col("last")?;
            let volume_col = panel.col("volume")?;
            let mut sign_sum = vec![0.0; panel.groups.len()];
            let mut vol_ratio = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    continue;
                }
                let l0 = last[n - 1];
                let l1 = delay_value_or_nan(last, delay1);
                let l2 = delay_value_or_nan(last, delay2);
                let l3 = delay_value_or_nan(last, delay3);
                let s1 = if l0.is_finite() && l1.is_finite() {
                    (l0 - l1).signum()
                } else {
                    0.0
                };
                let s2 = if l1.is_finite() && l2.is_finite() {
                    (l1 - l2).signum()
                } else {
                    0.0
                };
                let s3 = if l2.is_finite() && l3.is_finite() {
                    (l2 - l3).signum()
                } else {
                    0.0
                };
                sign_sum[gi] = s1 + s2 + s3;
                let s_short = {
                    let s = n.saturating_sub(sum_window_short);
                    sum_valid(&volume[s..n])
                };
                let s_long = {
                    let s = n.saturating_sub(sum_window_long);
                    sum_valid(&volume[s..n])
                };
                vol_ratio[gi] = s_short / (s_long + EPS);
            }
            let rank_sign = cs_rank_by_dt(&panel.groups, &sign_sum);
            for i in 0..out.len() {
                out[i] = (1.0 - rank_sign[i]) * vol_ratio[i];
            }
        }
        "alpha031_close_decay_momentum_v1" => {
            let delta_window = spec.param_i64("delta_window", 10).max(1) as usize;
            let decay_window = spec.param_i64("decay_window", 10).max(1) as usize;
            let delta_short_window = spec.param_i64("delta_short_window", 3).max(1) as usize;
            let corr_window = spec.param_i64("corr_window", 12).max(2) as usize;
            let adv_window = spec.param_i64("adv_window", 10).max(1) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "low")?;
            let last_col = panel.col("last")?;
            let amount_col = panel.col("amount")?;
            let low_col = panel.col("low")?;
            let mut decay_raw = vec![0.0; panel.groups.len()];
            let mut short_raw = vec![0.0; panel.groups.len()];
            let mut corr_raw = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let amount = &amount_col[g.start..g.end];
                let low = &low_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    continue;
                }
                let delta_close = diff(last, delta_window);
                let rank_delta = pct_rank(&delta_close);
                let tail_s = n.saturating_sub(decay_window);
                let mut num = 0.0;
                let mut den = 0.0;
                let mut w = 1.0;
                for &v in &rank_delta[tail_s..n] {
                    if v.is_finite() {
                        num += (-v) * w;
                    }
                    den += w;
                    w += 1.0;
                }
                decay_raw[gi] = if den > 0.0 { num / den } else { 0.0 };
                short_raw[gi] = -delta_last_from_slice(last, delta_short_window);
                let mut adv = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(adv_window);
                    adv[i] = mean_valid(&amount[s..=i]);
                }
                corr_raw[gi] = corr_last_window(&adv, low, corr_window);
            }
            let rank_decay = cs_rank_by_dt(&panel.groups, &decay_raw);
            let rank_short = cs_rank_by_dt(&panel.groups, &short_raw);
            let rank_corr = cs_rank_by_dt(&panel.groups, &corr_raw);
            for i in 0..out.len() {
                let corr_sign = (rank_corr[i] - 0.5).signum();
                out[i] = rank_decay[i] + rank_short[i] + corr_sign;
            }
        }
        "alpha032_vwap_close_mean_reversion_v1" => {
            let sum_window = spec.param_i64("sum_window", 7).max(1) as usize;
            let corr_window = spec.param_i64("corr_window", 60).max(2) as usize;
            let delay_window = spec.param_i64("delay_window", 5).max(1) as usize;
            let corr_scale = spec.param_f64("corr_scale", 20.0);
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            let last_col = panel.col("last")?;
            let amount_col = panel.col("amount")?;
            let volume_col = panel.col("volume")?;
            let mut raw = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let amount = &amount_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    continue;
                }
                let avg_close = rolling_mean_last(last, sum_window);
                let diff1 = avg_close - last[n - 1];
                let mut vwap = vec![f64::NAN; n];
                for i in 0..n {
                    vwap[i] = amount[i] / (volume[i] + EPS);
                }
                let mut delayed_last = vec![f64::NAN; n];
                for i in 0..n {
                    delayed_last[i] = if i >= delay_window {
                        last[i - delay_window]
                    } else {
                        f64::NAN
                    };
                }
                let corr = corr_last_window(&vwap, &delayed_last, corr_window);
                raw[gi] = diff1 + corr_scale * corr;
            }
            out = cs_rank_by_dt(&panel.groups, &raw);
        }
        "alpha033_open_close_ratio_v1" => {
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "open")?;
            ensure_col(py, panel, panel_df, "ask_price1")?;
            ensure_col(py, panel, panel_df, "bid_price1")?;
            let last_col = panel.col("last")?;
            let open_col = panel.col("open")?;
            let ask_col = panel.col("ask_price1")?;
            let bid_col = panel.col("bid_price1")?;
            let mut raw = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let r = g.end - 1;
                let last = last_col[r];
                let open_like = open_like_at(open_col[r], ask_col[r], bid_col[r]);
                let ratio = 1.0 - open_like / (last + EPS);
                raw[gi] = -ratio;
            }
            out = cs_rank_by_dt(&panel.groups, &raw);
        }
        "alpha034_return_volatility_rank_v1" => {
            let stddev_window_short = spec.param_i64("stddev_window_short", 2).max(2) as usize;
            let stddev_window_long = spec.param_i64("stddev_window_long", 5).max(2) as usize;
            let delta_window = spec.param_i64("delta_window", 1).max(1) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            let has_prev = has_col(py, panel_df, "prev_bar_close")?;
            let has_pre_close = has_col(py, panel_df, "pre_close")?;
            if has_prev {
                ensure_col(py, panel, panel_df, "prev_bar_close")?;
            } else if has_pre_close {
                ensure_col(py, panel, panel_df, "pre_close")?;
            }
            let last_col = panel.col("last")?;
            let mut vol_ratio = vec![0.0; panel.groups.len()];
            let mut delta_close = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    continue;
                }
                let mut pre = vec![f64::NAN; n];
                for i in 0..n {
                    pre[i] = if has_prev {
                        panel.col("prev_bar_close")?[g.start + i]
                    } else if has_pre_close {
                        panel.col("pre_close")?[g.start + i]
                    } else if i > 0 {
                        last[i - 1]
                    } else {
                        f64::NAN
                    };
                }
                let mut ret = vec![f64::NAN; n];
                for i in 0..n {
                    ret[i] = if last[i].is_finite() && pre[i].is_finite() && pre[i].abs() > EPS {
                        (last[i] - pre[i]) / (pre[i] + EPS)
                    } else {
                        f64::NAN
                    };
                }
                let s1 = n.saturating_sub(stddev_window_short);
                let s2 = n.saturating_sub(stddev_window_long);
                let std_short = {
                    let v = std_sample(&ret[s1..n]);
                    if v.is_finite() {
                        v
                    } else {
                        0.0
                    }
                };
                let std_long = {
                    let v = std_sample(&ret[s2..n]);
                    if v.is_finite() {
                        v
                    } else {
                        0.0
                    }
                };
                vol_ratio[gi] = std_short / (std_long + EPS);
                delta_close[gi] = delta_last_from_slice(last, delta_window);
            }
            let rank_vol = cs_rank_by_dt(&panel.groups, &vol_ratio);
            let rank_delta = cs_rank_by_dt(&panel.groups, &delta_close);
            for i in 0..out.len() {
                out[i] = (1.0 - rank_vol[i]) + (1.0 - rank_delta[i]);
            }
        }
        "alpha035_volume_price_momentum_v1" => {
            let ts_rank_window_long = spec.param_i64("ts_rank_window_long", 20).max(1) as usize;
            let ts_rank_window_short = spec.param_i64("ts_rank_window_short", 16).max(1) as usize;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "high")?;
            ensure_col(py, panel, panel_df, "low")?;
            let has_prev = has_col(py, panel_df, "prev_bar_close")?;
            let has_pre_close = has_col(py, panel_df, "pre_close")?;
            if has_prev {
                ensure_col(py, panel, panel_df, "prev_bar_close")?;
            } else if has_pre_close {
                ensure_col(py, panel, panel_df, "pre_close")?;
            }
            let volume_col = panel.col("volume")?;
            let last_col = panel.col("last")?;
            let high_col = panel.col("high")?;
            let low_col = panel.col("low")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let volume = &volume_col[g.start..g.end];
                let last = &last_col[g.start..g.end];
                let high = &high_col[g.start..g.end];
                let low = &low_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let ts_rank_vol = ts_rank_last(volume, ts_rank_window_long);
                let mut price_range = vec![f64::NAN; n];
                for i in 0..n {
                    price_range[i] = (last[i] + high[i]) - low[i];
                }
                let ts_rank_price = 1.0 - ts_rank_last(&price_range, ts_rank_window_short);
                let mut pre = vec![f64::NAN; n];
                for i in 0..n {
                    pre[i] = if has_prev {
                        panel.col("prev_bar_close")?[g.start + i]
                    } else if has_pre_close {
                        panel.col("pre_close")?[g.start + i]
                    } else if i > 0 {
                        last[i - 1]
                    } else {
                        f64::NAN
                    };
                }
                let mut ret = vec![f64::NAN; n];
                for i in 0..n {
                    ret[i] = if last[i].is_finite() && pre[i].is_finite() && pre[i].abs() > EPS {
                        (last[i] - pre[i]) / (pre[i] + EPS)
                    } else {
                        f64::NAN
                    };
                }
                let ts_rank_ret = 1.0 - ts_rank_last(&ret, ts_rank_window_long);
                out[gi] = ts_rank_vol * ts_rank_price * ts_rank_ret;
            }
        }
        "alpha036_complex_correlation_signal_v1" => {
            let corr_window_1 = spec.param_i64("corr_window_1", 15).max(2) as usize;
            let corr_window_2 = spec.param_i64("corr_window_2", 6).max(2) as usize;
            let sum_window = spec.param_i64("sum_window", 60).max(1) as usize;
            let ts_rank_window = spec.param_i64("ts_rank_window", 5).max(1) as usize;
            let delay_window = spec.param_i64("delay_window", 6).max(1) as usize;
            let adv_window = spec.param_i64("adv_window", 10).max(1) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "open")?;
            ensure_col(py, panel, panel_df, "ask_price1")?;
            ensure_col(py, panel, panel_df, "bid_price1")?;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "amount")?;
            let has_prev = has_col(py, panel_df, "prev_bar_close")?;
            let has_pre_close = has_col(py, panel_df, "pre_close")?;
            if has_prev {
                ensure_col(py, panel, panel_df, "prev_bar_close")?;
            } else if has_pre_close {
                ensure_col(py, panel, panel_df, "pre_close")?;
            }
            let last_col = panel.col("last")?;
            let open_col = panel.col("open")?;
            let ask_col = panel.col("ask_price1")?;
            let bid_col = panel.col("bid_price1")?;
            let volume_col = panel.col("volume")?;
            let amount_col = panel.col("amount")?;
            let mut t1 = vec![0.0; panel.groups.len()];
            let mut t2 = vec![0.0; panel.groups.len()];
            let mut t3 = vec![0.0; panel.groups.len()];
            let mut t4 = vec![0.0; panel.groups.len()];
            let mut t5 = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let open = &open_col[g.start..g.end];
                let ask = &ask_col[g.start..g.end];
                let bid = &bid_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let amount = &amount_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    continue;
                }
                let mut open_like = vec![f64::NAN; n];
                let mut diff1 = vec![f64::NAN; n];
                for i in 0..n {
                    open_like[i] = open_like_at(open[i], ask[i], bid[i]);
                    diff1[i] = if last[i].is_finite() && open_like[i].is_finite() {
                        last[i] - open_like[i]
                    } else {
                        f64::NAN
                    };
                }
                let mut delay_vol = vec![f64::NAN; n];
                for i in 0..n {
                    delay_vol[i] = if i >= 1 { volume[i - 1] } else { f64::NAN };
                }
                t1[gi] = corr_last_window(&diff1, &delay_vol, corr_window_1);
                t2[gi] = open_like[n - 1] - last[n - 1];

                let mut pre = vec![f64::NAN; n];
                for i in 0..n {
                    pre[i] = if has_prev {
                        panel.col("prev_bar_close")?[g.start + i]
                    } else if has_pre_close {
                        panel.col("pre_close")?[g.start + i]
                    } else if i > 0 {
                        last[i - 1]
                    } else {
                        f64::NAN
                    };
                }
                let mut ret = vec![f64::NAN; n];
                for i in 0..n {
                    ret[i] = if last[i].is_finite() && pre[i].is_finite() && pre[i].abs() > EPS {
                        (last[i] - pre[i]) / (pre[i] + EPS)
                    } else {
                        f64::NAN
                    };
                }
                let mut delay_ret = vec![f64::NAN; n];
                for i in 0..n {
                    let x = if ret[i].is_finite() {
                        -ret[i]
                    } else {
                        f64::NAN
                    };
                    delay_ret[i] = if i >= delay_window { x } else { f64::NAN };
                }
                if delay_window > 0 {
                    for i in (0..n).rev() {
                        delay_ret[i] = if i >= delay_window {
                            delay_ret[i - delay_window]
                        } else {
                            f64::NAN
                        };
                    }
                }
                t3[gi] = ts_rank_last(&delay_ret, ts_rank_window);

                let mut vwap = vec![f64::NAN; n];
                let mut adv = vec![f64::NAN; n];
                for i in 0..n {
                    vwap[i] = amount[i] / (volume[i] + EPS);
                    let s = i + 1usize - (i + 1).min(adv_window);
                    adv[i] = mean_valid(&amount[s..=i]);
                }
                t4[gi] = corr_last_window(&vwap, &adv, corr_window_2).abs();
                let avg_close = rolling_mean_last(last, sum_window);
                t5[gi] = (avg_close - open_like[n - 1]) * (last[n - 1] - open_like[n - 1]);
            }
            let r1 = cs_rank_by_dt(&panel.groups, &t1);
            let r2 = cs_rank_by_dt(&panel.groups, &t2);
            let r3 = cs_rank_by_dt(&panel.groups, &t3);
            let r4 = cs_rank_by_dt(&panel.groups, &t4);
            let r5 = cs_rank_by_dt(&panel.groups, &t5);
            for i in 0..out.len() {
                out[i] = 2.21 * r1[i] + 0.70 * r2[i] + 0.73 * r3[i] + r4[i] + 0.60 * r5[i];
            }
        }
        "alpha037_open_close_correlation_v1" => {
            let corr_window = spec.param_i64("corr_window", 30).max(2) as usize;
            let delay_window = spec.param_i64("delay_window", 1).max(1) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "open")?;
            ensure_col(py, panel, panel_df, "ask_price1")?;
            ensure_col(py, panel, panel_df, "bid_price1")?;
            let last_col = panel.col("last")?;
            let open_col = panel.col("open")?;
            let ask_col = panel.col("ask_price1")?;
            let bid_col = panel.col("bid_price1")?;
            let mut corr_term = vec![0.0; panel.groups.len()];
            let mut diff_term = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let open = &open_col[g.start..g.end];
                let ask = &ask_col[g.start..g.end];
                let bid = &bid_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    continue;
                }
                let mut diff = vec![f64::NAN; n];
                for i in 0..n {
                    let o = open_like_at(open[i], ask[i], bid[i]);
                    diff[i] = o - last[i];
                }
                let mut delay_diff = vec![f64::NAN; n];
                for i in 0..n {
                    delay_diff[i] = if i >= delay_window {
                        diff[i - delay_window]
                    } else {
                        f64::NAN
                    };
                }
                corr_term[gi] = corr_last_window(&delay_diff, last, corr_window);
                diff_term[gi] = diff[n - 1];
            }
            let rc = cs_rank_by_dt(&panel.groups, &corr_term);
            let rd = cs_rank_by_dt(&panel.groups, &diff_term);
            for i in 0..out.len() {
                out[i] = rc[i] + rd[i];
            }
        }
        "alpha038_close_rank_ratio_v1" => {
            let ts_rank_window = spec.param_i64("ts_rank_window", 10).max(1) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "open")?;
            ensure_col(py, panel, panel_df, "ask_price1")?;
            ensure_col(py, panel, panel_df, "bid_price1")?;
            let last_col = panel.col("last")?;
            let open_col = panel.col("open")?;
            let ask_col = panel.col("ask_price1")?;
            let bid_col = panel.col("bid_price1")?;
            let mut rank_close = vec![0.0; panel.groups.len()];
            let mut ratio = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    continue;
                }
                rank_close[gi] = ts_rank_last(last, ts_rank_window);
                let r = g.end - 1;
                let open_like = open_like_at(open_col[r], ask_col[r], bid_col[r]);
                ratio[gi] = last[n - 1] / (open_like + EPS);
            }
            let rc = cs_rank_by_dt(&panel.groups, &rank_close);
            let rr = cs_rank_by_dt(&panel.groups, &ratio);
            for i in 0..out.len() {
                out[i] = (-rc[i]) * rr[i];
            }
        }
        "alpha039_volume_decay_momentum_v1" => {
            let adv_window = spec.param_i64("adv_window", 10).max(1) as usize;
            let decay_window = spec.param_i64("decay_window", 9).max(1) as usize;
            let delta_window = spec.param_i64("delta_window", 7).max(1) as usize;
            let sum_window = spec.param_i64("sum_window", 60).max(1) as usize;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "last")?;
            let has_prev = has_col(py, panel_df, "prev_bar_close")?;
            let has_pre_close = has_col(py, panel_df, "pre_close")?;
            if has_prev {
                ensure_col(py, panel, panel_df, "prev_bar_close")?;
            } else if has_pre_close {
                ensure_col(py, panel, panel_df, "pre_close")?;
            }
            let amount_col = panel.col("amount")?;
            let volume_col = panel.col("volume")?;
            let last_col = panel.col("last")?;
            let mut term = vec![0.0; panel.groups.len()];
            let mut sum_ret = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let amount = &amount_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let last = &last_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    continue;
                }
                let mut vol_ratio = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(adv_window);
                    let adv = mean_valid(&amount[s..=i]);
                    vol_ratio[i] = volume[i] / (adv + EPS);
                }
                let vol_decay_rank = ts_rank_last(&vol_ratio, decay_window);
                let delta_close = delta_last_from_slice(last, delta_window);
                term[gi] = delta_close * (1.0 - vol_decay_rank);
                let mut pre = vec![f64::NAN; n];
                for i in 0..n {
                    pre[i] = if has_prev {
                        panel.col("prev_bar_close")?[g.start + i]
                    } else if has_pre_close {
                        panel.col("pre_close")?[g.start + i]
                    } else if i > 0 {
                        last[i - 1]
                    } else {
                        f64::NAN
                    };
                }
                let mut ret = vec![f64::NAN; n];
                for i in 0..n {
                    ret[i] = if last[i].is_finite() && pre[i].is_finite() && pre[i].abs() > EPS {
                        (last[i] - pre[i]) / (pre[i] + EPS)
                    } else {
                        f64::NAN
                    };
                }
                let s = n.saturating_sub(sum_window);
                sum_ret[gi] = sum_valid(&ret[s..n]);
            }
            let r_term = cs_rank_by_dt(&panel.groups, &term);
            let sum_plus_one: Vec<f64> = sum_ret.iter().map(|v| v + 1.0).collect();
            let r_sum = cs_rank_by_dt(&panel.groups, &sum_plus_one);
            for i in 0..out.len() {
                out[i] = (-r_term[i]) * (1.0 + r_sum[i]);
            }
        }
        "alpha040_high_volatility_corr_v1" => {
            let stddev_window = spec.param_i64("stddev_window", 10).max(2) as usize;
            let corr_window = spec.param_i64("corr_window", 10).max(2) as usize;
            ensure_col(py, panel, panel_df, "high")?;
            ensure_col(py, panel, panel_df, "volume")?;
            let high_col = panel.col("high")?;
            let volume_col = panel.col("volume")?;
            let mut std_high = vec![0.0; panel.groups.len()];
            let mut corr_hv = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let high = &high_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let n = high.len();
                if n == 0 {
                    continue;
                }
                let s = n.saturating_sub(stddev_window);
                let v = std_sample(&high[s..n]);
                std_high[gi] = if v.is_finite() { v } else { 0.0 };
                corr_hv[gi] = corr_last_window(high, volume, corr_window);
            }
            let rank_std = cs_rank_by_dt(&panel.groups, &std_high);
            for i in 0..out.len() {
                out[i] = (-rank_std[i]) * corr_hv[i];
            }
        }
        "alpha041_geometric_mean_vwap_v1" => {
            ensure_col(py, panel, panel_df, "high")?;
            ensure_col(py, panel, panel_df, "low")?;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            let high_col = panel.col("high")?;
            let low_col = panel.col("low")?;
            let amount_col = panel.col("amount")?;
            let volume_col = panel.col("volume")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let r = g.end - 1;
                let high = high_col[r].max(0.0);
                let low = low_col[r].max(0.0);
                let geo = (high * low).sqrt();
                let vwap = amount_col[r] / (volume_col[r] + EPS);
                out[gi] = geo - vwap;
            }
        }
        "alpha042_vwap_close_rank_ratio_v1" => {
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "last")?;
            let amount_col = panel.col("amount")?;
            let volume_col = panel.col("volume")?;
            let last_col = panel.col("last")?;
            let mut diff_raw = vec![0.0; panel.groups.len()];
            let mut sum_raw = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let r = g.end - 1;
                let vwap = amount_col[r] / (volume_col[r] + EPS);
                let last = last_col[r];
                diff_raw[gi] = vwap - last;
                sum_raw[gi] = vwap + last;
            }
            let rd = cs_rank_by_dt(&panel.groups, &diff_raw);
            let rs = cs_rank_by_dt(&panel.groups, &sum_raw);
            for i in 0..out.len() {
                out[i] = rd[i] / (rs[i] + EPS);
            }
        }
        "alpha043_volume_delay_momentum_v1" => {
            let adv_window = spec.param_i64("adv_window", 10).max(1) as usize;
            let ts_rank_window_1 = spec.param_i64("ts_rank_window_1", 10).max(1) as usize;
            let delta_window = spec.param_i64("delta_window", 5).max(1) as usize;
            let ts_rank_window_2 = spec.param_i64("ts_rank_window_2", 5).max(1) as usize;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "last")?;
            let amount_col = panel.col("amount")?;
            let volume_col = panel.col("volume")?;
            let last_col = panel.col("last")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let amount = &amount_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let last = &last_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let mut vol_ratio = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(adv_window);
                    let adv = mean_valid(&amount[s..=i]);
                    vol_ratio[i] = volume[i] / (adv + EPS);
                }
                let ts_rank_vol = ts_rank_last(&vol_ratio, ts_rank_window_1);
                let delta_close = diff(last, delta_window);
                let mut neg_delta = vec![f64::NAN; n];
                for i in 0..n {
                    neg_delta[i] = if delta_close[i].is_finite() {
                        -delta_close[i]
                    } else {
                        f64::NAN
                    };
                }
                let ts_rank_delta = ts_rank_last(&neg_delta, ts_rank_window_2);
                out[gi] = ts_rank_vol * ts_rank_delta;
            }
        }
        "alpha044_high_volume_rank_corr_v1" => {
            let corr_window = spec.param_i64("corr_window", 5).max(2) as usize;
            ensure_col(py, panel, panel_df, "high")?;
            ensure_col(py, panel, panel_df, "volume")?;
            let high_col = panel.col("high")?;
            let volume_col = panel.col("volume")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let high = &high_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let rank_vol = pct_rank(volume);
                out[gi] = -corr_last_window(high, &rank_vol, corr_window);
            }
        }
        "alpha045_close_sum_corr_v1" => {
            let delay_window = spec.param_i64("delay_window", 5).max(1) as usize;
            let sum_window_long = spec.param_i64("sum_window_long", 20).max(1) as usize;
            let corr_window_1 = spec.param_i64("corr_window_1", 2).max(2) as usize;
            let sum_window_short = spec.param_i64("sum_window_short", 5).max(1) as usize;
            let corr_window_2 = spec.param_i64("corr_window_2", 2).max(2) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "volume")?;
            let last_col = panel.col("last")?;
            let volume_col = panel.col("volume")?;
            let mut avg_delay = vec![0.0; panel.groups.len()];
            let mut corr1 = vec![0.0; panel.groups.len()];
            let mut corr2 = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    continue;
                }
                let mut delayed = vec![f64::NAN; n];
                for i in 0..n {
                    delayed[i] = if i >= delay_window {
                        last[i - delay_window]
                    } else {
                        f64::NAN
                    };
                }
                avg_delay[gi] = rolling_mean_last(&delayed, sum_window_long);
                corr1[gi] = corr_last_window(last, volume, corr_window_1);
                let mut sum_short = vec![f64::NAN; n];
                let mut sum_long = vec![f64::NAN; n];
                for i in 0..n {
                    let s1 = i + 1usize - (i + 1).min(sum_window_short);
                    let s2 = i + 1usize - (i + 1).min(sum_window_long);
                    sum_short[i] = sum_valid(&last[s1..=i]);
                    sum_long[i] = sum_valid(&last[s2..=i]);
                }
                corr2[gi] = corr_last_window(&sum_short, &sum_long, corr_window_2);
            }
            let rank_avg = cs_rank_by_dt(&panel.groups, &avg_delay);
            let rank_c2 = cs_rank_by_dt(&panel.groups, &corr2);
            for i in 0..out.len() {
                out[i] = -(rank_avg[i] * corr1[i] * rank_c2[i]);
            }
        }
        "alpha046_close_delay_trend_v1" => {
            let delay_window_long = spec.param_i64("delay_window_long", 10).max(1) as usize;
            let delay_window_short = spec.param_i64("delay_window_short", 5).max(1) as usize;
            let trend_scale = spec.param_f64("trend_scale", 10.0);
            let threshold_up = spec.param_f64("threshold_up", 0.25);
            let threshold_down = spec.param_f64("threshold_down", 0.0);
            ensure_col(py, panel, panel_df, "last")?;
            let last_col = panel.col("last")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let d_long = delay_value_or_nan(last, delay_window_long);
                let d_short = delay_value_or_nan(last, delay_window_short);
                let diff1 = (d_long - d_short) / trend_scale;
                let diff2 = (d_short - last[n - 1]) / trend_scale;
                let trend = if diff1.is_finite() && diff2.is_finite() {
                    diff1 - diff2
                } else {
                    0.0
                };
                let delta_close = if n >= 2 {
                    last[n - 1] - last[n - 2]
                } else {
                    0.0
                };
                out[gi] = if trend > threshold_up {
                    -1.0
                } else if trend < threshold_down {
                    1.0
                } else {
                    -delta_close
                };
            }
        }
        "alpha047_inverse_close_volume_v1" => {
            let adv_window = spec.param_i64("adv_window", 10).max(1) as usize;
            let sum_window = spec.param_i64("sum_window", 5).max(1) as usize;
            let delay_window = spec.param_i64("delay_window", 5).max(1) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "high")?;
            let last_col = panel.col("last")?;
            let amount_col = panel.col("amount")?;
            let volume_col = panel.col("volume")?;
            let high_col = panel.col("high")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let amount = &amount_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let high = &high_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let mut inv_close = vec![f64::NAN; n];
                for i in 0..n {
                    inv_close[i] = 1.0 / (last[i] + EPS);
                }
                let rank_inv = pct_rank(&inv_close);
                let mut vol_ratio = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(adv_window);
                    let adv = mean_valid(&amount[s..=i]);
                    vol_ratio[i] = (rank_inv[i] * volume[i]) / (adv + EPS);
                }
                let mut high_diff = vec![f64::NAN; n];
                for i in 0..n {
                    high_diff[i] = high[i] - last[i];
                }
                let rank_high_diff = pct_rank(&high_diff);
                let mut high_factor = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(sum_window);
                    let avg_high = mean_valid(&high[s..=i]);
                    high_factor[i] = (high[i] * rank_high_diff[i]) / (avg_high + EPS);
                }
                let mut vwap = vec![f64::NAN; n];
                for i in 0..n {
                    vwap[i] = amount[i] / (volume[i] + EPS);
                }
                let mut delay_vwap = vec![f64::NAN; n];
                for i in 0..n {
                    delay_vwap[i] = if i >= delay_window {
                        vwap[i - delay_window]
                    } else {
                        f64::NAN
                    };
                }
                let mut vwap_diff = vec![f64::NAN; n];
                for i in 0..n {
                    vwap_diff[i] = if vwap[i].is_finite() && delay_vwap[i].is_finite() {
                        vwap[i] - delay_vwap[i]
                    } else {
                        f64::NAN
                    };
                }
                let rank_diff = pct_rank(&vwap_diff);
                let val = vol_ratio[n - 1] * high_factor[n - 1] - rank_diff[n - 1];
                out[gi] = if val.is_finite() { val } else { 0.0 };
            }
        }
        "alpha049_close_delay_threshold_v1" => {
            let delay_window_long = spec.param_i64("delay_window_long", 10).max(1) as usize;
            let delay_window_short = spec.param_i64("delay_window_short", 5).max(1) as usize;
            let trend_scale = spec.param_f64("trend_scale", 10.0);
            let threshold = spec.param_f64("threshold", -0.1);
            ensure_col(py, panel, panel_df, "last")?;
            let last_col = panel.col("last")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let d_long = delay_value_or_nan(last, delay_window_long);
                let d_short = delay_value_or_nan(last, delay_window_short);
                let diff1 = (d_long - d_short) / trend_scale;
                let diff2 = (d_short - last[n - 1]) / trend_scale;
                let trend = if diff1.is_finite() && diff2.is_finite() {
                    diff1 - diff2
                } else {
                    0.0
                };
                let delta_close = if n >= 2 {
                    last[n - 1] - last[n - 2]
                } else {
                    0.0
                };
                out[gi] = if trend < threshold { 1.0 } else { -delta_close };
            }
        }
        "alpha050_volume_vwap_corr_max_v1" => {
            let corr_window = spec.param_i64("corr_window", 5).max(2) as usize;
            let ts_max_window = spec.param_i64("ts_max_window", 5).max(1) as usize;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            let amount_col = panel.col("amount")?;
            let volume_col = panel.col("volume")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let amount = &amount_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let n = volume.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let mut vwap = vec![f64::NAN; n];
                for i in 0..n {
                    vwap[i] = amount[i] / (volume[i] + EPS);
                }
                let rank_vol = pct_rank(volume);
                let rank_vwap = pct_rank(&vwap);
                let mut corr = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(corr_window);
                    corr[i] = corr_slice(&rank_vol[s..=i], &rank_vwap[s..=i]);
                }
                let rank_corr = pct_rank(&corr);
                let ts_max = rolling_max_last(&rank_corr, ts_max_window);
                out[gi] = if ts_max.is_finite() { -ts_max } else { 0.0 };
            }
        }
        "alpha051_close_delay_threshold_v2_v1" => {
            let delay_window_long = spec.param_i64("delay_window_long", 10).max(1) as usize;
            let delay_window_short = spec.param_i64("delay_window_short", 5).max(1) as usize;
            let trend_scale = spec.param_f64("trend_scale", 10.0);
            let threshold = spec.param_f64("threshold", -0.05);
            ensure_col(py, panel, panel_df, "last")?;
            let last_col = panel.col("last")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let d_long = delay_value_or_nan(last, delay_window_long);
                let d_short = delay_value_or_nan(last, delay_window_short);
                let diff1 = (d_long - d_short) / trend_scale;
                let diff2 = (d_short - last[n - 1]) / trend_scale;
                let trend = if diff1.is_finite() && diff2.is_finite() {
                    diff1 - diff2
                } else {
                    0.0
                };
                let delta_close = if n >= 2 {
                    last[n - 1] - last[n - 2]
                } else {
                    0.0
                };
                out[gi] = if trend < threshold { 1.0 } else { -delta_close };
            }
        }
        "alpha052_low_momentum_volume_v1" => {
            let ts_min_window = spec.param_i64("ts_min_window", 5).max(1) as usize;
            let delay_window = spec.param_i64("delay_window", 5).max(1) as usize;
            let sum_window_long = spec.param_i64("sum_window_long", 60).max(1) as usize;
            let sum_window_short = spec.param_i64("sum_window_short", 20).max(1) as usize;
            let ts_rank_window = spec.param_i64("ts_rank_window", 5).max(1) as usize;
            ensure_col(py, panel, panel_df, "low")?;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "last")?;
            let has_prev = has_col(py, panel_df, "prev_bar_close")?;
            let has_pre_close = has_col(py, panel_df, "pre_close")?;
            if has_prev {
                ensure_col(py, panel, panel_df, "prev_bar_close")?;
            } else if has_pre_close {
                ensure_col(py, panel, panel_df, "pre_close")?;
            }
            let low_col = panel.col("low")?;
            let volume_col = panel.col("volume")?;
            let last_col = panel.col("last")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let low = &low_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let last = &last_col[g.start..g.end];
                let n = low.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let mut ts_min_low = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(ts_min_window);
                    ts_min_low[i] = min_valid(&low[s..=i]);
                }
                let mut delay_min = vec![f64::NAN; n];
                for i in 0..n {
                    delay_min[i] = if i >= delay_window {
                        ts_min_low[i - delay_window]
                    } else {
                        f64::NAN
                    };
                }
                let mut low_diff = vec![f64::NAN; n];
                for i in 0..n {
                    low_diff[i] = -ts_min_low[i] + delay_min[i];
                }
                let mut pre = vec![f64::NAN; n];
                for i in 0..n {
                    pre[i] = if has_prev {
                        panel.col("prev_bar_close")?[g.start + i]
                    } else if has_pre_close {
                        panel.col("pre_close")?[g.start + i]
                    } else if i > 0 {
                        last[i - 1]
                    } else {
                        f64::NAN
                    };
                }
                let mut ret = vec![f64::NAN; n];
                for i in 0..n {
                    ret[i] = if last[i].is_finite() && pre[i].is_finite() && pre[i].abs() > EPS {
                        (last[i] - pre[i]) / (pre[i] + EPS)
                    } else {
                        f64::NAN
                    };
                }
                let mut sum_long = vec![f64::NAN; n];
                let mut sum_short = vec![f64::NAN; n];
                for i in 0..n {
                    let s1 = i + 1usize - (i + 1).min(sum_window_long);
                    let s2 = i + 1usize - (i + 1).min(sum_window_short);
                    sum_long[i] = sum_valid(&ret[s1..=i]);
                    sum_short[i] = sum_valid(&ret[s2..=i]);
                }
                let denom = (sum_window_long.saturating_sub(sum_window_short)).max(1) as f64;
                let mut ret_diff = vec![f64::NAN; n];
                for i in 0..n {
                    ret_diff[i] = (sum_long[i] - sum_short[i]) / denom;
                }
                let rank_ret = pct_rank(&ret_diff);
                let ts_rank_vol = ts_rank_last(volume, ts_rank_window);
                let alpha_last = low_diff[n - 1] * rank_ret[n - 1] * ts_rank_vol;
                out[gi] = if alpha_last.is_finite() {
                    alpha_last
                } else {
                    0.0
                };
            }
        }
        "alpha053_price_position_delta_v1" => {
            let delta_window = spec.param_i64("delta_window", 9).max(1) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "high")?;
            ensure_col(py, panel, panel_df, "low")?;
            let last_col = panel.col("last")?;
            let high_col = panel.col("high")?;
            let low_col = panel.col("low")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let high = &high_col[g.start..g.end];
                let low = &low_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let mut position = vec![f64::NAN; n];
                for i in 0..n {
                    let numerator = (last[i] - low[i]) - (high[i] - last[i]);
                    let denominator = last[i] - low[i];
                    position[i] = numerator / (denominator + EPS);
                }
                let delta_val = if n > delta_window {
                    position[n - 1] - position[n - 1 - delta_window]
                } else {
                    f64::NAN
                };
                out[gi] = if delta_val.is_finite() {
                    -delta_val
                } else {
                    0.0
                };
            }
        }
        "alpha054_price_power_ratio_v1" => {
            let power = spec.param_i64("power", 5);
            ensure_col(py, panel, panel_df, "low")?;
            ensure_col(py, panel, panel_df, "high")?;
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "open")?;
            ensure_col(py, panel, panel_df, "ask_price1")?;
            ensure_col(py, panel, panel_df, "bid_price1")?;
            let low_col = panel.col("low")?;
            let high_col = panel.col("high")?;
            let last_col = panel.col("last")?;
            let open_col = panel.col("open")?;
            let ask_col = panel.col("ask_price1")?;
            let bid_col = panel.col("bid_price1")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let r = g.end - 1;
                let low = low_col[r];
                let high = high_col[r];
                let last = last_col[r];
                let mid = open_like_at(open_col[r], ask_col[r], bid_col[r]);
                let numerator = -(low - last) * mid.powf(power as f64);
                let denominator = (low - high) * last.powf(power as f64);
                let alpha = numerator / (denominator + EPS);
                out[gi] = if alpha.is_finite() { alpha } else { 0.0 };
            }
        }
        "alpha055_close_range_volume_corr_v1" => {
            let ts_window = spec.param_i64("ts_window", 12).max(1) as usize;
            let corr_window = spec.param_i64("corr_window", 6).max(2) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "low")?;
            ensure_col(py, panel, panel_df, "high")?;
            ensure_col(py, panel, panel_df, "volume")?;
            let last_col = panel.col("last")?;
            let low_col = panel.col("low")?;
            let high_col = panel.col("high")?;
            let volume_col = panel.col("volume")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let low = &low_col[g.start..g.end];
                let high = &high_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let mut range_pos = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(ts_window);
                    let ts_min_low = min_valid(&low[s..=i]);
                    let ts_max_high = max_valid(&high[s..=i]);
                    range_pos[i] = (last[i] - ts_min_low) / (ts_max_high - ts_min_low + EPS);
                }
                let rank_range = pct_rank(&range_pos);
                let rank_volume = pct_rank(volume);
                out[gi] = -corr_last_window(&rank_range, &rank_volume, corr_window);
            }
        }
        "alpha057_close_vwap_decay_v1" => {
            let ts_argmax_window = spec.param_i64("ts_argmax_window", 10).max(1) as usize;
            let decay_window = spec.param_i64("decay_window", 2).max(1) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            let last_col = panel.col("last")?;
            let amount_col = panel.col("amount")?;
            let volume_col = panel.col("volume")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let amount = &amount_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let mut vwap = vec![f64::NAN; n];
                let mut diff = vec![f64::NAN; n];
                for i in 0..n {
                    vwap[i] = amount[i] / (volume[i] + EPS);
                    diff[i] = if last[i].is_finite() && vwap[i].is_finite() {
                        last[i] - vwap[i]
                    } else {
                        f64::NAN
                    };
                }
                let ts_argmax = rolling_argmax_position_series(last, ts_argmax_window);
                let rank_max = pct_rank(&ts_argmax);
                let decay = rolling_linear_decay_series(&rank_max, decay_window);
                let alpha_last = -(diff[n - 1] / (decay[n - 1] + EPS));
                out[gi] = if alpha_last.is_finite() {
                    alpha_last
                } else {
                    0.0
                };
            }
        }
        "alpha060_price_range_volume_scale_v1" => {
            let ts_argmax_window = spec.param_i64("ts_argmax_window", 10).max(1) as usize;
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "high")?;
            ensure_col(py, panel, panel_df, "low")?;
            ensure_col(py, panel, panel_df, "volume")?;
            let last_col = panel.col("last")?;
            let high_col = panel.col("high")?;
            let low_col = panel.col("low")?;
            let volume_col = panel.col("volume")?;
            let mut pos_vol = vec![0.0; panel.groups.len()];
            let mut ts_arg = vec![0.0; panel.groups.len()];
            for (gi, g) in panel.groups.iter().enumerate() {
                let last = &last_col[g.start..g.end];
                let high = &high_col[g.start..g.end];
                let low = &low_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let n = last.len();
                if n == 0 {
                    continue;
                }
                let i = n - 1;
                let numerator = (last[i] - low[i]) - (high[i] - last[i]);
                let denominator = high[i] - low[i];
                let position = numerator / (denominator + EPS);
                pos_vol[gi] = position * volume[i];
                ts_arg[gi] = argmax_pos_last(last, ts_argmax_window);
            }
            let rank_pos = cs_rank_by_dt(&panel.groups, &pos_vol);
            let scale_pos = cs_scale_by_dt(&panel.groups, &rank_pos);
            let rank_scaled = cs_rank_by_dt(&panel.groups, &scale_pos);
            let rank_arg = cs_rank_by_dt(&panel.groups, &ts_arg);
            let scale_max = cs_scale_by_dt(&panel.groups, &rank_arg);
            for i in 0..out.len() {
                out[i] = -((2.0 * rank_scaled[i]) - scale_max[i]);
            }
        }
        "alpha062_vwap_open_rank_compare_v1" => {
            let adv_window = spec.param_i64("adv_window", 20).max(1) as usize;
            let sum_window = spec.param_i64("sum_window", 22).max(1) as usize;
            let corr_window = spec.param_i64("corr_window", 10).max(2) as usize;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "high")?;
            ensure_col(py, panel, panel_df, "low")?;
            ensure_col(py, panel, panel_df, "ask_price1")?;
            ensure_col(py, panel, panel_df, "bid_price1")?;
            let amount_col = panel.col("amount")?;
            let volume_col = panel.col("volume")?;
            let high_col = panel.col("high")?;
            let low_col = panel.col("low")?;
            let ask_col = panel.col("ask_price1")?;
            let bid_col = panel.col("bid_price1")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let amount = &amount_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let high = &high_col[g.start..g.end];
                let low = &low_col[g.start..g.end];
                let ask = &ask_col[g.start..g.end];
                let bid = &bid_col[g.start..g.end];
                let n = amount.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let mut vwap = vec![f64::NAN; n];
                let mut mid1 = vec![f64::NAN; n];
                let mut adv = vec![f64::NAN; n];
                let mut sum_adv = vec![f64::NAN; n];
                for i in 0..n {
                    vwap[i] = amount[i] / (volume[i] + EPS);
                    mid1[i] = if ask[i].is_finite() && bid[i].is_finite() {
                        (ask[i] + bid[i]) / 2.0
                    } else {
                        f64::NAN
                    };
                    let s = i + 1usize - (i + 1).min(adv_window);
                    adv[i] = mean_valid(&amount[s..=i]);
                    let s2 = i + 1usize - (i + 1).min(sum_window);
                    sum_adv[i] = sum_valid(&adv[s2..=i]);
                }
                let mut corr = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(corr_window);
                    corr[i] = corr_slice(&vwap[s..=i], &sum_adv[s..=i]);
                }
                let rank_corr = pct_rank(&corr);
                let rc = rank_corr[n - 1];
                let rank_open = pct_rank(&mid1);
                let mut mid_price = vec![f64::NAN; n];
                for i in 0..n {
                    mid_price[i] = (high[i] + low[i]) / 2.0;
                }
                let rank_mid = pct_rank(&mid_price);
                let rank_high = pct_rank(high);
                let mut compare = vec![0.0; n];
                for i in 0..n {
                    let left = rank_open[i] * 2.0;
                    let right = rank_mid[i] + rank_high[i];
                    compare[i] = if left.is_finite() && right.is_finite() && left < right {
                        1.0
                    } else {
                        0.0
                    };
                }
                let rank_compare = pct_rank(&compare);
                let rp = rank_compare[n - 1];
                out[gi] = if rc.is_finite() && rp.is_finite() {
                    if rc < rp {
                        -1.0
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
            }
        }
        "alpha065_open_vwap_min_signal_v1" => {
            let weight = spec.param_f64("weight", 0.008);
            let adv_window = spec.param_i64("adv_window", 30).max(1) as usize;
            let sum_window = spec.param_i64("sum_window", 9).max(1) as usize;
            let corr_window = spec.param_i64("corr_window", 6).max(2) as usize;
            let ts_min_window = spec.param_i64("ts_min_window", 14).max(1) as usize;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "ask_price1")?;
            ensure_col(py, panel, panel_df, "bid_price1")?;
            let amount_col = panel.col("amount")?;
            let volume_col = panel.col("volume")?;
            let ask_col = panel.col("ask_price1")?;
            let bid_col = panel.col("bid_price1")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let amount = &amount_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let ask = &ask_col[g.start..g.end];
                let bid = &bid_col[g.start..g.end];
                let n = amount.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let mut mid1 = vec![f64::NAN; n];
                let mut vwap = vec![f64::NAN; n];
                let mut weighted = vec![f64::NAN; n];
                let mut adv = vec![f64::NAN; n];
                let mut sum_adv = vec![f64::NAN; n];
                for i in 0..n {
                    mid1[i] = if ask[i].is_finite() && bid[i].is_finite() {
                        (ask[i] + bid[i]) / 2.0
                    } else {
                        f64::NAN
                    };
                    vwap[i] = amount[i] / (volume[i] + EPS);
                    weighted[i] = mid1[i] * weight + vwap[i] * (1.0 - weight);
                    let s = i + 1usize - (i + 1).min(adv_window);
                    adv[i] = mean_valid(&amount[s..=i]);
                    let s2 = i + 1usize - (i + 1).min(sum_window);
                    sum_adv[i] = sum_valid(&adv[s2..=i]);
                }
                let mut corr = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(corr_window);
                    corr[i] = corr_slice(&weighted[s..=i], &sum_adv[s..=i]);
                }
                let rank_corr = pct_rank(&corr);
                let rc = rank_corr[n - 1];
                let mut ts_min_open = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(ts_min_window);
                    ts_min_open[i] = min_valid(&mid1[s..=i]);
                }
                let mut diff = vec![f64::NAN; n];
                for i in 0..n {
                    diff[i] = mid1[i] - ts_min_open[i];
                }
                let rank_diff = pct_rank(&diff);
                let rd = rank_diff[n - 1];
                out[gi] = if rc.is_finite() && rd.is_finite() {
                    if rc < rd {
                        -1.0
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
            }
        }
        "alpha066_vwap_low_decay_v1" => {
            let delta_window = spec.param_i64("delta_window", 4).max(1) as usize;
            let decay_window_1 = spec.param_i64("decay_window_1", 7).max(1) as usize;
            let weight = spec.param_f64("weight", 0.966);
            let decay_window_2 = spec.param_i64("decay_window_2", 11).max(1) as usize;
            let ts_rank_window = spec.param_i64("ts_rank_window", 7).max(1) as usize;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "low")?;
            ensure_col(py, panel, panel_df, "high")?;
            ensure_col(py, panel, panel_df, "ask_price1")?;
            ensure_col(py, panel, panel_df, "bid_price1")?;
            let amount_col = panel.col("amount")?;
            let volume_col = panel.col("volume")?;
            let low_col = panel.col("low")?;
            let high_col = panel.col("high")?;
            let ask_col = panel.col("ask_price1")?;
            let bid_col = panel.col("bid_price1")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let amount = &amount_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let low = &low_col[g.start..g.end];
                let high = &high_col[g.start..g.end];
                let ask = &ask_col[g.start..g.end];
                let bid = &bid_col[g.start..g.end];
                let n = amount.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let mut vwap = vec![f64::NAN; n];
                let mut mid1 = vec![f64::NAN; n];
                for i in 0..n {
                    vwap[i] = amount[i] / (volume[i] + EPS);
                    mid1[i] = if ask[i].is_finite() && bid[i].is_finite() {
                        (ask[i] + bid[i]) / 2.0
                    } else {
                        f64::NAN
                    };
                }
                let delta_vwap = diff(&vwap, delta_window);
                let decay1 = rolling_linear_decay_series(&delta_vwap, decay_window_1);
                let rank_decay1 = pct_rank(&decay1);
                let mut weighted_low = vec![f64::NAN; n];
                let mut mid_price = vec![f64::NAN; n];
                let mut ratio = vec![f64::NAN; n];
                for i in 0..n {
                    weighted_low[i] = low[i] * weight + low[i] * (1.0 - weight);
                    mid_price[i] = (high[i] + low[i]) / 2.0;
                    ratio[i] = (weighted_low[i] - vwap[i]) / (mid1[i] - mid_price[i] + EPS);
                }
                let decay2 = rolling_linear_decay_series(&ratio, decay_window_2);
                let ts_rank = ts_rank_last(&decay2, ts_rank_window);
                let mut r1 = rank_decay1[n - 1];
                if !r1.is_finite() {
                    r1 = 0.0;
                }
                let r2 = if ts_rank.is_finite() { ts_rank } else { 0.0 };
                out[gi] = -(r1 + r2);
            }
        }
        "alpha068_high_adv_rank_signal_v1" => {
            let adv_window = spec.param_i64("adv_window", 15).max(1) as usize;
            let corr_window = spec.param_i64("corr_window", 9).max(2) as usize;
            let ts_rank_window = spec.param_i64("ts_rank_window", 14).max(1) as usize;
            let weight = spec.param_f64("weight", 0.518);
            let delta_window = spec.param_i64("delta_window", 1).max(1) as usize;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "high")?;
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "low")?;
            let amount_col = panel.col("amount")?;
            let high_col = panel.col("high")?;
            let last_col = panel.col("last")?;
            let low_col = panel.col("low")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let amount = &amount_col[g.start..g.end];
                let high = &high_col[g.start..g.end];
                let last = &last_col[g.start..g.end];
                let low = &low_col[g.start..g.end];
                let n = amount.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let mut adv = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(adv_window);
                    adv[i] = mean_valid(&amount[s..=i]);
                }
                let rank_high = pct_rank(high);
                let rank_adv = pct_rank(&adv);
                let mut corr = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(corr_window);
                    corr[i] = corr_slice(&rank_high[s..=i], &rank_adv[s..=i]);
                }
                let ts_rank_corr = rolling_last_rank_pct_series(&corr, ts_rank_window);
                let rank_ts = pct_rank(&ts_rank_corr);
                let mut weighted = vec![f64::NAN; n];
                for i in 0..n {
                    weighted[i] = last[i] * weight + low[i] * (1.0 - weight);
                }
                let delta_weighted = diff(&weighted, delta_window);
                let rank_delta = pct_rank(&delta_weighted);
                let rts = rank_ts[n - 1];
                let rd = rank_delta[n - 1];
                out[gi] = if rts.is_finite() && rd.is_finite() {
                    if rts < rd {
                        -1.0
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
            }
        }
        "alpha072_vwap_volume_decay_ratio_v1" => {
            let adv_window = spec.param_i64("adv_window", 20).max(1) as usize;
            let corr_window_1 = spec.param_i64("corr_window_1", 9).max(2) as usize;
            let decay_window_1 = spec.param_i64("decay_window_1", 10).max(1) as usize;
            let ts_rank_window_1 = spec.param_i64("ts_rank_window_1", 4).max(1) as usize;
            let ts_rank_window_2 = spec.param_i64("ts_rank_window_2", 19).max(1) as usize;
            let corr_window_2 = spec.param_i64("corr_window_2", 7).max(2) as usize;
            let decay_window_2 = spec.param_i64("decay_window_2", 3).max(1) as usize;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "high")?;
            ensure_col(py, panel, panel_df, "low")?;
            let amount_col = panel.col("amount")?;
            let volume_col = panel.col("volume")?;
            let high_col = panel.col("high")?;
            let low_col = panel.col("low")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let amount = &amount_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let high = &high_col[g.start..g.end];
                let low = &low_col[g.start..g.end];
                let n = amount.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let mut vwap = vec![f64::NAN; n];
                let mut adv = vec![f64::NAN; n];
                let mut mid = vec![f64::NAN; n];
                for i in 0..n {
                    vwap[i] = amount[i] / (volume[i] + EPS);
                    let s = i + 1usize - (i + 1).min(adv_window);
                    adv[i] = mean_valid(&amount[s..=i]);
                    mid[i] = (high[i] + low[i]) / 2.0;
                }
                let mut corr1 = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(corr_window_1);
                    corr1[i] = corr_slice(&mid[s..=i], &adv[s..=i]);
                }
                let decay1 = rolling_linear_decay_series(&corr1, decay_window_1);
                let rank_decay1 = pct_rank(&decay1);
                let ts_rank_vwap = rolling_last_rank_pct_series(&vwap, ts_rank_window_1);
                let ts_rank_vol = rolling_last_rank_pct_series(volume, ts_rank_window_2);
                let mut corr2 = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(corr_window_2);
                    corr2[i] = corr_slice(&ts_rank_vwap[s..=i], &ts_rank_vol[s..=i]);
                }
                let decay2 = rolling_linear_decay_series(&corr2, decay_window_2);
                let rank_decay2 = pct_rank(&decay2);
                let mut r1 = rank_decay1[n - 1];
                let mut r2 = rank_decay2[n - 1];
                if !r1.is_finite() {
                    r1 = 0.0;
                }
                if !r2.is_finite() {
                    r2 = 0.0;
                }
                out[gi] = r1 / (r2 + EPS);
            }
        }
        "alpha073_vwap_open_decay_max_v1" => {
            let delta_window_1 = spec.param_i64("delta_window_1", 5).max(1) as usize;
            let decay_window_1 = spec.param_i64("decay_window_1", 3).max(1) as usize;
            let weight = spec.param_f64("weight", 0.147);
            let delta_window_2 = spec.param_i64("delta_window_2", 2).max(1) as usize;
            let decay_window_2 = spec.param_i64("decay_window_2", 3).max(1) as usize;
            let ts_rank_window = spec.param_i64("ts_rank_window", 17).max(1) as usize;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "low")?;
            ensure_col(py, panel, panel_df, "ask_price1")?;
            ensure_col(py, panel, panel_df, "bid_price1")?;
            let amount_col = panel.col("amount")?;
            let volume_col = panel.col("volume")?;
            let low_col = panel.col("low")?;
            let ask_col = panel.col("ask_price1")?;
            let bid_col = panel.col("bid_price1")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let amount = &amount_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let low = &low_col[g.start..g.end];
                let ask = &ask_col[g.start..g.end];
                let bid = &bid_col[g.start..g.end];
                let n = amount.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let mut vwap = vec![f64::NAN; n];
                let mut mid1 = vec![f64::NAN; n];
                for i in 0..n {
                    vwap[i] = amount[i] / (volume[i] + EPS);
                    mid1[i] = if ask[i].is_finite() && bid[i].is_finite() {
                        (ask[i] + bid[i]) / 2.0
                    } else {
                        f64::NAN
                    };
                }
                let delta_vwap = diff(&vwap, delta_window_1);
                let decay1 = rolling_linear_decay_series(&delta_vwap, decay_window_1);
                let rank_decay1 = pct_rank(&decay1);
                let mut weighted = vec![f64::NAN; n];
                for i in 0..n {
                    weighted[i] = mid1[i] * weight + low[i] * (1.0 - weight);
                }
                let delta_weighted = diff(&weighted, delta_window_2);
                let mut ratio = vec![f64::NAN; n];
                for i in 0..n {
                    ratio[i] = delta_weighted[i] / (weighted[i] + EPS);
                }
                let neg_ratio: Vec<f64> = ratio
                    .iter()
                    .map(|v| if v.is_finite() { -*v } else { f64::NAN })
                    .collect();
                let decay2 = rolling_linear_decay_series(&neg_ratio, decay_window_2);
                let ts_rank = ts_rank_last(&decay2, ts_rank_window);
                let mut r1 = rank_decay1[n - 1];
                if !r1.is_finite() {
                    r1 = 0.0;
                }
                let r2 = if ts_rank.is_finite() { ts_rank } else { 0.0 };
                out[gi] = -r1.max(r2);
            }
        }
        "alpha074_close_adv_rank_corr_v1" => {
            let adv_window = spec.param_i64("adv_window", 20).max(1) as usize;
            let sum_window = spec.param_i64("sum_window", 37).max(1) as usize;
            let corr_window_1 = spec.param_i64("corr_window_1", 15).max(2) as usize;
            let weight = spec.param_f64("weight", 0.026);
            let corr_window_2 = spec.param_i64("corr_window_2", 11).max(2) as usize;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "last")?;
            ensure_col(py, panel, panel_df, "high")?;
            let amount_col = panel.col("amount")?;
            let volume_col = panel.col("volume")?;
            let last_col = panel.col("last")?;
            let high_col = panel.col("high")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let amount = &amount_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let last = &last_col[g.start..g.end];
                let high = &high_col[g.start..g.end];
                let n = amount.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let mut adv = vec![f64::NAN; n];
                let mut sum_adv = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(adv_window);
                    adv[i] = mean_valid(&amount[s..=i]);
                    let s2 = i + 1usize - (i + 1).min(sum_window);
                    sum_adv[i] = sum_valid(&adv[s2..=i]);
                }
                let mut corr1 = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(corr_window_1);
                    corr1[i] = corr_slice(&last[s..=i], &sum_adv[s..=i]);
                }
                let rank_corr1 = pct_rank(&corr1);
                let mut vwap = vec![f64::NAN; n];
                let mut weighted = vec![f64::NAN; n];
                for i in 0..n {
                    vwap[i] = amount[i] / (volume[i] + EPS);
                    weighted[i] = high[i] * weight + vwap[i] * (1.0 - weight);
                }
                let rank_weighted = pct_rank(&weighted);
                let rank_vol = pct_rank(volume);
                let mut corr2 = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(corr_window_2);
                    corr2[i] = corr_slice(&rank_weighted[s..=i], &rank_vol[s..=i]);
                }
                let rank_corr2 = pct_rank(&corr2);
                let r1 = rank_corr1[n - 1];
                let r2 = rank_corr2[n - 1];
                out[gi] = if r1.is_finite() && r2.is_finite() {
                    if r1 < r2 {
                        -1.0
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
            }
        }
        "alpha075_vwap_volume_low_adv_corr_v1" => {
            let corr_window_1 = spec.param_i64("corr_window_1", 4).max(2) as usize;
            let adv_window = spec.param_i64("adv_window", 30).max(1) as usize;
            let corr_window_2 = spec.param_i64("corr_window_2", 12).max(2) as usize;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "low")?;
            let amount_col = panel.col("amount")?;
            let volume_col = panel.col("volume")?;
            let low_col = panel.col("low")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let amount = &amount_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let low = &low_col[g.start..g.end];
                let n = amount.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let mut vwap = vec![f64::NAN; n];
                for i in 0..n {
                    vwap[i] = amount[i] / (volume[i] + EPS);
                }
                let mut corr1 = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(corr_window_1);
                    corr1[i] = corr_slice(&vwap[s..=i], &volume[s..=i]);
                }
                let rank_corr1 = pct_rank(&corr1);
                let mut adv = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(adv_window);
                    adv[i] = mean_valid(&amount[s..=i]);
                }
                let rank_low = pct_rank(low);
                let rank_adv = pct_rank(&adv);
                let mut corr2 = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(corr_window_2);
                    corr2[i] = corr_slice(&rank_low[s..=i], &rank_adv[s..=i]);
                }
                let rank_corr2 = pct_rank(&corr2);
                let r1 = rank_corr1[n - 1];
                let r2 = rank_corr2[n - 1];
                out[gi] = if r1.is_finite() && r2.is_finite() {
                    if r1 < r2 {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
            }
        }
        "alpha077_mid_price_adv_decay_min_v1" => {
            let decay_window_1 = spec.param_i64("decay_window_1", 10).max(1) as usize;
            let adv_window = spec.param_i64("adv_window", 20).max(1) as usize;
            let corr_window = spec.param_i64("corr_window", 3).max(2) as usize;
            let decay_window_2 = spec.param_i64("decay_window_2", 6).max(1) as usize;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "high")?;
            ensure_col(py, panel, panel_df, "low")?;
            let amount_col = panel.col("amount")?;
            let volume_col = panel.col("volume")?;
            let high_col = panel.col("high")?;
            let low_col = panel.col("low")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let amount = &amount_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let high = &high_col[g.start..g.end];
                let low = &low_col[g.start..g.end];
                let n = amount.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let mut vwap = vec![f64::NAN; n];
                let mut mid = vec![f64::NAN; n];
                for i in 0..n {
                    vwap[i] = amount[i] / (volume[i] + EPS);
                    mid[i] = (high[i] + low[i]) / 2.0;
                }
                let mut diff1 = vec![f64::NAN; n];
                for i in 0..n {
                    diff1[i] = mid[i] - vwap[i];
                }
                let decay1 = rolling_linear_decay_series(&diff1, decay_window_1);
                let rank_decay1 = pct_rank(&decay1);
                let mut adv = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(adv_window);
                    adv[i] = mean_valid(&amount[s..=i]);
                }
                let mut corr = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(corr_window);
                    corr[i] = corr_slice(&mid[s..=i], &adv[s..=i]);
                }
                let decay2 = rolling_linear_decay_series(&corr, decay_window_2);
                let rank_decay2 = pct_rank(&decay2);
                let mut r1 = rank_decay1[n - 1];
                let mut r2 = rank_decay2[n - 1];
                if !r1.is_finite() {
                    r1 = 0.0;
                }
                if !r2.is_finite() {
                    r2 = 0.0;
                }
                out[gi] = r1.min(r2);
            }
        }
        "alpha078_low_vwap_adv_corr_v1" => {
            let weight = spec.param_f64("weight", 0.352);
            let sum_window_1 = spec.param_i64("sum_window_1", 20).max(1) as usize;
            let adv_window = spec.param_i64("adv_window", 20).max(1) as usize;
            let sum_window_2 = spec.param_i64("sum_window_2", 20).max(1) as usize;
            let corr_window_1 = spec.param_i64("corr_window_1", 7).max(2) as usize;
            let corr_window_2 = spec.param_i64("corr_window_2", 6).max(2) as usize;
            ensure_col(py, panel, panel_df, "amount")?;
            ensure_col(py, panel, panel_df, "volume")?;
            ensure_col(py, panel, panel_df, "low")?;
            let amount_col = panel.col("amount")?;
            let volume_col = panel.col("volume")?;
            let low_col = panel.col("low")?;
            for (gi, g) in panel.groups.iter().enumerate() {
                let amount = &amount_col[g.start..g.end];
                let volume = &volume_col[g.start..g.end];
                let low = &low_col[g.start..g.end];
                let n = amount.len();
                if n == 0 {
                    out[gi] = 0.0;
                    continue;
                }
                let mut vwap = vec![f64::NAN; n];
                let mut weighted = vec![f64::NAN; n];
                for i in 0..n {
                    vwap[i] = amount[i] / (volume[i] + EPS);
                    weighted[i] = low[i] * weight + vwap[i] * (1.0 - weight);
                }
                let mut sum_weighted = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(sum_window_1);
                    sum_weighted[i] = sum_valid(&weighted[s..=i]);
                }
                let mut adv = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(adv_window);
                    adv[i] = mean_valid(&amount[s..=i]);
                }
                let mut sum_adv = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(sum_window_2);
                    sum_adv[i] = sum_valid(&adv[s..=i]);
                }
                let mut corr1 = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(corr_window_1);
                    corr1[i] = corr_slice(&sum_weighted[s..=i], &sum_adv[s..=i]);
                }
                let rank_corr1 = pct_rank(&corr1);
                let rank_vwap = pct_rank(&vwap);
                let rank_vol = pct_rank(volume);
                let mut corr2 = vec![f64::NAN; n];
                for i in 0..n {
                    let s = i + 1usize - (i + 1).min(corr_window_2);
                    corr2[i] = corr_slice(&rank_vwap[s..=i], &rank_vol[s..=i]);
                }
                let rank_corr2 = pct_rank(&corr2);
                let mut r1 = rank_corr1[n - 1];
                let mut r2 = rank_corr2[n - 1];
                if !r1.is_finite() {
                    r1 = 0.0;
                }
                if !r2.is_finite() {
                    r2 = 0.0;
                }
                out[gi] = r1.powf(r2);
            }
        }
        "daily_sharpe_mean_v1" => {
            let source = spec.param_str("source", "market_cbond.daily_twap");
            let price_col = spec.param_str("price_col", "twap_1442_1457");
            let lookback_days = spec.param_i64("lookback_days", 20).max(2) as usize;
            let smooth_days = spec.param_i64("smooth_days", 5).max(1) as usize;
            let min_periods = spec.param_i64("min_periods", lookback_days as i64).max(2) as usize;
            let annualize = spec.param_bool("annualize", false);
            for (gi, g) in panel.groups.iter().enumerate() {
                let code = norm_code(&g.code);
                let dt = norm_dt(&g.dt);
                if code.is_empty() || dt.is_empty() {
                    out[gi] = 0.0;
                    continue;
                }
                let need_days = lookback_days + smooth_days + 2usize;
                let price_series =
                    daily_values_asof(aux, &source, &code, &dt, &price_col, need_days);
                out[gi] = rolling_sharpe_last_mean(
                    &price_series,
                    lookback_days,
                    smooth_days,
                    min_periods,
                    annualize,
                );
            }
        }
        _ => {
            return Err(PyErr::new::<PyRuntimeError, _>(format!(
                "rust factor kernel not implemented: {}",
                spec.factor
            )));
        }
    }
    Ok(out)
}
