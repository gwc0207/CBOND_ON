use std::collections::{BTreeSet, HashMap};
use std::time::Instant;
use chrono::NaiveDate;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

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
struct AuxData {
    stock_latest: HashMap<(String, String), LatestQuote>,
    bond_stock_map: HashMap<String, String>,
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
fn compute_factor_frame(
    py: Python<'_>,
    panel_df: &Bound<'_, PyAny>,
    specs_payload: &Bound<'_, PyAny>,
    stock_df: Option<&Bound<'_, PyAny>>,
    map_df: Option<&Bound<'_, PyAny>>,
    _compute_params: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let mut panel = parse_panel(py, panel_df)?;
    let specs = parse_specs(specs_payload)?;
    let plan_limits = parse_plan_limits(_compute_params)?;
    let plan = extract_factor_plan(&specs, &plan_limits)?;
    let window_cache = build_window_start_cache(&panel, &plan.windows);
    let aux = parse_aux_data(py, stock_df, map_df)?;
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
        let values = compute_factor_values(
            py,
            &mut panel,
            panel_df,
            &aux,
            &window_cache,
            spec,
        )?;
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

fn ensure_col(py: Python<'_>, panel: &mut PanelData, panel_df: &Bound<'_, PyAny>, col: &str) -> PyResult<()> {
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
    let lookback = window_minutes.saturating_mul(60).saturating_mul(1_000_000_000i64);
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
    idx.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal));
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
    if r.is_finite() { r } else { 0.0 }
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
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    for v in &vals {
        let d = *v - mean;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= n as f64;
    m3 /= n as f64;
    if m2 <= EPS {
        return 0.0;
    }
    let g1 = m3 / m2.powf(1.5);
    if n <= 2 {
        return 0.0;
    }
    let adj = ((n * (n - 1)) as f64).sqrt() / ((n - 2) as f64);
    let out = adj * g1;
    if out.is_finite() { out } else { 0.0 }
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

fn diff(values: &[f64], lag: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; values.len()];
    if lag == 0 {
        return out;
    }
    for i in lag..values.len() {
        let a = values[i];
        let b = values[i - lag];
        out[i] = if a.is_finite() && b.is_finite() { a - b } else { f64::NAN };
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
                let start_ns = base_ns
                    + (sh as i64 * 3600 + sm as i64 * 60) * 1_000_000_000i64;
                let end_ns = base_ns
                    + (eh as i64 * 3600 + em as i64 * 60) * 1_000_000_000i64;
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
                out[gi] = if lo.is_finite() && hi.is_finite() && last.is_finite() && last.abs() > EPS {
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
                out[gi] = if denom > EPS { (bid - ask) / denom } else { 0.0 };
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
                out[gi] = if denom > EPS { (bid - ask) / denom } else { 0.0 };
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
                    if wsum > EPS { s / wsum } else { 0.0 }
                };
                let bid_slope = calc_side(&bp, &bv);
                let ask_slope = calc_side(&ap, &av);
                let mid = (panel.col("ask_price1")?[r] + panel.col("bid_price1")?[r]) * 0.5;
                out[gi] = if mid > EPS { (ask_slope - bid_slope) / mid } else { 0.0 };
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
                out[gi] = if bid.is_finite() && ask.is_finite() { (bid - ask) / denom } else { 0.0 };
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
                out[gi] = if bid.is_finite() && ask.is_finite() { (bid - ask) / denom } else { 0.0 };
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
                out[gi] = if spread > EPS { (last - low) / (spread + EPS) } else { 0.0 };
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
                    ret[i] = if op.abs() > EPS && last[i].is_finite() { (last[i] - op) / (op + EPS) } else { f64::NAN };
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
                    vwap[i] = if volume[i].abs() > EPS { amount[i] / (volume[i] + EPS) } else { f64::NAN };
                }
                let mut open_like = vec![f64::NAN; open.len()];
                for i in 0..open.len() {
                    open_like[i] = open_like_at(open[i], ask[i], bid[i]);
                }
                let avg_vwap_last = rolling_mean_last(&vwap, vwap_window);
                let open_last = last_valid(&open_like);
                let vwap_last = last_valid(&vwap);
                let last_px = last_valid(last);
                gap1[gi] = if open_last.is_finite() && avg_vwap_last.is_finite() { open_last - avg_vwap_last } else { 0.0 };
                gap2[gi] = if last_px.is_finite() && vwap_last.is_finite() { last_px - vwap_last } else { 0.0 };
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
                    let sign = if d_last.is_finite() { d_last.signum() } else { 0.0 };
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
                    ret[i] = if o[i].abs() > EPS { (last[i] - o[i]) / (o[i] + EPS) } else { f64::NAN };
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
                let delayed = if n > delay_window { prod[delay_idx] } else { f64::NAN };
                let delay_val = if delayed.is_finite() { delayed } else { prod[0] };
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
                    if last_d.is_finite() { last_d } else { 0.0 }
                } else if ts_max < 0.0 {
                    if last_d.is_finite() { last_d } else { 0.0 }
                } else {
                    if last_d.is_finite() { -last_d } else { 0.0 }
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
                    if last_d.is_finite() { last_d } else { 0.0 }
                } else if ts_max < 0.0 {
                    if last_d.is_finite() { last_d } else { 0.0 }
                } else {
                    if last_d.is_finite() { -last_d } else { 0.0 }
                };
            }
            out = cs_rank_by_dt(&panel.groups, &raw);
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

