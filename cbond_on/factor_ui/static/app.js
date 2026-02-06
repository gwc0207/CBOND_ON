async function fetchJson(url, options = {}) {
  const resp = await fetch(url, options);
  const data = await resp.json();
  if (!resp.ok) {
    const msg = data && data.error ? data.error : "请求失败";
    throw new Error(msg);
  }
  return data;
}

function setMetric(id, value) {
  const el = document.getElementById(id);
  if (el) {
    el.textContent = Number.isFinite(value) ? value.toFixed(4) : "-";
  }
}

function getDateInputValue(id) {
  const el = document.getElementById(id);
  return el && el.value ? el.value : null;
}

function fillDatesFromIndex(index) {
  if (!index || !index.start || !index.end) {
    return;
  }
  const startInput = document.getElementById("global-start");
  const endInput = document.getElementById("global-end");
  if (startInput && !startInput.value) {
    startInput.value = index.start;
  }
  if (endInput && !endInput.value) {
    endInput.value = index.end;
  }
}

function renderRankTable(items, metric) {
  const body = document.getElementById("rank-body");
  body.innerHTML = "";
  const sorted = [...items].sort((a, b) => (b[metric] ?? 0) - (a[metric] ?? 0));
  sorted.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.factor_name}</td>
      <td>${(row[metric] ?? 0).toFixed(4)}</td>
      <td>${(row.rank_ic_mean ?? 0).toFixed(4)}</td>
      <td>${(row.sharpe ?? 0).toFixed(4)}</td>
    `;
    tr.addEventListener("click", () => {
      const select = document.getElementById("factor-select");
      select.value = row.factor_name;
      refreshSingle();
    });
    body.appendChild(tr);
  });
}

function plotMultiNav(items) {
  const traces = items.slice(0, 20).map((item) => ({
    x: item.nav_series?.map((p) => p.trade_time) || [],
    y: item.nav_series?.map((p) => p.nav) || [],
    type: "scatter",
    mode: "lines",
    name: item.factor_name,
  }));
  Plotly.newPlot("multi-nav", traces, {
    margin: { t: 20, r: 10, l: 40, b: 40 },
    title: "多因子 NAV",
  });
}

async function refreshMulti() {
  const start = getDateInputValue("global-start");
  const end = getDateInputValue("global-end");
  const metric = document.getElementById("rank-metric").value;
  const status = document.getElementById("multi-status");
  status.textContent = "多因子计算中...";
  const payload = { start, end, include_nav: true };
  try {
    const data = await fetchJson("/api/factors/summary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const items = data.items || [];
    renderRankTable(items, metric);
    const traces = items.slice(0, 20).map((item) => ({
      x: item.nav_series?.map((p) => p.trade_time) || [],
      y: item.nav_series?.map((p) => p.nav) || [],
      type: "scatter",
      mode: "lines",
      name: item.factor_name,
    }));
    Plotly.newPlot("multi-nav", traces, {
      margin: { t: 20, r: 10, l: 40, b: 40 },
      title: "多因子 NAV",
    });
    status.textContent = items.length ? "" : "多因子结果为空（请确认日期范围与数据）。";
  } catch (err) {
    status.textContent = `多因子计算失败：${err.message}`;
  }
}

function buildBinSelector(binColumns) {
    const container = document.getElementById("bin-selector");
    container.innerHTML = "";
    binColumns.forEach((col) => {
        const label = document.createElement("label");
        label.innerHTML = `<input type="checkbox" data-bin="${col}" checked />bin ${col}`;
        container.appendChild(label);
    });
}

function selectedBins() {
  const boxes = document.querySelectorAll("#bin-selector input[type=checkbox]");
  const picked = [];
  boxes.forEach((box) => {
    if (box.checked) {
      picked.push(box.getAttribute("data-bin"));
    }
  });
  return picked;
}

function plotBinCharts(binReturns, binColumns) {
  window._lastBinReturns = binReturns || {};
  if (!binColumns || binColumns.length === 0) {
    Plotly.newPlot("bin-nav", [], { title: "分箱 NAV" });
    Plotly.newPlot("bin-bar", [], { title: "分箱收益" });
    return;
  }
  const picked = selectedBins();
  const dates = Object.keys(binReturns);
  const binNavTraces = binColumns.map((bin) => {
    const series = dates.map((dt) => binReturns[dt]?.[bin] ?? 0);
    let nav = 1;
    const navSeries = series.map((v) => {
      nav *= 1 + (v || 0);
      return nav;
    });
    const isPicked = picked.includes(String(bin));
    return {
      x: dates,
      y: navSeries,
      type: "scatter",
      mode: "lines",
      name: `bin ${bin}`,
      line: { color: isPicked ? undefined : "rgba(150,150,150,0.3)" },
    };
  });
  Plotly.newPlot("bin-nav", binNavTraces, {
    margin: { t: 20, r: 10, l: 40, b: 40 },
    title: "分箱 NAV",
  });

  const totals = binColumns.map((bin) => {
    let nav = 1;
    dates.forEach((dt) => {
      nav *= 1 + (binReturns[dt]?.[bin] ?? 0);
    });
    return nav - 1;
  });
  Plotly.newPlot(
    "bin-bar",
    [
      {
        x: binColumns.map((b) => `bin ${b}`),
        y: totals,
        type: "bar",
        text: totals.map((v) => v.toFixed(4)),
        textposition: "outside",
      },
    ],
    { margin: { t: 20, r: 10, l: 40, b: 40 }, title: "分箱累计收益" }
  );
}

async function refreshSingle() {
  const factorName = document.getElementById("factor-select").value;
  if (!factorName) {
    return;
  }
  const status = document.getElementById("single-status");
  status.textContent = "单因子计算中...";
  const start = getDateInputValue("single-start") || getDateInputValue("global-start");
  const end = getDateInputValue("single-end") || getDateInputValue("global-end");
  const payload = { factor_name: factorName, start, end };
  try {
    const data = await fetchJson("/api/factor/analysis", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const summary = data.summary || {};
    setMetric("metric-sharpe", summary.sharpe);
    setMetric("metric-maxdd", summary.maxdd);
    setMetric("metric-win", summary.win_rate);

    const navTrace = {
      x: data.nav?.map((p) => p.trade_time) || [],
      y: data.nav?.map((p) => p.nav) || [],
      type: "scatter",
      mode: "lines",
      name: "Factor NAV",
    };
    Plotly.newPlot("single-nav", [navTrace], {
      margin: { t: 20, r: 10, l: 40, b: 40 },
      title: "单因子 NAV",
    });

    const icTrace = {
      x: data.ic_series?.map((p) => p.trade_time) || [],
      y: data.ic_series?.map((p) => p.rank_ic) || [],
      type: "scatter",
      mode: "lines",
      name: "Rank IC",
    };
    Plotly.newPlot("ic-chart", [icTrace], {
      margin: { t: 20, r: 10, l: 40, b: 40 },
      title: "Rank IC",
    });

    buildBinSelector(data.bin_columns || []);
    plotBinCharts(data.bin_returns || {}, data.bin_columns || []);
    status.textContent = data.ic_series?.length ? "" : "单因子结果为空（请确认日期范围与数据）。";
  } catch (err) {
    status.textContent = `单因子计算失败：${err.message}`;
  }
}

async function init() {
  const index = await fetchJson("/api/index");
  fillDatesFromIndex(index);
  const select = document.getElementById("factor-select");
  if (index && index.factors) {
    index.factors.forEach((f) => {
      const option = document.createElement("option");
      option.value = f.name;
      option.textContent = f.name;
      select.appendChild(option);
    });
  }
  document.getElementById("apply-global").addEventListener("click", () => {
    refreshMulti();
    refreshSingle();
  });
  document.getElementById("refresh-multi").addEventListener("click", refreshMulti);
  document.getElementById("refresh-single").addEventListener("click", refreshSingle);
  document.getElementById("rank-metric").addEventListener("change", refreshMulti);
  document.getElementById("bin-selector").addEventListener("change", () => {
    const binColumns = Array.from(document.querySelectorAll("#bin-selector input")).map((b) =>
      b.getAttribute("data-bin")
    );
    plotBinCharts(window._lastBinReturns || {}, binColumns);
  });

  await refreshMulti();
  await refreshSingle();
}

window.addEventListener("load", init);
