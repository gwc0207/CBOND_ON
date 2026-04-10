const logBox = document.getElementById("log-box");
const logPath = document.getElementById("log-path");
const holdingsEl = document.getElementById("holdings");
const syncStatus = document.getElementById("sync-status");
const logFollow = document.getElementById("log-follow");
const logDaySelect = document.getElementById("log-day");
const processList = document.getElementById("process-list");
const processCount = document.getElementById("process-count");
const calendarAnchor = document.getElementById("calendar-anchor");
const dataCalendar = document.getElementById("data-calendar");
const perfLookbackInput = document.getElementById("perf-lookback");
const perfMeta = document.getElementById("perf-meta");
const perfMetricsCanvas = document.getElementById("perf-metrics-chart");
const perfNavCanvas = document.getElementById("perf-nav-chart");

let perfMetricsChart = null;
let perfNavChart = null;
let calendarSelectedDay = "";

let followLogs = true;
function atBottom(el) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < 10;
}

function escapeHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function normalizeDay(value) {
  return String(value ?? "").replaceAll("-", "").trim();
}

function selectDayInDropdown(dayValue) {
  if (!logDaySelect) return false;
  const target = normalizeDay(dayValue);
  if (!target) return false;
  for (const opt of Array.from(logDaySelect.options)) {
    if (normalizeDay(opt.value) === target) {
      logDaySelect.value = opt.value;
      calendarSelectedDay = target;
      return true;
    }
  }
  return false;
}

function ensureDayOption(dayValue, dayLabel) {
  if (!logDaySelect) return;
  const target = normalizeDay(dayValue);
  if (!target) return;
  for (const opt of Array.from(logDaySelect.options)) {
    if (normalizeDay(opt.value) === target) {
      return;
    }
  }
  const option = document.createElement("option");
  option.value = target;
  option.textContent = dayLabel || target;
  logDaySelect.insertBefore(option, logDaySelect.firstChild);
}

logBox.addEventListener("scroll", () => {
  if (!logFollow) {
    return;
  }
  followLogs = atBottom(logBox);
  if (logFollow) {
    logFollow.checked = followLogs;
  }
});

async function refreshLogs() {
  const selectedDay = logDaySelect && logDaySelect.value ? logDaySelect.value : "";
  const res = await axios.get("/api/logs", { params: selectedDay ? { day: selectedDay } : {} });
  logPath.textContent = res.data.path || "";
  const wasAtBottom = atBottom(logBox);
  logBox.textContent = (res.data.lines || []).join("\n");
  if (logFollow ? logFollow.checked : true) {
    if (followLogs && wasAtBottom) {
      logBox.scrollTop = logBox.scrollHeight;
    }
  }
}

async function refreshLogsSafe() {
  try {
    await refreshLogs();
  } catch (err) {
    // keep UI stable when backend log file rotates/misses briefly
  }
}

async function loadLogDays() {
  if (!logDaySelect) return;
  const prev = logDaySelect.value;
  const res = await axios.get("/api/log_days");
  const days = [...(res.data.days || [])];
  const currentDay = res.data.current_day || "";
  if (prev && !days.some((d) => normalizeDay(d) === normalizeDay(prev))) {
    days.unshift(prev);
  }
  logDaySelect.innerHTML = days
    .map((d) => `<option value="${d}">${d}</option>`)
    .join("");
  if (selectDayInDropdown(prev)) {
    return;
  }
  if (selectDayInDropdown(currentDay)) {
    return;
  }
  if (days.length) {
    logDaySelect.value = days[0];
    calendarSelectedDay = normalizeDay(days[0]);
  }
}

async function applyCalendarDay(day) {
  if (!day || !logDaySelect) return false;
  const compactDay = normalizeDay(day);
  if (!compactDay) return false;

  if (!selectDayInDropdown(compactDay)) {
    await loadLogDays();
    if (!selectDayInDropdown(compactDay)) {
      ensureDayOption(compactDay, day);
      logDaySelect.value = compactDay;
    }
  }
  return true;
}

async function refreshBySelectedDay() {
  await refreshLogs();
  await refreshHoldings();
  await refreshPerformance();
  await refreshDataCalendar();
}

async function onCalendarDayClick(day) {
  calendarSelectedDay = normalizeDay(day);
  const ok = await applyCalendarDay(day);
  if (!ok) return;
  await refreshBySelectedDay();
}

if (dataCalendar) {
  dataCalendar.addEventListener("click", async (evt) => {
    const target = evt.target;
    if (!(target instanceof HTMLElement)) return;
    const day = target.getAttribute("data-day");
    if (!day) return;
    await onCalendarDayClick(day);
  });
}

async function refreshHoldings() {
  const selectedDay = logDaySelect && logDaySelect.value ? logDaySelect.value : "";
  const res = await axios.get("/api/holdings", { params: selectedDay ? { day: selectedDay } : {} });
  const rows = res.data.rows || [];
  if (!rows.length) {
    holdingsEl.innerHTML = "<div class='text-muted'>No holdings</div>";
    return;
  }
  holdingsEl.innerHTML = rows
    .map((row) => {
      const w = row.weight == null ? "" : Number(row.weight).toFixed(6);
      return `
        <div class="holding-card">
          <div class="sym">${row.symbol}</div>
          <div class="text-end">
            <div>weight: ${w}</div>
          </div>
        </div>
      `;
    })
    .join("");
}

function destroyCharts() {
  if (perfMetricsChart) {
    perfMetricsChart.destroy();
    perfMetricsChart = null;
  }
  if (perfNavChart) {
    perfNavChart.destroy();
    perfNavChart = null;
  }
}

async function refreshPerformance() {
  if (!perfLookbackInput || !perfMeta || !perfMetricsCanvas || !perfNavCanvas) {
    return;
  }
  try {
    const selectedDay = logDaySelect && logDaySelect.value ? logDaySelect.value : "";
    let lookback = Number.parseInt(perfLookbackInput.value || "20", 10);
    if (!Number.isFinite(lookback) || lookback <= 0) {
      lookback = 20;
    }
    perfLookbackInput.value = String(lookback);

    const res = await axios.get("/api/perf_summary", {
      params: { ...(selectedDay ? { day: selectedDay } : {}), lookback },
    });
    const payload = res.data || {};
    const series = payload.series || [];
    if (!series.length) {
      destroyCharts();
      perfMeta.textContent = "No performance data";
      return;
    }

    const metrics = payload.metrics || {};
    const sharpe = Number(metrics.sharpe || 0);
    const vol = Number(metrics.volatility || 0);
    const benchSharpe = Number(metrics.benchmark_sharpe || 0);
    const benchVol = Number(metrics.benchmark_volatility || 0);

    const labels = series.map((x) => x.trade_date);
    const nav = series.map((x) => Number(x.strategy_nav || 0));
    const benchNav = series.map((x) => Number(x.benchmark_nav || 0));

    perfMeta.textContent = `asof: ${payload.asof_day || "-"} | samples: ${payload.count_days || 0} | lookback: ${payload.lookback || lookback}`;

    if (perfMetricsChart) {
      perfMetricsChart.destroy();
    }
    perfMetricsChart = new Chart(perfMetricsCanvas.getContext("2d"), {
      type: "bar",
      data: {
        labels: ["Sharpe", "Volatility"],
        datasets: [
          {
            label: "Strategy",
            data: [sharpe, vol],
            backgroundColor: "#2b7fff",
          },
          {
            label: "Benchmark",
            data: [benchSharpe, benchVol],
            backgroundColor: "#30a46c",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true, position: "bottom" },
        },
      },
    });

    if (perfNavChart) {
      perfNavChart.destroy();
    }
    perfNavChart = new Chart(perfNavCanvas.getContext("2d"), {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Strategy NAV",
            data: nav,
            borderColor: "#2b7fff",
            backgroundColor: "rgba(43,127,255,0.12)",
            tension: 0.25,
            pointRadius: 2,
            pointHoverRadius: 5,
            borderWidth: 2,
          },
          {
            label: "Benchmark NAV",
            data: benchNav,
            borderColor: "#30a46c",
            backgroundColor: "rgba(48,164,108,0.10)",
            tension: 0.25,
            pointRadius: 2,
            pointHoverRadius: 5,
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: "index",
          intersect: false,
        },
        plugins: {
          legend: { display: true, position: "bottom" },
          tooltip: {
            enabled: true,
            callbacks: {
              title: (items) => {
                if (!items || !items.length) return "";
                return `Date: ${items[0].label}`;
              },
              label: (ctx) => {
                const name = ctx.dataset?.label || "";
                const val = Number(ctx.parsed?.y ?? NaN);
                if (!Number.isFinite(val)) return `${name}: -`;
                return `${name}: ${val.toFixed(6)}`;
              },
            },
          },
        },
        scales: {
          x: { ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 8 } },
        },
      },
    });
  } catch (err) {
    destroyCharts();
    perfMeta.textContent = "Performance load failed";
  }
}

async function refreshProcesses() {
  const res = await axios.get("/api/processes");
  const items = res.data.items || [];
  if (processCount) {
    processCount.textContent = `Running: ${items.length}`;
  }
  if (!processList) return;
  if (!items.length) {
    processList.innerHTML = "<div class='text-muted'>No live processes</div>";
    return;
  }
  processList.innerHTML = items
    .map((item) => {
      const cmd = (item.cmd || "").replace(/.*liveLaunch\\\\/, "liveLaunch\\");
      return `
        <div class="process-card">
          <div class="pid">PID: ${item.pid}</div>
          <div class="text-muted">start: ${item.start || "-"}</div>
          <div class="text-muted">cmd: ${cmd}</div>
        </div>
      `;
    })
    .join("");
}

async function refreshDataCalendar() {
  if (!dataCalendar) return;
  const selectedDay = calendarSelectedDay || (logDaySelect && logDaySelect.value ? normalizeDay(logDaySelect.value) : "");
  const res = await axios.get("/api/data_calendar", {
    params: { months: 6 },
  });
  const payload = res.data || {};
  const months = (payload.months || [])
    .slice()
    .sort((a, b) => {
      const ka = String((a && (a.month || a.title)) || "");
      const kb = String((b && (b.month || b.title)) || "");
      // Desc order so current/latest month shows first.
      return kb.localeCompare(ka);
    });
  if (calendarAnchor) {
    const expected = Number(payload.expected_factor_count || 0);
    const panel = payload.label ? ` ${payload.label}` : "";
    const label = payload.anchor_day ? `asof ${payload.anchor_day}${panel} | factors ${expected}` : "";
    calendarAnchor.textContent = label;
  }
  if (!months.length) {
    dataCalendar.innerHTML = "<div class='text-muted small'>No calendar data</div>";
    return;
  }
  const weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  dataCalendar.innerHTML = months
    .map((m) => {
      const weeks = m.weeks || [];
      const cells = weeks
        .map((week) =>
          (week || [])
            .map((cell) => {
              if (!cell) return "<div class='cal-cell cal-empty'></div>";
              const isSelected = normalizeDay(cell.day) === normalizeDay(selectedDay);
              const cls = `cal-cell cal-${cell.status || "off"}${isSelected ? " cal-selected" : ""}`;
              const title = escapeHtml(`${cell.day} | ${cell.detail || ""}`);
              return `<button type="button" class="${cls}" data-day="${cell.day}" title="${title}">${cell.day_num}</button>`;
            })
            .join("")
        )
        .join("");
      return `
        <div class="cal-month">
          <div class="cal-month-title">${escapeHtml(m.title || "")}</div>
          <div class="cal-weekday">${weekday.map((w) => `<div>${w}</div>`).join("")}</div>
          <div class="cal-grid">${cells}</div>
        </div>
      `;
    })
    .join("");
}

async function callAction(url) {
  try {
    const res = await axios.post(url);
    return res.data;
  } catch (err) {
    let msg = "request failed";
    if (err && err.response) {
      const status = err.response.status;
      const body = err.response.data;
      if (typeof body === "string" && body.trim()) {
        msg = `HTTP ${status}: ${body}`;
      } else if (body && typeof body === "object" && body.error) {
        msg = `HTTP ${status}: ${body.error}`;
      } else {
        msg = `HTTP ${status}`;
      }
    } else if (err && err.message) {
      msg = err.message;
    }
    throw new Error(msg);
  }
}

async function loadConfig() {
  const res = await axios.get("/api/config");
  const cfg = res.data.config || {};
  const meta = res.data.meta || { read_only: [], types: {} };
  const readonly = new Set(meta.read_only || []);
  const hiddenRootKeys = new Set(["start", "target"]);
  const list = document.getElementById("config-list");

  const groups = { bool: [], data: [], readonly: [] };

  function inferType(val) {
    if (typeof val === "boolean") return "bool";
    if (typeof val === "number") return Number.isInteger(val) ? "int" : "float";
    if (Array.isArray(val)) return "json";
    if (val !== null && typeof val === "object") return "object";
    return "str";
  }

  function pushItem(path, val, ro) {
    const type = inferType(val);
    if (type === "object") {
      const childKeys = Object.keys(val || {}).sort();
      childKeys.forEach((k) => pushItem(`${path}.${k}`, val[k], ro));
      return;
    }
    const item = { key: path, val, type, ro };
    if (ro) {
      groups.readonly.push(item);
    } else if (type === "bool") {
      groups.bool.push(item);
    } else {
      groups.data.push(item);
    }
  }

  const keys = Object.keys(cfg).sort();
  keys.forEach((key) => {
    if (hiddenRootKeys.has(key)) {
      return;
    }
    const ro = readonly.has(key);
    pushItem(key, cfg[key], ro);
  });

  function renderItem({ key, val, type, ro }) {
    let input = "";
    if (type === "bool") {
      input = `<input type="checkbox" class="form-check-input" data-key="${key}" data-type="bool" ${val ? "checked" : ""} ${ro ? "disabled" : ""}>`;
    } else {
      if (type === "json") {
        const jsonVal = JSON.stringify(val);
        input = `<input type="text" class="form-control form-control-sm" data-key="${key}" data-type="json" value='${jsonVal}' ${ro ? "disabled" : ""}>`;
      } else {
        const inputType = type === "int" || type === "float" ? "number" : "text";
        const safeVal = val == null ? "" : String(val);
        input = `<input type="${inputType}" class="form-control form-control-sm" data-key="${key}" data-type="${type}" value="${safeVal}" ${ro ? "disabled" : ""}>`;
      }
    }
    return `
        <div class="config-item ${ro ? "readonly" : ""}">
          <div class="label">${key}</div>
          ${input}
        </div>
      `;
  }

  function renderSection(title, items) {
    if (!items.length) return "";
    return `
      <div class="config-section">
        <div class="config-grid">
          ${items.map(renderItem).join("")}
        </div>
      </div>
    `;
  }

  list.innerHTML =
    renderSection("", groups.bool) +
    renderSection("", groups.data) +
    renderSection("", groups.readonly);
}

async function saveConfig() {
  const list = document.getElementById("config-list");
  const inputs = list.querySelectorAll("input[data-key]");
  const payload = {};

  function setByPath(obj, dottedKey, value) {
    const parts = dottedKey.split(".");
    let cur = obj;
    for (let i = 0; i < parts.length - 1; i += 1) {
      const p = parts[i];
      if (!cur[p] || typeof cur[p] !== "object" || Array.isArray(cur[p])) {
        cur[p] = {};
      }
      cur = cur[p];
    }
    cur[parts[parts.length - 1]] = value;
  }

  inputs.forEach((el) => {
    const key = el.getAttribute("data-key");
    const type = el.getAttribute("data-type");
    if (el.disabled) return;
    let value;
    if (type === "bool") {
      value = el.checked;
    } else if (type === "int") {
      value = el.value === "" ? null : parseInt(el.value, 10);
    } else if (type === "float") {
      value = el.value === "" ? null : parseFloat(el.value);
    } else if (type === "json") {
      try {
        value = el.value === "" ? [] : JSON.parse(el.value);
      } catch (err) {
        value = [];
      }
    } else {
      value = el.value;
    }
    setByPath(payload, key, value);
  });
  await axios.post("/api/config", payload);
  syncStatus.textContent = "Config saved";
}

document.getElementById("btn-open").addEventListener("click", async () => {
  try {
    await callAction("/api/start_open");
    syncStatus.textContent = "Start Open triggered";
    await refreshLogsSafe();
    await refreshHoldings();
    await refreshPerformance();
  } catch (err) {
    syncStatus.textContent = `Start Open failed: ${err.message || err}`;
  }
});

document.getElementById("btn-restart").addEventListener("click", async () => {
  try {
    await callAction("/api/restart_scheduler");
    syncStatus.textContent = "Scheduler restarted";
    await refreshLogsSafe();
    await refreshProcesses();
  } catch (err) {
    syncStatus.textContent = `Restart failed: ${err.message || err}`;
  }
});

document.getElementById("btn-stop").addEventListener("click", async () => {
  try {
    await callAction("/api/emergency_stop");
    syncStatus.textContent = "Emergency Stop triggered";
    await refreshLogsSafe();
    await refreshProcesses();
  } catch (err) {
    syncStatus.textContent = `Emergency Stop failed: ${err.message || err}`;
  }
});

document.getElementById("btn-sync").addEventListener("click", async () => {
  try {
    const res = await callAction("/api/sync_holdings");
    if (res.ok) {
      syncStatus.textContent = `Synced holdings: ${res.count}`;
    } else {
      syncStatus.textContent = `Sync failed: ${res.error || ""}`;
    }
    await refreshHoldings();
    await refreshPerformance();
    await refreshLogsSafe();
  } catch (err) {
    syncStatus.textContent = `Sync failed: ${err.message || err}`;
  }
});

document.getElementById("btn-shutdown").addEventListener("click", async () => {
  try {
    await callAction("/api/shutdown");
    setTimeout(() => {
      window.close();
    }, 300);
  } catch (err) {
    syncStatus.textContent = `Shutdown failed: ${err.message || err}`;
  }
});

document.getElementById("btn-save-config").addEventListener("click", async () => {
  await saveConfig();
});

if (logDaySelect) {
  logDaySelect.addEventListener("change", async () => {
    calendarSelectedDay = normalizeDay(logDaySelect.value);
    await refreshLogs();
    await refreshHoldings();
    await refreshPerformance();
    await refreshDataCalendar();
  });
}

const btnPerfRefresh = document.getElementById("btn-perf-refresh");
if (btnPerfRefresh) {
  btnPerfRefresh.addEventListener("click", async () => {
    await refreshPerformance();
  });
}

if (perfLookbackInput) {
  perfLookbackInput.addEventListener("change", async () => {
    await refreshPerformance();
  });
}

async function tick() {
  await refreshLogsSafe();
}

loadLogDays().then(async () => {
  await tick();
  await refreshPerformance();
  await refreshDataCalendar();
});
loadConfig();
setInterval(tick, 3000);
setInterval(loadLogDays, 30000);
refreshProcesses();
setInterval(refreshProcesses, 30000);
setInterval(refreshDataCalendar, 60000);
