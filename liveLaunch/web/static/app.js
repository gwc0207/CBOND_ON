const el = (id) => document.getElementById(id);

const liveStateBadge = el("live-state-badge");
const liveStateReason = el("live-state-reason");
const tradeDateEl = el("trade-date");
const targetDateEl = el("target-date");
const heartbeatSummary = el("heartbeat-summary");
const envName = el("env-name");
const lastUpdated = el("last-updated");
const statusCards = el("status-cards");
const liveTimeline = el("live-timeline");
const nextAction = el("next-action");
const alertSummary = el("alert-summary");
const recentEvents = el("recent-events");
const logSummaryMeta = el("log-summary-meta");
const profileSummary = el("profile-summary");
const coverageSummary = el("coverage-summary");

const logBox = el("log-box");
const logPath = el("log-path");
const holdingsEl = el("holdings");
const returnOverviewEl = el("return-overview");
const returnRankingEl = el("return-ranking");
const syncStatus = el("sync-status");
const logFollow = el("log-follow");
const logDaySelect = el("log-day");
const sellTwapSelect = el("sell-twap-col");
const tradeRefreshBtn = el("btn-trade-refresh");
const calendarAnchor = el("calendar-anchor");
const dataCalendar = el("data-calendar");
const perfLookbackInput = el("perf-lookback");
const perfMeta = el("perf-meta");
const perfMetricsCanvas = el("perf-metrics-chart");
const perfNavCanvas = el("perf-nav-chart");
const configMode = el("config-mode");

let perfMetricsChart = null;
let perfNavChart = null;
let returnDistributionChart = null;
let contributionDonutChart = null;
let calendarSelectedDay = "";
let followLogs = true;
let latestLiveStatus = null;
const SELL_TWAP_STORAGE_KEY = "cbond_on.dashboard.sell_twap_col";
const SELL_TWAP_OPTIONS = [
  "twap_0930_0935",
  "twap_0930_0938",
  "twap_0930_0939",
  "twap_0930_0945",
  "twap_0930_1000",
];

function markTradeFilterDirty() {
  if (!syncStatus) return;
  syncStatus.textContent = "筛选已修改，点击“刷新”更新持仓与绩效。";
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

function isValidTwapCol(value) {
  return /^twap_\d{4}_\d{4}$/.test(String(value ?? "").trim());
}

function formatTwapCol(col) {
  const text = String(col ?? "").trim();
  const m = /^twap_(\d{4})_(\d{4})$/.exec(text);
  if (!m) return text || "-";
  const from = `${m[1].slice(0, 2)}:${m[1].slice(2, 4)}`;
  const to = `${m[2].slice(0, 2)}:${m[2].slice(2, 4)}`;
  return `${from}-${to}`;
}

function getSelectedSellTwapCol() {
  const value = String(sellTwapSelect?.value || "").trim();
  return isValidTwapCol(value) ? value : "";
}

function ensureSellTwapOption(col) {
  if (!sellTwapSelect || !isValidTwapCol(col)) return;
  for (const opt of Array.from(sellTwapSelect.options || [])) {
    if (String(opt.value).trim() === col) return;
  }
  const option = document.createElement("option");
  option.value = col;
  option.textContent = formatTwapCol(col);
  sellTwapSelect.appendChild(option);
}

function initSellTwapSelector(defaultSellCol) {
  if (!sellTwapSelect) return;
  sellTwapSelect.innerHTML = SELL_TWAP_OPTIONS
    .map((col) => `<option value="${escapeHtml(col)}">${escapeHtml(formatTwapCol(col))}</option>`)
    .join("");
  const saved = String(window.localStorage.getItem(SELL_TWAP_STORAGE_KEY) || "").trim();
  const selected = isValidTwapCol(saved)
    ? saved
    : (isValidTwapCol(defaultSellCol) ? String(defaultSellCol).trim() : SELL_TWAP_OPTIONS[0]);
  ensureSellTwapOption(selected);
  sellTwapSelect.value = selected;
  if (!sellTwapSelect.value) {
    sellTwapSelect.value = SELL_TWAP_OPTIONS[0];
  }
}

function atBottom(node) {
  if (!node) return true;
  return node.scrollHeight - node.scrollTop - node.clientHeight < 10;
}

function formatAge(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) {
    return "-";
  }
  const s = Math.max(0, Number(seconds));
  if (s < 60) return `${Math.round(s)} 秒前`;
  const m = Math.floor(s / 60);
  const rest = Math.round(s % 60);
  if (m < 60) return `${m} 分 ${rest} 秒前`;
  const h = Math.floor(m / 60);
  return `${h} 小时 ${m % 60} 分前`;
}

const LABEL_ZH = {
  "Idle After Run": "本轮已完成",
  "Live Completed": "实盘完成",
  "Live Running": "实盘运行中",
  "Waiting Cutoff": "等待截点",
  "Heartbeat Stale": "心跳过期",
  "Status Error": "状态异常",
  Running: "运行中",
  Ready: "已就绪",
  Waiting: "等待中",
  Completed: "已完成",
  "Not Run": "未运行",
  "Not Generated": "未生成",
  "Run succeeded": "写入完成",
  "Enabled by config": "配置已启用",
  "Disabled by config": "配置关闭",
  Unknown: "未知",
  Fresh: "正常",
  Stale: "过期",
  "Trade Day": "交易日",
  "Ready Gate": "就绪检查",
  "Build Panel": "构建面板",
  "Compute Factors": "计算因子",
  "Model Score": "模型打分",
  "Strategy Select": "策略选券",
  "Trade List": "交易清单",
  "DB Write": "数据库写入",
  Monitor: "持续监控",
  "Waiting for cutoff": "等待截点",
  "Check scheduler": "检查调度器",
  "Heartbeat normal": "心跳正常",
};

const STATUS_ZH = {
  success: "完成",
  running: "运行中",
  waiting: "等待",
  warning: "警告",
  failed: "失败",
  disabled: "关闭",
  unknown: "未知",
  stale: "过期",
  fresh: "正常",
  not_started: "未开始",
};

const REASON_ZH = new Map([
  ["target already ran", "目标日已经完成，本轮进入监控状态"],
  ["cutoff time not reached", "尚未到达截点时间"],
  ["no manual action required", "无需人工操作"],
  ["process exists and heartbeat is fresh", "调度进程存在，心跳正常"],
  ["scheduler heartbeat is stale", "调度器心跳已过期"],
  ["heartbeat is fresh", "心跳正常"],
  ["heartbeat missing or stale", "心跳缺失或已过期"],
  ["ready gate has passed for current live cycle", "当前实盘周期已通过就绪检查"],
  ["loaded factor profile live/live_factors", "已加载实盘因子配置 live/live_factors"],
  ["live run completed", "实盘链路已完成"],
  ["live run completed with db_write enabled", "实盘链路完成，数据库写入已启用"],
  ["waiting for factors", "等待因子输出"],
  ["waiting for strategy output", "等待策略输出"],
  ["latest trade list found for 2026-05-13", "已找到 2026-05-13 的交易清单"],
  ["manual sync available", "可手动同步持仓"],
  ["manual restart available", "可手动重启调度器"],
  ["stops only the dashboard UI server", "只关闭控制台服务，不停止调度器"],
  ["sets STOP flag and kills scheduler process", "写入 STOP 标记并停止调度器进程"],
  ["disabled in production; set CBOND_ON_DASHBOARD_ALLOW_CONFIG_WRITE=1 to enable", "生产模式禁止保存配置；本地调试需设置 CBOND_ON_DASHBOARD_ALLOW_CONFIG_WRITE=1"],
  ["scheduler running; action will restart the open cycle", "调度器正在运行；执行后会重启开盘链路"],
]);

function zhLabel(value) {
  const text = String(value ?? "").trim();
  const expected = text.match(/^(\d+)\s+expected$/i);
  if (expected) return `${expected[1]} 个预期`;
  const rows = text.match(/^(\d+)\s+rows?$/i);
  if (rows) return `${rows[1]} 行`;
  return LABEL_ZH[text] || text || "未知";
}

function zhStatus(value) {
  const text = String(value ?? "unknown").trim().toLowerCase();
  return STATUS_ZH[text] || text || "未知";
}

function zhReason(value) {
  const text = String(value ?? "").trim();
  if (!text) return "";
  if (REASON_ZH.has(text)) return REASON_ZH.get(text);
  return text
    .replaceAll("live run completed", "实盘链路已完成")
    .replaceAll("waiting for previous step", "等待上一步完成")
    .replaceAll("waiting for ready gate", "等待就绪检查")
    .replaceAll("waiting for panel", "等待面板数据")
    .replaceAll("waiting for score", "等待模型分数")
    .replaceAll("source file not found", "源文件不存在")
    .replaceAll("not_available", "暂无可用数据")
    .replaceAll("no scheduler state", "暂无调度器状态")
    .replaceAll("target already ran", "目标日已经完成");
}

function zhProfileValue(value) {
  const text = String(value ?? "-");
  return text
    .replaceAll("enabled", "已启用")
    .replaceAll("disabled by config", "配置关闭")
    .replaceAll("redis_snapshot incremental", "Redis 快照增量")
    .replaceAll("unknown", "未知");
}

function zhEventMessage(message) {
  const text = String(message ?? "");
  return text
    .replace("[run] start", "[运行] 开始")
    .replace("[run] success", "[运行] 成功")
    .replace("[run] failed", "[运行] 失败")
    .replace("[dashboard] shutdown", "[控制台] 关闭")
    .replace("[dashboard] restart_scheduler", "[控制台] 重启调度器")
    .replace("[dashboard] emergency_stop", "[控制台] 紧急停止")
    .replace("[dashboard] sync_holdings", "[控制台] 同步持仓")
    .replace("target=", "目标日=")
    .replace("out=", "输出=");
}

function toneOf(item) {
  const health = String(item?.health || "").toLowerCase();
  const status = String(item?.status || "").toLowerCase();
  if (health === "error" || status === "failed") return "danger";
  if (health === "stale" || status === "stale") return "danger";
  if (health === "warning" || status === "warning") return "warning";
  if (status === "running") return "running";
  if (health === "ok" || status === "success" || status === "fresh") return "success";
  if (status === "waiting") return "waiting";
  if (status === "disabled") return "disabled";
  return "unknown";
}

function badgeClass(item) {
  return `status-badge status-${toneOf(item)}`;
}

function setStatusBadge(node, item) {
  if (!node) return;
  node.className = badgeClass(item);
  node.textContent = zhLabel(item?.label || item?.state || "Unknown");
  node.title = zhReason(item?.reason || "");
}

function renderHeader(payload) {
  const live = payload.live || {};
  const freshness = payload.freshness || {};
  setStatusBadge(liveStateBadge, live);
  if (liveStateReason) {
    liveStateReason.textContent = zhReason(live.reason || "");
  }
  if (tradeDateEl) tradeDateEl.textContent = payload.trade_date || "-";
  if (targetDateEl) targetDateEl.textContent = payload.target_date || "-";
  if (envName) envName.textContent = payload.env || "-";
  if (lastUpdated) lastUpdated.textContent = payload.asof || "-";
  if (heartbeatSummary) {
    const last = freshness.last_heartbeat || "-";
    const age = formatAge(freshness.heartbeat_age_sec);
    heartbeatSummary.textContent = last === "-" ? "-" : `${last} · ${age}`;
    heartbeatSummary.className = `meta-value heartbeat-${toneOf(freshness)}`;
  }
  if (nextAction) {
    const na = payload.next_action || {};
    const label = zhLabel(na.label || "Monitor");
    const reason = zhReason(na.reason || "");
    nextAction.textContent = `${label}${reason ? ` · ${reason}` : ""}`;
  }
}

function renderStatusCards(cards) {
  if (!statusCards) return;
  const order = [
    ["scheduler", "调度器"],
    ["data_ready", "数据就绪"],
    ["factors", "因子"],
    ["model", "模型打分"],
    ["trade_list", "交易清单"],
    ["db_write", "数据库写入"],
  ];
  statusCards.innerHTML = order
    .map(([key, title]) => {
      const item = cards?.[key] || {};
      const tone = toneOf(item);
      const detail = zhReason(item.reason || "");
      let metric = "";
      if (key === "scheduler" && item.pid) metric = `PID ${item.pid}`;
      if (key === "factors" && Number.isFinite(Number(item.total))) metric = `${item.total} 个因子`;
      if (key === "trade_list" && Number.isFinite(Number(item.count))) metric = `${item.count} 行`;
      if (key === "model" && item.ref) metric = item.ref;
      return `
        <article class="status-card status-card-${tone}">
          <div class="status-card-title">${escapeHtml(title)}</div>
          <div class="status-card-value">${escapeHtml(zhLabel(item.label || "Unknown"))}</div>
          <div class="status-card-detail">${escapeHtml(metric || detail || "-")}</div>
        </article>
      `;
    })
    .join("");
}

function renderTimeline(items) {
  if (!liveTimeline) return;
  const rows = items || [];
  liveTimeline.innerHTML = rows
    .map((item) => {
      const tone = toneOf(item);
      return `
        <div class="timeline-step timeline-${tone}" title="${escapeHtml(zhReason(item.reason || ""))}">
          <div class="timeline-dot"></div>
          <div class="timeline-label">${escapeHtml(zhLabel(item.label || item.key || ""))}</div>
          <div class="timeline-status">${escapeHtml(zhStatus(item.status || "unknown"))}</div>
        </div>
      `;
    })
    .join("");
}

function applyActionState(actions) {
  const map = {
    start_open: el("btn-open"),
    sync_holdings: el("btn-sync"),
    restart_scheduler: el("btn-restart"),
    emergency_stop: el("btn-stop"),
    shutdown_ui: el("btn-shutdown"),
    save_config: el("btn-save-config"),
  };
  Object.entries(map).forEach(([key, node]) => {
    if (!node) return;
    const spec = actions?.[key] || {};
    const enabled = spec.enabled !== false;
    node.disabled = !enabled;
    node.title = zhReason(spec.reason || "");
  });
}

function renderAlerts(payload) {
  const alerts = payload.alerts || {};
  const freshness = payload.freshness || {};
  if (logSummaryMeta) {
    const heartbeat = freshness.last_heartbeat ? `${freshness.last_heartbeat} · ${formatAge(freshness.heartbeat_age_sec)}` : "-";
    logSummaryMeta.textContent = `最近心跳：${heartbeat}`;
  }
  if (alertSummary) {
    const errorCount = Number(alerts.errors || 0);
    const warningCount = Number(alerts.warnings || 0);
    const hb = alerts.heartbeat || {};
    const hbText = hb.count ? `心跳重复 ${hb.count} 次` : zhLabel(hb.summary || "Heartbeat normal");
    alertSummary.innerHTML = `
      <div class="alert-pill ${errorCount ? "alert-danger-soft" : "alert-ok-soft"}">错误 ${errorCount}</div>
      <div class="alert-pill ${warningCount ? "alert-warning-soft" : "alert-ok-soft"}">警告 ${warningCount}</div>
      <div class="alert-pill alert-muted-soft">${escapeHtml(hbText)}</div>
    `;
  }
  if (recentEvents) {
    const events = alerts.recent_events || [];
    if (!events.length) {
      recentEvents.innerHTML = "<div class='empty-mini'>暂无关键事件</div>";
      return;
    }
    recentEvents.innerHTML = events
      .map((item) => `<div class="event-line event-${escapeHtml(item.level || "info")}">${escapeHtml(zhEventMessage(item.message || ""))}</div>`)
      .join("");
  }
}

function renderProfileSummary(payload) {
  const summary = payload.profile_summary || {};
  if (profileSummary) {
    const items = [
      ["配置文件", summary.profile],
      ["模型版本", summary.model_ref],
      ["策略", summary.strategy],
      ["因子配置", summary.factor_profile],
      ["数据模式", summary.data_mode],
      ["就绪检查", summary.ready_gate],
      ["数据库写入", summary.db_write],
    ];
    profileSummary.innerHTML = items
      .map(([k, v]) => `
        <div class="profile-item">
          <span>${escapeHtml(k)}</span>
          <strong>${escapeHtml(zhProfileValue(v || "-"))}</strong>
        </div>
      `)
      .join("");
  }
  if (coverageSummary) {
    const factors = payload.cards?.factors || {};
    coverageSummary.innerHTML = `
      <div class="coverage-chip">${escapeHtml(zhLabel(factors.label || "未知因子状态"))}</div>
      <div class="coverage-chip">${escapeHtml(zhReason(factors.reason || "暂无覆盖摘要"))}</div>
    `;
  }
}

async function refreshLiveStatus() {
  try {
    const res = await axios.get("/api/live_status");
    latestLiveStatus = res.data || {};
    renderHeader(latestLiveStatus);
    renderStatusCards(latestLiveStatus.cards || {});
    renderTimeline(latestLiveStatus.timeline || []);
    applyActionState(latestLiveStatus.actions || {});
    renderAlerts(latestLiveStatus);
    renderProfileSummary(latestLiveStatus);
  } catch (err) {
    setStatusBadge(liveStateBadge, {
      label: "Status Error",
      status: "failed",
      health: "error",
      reason: err?.message || "status request failed",
    });
    if (liveStateReason) liveStateReason.textContent = `状态接口请求失败：${err?.message || "未知错误"}`;
  }
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
    if (normalizeDay(opt.value) === target) return;
  }
  const option = document.createElement("option");
  option.value = target;
  option.textContent = dayLabel || target;
  logDaySelect.insertBefore(option, logDaySelect.firstChild);
}

if (logBox) {
  logBox.addEventListener("scroll", () => {
    followLogs = atBottom(logBox);
    if (logFollow) logFollow.checked = followLogs;
  });
}

async function refreshLogs() {
  if (!logBox || !logPath) return;
  const selectedDay = logDaySelect && logDaySelect.value ? logDaySelect.value : "";
  const res = await axios.get("/api/logs", { params: selectedDay ? { day: selectedDay } : {} });
  logPath.textContent = res.data.path || "";
  const wasAtBottom = atBottom(logBox);
  logBox.textContent = (res.data.lines || []).join("\n");
  if ((logFollow ? logFollow.checked : true) && followLogs && wasAtBottom) {
    logBox.scrollTop = logBox.scrollHeight;
  }
}

async function refreshLogsSafe() {
  try {
    await refreshLogs();
  } catch (err) {
    // Logs rotate during live runs; keep the dashboard stable.
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
  logDaySelect.innerHTML = days.map((d) => `<option value="${escapeHtml(d)}">${escapeHtml(d)}</option>`).join("");
  if (selectDayInDropdown(prev)) return;
  if (selectDayInDropdown(currentDay)) return;
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
  await refreshLogsSafe();
  await refreshHoldings();
  await refreshPerformance();
  await refreshDataCalendar();
  await refreshLiveStatus();
}

async function refreshTradeWindowOnly() {
  const results = await Promise.allSettled([
    refreshHoldings(),
    refreshPerformance(),
  ]);
  return results.every((x) => x.status === "fulfilled");
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

document.querySelectorAll('button[data-bs-toggle="tab"]').forEach((button) => {
  button.addEventListener("shown.bs.tab", () => {
    if (returnDistributionChart) returnDistributionChart.resize();
    if (contributionDonutChart) contributionDonutChart.resize();
  });
});

function fmtFixed(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function fmtPct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const num = Number(value);
  const sign = num > 0 ? "+" : "";
  return `${sign}${(num * 100).toFixed(2)}%`;
}

function holdingsStatusLabel(row) {
  const status = String(row?.status || "unknown");
  if (row?.status_label) return row.status_label;
  if (status === "ready") return "已出收益";
  if (status === "pending") return "等待收益";
  if (status === "halted") return "停牌";
  if (status === "unavailable") return "缺少行情";
  return "未知";
}

function returnClass(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "return-muted";
  const num = Number(value);
  if (num > 0) return "return-positive";
  if (num < 0) return "return-negative";
  return "return-flat";
}

function numberValue(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return null;
  return Number(value);
}

function median(values) {
  const arr = values.filter((x) => Number.isFinite(Number(x))).map(Number).sort((a, b) => a - b);
  if (!arr.length) return null;
  const mid = Math.floor(arr.length / 2);
  return arr.length % 2 ? arr[mid] : (arr[mid - 1] + arr[mid]) / 2;
}

function mean(values) {
  const arr = values.filter((x) => Number.isFinite(Number(x))).map(Number);
  if (!arr.length) return null;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function sum(values) {
  return values.filter((x) => Number.isFinite(Number(x))).map(Number).reduce((a, b) => a + b, 0);
}

function clamp01(value) {
  return Math.min(1, Math.max(0, Number(value) || 0));
}

function hexToRgb(hex) {
  const clean = String(hex || "").replace("#", "");
  const value = Number.parseInt(clean, 16);
  return {
    r: (value >> 16) & 255,
    g: (value >> 8) & 255,
    b: value & 255,
  };
}

function rgbToHex({ r, g, b }) {
  const part = (value) => Math.round(value).toString(16).padStart(2, "0");
  return `#${part(r)}${part(g)}${part(b)}`;
}

function mixHexColor(light, dark, ratio) {
  const t = clamp01(ratio);
  const a = hexToRgb(light);
  const b = hexToRgb(dark);
  return rgbToHex({
    r: a.r + (b.r - a.r) * t,
    g: a.g + (b.g - a.g) * t,
    b: a.b + (b.b - a.b) * t,
  });
}

function contributionColor(value, maxAbs) {
  const absValue = Math.abs(Number(value) || 0);
  const ratio = maxAbs > 0 ? 0.18 + 0.82 * (absValue / maxAbs) : 0.18;
  if (Number(value) >= 0) return mixHexColor("#fee2e2", "#991b1b", ratio);
  return mixHexColor("#dcfce7", "#166534", ratio);
}

function returnRows(rows) {
  return (rows || []).filter((row) => row.status === "ready" && numberValue(row.return_net) !== null);
}

function returnSummary(rows, payload) {
  const ready = returnRows(rows);
  const returns = ready.map((row) => Number(row.return_net));
  const contributions = ready.map((row) => Number(row.weighted_return || 0));
  const positiveRows = ready.filter((row) => Number(row.return_net) > 0);
  const negativeRows = ready.filter((row) => Number(row.return_net) < 0);
  const best = ready.length ? ready.reduce((a, b) => (Number(a.return_net) > Number(b.return_net) ? a : b)) : null;
  const worst = ready.length ? ready.reduce((a, b) => (Number(a.return_net) < Number(b.return_net) ? a : b)) : null;
  return {
    ready,
    returns,
    contributions,
    dayReturn: ready.length ? sum(contributions) : null,
    avgReturn: mean(returns),
    medianReturn: median(returns),
    positiveCount: positiveRows.length,
    negativeCount: negativeRows.length,
    pendingCount: Number(payload?.pending_count || rows.filter((row) => row.status === "pending").length || 0),
    haltedCount: Number(payload?.halted_count || rows.filter((row) => row.status === "halted").length || 0),
    unavailableCount: Number(payload?.unavailable_count || rows.filter((row) => row.status === "unavailable").length || 0),
    best,
    worst,
    positiveContribution: ready.length ? sum(positiveRows.map((row) => row.weighted_return || 0)) : 0,
    negativeContribution: ready.length ? sum(negativeRows.map((row) => row.weighted_return || 0)) : 0,
  };
}

function analysisEmpty(title, text) {
  return `
    <div class="analysis-card">
      <div class="empty-state">
        <div class="empty-title">${escapeHtml(title)}</div>
        <div class="empty-text">${escapeHtml(text)}</div>
      </div>
    </div>
  `;
}

function buildSignedBins(values, maxBins, color) {
  if (!values.length) return [];
  let minVal = Math.min(...values);
  let maxVal = Math.max(...values);
  if (minVal === maxVal) {
    const pad = Math.max(Math.abs(minVal) * 0.08, 0.05);
    minVal -= pad;
    maxVal += pad;
  }
  const binCount = Math.min(maxBins, Math.max(3, Math.ceil(Math.sqrt(values.length) * 1.5)));
  const step = (maxVal - minVal) / binCount;
  const counts = new Array(binCount).fill(0);
  values.forEach((value) => {
    const idx = Math.min(binCount - 1, Math.max(0, Math.floor((value - minVal) / step)));
    counts[idx] += 1;
  });
  return counts.map((count, idx) => {
    const left = minVal + step * idx;
    const right = left + step;
    return {
      label: `${left.toFixed(1)}~${right.toFixed(1)}`,
      count,
      color,
    };
  });
}

function buildHistogram(values) {
  const arr = values.filter((x) => Number.isFinite(Number(x))).map((x) => Number(x) * 100);
  if (!arr.length) return { labels: [], counts: [], colors: [] };
  const negative = arr.filter((value) => value < 0);
  const zeroCount = arr.filter((value) => value === 0).length;
  const positive = arr.filter((value) => value > 0);
  const hasBothSides = negative.length > 0 && positive.length > 0;
  const maxSideBins = hasBothSides ? 7 : 12;
  const bins = [
    ...buildSignedBins(negative, maxSideBins, "#16a34a"),
    ...(hasBothSides || zeroCount ? [{ label: "0", count: zeroCount, color: "#cbd5e1" }] : []),
    ...buildSignedBins(positive, maxSideBins, "#dc2626"),
  ];
  const labels = bins.map((bin) => bin.label);
  const counts = bins.map((bin) => bin.count);
  const colors = bins.map((bin) => bin.color);
  return { labels, counts, colors };
}

function renderReturnOverview(rows, payload) {
  if (!returnOverviewEl) return;
  if (returnDistributionChart) {
    returnDistributionChart.destroy();
    returnDistributionChart = null;
  }
  if (!rows.length) {
    returnOverviewEl.innerHTML = analysisEmpty("暂无收益总览", "所选日期没有可分析的持仓数据。");
    return;
  }
  const s = returnSummary(rows, payload);
  returnOverviewEl.innerHTML = `
    <div class="analysis-card overview-card">
      <div class="analysis-head">
        <div>
          <div class="analysis-title">当日收益总览</div>
          <div class="analysis-subtitle">日期 ${escapeHtml(payload?.day || "-")} · 卖出日 ${escapeHtml(payload?.next_day || "-")}</div>
        </div>
        <div class="analysis-status-line">已出 ${s.ready.length} · 等待 ${s.pendingCount} · 停牌 ${s.haltedCount} · 缺行情 ${s.unavailableCount}</div>
      </div>
      <div class="overview-layout">
        <div class="overview-metrics">
          <div class="metric-tile metric-primary"><span>当日组合收益</span><strong class="${returnClass(s.dayReturn)}">${escapeHtml(fmtPct(s.dayReturn))}</strong></div>
          <div class="metric-tile"><span>平均单券收益</span><strong class="${returnClass(s.avgReturn)}">${escapeHtml(fmtPct(s.avgReturn))}</strong></div>
          <div class="metric-tile"><span>中位数收益</span><strong class="${returnClass(s.medianReturn)}">${escapeHtml(fmtPct(s.medianReturn))}</strong></div>
          <div class="metric-tile"><span>正/负收益</span><strong>${s.positiveCount} / ${s.negativeCount}</strong></div>
          <div class="metric-tile"><span>最好单券</span><strong class="${returnClass(s.best?.return_net)}">${escapeHtml(s.best ? `${s.best.symbol} ${fmtPct(s.best.return_net)}` : "-")}</strong></div>
          <div class="metric-tile"><span>最差单券</span><strong class="${returnClass(s.worst?.return_net)}">${escapeHtml(s.worst ? `${s.worst.symbol} ${fmtPct(s.worst.return_net)}` : "-")}</strong></div>
        </div>
        <div class="analysis-chart-box return-distribution-box">
          <div class="chart-title">收益分布（票数）</div>
          ${s.ready.length ? '<canvas id="return-distribution-chart"></canvas>' : '<div class="empty-mini">当天收益尚未出齐，等待 T+1 早盘 TWAP。</div>'}
        </div>
      </div>
    </div>
  `;
  if (!s.ready.length) return;
  const hist = buildHistogram(s.returns);
  const canvas = el("return-distribution-chart");
  if (!canvas) return;
  returnDistributionChart = new Chart(canvas.getContext("2d"), {
    type: "bar",
    data: {
      labels: hist.labels,
      datasets: [{
        label: "票数",
        data: hist.counts,
        backgroundColor: hist.colors,
        borderRadius: 6,
        borderSkipped: false,
        categoryPercentage: 0.72,
        barPercentage: 0.9,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { top: 6, right: 8, bottom: 0, left: 4 } },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => (items.length ? `${items[0].label}%` : ""),
            label: (ctx) => `票数：${ctx.raw}`,
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 8, font: { size: 12, weight: 650 } },
          title: { display: true, text: "单券收益区间（%）", color: "#64748b", font: { size: 12, weight: 700 } },
        },
        y: {
          beginAtZero: true,
          suggestedMax: Math.max(...hist.counts, 1) + 1,
          ticks: { precision: 0, stepSize: 1, font: { size: 12, weight: 650 } },
          title: { display: true, text: "票数", color: "#64748b", font: { size: 12, weight: 700 } },
        },
      },
    },
  });
}

function rankingRowsHtml(rows, title) {
  if (!rows.length) {
    return `<div class="ranking-list"><div class="ranking-title">${escapeHtml(title)}</div><div class="empty-mini">暂无数据</div></div>`;
  }
  return `
    <div class="ranking-list">
      <div class="ranking-title">${escapeHtml(title)}</div>
      <table class="table table-sm ranking-table">
        <thead><tr><th>代码</th><th class="text-end">收益</th><th class="text-end">贡献</th><th class="text-end">权重</th></tr></thead>
        <tbody>
          ${rows.map((row) => `
            <tr class="position-row-${escapeHtml(row.status || "")}">
              <td><span class="bond-code">${escapeHtml(row.symbol || "")}</span></td>
              <td class="text-end"><span class="${returnClass(row.return_net)}">${escapeHtml(fmtPct(row.return_net))}</span></td>
              <td class="text-end"><span class="${returnClass(row.weighted_return)}">${escapeHtml(fmtPct(row.weighted_return))}</span></td>
              <td class="text-end">${escapeHtml(fmtFixed(row.weight, 4))}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderReturnRanking(rows, payload) {
  if (!returnRankingEl) return;
  if (contributionDonutChart) {
    contributionDonutChart.destroy();
    contributionDonutChart = null;
  }
  if (!rows.length) {
    returnRankingEl.innerHTML = analysisEmpty("暂无排名与贡献", "所选日期没有可分析的持仓数据。");
    return;
  }
  const s = returnSummary(rows, payload);
  const ready = returnRows(rows);
  const positiveRows = [...ready]
    .filter((row) => Number(row.return_net || 0) > 0)
    .sort((a, b) => Number(b.return_net || 0) - Number(a.return_net || 0));
  const negativeRows = [...ready]
    .filter((row) => Number(row.return_net || 0) < 0)
    .sort((a, b) => Number(a.return_net || 0) - Number(b.return_net || 0));
  const positiveContributionRows = [...ready]
    .filter((row) => Number(row.weighted_return || 0) > 0)
    .sort((a, b) => Number(b.weighted_return || 0) - Number(a.weighted_return || 0));
  const negativeContributionRows = [...ready]
    .filter((row) => Number(row.weighted_return || 0) < 0)
    .sort((a, b) => Number(a.weighted_return || 0) - Number(b.weighted_return || 0));
  const contributionRows = [...positiveContributionRows, ...negativeContributionRows];
  const maxAbsContribution = Math.max(...contributionRows.map((row) => Math.abs(Number(row.weighted_return || 0))), 0);
  const contributionLabels = contributionRows.map((row) => row.symbol || "-");
  const contributionValues = contributionRows.map((row) => Math.abs(Number(row.weighted_return || 0)));
  const contributionColors = contributionRows.map((row) => contributionColor(row.weighted_return, maxAbsContribution));
  const hasContribution = contributionValues.some((value) => value > 0);
  returnRankingEl.innerHTML = `
    <div class="analysis-card">
      <div class="analysis-head">
        <div>
          <div class="analysis-title">收益排名与贡献</div>
          <div class="analysis-subtitle">左侧按单券收益分组，右侧看正负收益对应的贡献结构和数据状态。</div>
        </div>
        <div class="analysis-status-line">日期 ${escapeHtml(payload?.day || "-")} · 组合收益 ${escapeHtml(fmtPct(s.dayReturn))}</div>
      </div>
      <div class="ranking-contribution-layout">
        <div class="ranking-stack">
          ${rankingRowsHtml(positiveRows, `正收益全部 ${positiveRows.length}`)}
          ${rankingRowsHtml(negativeRows, `负收益全部 ${negativeRows.length}`)}
        </div>
        <div class="contribution-side contribution-merged">
          <div class="analysis-chart-box contribution-chart-box">
            <div class="chart-title">逐券贡献圆环</div>
            ${hasContribution ? '<div class="donut-canvas-wrap"><canvas id="contribution-donut-chart"></canvas></div><div class="chart-note">红色为正贡献，绿色为负贡献；颜色越深，贡献绝对值越大。</div>' : '<div class="empty-mini">当天收益尚未出齐，贡献结构等待计算。</div>'}
          </div>
          <div class="status-breakdown">
            <span>已出 ${s.ready.length}</span>
            <span>等待 ${s.pendingCount}</span>
            <span>停牌 ${s.haltedCount}</span>
            <span>缺行情 ${s.unavailableCount}</span>
          </div>
        </div>
      </div>
    </div>
  `;
  if (!hasContribution) return;
  const canvas = el("contribution-donut-chart");
  if (!canvas) return;
  contributionDonutChart = new Chart(canvas.getContext("2d"), {
    type: "doughnut",
    data: {
      labels: contributionLabels,
      datasets: [{
        data: contributionValues,
        backgroundColor: contributionColors,
        borderColor: "#ffffff",
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      aspectRatio: 1,
      cutout: "62%",
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const row = contributionRows[ctx.dataIndex] || {};
              return `${ctx.label}: 贡献 ${fmtPct(row.weighted_return)}，收益 ${fmtPct(row.return_net)}`;
            },
          },
        },
      },
    },
  });
}

async function refreshHoldings() {
  if (!holdingsEl) return;
  const selectedDay = logDaySelect && logDaySelect.value ? logDaySelect.value : "";
  const selectedSellCol = getSelectedSellTwapCol();
  const params = {
    ...(selectedDay ? { day: selectedDay } : {}),
    ...(selectedSellCol ? { sell_col: selectedSellCol } : {}),
  };
  const res = await axios.get("/api/holdings", { params });
  const rows = res.data.rows || [];
  if (!rows.length) {
    renderReturnOverview([], res.data || {});
    renderReturnRanking([], res.data || {});
    holdingsEl.innerHTML = `
      <div class="empty-state">
        <div class="empty-title">暂无持仓数据</div>
        <div class="empty-text">所选日期尚未生成 trade_list.csv，或当前目标日还未完成。</div>
      </div>
    `;
    return;
  }
  const meta = [
    `日期 ${res.data.day || "-"}`,
    res.data.next_day ? `卖出日 ${res.data.next_day}` : "卖出日 -",
    `已出 ${res.data.ready_count || 0}`,
    `等待 ${res.data.pending_count || 0}`,
    `停牌 ${res.data.halted_count || 0}`,
  ].join(" · ");
  const metaWithSell = [meta, `卖出列 ${formatTwapCol(res.data.sell_col || selectedSellCol || "-")}`].join(" · ");
  holdingsEl.innerHTML = `
    <div class="position-return-meta">${escapeHtml(metaWithSell)}</div>
    <div class="table-responsive">
      <table class="table table-sm align-middle position-table position-return-table">
        <colgroup>
          <col class="position-code-col">
          <col class="position-rank-col">
          <col class="position-weight-col">
          <col class="position-score-col">
          <col class="position-price-col">
          <col class="position-price-col">
          <col class="position-return-col">
          <col class="position-return-col">
          <col class="position-status-col">
        </colgroup>
        <thead>
          <tr>
            <th>转债代码</th>
            <th class="text-end">排名</th>
            <th class="text-end">权重</th>
            <th class="text-end">分数</th>
            <th class="text-end">买入TWAP</th>
            <th class="text-end">${escapeHtml(`卖出TWAP(${formatTwapCol(res.data.sell_col || selectedSellCol || "")})`)}</th>
            <th class="text-end">单券收益</th>
            <th class="text-end">贡献</th>
            <th class="text-end">状态</th>
          </tr>
        </thead>
        <tbody>
          ${rows
            .map((row) => {
              const w = row.weight == null ? "-" : Number(row.weight).toFixed(4);
              const retClass = returnClass(row.return_net);
              const contribClass = returnClass(row.weighted_return);
              const status = String(row.status || "unknown");
              return `
                <tr class="position-row-${escapeHtml(status)}" title="${escapeHtml(row.reason || "")}">
                  <td><span class="bond-code">${escapeHtml(row.symbol || "")}</span></td>
                  <td class="text-end">${escapeHtml(row.rank ?? "-")}</td>
                  <td class="text-end"><span class="weight-value">${escapeHtml(w)}</span></td>
                  <td class="text-end">${escapeHtml(fmtFixed(row.score, 6))}</td>
                  <td class="text-end">${escapeHtml(fmtFixed(row.buy_twap, 3))}</td>
                  <td class="text-end">${escapeHtml(fmtFixed(row.sell_twap_next, 3))}</td>
                  <td class="text-end"><span class="${retClass}">${escapeHtml(fmtPct(row.return_net))}</span></td>
                  <td class="text-end"><span class="${contribClass}">${escapeHtml(fmtPct(row.weighted_return))}</span></td>
                  <td class="text-end"><span class="position-status-pill status-${escapeHtml(status)}">${escapeHtml(holdingsStatusLabel(row))}</span></td>
                </tr>
              `;
            })
            .join("")}
        </tbody>
      </table>
    </div>
  `;
  renderReturnOverview(rows, res.data || {});
  renderReturnRanking(rows, res.data || {});
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
  if (!perfLookbackInput || !perfMeta || !perfMetricsCanvas || !perfNavCanvas) return;
  try {
    const selectedDay = logDaySelect && logDaySelect.value ? logDaySelect.value : "";
    const selectedSellCol = getSelectedSellTwapCol();
    let lookback = Number.parseInt(perfLookbackInput.value || "20", 10);
    if (!Number.isFinite(lookback) || lookback <= 0) lookback = 20;
    perfLookbackInput.value = String(lookback);
    const res = await axios.get("/api/perf_summary", {
      params: {
        ...(selectedDay ? { day: selectedDay } : {}),
        ...(selectedSellCol ? { sell_col: selectedSellCol } : {}),
        lookback,
      },
    });
    const payload = res.data || {};
    const series = payload.series || [];
    if (!series.length) {
      destroyCharts();
      perfMeta.textContent = "暂无绩效数据";
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
    perfMeta.textContent = `截至 ${payload.asof_day || "-"} · 样本 ${payload.count_days || 0} · 回看 ${payload.lookback || lookback} · 卖出列 ${formatTwapCol(payload.sell_col || selectedSellCol || "-")}`;

    if (perfMetricsChart) perfMetricsChart.destroy();
    perfMetricsChart = new Chart(perfMetricsCanvas.getContext("2d"), {
      type: "bar",
      data: {
        labels: ["夏普", "年化波动"],
        datasets: [
          { label: "策略", data: [sharpe, vol], backgroundColor: "#2563eb" },
          { label: "基准", data: [benchSharpe, benchVol], backgroundColor: "#16a34a" },
        ],
      },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: true, position: "bottom" } } },
    });

    if (perfNavChart) perfNavChart.destroy();
    perfNavChart = new Chart(perfNavCanvas.getContext("2d"), {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "策略净值",
            data: nav,
            borderColor: "#2563eb",
            backgroundColor: "rgba(37,99,235,0.12)",
            tension: 0.25,
            pointRadius: 2,
            pointHoverRadius: 5,
            borderWidth: 2,
          },
          {
            label: "基准净值",
            data: benchNav,
            borderColor: "#16a34a",
            backgroundColor: "rgba(22,163,74,0.10)",
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
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { display: true, position: "bottom" } },
        scales: { x: { ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 8 } } },
      },
    });
  } catch (err) {
    destroyCharts();
    perfMeta.textContent = "绩效数据加载失败";
  }
}

async function refreshDataCalendar() {
  if (!dataCalendar) return;
  const selectedDay = calendarSelectedDay || (logDaySelect && logDaySelect.value ? normalizeDay(logDaySelect.value) : "");
  const res = await axios.get("/api/data_calendar", { params: { months: 4 } });
  const payload = res.data || {};
  const months = (payload.months || [])
    .slice()
    .sort((a, b) => String(b?.month || b?.title || "").localeCompare(String(a?.month || a?.title || "")));
  if (calendarAnchor) {
    const expected = Number(payload.expected_factor_count || 0);
    const panel = payload.label ? ` ${payload.label}` : "";
    calendarAnchor.textContent = payload.anchor_day ? `截至 ${payload.anchor_day}${panel} · 因子 ${expected}` : "";
  }
  if (!months.length) {
    dataCalendar.innerHTML = "<div class='empty-mini'>暂无日历数据</div>";
    return;
  }
  const weekday = ["一", "二", "三", "四", "五", "六", "日"];
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
              return `<button type="button" class="${cls}" data-day="${escapeHtml(cell.day)}" title="${title}">${cell.day_num}</button>`;
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
    let msg = "请求失败";
    if (err?.response) {
      const status = err.response.status;
      const body = err.response.data;
      if (typeof body === "string" && body.trim()) msg = `HTTP ${status}: ${body}`;
      else if (body && typeof body === "object" && body.error) msg = `HTTP ${status}: ${body.error}`;
      else msg = `HTTP ${status}`;
    } else if (err?.message) {
      msg = err.message;
    }
    throw new Error(msg);
  }
}

function confirmMaintenance(message) {
  return window.confirm(message);
}

function confirmEmergencyStop() {
  const text = window.prompt("确认紧急停止。请输入 STOP 来停止调度器并写入 STOP 标记。");
  return text === "STOP";
}

function isSecretKey(key) {
  return /(password|passwd|secret|token|credential)/i.test(String(key || ""));
}

async function refreshAfterAction() {
  await refreshLiveStatus();
  await refreshLogsSafe();
  await refreshHoldings();
  await refreshPerformance();
}

async function loadConfig() {
  const res = await axios.get("/api/config");
  const cfg = res.data.config || {};
  const meta = res.data.meta || { read_only: [], types: {}, write_enabled: false };
  const writeEnabled = Boolean(meta.write_enabled);
  const list = el("config-list");
  if (configMode) {
    configMode.textContent = writeEnabled ? "本地调试可编辑" : "生产只读模式";
  }
  const saveBtn = el("btn-save-config");
  if (saveBtn) {
    saveBtn.disabled = !writeEnabled;
    saveBtn.className = writeEnabled ? "btn btn-sm btn-primary" : "btn btn-sm btn-outline-secondary";
  }
  const defaultSellCol = String(cfg?.output?.sell_twap_col || cfg?.data?.sell_twap_col || "").trim();
  initSellTwapSelector(defaultSellCol);
  if (!list) return;

  const rows = [];
  function pushRows(path, value) {
    if (value !== null && typeof value === "object" && !Array.isArray(value)) {
      Object.keys(value)
        .sort()
        .forEach((key) => pushRows(`${path}.${key}`, value[key]));
      return;
    }
    rows.push({ key: path, value });
  }
  Object.keys(cfg)
    .sort()
    .forEach((key) => pushRows(key, cfg[key]));

  list.innerHTML = rows
    .map(({ key, value }) => {
      const isBool = typeof value === "boolean";
      const display = Array.isArray(value) ? JSON.stringify(value) : value == null ? "" : String(value);
      const secret = isSecretKey(key);
      const disabled = !writeEnabled || secret;
      const input = isBool
        ? `<input type="checkbox" class="form-check-input" data-key="${escapeHtml(key)}" data-type="bool" ${value ? "checked" : ""} ${disabled ? "disabled" : ""}>`
        : `<input type="text" class="form-control form-control-sm" data-key="${escapeHtml(key)}" data-type="str" value="${escapeHtml(secret ? "********" : display)}" ${disabled ? "disabled" : ""}>`;
      return `
        <div class="config-item ${disabled ? "readonly" : ""}">
          <div class="label">${escapeHtml(key)}</div>
          ${input}
        </div>
      `;
    })
    .join("");
}

async function saveConfig() {
  const actionSpec = latestLiveStatus?.actions?.save_config || {};
  if (actionSpec.enabled === false) {
    syncStatus.textContent = zhReason(actionSpec.reason || "配置写入已禁用");
    return;
  }
  if (!confirmMaintenance("确认保存配置变更？该操作只应在本地调试模式使用。")) return;
  const list = el("config-list");
  const inputs = list ? list.querySelectorAll("input[data-key]") : [];
  const payload = {};

  function setByPath(obj, dottedKey, value) {
    const parts = dottedKey.split(".");
    let cur = obj;
    for (let i = 0; i < parts.length - 1; i += 1) {
      const p = parts[i];
      if (!cur[p] || typeof cur[p] !== "object" || Array.isArray(cur[p])) cur[p] = {};
      cur = cur[p];
    }
    cur[parts[parts.length - 1]] = value;
  }

  inputs.forEach((node) => {
    if (node.disabled) return;
    const key = node.getAttribute("data-key");
    const type = node.getAttribute("data-type");
    const value = type === "bool" ? node.checked : node.value;
    setByPath(payload, key, value);
  });
  await axios.post("/api/config", payload);
  syncStatus.textContent = "配置已保存";
  await refreshLiveStatus();
}

function bindActions() {
  el("btn-open")?.addEventListener("click", async () => {
    try {
      await callAction("/api/start_open");
      syncStatus.textContent = "已触发开盘链路";
      await refreshAfterAction();
    } catch (err) {
      syncStatus.textContent = `启动开盘链路失败：${err.message || err}`;
    }
  });

  el("btn-restart")?.addEventListener("click", async () => {
    if (!confirmMaintenance("确认重启调度器？当前调度进程会被停止并重新启动。")) return;
    try {
      await callAction("/api/restart_scheduler");
      syncStatus.textContent = "调度器已重启";
      await refreshAfterAction();
    } catch (err) {
      syncStatus.textContent = `重启失败：${err.message || err}`;
    }
  });

  el("btn-stop")?.addEventListener("click", async () => {
    if (!confirmEmergencyStop()) return;
    try {
      await callAction("/api/emergency_stop");
      syncStatus.textContent = "已触发紧急停止";
      await refreshAfterAction();
    } catch (err) {
      syncStatus.textContent = `紧急停止失败：${err.message || err}`;
    }
  });

  el("btn-sync")?.addEventListener("click", async () => {
    try {
      const res = await callAction("/api/sync_holdings");
      syncStatus.textContent = res.ok ? `已同步持仓：${res.count}` : `同步失败：${res.error || ""}`;
      await refreshAfterAction();
    } catch (err) {
      syncStatus.textContent = `同步失败：${err.message || err}`;
    }
  });

  el("btn-shutdown")?.addEventListener("click", async () => {
    if (!confirmMaintenance("确认关闭控制台服务？该操作不会停止调度器。")) return;
    try {
      await callAction("/api/shutdown");
      setTimeout(() => window.close(), 300);
    } catch (err) {
      syncStatus.textContent = `关闭控制台服务失败：${err.message || err}`;
    }
  });

  el("btn-save-config")?.addEventListener("click", async () => {
    try {
      await saveConfig();
    } catch (err) {
      syncStatus.textContent = `保存配置失败：${err.message || err}`;
    }
  });
}

if (logDaySelect) {
  logDaySelect.addEventListener("change", () => {
    calendarSelectedDay = normalizeDay(logDaySelect.value);
    markTradeFilterDirty();
  });
}

if (sellTwapSelect) {
  sellTwapSelect.addEventListener("change", () => {
    const selected = getSelectedSellTwapCol();
    if (selected) {
      window.localStorage.setItem(SELL_TWAP_STORAGE_KEY, selected);
    } else {
      window.localStorage.removeItem(SELL_TWAP_STORAGE_KEY);
    }
    markTradeFilterDirty();
  });
}

if (tradeRefreshBtn) {
  tradeRefreshBtn.addEventListener("click", async () => {
    tradeRefreshBtn.disabled = true;
    const prevText = tradeRefreshBtn.textContent;
    tradeRefreshBtn.textContent = "刷新中...";
    const unlockTimer = setTimeout(() => {
      tradeRefreshBtn.disabled = false;
      tradeRefreshBtn.textContent = prevText || "刷新";
    }, 15000);
    try {
      const ok = await refreshTradeWindowOnly();
      if (syncStatus) {
        syncStatus.textContent = ok ? "持仓与绩效已刷新。" : "部分模块刷新失败，请重试。";
      }
    } catch (err) {
      if (syncStatus) {
        syncStatus.textContent = `刷新失败：${err?.message || err}`;
      }
    } finally {
      clearTimeout(unlockTimer);
      tradeRefreshBtn.disabled = false;
      tradeRefreshBtn.textContent = prevText || "刷新";
    }
  });
}

el("btn-perf-refresh")?.addEventListener("click", refreshPerformance);
perfLookbackInput?.addEventListener("change", refreshPerformance);

async function bootstrapDashboard() {
  bindActions();
  await loadLogDays();
  await loadConfig();
  await refreshLiveStatus();
  await refreshLogsSafe();
  await refreshHoldings();
  await refreshPerformance();
  await refreshDataCalendar();
}

bootstrapDashboard();
setInterval(refreshLiveStatus, 5000);
setInterval(refreshLogsSafe, 10000);
setInterval(loadLogDays, 30000);
setInterval(refreshDataCalendar, 60000);
