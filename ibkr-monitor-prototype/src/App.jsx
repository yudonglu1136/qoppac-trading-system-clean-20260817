import { useEffect, useMemo, useState } from "react";

const LONG = "#dc2626";
const SHORT = "#12823a";
const BLUE = "#1f5eff";
const INK = "#121722";
const NAV_RANGES = ["1D", "1W", "1M", "3M", "YTD", "ALL"];
const INSTRUMENT_RANGES = ["1D", "1W", "1M", "3M", "YTD", "1Y"];
const RANGE_DAYS = {
  "1D": 1,
  "1W": 7,
  "1M": 31,
  "3M": 93,
  "1Y": 366,
};

function number(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function hasNumber(value) {
  if (value === null || value === undefined || value === "") return false;
  const parsed = Number(value);
  return Number.isFinite(parsed);
}

function money(value, digits = 0) {
  const amount = number(value);
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(amount);
}

function moneyOrDash(value, digits = 0) {
  return hasNumber(value) ? money(value, digits) : "-";
}

function compact(value, digits = 0) {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(number(value));
}

function compactOrDash(value, digits = 0) {
  return hasNumber(value) ? compact(value, digits) : "-";
}

function pct(value, digits = 2) {
  return `${(number(value) * 100).toFixed(digits)}%`;
}

function pctOrDash(value, digits = 2) {
  return hasNumber(value) ? pct(value, digits) : "-";
}

function niceTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("en-US", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function directionClass(direction) {
  if (direction === "LONG") return "long";
  if (direction === "SHORT") return "short";
  return "flat";
}

function isLocked(row) {
  return Boolean(row?.positionLockedByGuardrail || row?.positionBreak);
}

function isExitOnly(row) {
  return Boolean(row?.positionExitOnly || row?.instrumentExitOnly || row?.riskPolicy === "EXIT_ONLY");
}

function rowStatusTitle(row) {
  const parts = [];
  if (isLocked(row)) parts.push("Position locked: broker and local position reconciliation required");
  if (isExitOnly(row)) parts.push("Exit only: may reduce or close; cannot open, add, or reverse");
  if (row?.instrumentDataStatus) parts.push(`Data status: ${row.instrumentDataStatus}`);
  if (row?.instrumentLatestAdjustedDate) parts.push(`Latest adjusted: ${row.instrumentLatestAdjustedDate}`);
  if (row?.forecastDate) parts.push(`Forecast date: ${row.forecastDate}`);
  if (hasNumber(row?.ignoredForecast)) parts.push(`Ignored forecast: ${compact(row.ignoredForecast, 2)}`);
  if (row?.forecastIgnoredReason) parts.push(`Forecast ignored: ${row.forecastIgnoredReason}`);
  if (row?.instrumentDataReasons?.length) parts.push(`Reasons: ${row.instrumentDataReasons.join(", ")}`);
  return parts.join("\n");
}

function forecastNote(row) {
  if (isExitOnly(row)) return "exit only";
  if (row?.forecastIgnored) return "ignored";
  if (isLocked(row)) return "locked";
  if (row?.instrumentDataStatus && row.instrumentDataStatus !== "PASS") return String(row.instrumentDataStatus).toLowerCase();
  return "";
}

function pathFrom(points) {
  return points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(" ");
}

function parseChartDate(row) {
  const raw = row?.timestamp || row?.date || row?.barTime || "";
  if (!raw) return null;
  const parsed = new Date(String(raw).replace(" ", "T"));
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function filterByRange(rows, range) {
  if (!rows?.length || range === "ALL") return rows || [];
  const datedRows = rows.map((row) => ({ row, date: parseChartDate(row) })).filter((item) => item.date);
  if (datedRows.length < 2) return rows;
  const lastDate = datedRows[datedRows.length - 1].date;
  let cutoff;
  if (range === "YTD") {
    cutoff = new Date(lastDate.getFullYear(), 0, 1);
  } else {
    cutoff = new Date(lastDate.getTime() - (RANGE_DAYS[range] || 0) * 24 * 60 * 60 * 1000);
  }
  const filtered = datedRows.filter((item) => item.date >= cutoff).map((item) => item.row);
  return filtered.length >= 2 ? filtered : rows.slice(-Math.min(rows.length, 2));
}

function sortByChartDate(rows) {
  return [...(rows || [])].sort((left, right) => {
    const leftDate = parseChartDate(left);
    const rightDate = parseChartDate(right);
    if (!leftDate && !rightDate) return 0;
    if (!leftDate) return 1;
    if (!rightDate) return -1;
    return leftDate.getTime() - rightDate.getTime();
  });
}

function scalePoints(rows, valueKey, width, height, padding = 28) {
  const values = rows.map((row) => number(row[valueKey], NaN)).filter(Number.isFinite);
  if (values.length === 0) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || Math.max(Math.abs(max), 1);
  return rows.map((row, index) => {
    const x = padding + (index / Math.max(rows.length - 1, 1)) * (width - padding * 2);
    const y = height - padding - ((number(row[valueKey]) - min) / span) * (height - padding * 2);
    return { ...row, x, y };
  });
}

function scaleTimePoints(rows, valueKey, width, height, padding = 28) {
  const datedRows = rows
    .map((row) => ({ row, date: parseChartDate(row) }))
    .filter((item) => item.date && Number.isFinite(number(item.row[valueKey], NaN)));
  if (datedRows.length !== rows.length || datedRows.length < 2) {
    return scalePoints(rows, valueKey, width, height, padding);
  }
  const values = datedRows.map((item) => number(item.row[valueKey]));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const valueSpan = max - min || Math.max(Math.abs(max), 1);
  const firstTime = datedRows[0].date.getTime();
  const lastTime = datedRows[datedRows.length - 1].date.getTime();
  const timeSpan = lastTime - firstTime || 1;
  return datedRows.map(({ row, date }) => ({
    ...row,
    x: padding + ((date.getTime() - firstTime) / timeSpan) * (width - padding * 2),
    y: height - padding - ((number(row[valueKey]) - min) / valueSpan) * (height - padding * 2),
  }));
}

function MiniNavChart({ rows }) {
  const width = 420;
  const height = 96;
  const cleanRows = rows.length > 1 ? rows : [{ nav: number(rows[0]?.nav) - 1, timestamp: "" }, ...(rows || [])];
  const points = scalePoints(cleanRows, "nav", width, height, 10);
  return (
    <svg className="mini-nav" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="NAV equity curve">
      <path className="grid-line" d={`M 0 ${height - 18} H ${width}`} />
      <path className="grid-line" d={`M 0 ${height / 2} H ${width}`} />
      <path className="nav-line" d={pathFrom(points)} />
    </svg>
  );
}

function RangeButtons({ options, selected, onSelect }) {
  return options.map((option) => (
    <button
      key={option}
      type="button"
      className={option === selected ? "selected" : ""}
      aria-pressed={option === selected}
      onClick={() => onSelect(option)}
    >
      {option}
    </button>
  ));
}

function MainNavChart({ rows }) {
  const [range, setRange] = useState("YTD");
  const width = 1040;
  const height = 250;
  const rangeRows = filterByRange(rows || [], range);
  const cleanRows = rangeRows.length > 1 ? rangeRows : [{ nav: number(rangeRows[0]?.nav) - 1, timestamp: "" }, ...(rangeRows || [])];
  const points = scalePoints(cleanRows, "nav", width, height, 34);
  const values = cleanRows.map((row) => number(row.nav));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const labels = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
    const index = Math.min(Math.round(ratio * Math.max(cleanRows.length - 1, 0)), cleanRows.length - 1);
    return niceTime(cleanRows[index]?.timestamp);
  });
  return (
    <div className="chart-frame">
      <div className="chart-title-row">
        <div className="panel-title">NAV / Equity Curve</div>
        <div className="legend-row">
          <span><i className="line-blue" />NAV</span>
          <span><i className="line-dash" />High Water Mark</span>
        </div>
        <div className="range-tabs">
          <RangeButtons options={NAV_RANGES} selected={range} onSelect={setRange} />
        </div>
      </div>
      <svg className="main-nav" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="NAV history">
        {[0.2, 0.4, 0.6, 0.8].map((tick) => (
          <path key={tick} className="grid-line" d={`M 42 ${(height * tick).toFixed(2)} H ${width - 18}`} />
        ))}
        <path className="nav-line" d={pathFrom(points)} />
        <path className="watermark-line" d={`M 42 ${Math.min(...points.map((p) => p.y)).toFixed(2)} H ${width - 18}`} />
        <text className="axis-label" x="8" y="44">{money(max, 0)}</text>
        <text className="axis-label" x="8" y={height - 18}>{money(min, 0)}</text>
      </svg>
      <div className="month-strip">
        {labels.map((label, index) => <span key={`${label}-${index}`}>{label}</span>)}
      </div>
      <div className="performance-strip">
        <b>Performance (net)</b><span>Today</span><strong>Live snapshots only</strong><span>State saved every 1h</span><strong>Trading check 15m</strong>
      </div>
    </div>
  );
}

function InstrumentChart({ instrument, series, position, orders }) {
  const [range, setRange] = useState("3M");
  const width = 690;
  const height = 252;
  const [hoverIndex, setHoverIndex] = useState(null);
  const cleanRows = filterByRange(sortByChartDate(series), range).filter((row) => Number.isFinite(number(row.price, NaN)));
  const pricePoints = scaleTimePoints(cleanRows, "price", width, height, 30);
  const forecastRows = cleanRows.map((row) => ({ ...row, forecastScaled: number(row.forecast, 0) }));
  const forecastPoints = scaleTimePoints(forecastRows, "forecastScaled", width, height, 30);
  const hovered = hoverIndex === null ? null : pricePoints[Math.min(hoverIndex, Math.max(pricePoints.length - 1, 0))];
  const selectedOrders = orders.filter((order) => order.instrument === instrument).slice(0, 5);

  useEffect(() => {
    setHoverIndex(null);
  }, [instrument, range]);

  function updateHover(event) {
    if (pricePoints.length === 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = (event.clientX - rect.left) / Math.max(rect.width, 1);
    const index = Math.round(ratio * Math.max(pricePoints.length - 1, 0));
    const nextIndex = Math.max(0, Math.min(index, pricePoints.length - 1));
    setHoverIndex((current) => (current === nextIndex ? current : nextIndex));
  }

  return (
    <div className="instrument-chart">
      <div className="tabs chart-tabs">
        <RangeButtons options={INSTRUMENT_RANGES} selected={range} onSelect={setRange} />
      </div>
      <svg className="detail-chart" viewBox={`0 0 ${width} ${height}`} onMouseMove={updateHover} onMouseLeave={() => setHoverIndex(null)}>
        {[0.2, 0.4, 0.6, 0.8].map((tick) => (
          <path key={tick} className="grid-line" d={`M 34 ${(height * tick).toFixed(2)} H ${width - 18}`} />
        ))}
        <path className="price-line" d={pathFrom(pricePoints)} />
        <path className="forecast-line" d={pathFrom(forecastPoints)} />
        {pricePoints.map((point, index) => {
          if (!point.action) return null;
          const color = point.action === "LONG" ? LONG : SHORT;
          const shape = point.action === "LONG"
            ? `${point.x},${point.y - 8} ${point.x - 5},${point.y + 4} ${point.x + 5},${point.y + 4}`
            : `${point.x},${point.y + 8} ${point.x - 5},${point.y - 4} ${point.x + 5},${point.y - 4}`;
          return <polygon key={`${point.date}-${index}`} points={shape} fill={color} stroke="white" strokeWidth="1.5" />;
        })}
        {hovered && (
          <>
            <path className="hover-line" d={`M ${hovered.x} 22 V ${height - 24}`} />
            <circle cx={hovered.x} cy={hovered.y} r="5" fill="#fff" stroke={BLUE} strokeWidth="2.5" />
          </>
        )}
      </svg>
      {hovered && (
        <div className="chart-tooltip">
          <b>{hovered.date}</b>
          <span>Instrument <strong>{instrument}</strong></span>
          <span>Price <strong>{compact(hovered.price, 4)}</strong></span>
          <span>Forecast <strong>{compactOrDash(hovered.forecast, 2)}</strong></span>
      <span>Position <strong>{compact(position?.position || 0)} contracts</strong></span>
          <span>Target <strong>{compact(position?.target || 0)} contracts</strong></span>
          <span>Execution <strong className={hovered.action === "SHORT" ? "short-text" : "long-text"}>{hovered.action === "LONG" ? "Buy" : hovered.action === "SHORT" ? "Sell" : "None"}</strong></span>
          <span>NAV <strong>{money(position?.nav || 0)}</strong></span>
          <span>Unrealized P&L <strong>{moneyOrDash(position?.unrealizedPnl)}</strong></span>
          <span>Source <strong>IBKR / local overlay</strong></span>
        </div>
      )}
      <div className="chart-footer">
        <span><i className="price-key" />Price</span>
        <span><i className="forecast-key" />Forecast</span>
        <span><i className="long-dot" />Executed Buy</span>
        <span><i className="short-dot" />Executed Sell</span>
      </div>
      {selectedOrders.length > 0 && (
        <div className="chart-order-note">
          Recorded order history: {selectedOrders.map((order) => `${order.action} ${order.quantity} (${order.status})`).join(" | ")}
        </div>
      )}
    </div>
  );
}

function Kpi({ label, value, sub, tone }) {
  return (
    <div className="kpi">
      <span>{label}</span>
      <strong className={tone || ""}>{value}</strong>
      <em className={tone || ""}>{sub}</em>
    </div>
  );
}

function StatusDot({ label, value }) {
  const text = String(value);
  const ratio = text.match(/^(\d+)\/(\d+)$/);
  const badRatio = ratio ? Number(ratio[1]) < Number(ratio[2]) : false;
  const bad = badRatio || /disconnected|stopped|error|fail|false|blocked|missing|stale|invalid/i.test(text);
  return (
    <div className="status-line">
      <span>{label}</span>
      <strong className={bad ? "status-bad" : ""}><i />{value}</strong>
    </div>
  );
}

function PositionsTable({ rows, selected, onSelect, summary }) {
  function positionText(row) {
    const position = hasNumber(row.position) ? number(row.position) : null;
    const target = hasNumber(row.target) ? number(row.target) : null;
    if (position === null && target === null) return "-";
    if (target === null) return compact(position, 0);
    if (position === null) return `- → ${compact(target, 0)}`;
    return `${compact(position, 0)} → ${compact(target, 0)}`;
  }

  const unrealizedTotal = hasNumber(summary?.unrealizedPnlEstimate)
    ? number(summary.unrealizedPnlEstimate)
    : rows.reduce((sum, row) => sum + number(row.unrealizedPnl), 0);
  const missingPnl = number(summary?.unrealizedPnlMissingCount, 0);

  return (
    <div className="positions-panel">
      <div className="panel-heading">
        <h2>Positions</h2>
        <span>{rows.filter((row) => number(row.position) !== 0).length} held / {rows.length} tracked</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>Instrument</th>
            <th>Position / Target</th>
            <th>Direction</th>
            <th>Market Price</th>
            <th>Forecast</th>
            <th>Unrealized P&L</th>
            <th>% NAV</th>
            <th>Avg Entry</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.instrument} className={row.instrument === selected ? "active-row" : ""} title={rowStatusTitle(row)} onClick={() => onSelect(row.instrument)}>
              <td><b>{row.instrument}</b><small>{row.localSymbol}</small></td>
              <td className={hasNumber(row.target) && number(row.position) !== number(row.target) ? "target-diff" : ""}>{positionText(row)}</td>
              <td>
                <span className={directionClass(row.direction)}>{row.direction}</span>
                {isLocked(row) && <small className="lock-note">Locked</small>}
                {isExitOnly(row) && <small className="exit-only-note">Exit only</small>}
              </td>
              <td>{compactOrDash(row.lastPrice, 4)}</td>
              <td>
                {compactOrDash(row.forecast, 2)}
                {forecastNote(row) && <small className="forecast-note">{forecastNote(row)}</small>}
              </td>
              <td className={hasNumber(row.unrealizedPnl) ? (number(row.unrealizedPnl) >= 0 ? "positive" : "negative") : "muted"}>{moneyOrDash(row.unrealizedPnl, 0)}</td>
              <td>{pctOrDash(row.unrealizedPctNav)}</td>
              <td>{compactOrDash(row.avgEntry, 4)}</td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr>
            <td>Total</td><td colSpan="4" />
            <td className={unrealizedTotal >= 0 ? "positive" : "negative"}>{money(unrealizedTotal, 0)}</td>
            <td colSpan="2">{missingPnl ? `Partial IB P&L (${compact(missingPnl, 0)} missing)` : "P&L from IB portfolio snapshot"}</td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}

function OrdersPanel({ orders }) {
  return (
    <div className="orders-panel">
      <div className="panel-heading">
        <h2>Orders & Execution</h2>
        <span>historical attempts</span>
      </div>
      <div className="order-list">
        {orders.slice(0, 5).map((order, index) => (
          <div className="order-row" key={`${order.run_id}-${order.instrument}-${index}`}>
            <span>{niceTime(order.timestamp_utc)}</span>
            <b>{order.instrument}</b>
            <strong className={order.action === "BUY" ? "long-text" : "short-text"}>{order.action}</strong>
            <span>{compact(order.quantity, 0)}</span>
            <em>{order.status}</em>
          </div>
        ))}
      </div>
    </div>
  );
}

export function App() {
  const [data, setData] = useState(null);
  const [selected, setSelected] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    async function load() {
      try {
        const response = await fetch("/api/monitor", { cache: "no-store" }).catch(() => fetch("/monitor-data.json", { cache: "no-store" }));
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        if (!alive) return;
        setData(payload);
        setError("");
        setSelected((current) => current || payload.selectedInstrument || payload.positions?.[0]?.instrument || "");
      } catch (err) {
        if (alive) setError(err.message || String(err));
      }
    }
    load();
    const timer = window.setInterval(load, 30000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  const selectedPosition = useMemo(() => {
    if (!data) return null;
    const row = data.positions.find((position) => position.instrument === selected) || data.positions[0];
    return row ? { ...row, nav: data.summary.nav } : null;
  }, [data, selected]);

  if (error && !data) {
    return <main className="app-shell"><div className="error-state">Monitor data failed to load: {error}</div></main>;
  }

  if (!data) {
    return <main className="app-shell"><div className="loading-state">Loading IBKR monitor...</div></main>;
  }

  const summary = data.summary;
  const status = data.status;
  const series = data.series[selectedPosition?.instrument] || [];
  const selectedPositionText = selectedPosition && number(selectedPosition.position) !== number(selectedPosition.target)
    ? `${compact(selectedPosition.position, 0)} → ${compact(selectedPosition.target, 0)}`
    : compact(selectedPosition?.position, 0);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Carver Systematic Futures Monitor</h1>
          <p>Strategy: {data.strategy} <span /> Account: {data.account} <span /> {data.mode}</p>
        </div>
        <div className="top-actions">
          <time>{niceTime(data.generatedAt)}</time>
          <button type="button" onClick={() => window.location.reload()}>Refresh Data</button>
        </div>
      </header>

      <section className="kpi-strip">
        <Kpi label="Net Asset Value (NAV)" value={money(summary.nav, 0)} sub={`${money(summary.dailyPnl, 0)} today`} tone={number(summary.dailyPnl) >= 0 ? "positive" : "negative"} />
        <Kpi label="Daily P&L" value={money(summary.dailyPnl, 0)} sub={pct(summary.dailyPnlPct)} tone={number(summary.dailyPnl) >= 0 ? "positive" : "negative"} />
        <Kpi label="YTD P&L" value={money(summary.ytdPnl, 0)} sub={pct(summary.ytdPnlPct)} tone={number(summary.ytdPnl) >= 0 ? "positive" : "negative"} />
        <Kpi label="Projected Margin" value={money(summary.projectedMargin, 0)} sub={pct(summary.projectedMarginPct)} />
        <Kpi label="Available Funds" value={money(summary.availableFunds, 0)} sub={`Excess ${money(summary.excessLiquidity, 0)}`} />
        <Kpi label="Buying Power" value={money(summary.buyingPower, 0)} sub="paper account" />
      </section>

      <section className="main-grid">
        <div className="nav-panel">
          <MainNavChart rows={data.navSeries} />
        </div>
        <aside className="side-panel">
          <section>
            <h2>System Status</h2>
            <StatusDot label="Trading Gate" value={status.tradingGate} />
            <StatusDot label="IBKR Connection" value={status.ibkrConnection} />
            <StatusDot label="Account Status" value={status.accountStatus} />
            <StatusDot label="Gate Evaluator" value={status.daemonPhase} />
            <StatusDot label="Core Services" value={`${status.launchAgentsLoaded}/${status.launchAgentsExpected}`} />
            <StatusDot label="Auto Recovery" value={`${status.autoRecoveryConfigured}/${status.launchAgentsExpected}`} />
          </section>
          <section>
            <h2>Data Freshness</h2>
            <StatusDot label="Business Date" value={status.businessDate || "Missing"} />
            <StatusDot label="Gate Freshness" value={status.gateFreshness || "Missing"} />
            <StatusDot label="Native Data Gate" value={status.nativeDataGate || "BLOCKED"} />
            <StatusDot label="Gate Snapshot" value={niceTime(status.daemonHeartbeatUtc)} />
            <StatusDot label="Live 15m Price" value={niceTime(status.latestMonitorPriceUtc || status.marketDataLatestBarUtc)} />
            <StatusDot label="Price Overlay" value={status.monitorPriceOverlaySource || "Missing"} />
            <StatusDot label="Rob Daily Price" value={niceTime(status.latestNativePriceUtc)} />
            <StatusDot label="Frontend Export" value={niceTime(status.monitorExportUtc)} />
            <StatusDot label="15m Collector" value={status.marketDataCollector || "Missing"} />
            <StatusDot label="Collector Run" value={niceTime(status.marketDataLastRunUtc)} />
            <StatusDot label="Latest 15m Bar" value={niceTime(status.marketDataLatestBarUtc)} />
            <StatusDot label="Archived Instruments" value={status.marketDataStored || "0/38"} />
            <StatusDot label="Archived Bars" value={compact(status.marketDataBars, 0)} />
            <StatusDot label="Full Snapshot" value={niceTime(status.lastSnapshotUtc)} />
            <StatusDot label="Snapshot Freshness" value={status.fullSnapshotFreshness || "Stale"} />
            <StatusDot label="Account Values" value={niceTime(status.latestAccountValuesUtc)} />
            <StatusDot label="Account Freshness" value={status.accountValuesFreshness || "Stale"} />
            <StatusDot label="Production Files" value={status.productionFiles} />
            <StatusDot label="Tradable Current" value={status.productionTradable} />
            <StatusDot label="Research Files" value={status.researchFiles} />
            <StatusDot label="Research Current" value={status.researchCurrent} />
            <StatusDot label="Frozen" value={compact(status.frozenInstruments, 0)} />
            <StatusDot label="Contract Gap Fails" value={compact(status.continuousGapFailures, 0)} />
          </section>
          <section>
            <h2>Target Integrity</h2>
            <StatusDot label="SHA256" value={status.targetSha256Verified ? `Verified ${String(status.targetSha256).slice(0, 10)}...` : "Failed / Missing"} />
            <StatusDot label="Target Date" value={status.targetBusinessDateVerified ? status.targetBusinessDate : "Failed / Missing"} />
            <StatusDot label="Target Universe" value={`${status.targetUniverseCount || 0}/38`} />
            <StatusDot label="Target Age" value={status.targetAgeHours == null ? "Missing" : `${compact(status.targetAgeHours, 1)} h`} />
            <StatusDot label="Intent Runs / Rows" value={`${status.intentRunCount}/${status.intentCount}`} />
            <StatusDot label="Last Intent" value={niceTime(status.lastIntentUtc)} />
          </section>
          <section>
            <h2>Orders & Execution</h2>
            <StatusDot label={status.tradingGate === "PASS" ? "Remaining Actionable" : "Blocked Target Deltas"} value={compact(status.remainingActionableOrders, 0)} />
            <StatusDot label="Open / held attempts" value={compact(status.openOrders, 0)} />
            <StatusDot label="Filled contracts" value={compact(status.filledToday, 0)} />
            <StatusDot label="Broker submissions" value={compact(status.brokerSubmissionCount, 0)} />
            <StatusDot
              label="Unresolved sends"
              value={status.brokerSubmissionUnresolved ? `${compact(status.brokerSubmissionUnresolved, 0)} BLOCKED` : "0"}
            />
          </section>
          <section>
            <h2>Risk Summary</h2>
            <StatusDot label="Margin Used" value={`${money(summary.marginUsed, 0)} (${pct(summary.marginUsedPct)})`} />
            <StatusDot label="Projected Margin" value={money(summary.projectedMargin, 0)} />
            <StatusDot label="Trading Check" value={status.tradingCheckCadence} />
            <StatusDot label="State Snapshot" value={status.stateSnapshotCadence} />
          </section>
          {status.gateReasons?.length > 0 && (
            <section className="gate-reasons">
              <h2>Gate Reasons</h2>
              <ul>{status.gateReasons.map((reason) => <li key={reason}>{reason}</li>)}</ul>
            </section>
          )}
        </aside>
      </section>

      <section className="lower-grid">
        <div>
          <PositionsTable rows={data.positions} selected={selectedPosition?.instrument} onSelect={setSelected} summary={summary} />
          <OrdersPanel orders={data.orders} />
        </div>
        <div className="selected-panel">
          <div className="selected-heading">
            <div>
              <span>Selected Instrument</span>
              <h2>{selectedPosition?.instrument}</h2>
            </div>
            <strong>{compact(selectedPosition?.lastPrice, 4)}</strong>
            <em className={directionClass(selectedPosition?.direction)} title={rowStatusTitle(selectedPosition)}>
              {selectedPositionText} {selectedPosition?.direction}
              {isLocked(selectedPosition) && <small className="lock-note">Locked</small>}
              {isExitOnly(selectedPosition) && <small className="exit-only-note">Exit only</small>}
            </em>
            <div>
              <span>Avg Entry</span>
              <b>{compactOrDash(selectedPosition?.avgEntry, 4)}</b>
            </div>
            <div>
              <span>Unrealized P&L</span>
              <b className={hasNumber(selectedPosition?.unrealizedPnl) ? (number(selectedPosition?.unrealizedPnl) >= 0 ? "positive" : "negative") : "muted"}>{moneyOrDash(selectedPosition?.unrealizedPnl, 0)} ({pctOrDash(selectedPosition?.unrealizedPctNav)})</b>
            </div>
          </div>
          <InstrumentChart instrument={selectedPosition?.instrument} series={series} position={selectedPosition} orders={data.orders} />
        </div>
      </section>

      <footer>
        <span>Data source: native Rob Gate and local IBKR paper snapshots</span>
        <span>Generated: {niceTime(data.generatedAt)}</span>
        <span><b className="long-text">Long = red</b> / <b className="short-text">Short = green</b></span>
      </footer>
    </main>
  );
}
