"use strict";
const $ = (s) => document.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(+v)) ? "–" : (+v).toLocaleString("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d });
const rs = (v) => (v === null || v === undefined) ? "–" : (v < 0 ? "−₹" : "₹") + Math.abs(+v).toLocaleString("en-IN", { maximumFractionDigits: 0 });
const cls = (status, side) => status === "WAIT" ? "wait" : status && status.startsWith("BUY") ? (side === "PE" ? "pe" : "ce") : "none";
const label = (status, side) => status === "BUY" ? "BUY " + side : status === "WAIT" ? "WAIT FOR CONFIRMATION" : status;
const pn = (v) => `<span class="${v > 0 ? "up" : v < 0 ? "dn" : ""}">${rs(v)}</span>`;
let state = {}, tab = "dash";

async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : undefined);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
function table(el, heads, rows, empty) {
  el.innerHTML = "<thead><tr>" + heads.map((h) => `<th>${h}</th>`).join("") + "</tr></thead><tbody>" +
    (rows.length ? rows.map((r) => "<tr>" + r.map((c) => `<td>${c}</td>`).join("") + "</tr>").join("")
      : `<tr><td class="l muted" colspan="${heads.length}">${empty || "Nothing yet."}</td></tr>`) + "</tbody>";
}
function trend(t) {
  return '<div class="tf">' + Object.entries(t || {}).map(([k, v]) => `<span class="${v === 1 ? "u" : v === -1 ? "d" : ""}" title="${k} trend">${k} ${v === 1 ? "▲" : v === -1 ? "▼" : "–"}</span>`).join("") + "</div>";
}

function renderDash() {
  $("#cards").innerHTML = (state.indices || []).map((w) => `<div class="card ${cls(w.status, w.side)}">
    <div class="top"><span class="sym">${esc(w.symbol)}</span> <span class="badge ${cls(w.status, w.side)}">${esc(w.status === "WAIT" ? "WAIT" : w.status || "–")}</span></div>
    <div class="px">${num(w.ltp)}</div>
    <div class="${w.change_pct > 0 ? "up" : w.change_pct < 0 ? "dn" : "muted"}">${w.change_pct == null ? "–" : (w.change_pct > 0 ? "+" : "") + num(w.change_pct) + "%"}</div>
    <div class="muted" style="font-size:12px">expiry ${esc(w.expiry || "–")} · lot ${esc(w.lot_size ?? "–")} · strike step ${esc(w.strike_step ?? "–")}</div>
    ${trend(w.trend)}</div>`).join("") || '<p class="muted">Waiting for the first cycle…</p>';
  const r = state.risk || {};
  const tile = (v, l) => `<div class="tile"><b>${v}</b><span>${l}</span></div>`;
  $("#risk").innerHTML = tile(pn(r.total), `day P&L (limit ${rs(-(r.max_daily_loss || 0))})`) + tile(pn(r.realized), "realised today") +
    tile(`${r.trades ?? 0} / ${r.max_trades ?? "–"}`, "trades today") + tile(`${(state.positions || []).length} / ${r.max_open ?? "–"}`, "open positions") +
    tile(`${r.consecutive_losses ?? 0} / ${r.max_consecutive_losses ?? "–"}`, "consecutive losses") + tile(rs(r.risk_per_trade), "risk per trade") +
    tile(esc(r.session || "open for entries"), "entry gate");
  $("#signals").innerHTML = (state.signals || []).slice(0, 30).map(sigCard).join("") || '<p class="muted">No signals yet today.</p>';
  table($("#positions"), ["Instrument", "Strategy", "Qty", "Entry", "LTP", "SL", "T1", "T2", "T3", "Stage", "P&L", ""],
    (state.positions || []).map((p) => [`<span class="badge ${p.side === "PE" ? "pe" : "ce"}">${p.side}</span> ${esc(p.instrument)}`, esc(p.plan.strategy_name || p.strategy), `${p.qty_open}/${p.qty}`,
      num(p.entry), num(p.last_price), num(p.stop), ...p.targets.map((t) => num(t)), "T" + p.stage, pn(p.unrealized + p.net), `<button data-close="${esc(p.id)}">Close</button>`]), "No open positions.");
  table($("#closed"), ["Instrument", "Strategy", "Qty", "Entry", "Exit reason", "Net", "R"],
    (state.closed_today || []).map((p) => [`<span class="badge ${p.side === "PE" ? "pe" : "ce"}">${p.side}</span> ${esc(p.instrument)}`, esc(p.plan.strategy_name || p.strategy), p.qty, num(p.entry), esc(p.exit_reason), pn(p.net), num(p.r_multiple)]), "No closed trades today.");
}
function sigCard(s) {
  const c = cls(s.status, s.side), p = s.plan;
  const checks = Object.entries(s.checks || {}).map(([k, v]) => `<span class="${v === true ? "y" : v === false ? "n" : ""}" title="${v === null ? "no data" : v ? "agrees" : "does not agree"}">${k}${v === true ? " ✓" : v === false ? " ✗" : " ·"}</span>`).join("");
  let body = "";
  if (p) {
    const total = 1 + (p.t3 - p.entry) / p.r, two = p.plan_mode === "percent";
    const pc = (v, sign) => (v ? ` <small class="muted">${sign}${num(v, 1)}%</small>` : "");
    body = `<div class="inst">${esc(p.instrument)}</div>
    <div class="muted" style="font-size:12px">${esc(p.direction || "")} · ${esc(p.timeframe || "")} · ${esc(p.signal_status || "PAPER TRADE")}</div>
    <dl><div><dt>Entry (live premium)</dt><dd>${num(p.entry)}</dd></div><div><dt>Stop-loss</dt><dd class="dn">${num(p.stop)}${pc(p.stop_pct, "−")}</dd></div><div><dt>Qty (lots)</dt><dd>${p.qty} (${p.lots})</dd></div><div><dt>Max loss</dt><dd>${rs(p.max_loss)}</dd></div>
    <div><dt>Target 1</dt><dd class="up">${num(p.t1)}${pc(p.t1_pct, "+")}</dd></div><div><dt>Target 2</dt><dd class="up">${num(p.t2)}${pc(p.t2_pct, "+")}</dd></div>${two ? "" : `<div><dt>Target 3</dt><dd class="up">${num(p.t3)}</dd></div>`}<div><dt>R:R (T2, net)</dt><dd>1 : ${num(p.rr)}</dd></div></dl>
    <div class="rr" role="img" aria-label="Risk ${num(p.r)} versus reward up to ${num(p.t3 - p.entry)} premium points"><i class="risk" style="width:${100 / total}%"></i><i class="rew" style="width:${100 - 100 / total}%"></i></div>
    <div class="rrlab"><span>SL ${num(p.stop)}</span><span>entry ${num(p.entry)}</span><span>${two ? "T2" : "T3"} ${num(p.t3)}</span></div>
    <div class="reason">Underlying ${num(p.und_entry)} · invalid ${p.side === "CE" ? "below" : "above"} ${num(p.und_stop)} (${esc(p.stop_basis)})</div>
    <details><summary>Exit conditions</summary><ul>${p.exits.map((e) => `<li>${esc(e)}</li>`).join("")}</ul></details>`;
  }
  return `<div class="sig ${c}"><div class="top"><span><b>${esc(s.symbol)}</b> <span class="badge ${c}">${esc(label(s.status, s.side))}</span></span><span class="muted">${esc((s.timestamp || "").slice(11, 16))}</span></div>
    <div class="muted" style="font-size:12px;margin-top:4px">${esc(s.strategy_name)}${(s.also || []).length ? " + " + s.also.map(esc).join(", ") : ""}</div>${body}
    <div class="rrlab" style="margin-top:8px"><span>Confidence (${esc(s.confidence_basis || "checks aligned")}, not a win probability)</span><span>${num(s.confidence, 0)}</span></div>
    <div class="conf"><i style="width:${Math.min(100, s.confidence || 0)}%"></i></div>
    <div class="checks">${checks}</div>${(s.why || []).length ? `<details><summary>Why this signal</summary><ul>${s.why.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></details>` : ""}<div class="reason">${esc(s.reason || "")}${s.premium ? " · " + esc(s.premium) : ""}</div></div>`;
}
function rankTable(el, rows, side) {
  const c = side === "CE" ? "ce" : "pe";
  table(el, ["#", "Stock", "LTP", "Score", "Factors", "Option", "Strike", "Expiry", "Entry", "SL", "T1", "T2", "T3", "R:R", "Conf.", "Strategy"],
    rows.slice(0, 15).map((r) => { const p = r.idea || {}; const has = p.entry !== undefined;
      const fx = Object.entries(r.factors || {}).map(([k, v]) => `<span class="${v === 1 ? "y" : v === -1 ? "n" : ""}" title="${k}: ${v === 1 ? "bullish" : v === -1 ? "bearish" : v === null ? "no data" : "neutral"}">${k}</span>`).join("");
      return [r.rank, `<b>${esc(r.symbol)}</b> <span class="badge ${has ? c : "none"}">${has ? "BUY " + side : r.idea ? "NO TRADE" : "RANKED"}</span>`, num(r.ltp), num(r.score, 0), `<div class="checks" style="margin:0">${fx}</div>`,
        has ? esc(p.instrument) : `<span class="muted">${esc(p.note || "plan is built for the top 3")}</span>`, has ? num(p.strike, 0) : "–", has ? esc(p.expiry) : "–", has ? num(p.entry) : "–", has ? num(p.stop) : "–",
        has ? num(p.t1) : "–", has ? num(p.t2) : "–", has ? num(p.t3) : "–", has ? "1 : " + num(p.rr) : "–", has ? num(p.confidence, 0) : "–", has ? esc(p.strategy_name) : "–"]; }),
    "No stock qualifies on this side right now: NO TRADE.");
}
function renderScanner() {
  const sc = state.scanner || {};
  rankTable($("#rankup"), sc.bullish || [], "CE"); rankTable($("#rankdn"), sc.bearish || [], "PE");
  const f = $("#filter").value.trim().toUpperCase(), st = $("#fstatus").value;
  const rows = (state.watchlist || []).filter((w) => (!f || w.symbol.includes(f)) && (!st || (w.status || "").startsWith(st)));
  $("#scaninfo").textContent = `${(state.health || {}).scanned ?? 0} of ${(state.watchlist || []).length} analysed (stocks rotate within the request budget) · ${sc.no_trade ?? 0} stocks NO TRADE (insufficient or conflicting)`;
  table($("#watch"), ["Symbol", "LTP", "Chg %", "Status", "Conf.", "Checks ▲ / ▼", "Trend", "RSI", "RVOL", "Lot", "Expiry", "Strategies"],
    rows.map((w) => [`<b>${esc(w.symbol)}</b>`, num(w.ltp), w.change_pct == null ? "–" : `<span class="${w.change_pct >= 0 ? "up" : "dn"}">${num(w.change_pct)}</span>`,
      `<span class="badge ${cls(w.status, w.side)}">${esc(w.status || "–")}</span>`, w.confidence ? num(w.confidence, 0) : "–",
      w.checks_avail ? `${w.checks_up} / ${w.checks_dn} of ${w.checks_avail}` : "–", trend(w.trend), num(w.rsi, 0), num(w.rvol, 1), esc(w.lot_size ?? "–"), esc(w.expiry || "–"),
      `<span class="muted">${esc((w.strategies || []).join(", ") || w.note || "")}</span>`]), "No instruments.");
}
async function renderStrategies() {
  const list = await api("/api/strategies");
  $("#stratlist").innerHTML = list.map((s) => `<div class="strat" data-key="${esc(s.key)}"><div class="head">
    <b>${s.number}. ${esc(s.name)}</b><span class="muted">${esc(s.category)} · ${esc(s.kind)}${s.needs.length ? " · needs " + esc(s.needs.join(", ")) : ""} · ${s.signals} signals</span>
    <label style="flex-direction:row;align-items:center;gap:6px;font-size:13px;color:inherit"><input type="checkbox" class="en" ${s.enabled ? "checked" : ""}> enabled</label></div>
    <div class="rules"><b class="up">BUY CE</b> ${esc(s.ce)}<br><b class="dn">BUY PE</b> ${esc(s.pe)}<br><b>Exit</b> ${esc(s.exit)}</div>
    <div class="params">${Object.entries(s.params).map(([k, v]) => `<label>${esc(k)}<input type="number" step="any" name="${esc(k)}" value="${esc(v)}"></label>`).join("")}
    ${Object.keys(s.params).length ? "<button class='save'>Save</button>" : ""}<span class="muted msg"></span></div></div>`).join("");
}
function perfTable(el, rows, first) {
  table(el, [first, "Signals", "Trades", "Wins", "Losses", "Win %", "PF", "Net P&L", "Avg profit", "Avg loss", "Avg R", "Max DD", "Sharpe"],
    rows.map((r) => [esc(r.group), r.signals ?? "–", r.trades, r.wins, r.losses, num(r.win_rate, 1), num(r.profit_factor), pn(r.net_pnl), rs(r.avg_profit), rs(r.avg_loss), num(r.avg_r, 2), rs(r.max_drawdown), num(r.sharpe)]), "No closed trades yet.");
}
async function renderPerformance() {
  const p = await api("/api/performance"), o = p.overall;
  const tile = (v, l) => `<div class="tile"><b>${v}</b><span>${l}</span></div>`;
  $("#perf").innerHTML = tile(o.signals ?? 0, "total signals") + tile(o.wins, "winning trades") + tile(o.losses, "losing trades") + tile(num(o.win_rate, 1) + "%", "win rate") +
    tile(num(o.profit_factor), "profit factor") + tile(pn(o.net_pnl), "net P&L") + tile(rs(o.avg_profit), "average profit") + tile(rs(o.avg_loss), "average loss") +
    tile(rs(o.max_drawdown), "maximum drawdown") + tile(o.sharpe == null ? "needs 20 days" : num(o.sharpe), "Sharpe ratio (daily)");
  lineChart($("#equity"), p.equity);
  perfTable($("#pstrategy"), p.by_strategy, "Strategy"); perfTable($("#psymbol"), p.by_symbol, "Symbol");
  perfTable($("#pside"), p.by_side, "Side"); perfTable($("#ptime"), p.by_time, "Entry time");
}
function lineChart(el, pts) {
  if (!pts || pts.length < 2) { el.innerHTML = '<p class="muted">The equity curve appears after two closed trades.</p>'; return; }
  const W = 900, H = 220, L = 64, R = 12, T = 10, B = 24, ys = pts.map((p) => p.equity);
  const lo = Math.min(0, ...ys), hi = Math.max(0, ...ys), span = hi - lo || 1;
  const x = (i) => L + (W - L - R) * i / (pts.length - 1), y = (v) => T + (H - T - B) * (1 - (v - lo) / span);
  const d = pts.map((p, i) => (i ? "L" : "M") + x(i).toFixed(1) + " " + y(p.equity).toFixed(1)).join(" ");
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Cumulative net profit and loss after each closed trade">
    <line x1="${L}" x2="${W - R}" y1="${y(0)}" y2="${y(0)}" stroke="var(--line)"/><text x="${L - 6}" y="${y(0) + 4}" text-anchor="end" font-size="11" fill="var(--mut)">0</text>
    <text x="${L - 6}" y="${y(hi) + 10}" text-anchor="end" font-size="11" fill="var(--mut)">${rs(hi)}</text><text x="${L - 6}" y="${y(lo)}" text-anchor="end" font-size="11" fill="var(--mut)">${rs(lo)}</text>
    <path d="${d}" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round"/><line id="cross" y1="${T}" y2="${H - B}" stroke="var(--mut)" stroke-dasharray="3 3" visibility="hidden"/>
    <text x="${L}" y="${H - 6}" font-size="11" fill="var(--mut)">${esc(pts[0].t.slice(0, 10))}</text><text x="${W - R}" y="${H - 6}" text-anchor="end" font-size="11" fill="var(--mut)">${esc(pts[pts.length - 1].t.slice(0, 10))}</text></svg>`;
  const svg = el.querySelector("svg"), cross = svg.querySelector("#cross"), tip = $("#tip");
  svg.onmousemove = (e) => {
    const b = svg.getBoundingClientRect(), i = Math.max(0, Math.min(pts.length - 1, Math.round(((e.clientX - b.left) / b.width * W - L) / (W - L - R) * (pts.length - 1))));
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
    tip.hidden = false; tip.textContent = `${pts[i].t.slice(0, 16).replace("T", " ")} · ${rs(pts[i].equity)}`; tip.style.left = e.clientX + 12 + "px"; tip.style.top = e.clientY + 12 + "px";
  };
  svg.onmouseleave = () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); };
}
async function renderTrades() {
  const t = await api("/api/trades");
  table($("#history"), ["Opened", "Instrument", "Strategy", "Qty", "Entry", "SL", "Exit reason", "Closed", "Charges", "Net", "R", "Source"],
    t.map((p) => [esc(String(p.opened).slice(0, 16)), `<span class="badge ${p.side === "PE" ? "pe" : "ce"}">${p.side}</span> ${esc(p.instrument)}`, esc((p.plan || {}).strategy_name || p.strategy), p.qty, num(p.entry), num(p.initial_stop),
      esc(p.status === "OPEN" ? "open" : p.exit_reason), esc(String(p.closed || "").slice(0, 16)), rs(p.charges), pn(p.net), num(p.r_multiple), esc(p.price_source)]), "No paper trades yet.");
}
async function renderBacktests() {
  const b = await api("/api/backtests");
  $("#btjob").textContent = b.job.state === "idle" ? "" : `${b.job.state}: ${b.job.message}`;
  table($("#btlist"), ["Run", "Period", "Symbols", "Trades", "Net"], b.reports.map((r) => [`<button data-bt="${esc(r.id)}">${esc(r.created.replace("T", " "))}</button>`, `${esc(r.from)} → ${esc(r.to)}`,
    esc(r.symbols.slice(0, 3).join(", ")) + (r.symbols.length > 3 ? ` +${r.symbols.length - 3}` : ""), r.trades ?? 0, pn(r.net_pnl)]), "No backtests yet. Run one above, or: python -m src.fno backtest --help");
  if (b.job.state === "running") setTimeout(() => tab === "backtests" && renderBacktests(), 2500);
}
async function showReport(id) {
  const r = await api("/api/backtests/" + id), o = r.overall;
  const row = (s) => { const a = s.all, i = s.in_sample, q = s.out_of_sample, wf = s.walk_forward; const why = Object.values(s.unavailable || {})[0];
    return [`${s.number}. ${esc(s.name)}`, a.trades, num(a.win_rate, 1), num(a.profit_factor), pn(a.net_pnl), num(a.avg_r, 3), rs(a.max_drawdown), num(a.sharpe), num(i.avg_r, 3), q.trades, num(q.avg_r, 3),
      wf ? `${wf.out_of_sample.trades} / ${num(wf.out_of_sample.avg_r, 3)}` : "–", `<span class="muted">${esc(a.trades ? (s.caution || "") : (why || "no signals"))}</span>`]; };
  $("#btreport").innerHTML = `<h2>${esc(r.from)} → ${esc(r.to)} · ${esc(r.symbols.join(", "))}</h2>
    <div class="muted">Price source: ${esc(r.price_sources.join(", ") || "–")} · ${r.days} days · out-of-sample from ${esc(r.oos_starts || "–")} · ${o.trades} trades · net ${rs(o.net_pnl)}</div>
    ${r.warnings.map((w) => `<div class="warn">${esc(w)}</div>`).join("")}<div id="bteq" class="chart"></div>
    <div class="scroll"><table id="bttab"></table></div>`;
  table($("#bttab"), ["Strategy", "Trades", "Win %", "PF", "Net", "Avg R", "Max DD", "Sharpe", "In-sample R", "OOS n", "OOS R", "Walk-fwd n / R", "Note"], r.strategies.map(row));
  lineChart($("#bteq"), r.equity);
}
function renderSystem() {
  const h = state.health || {}, f = h.feed || {}, w = h.websocket || {};
  const tile = (v, l) => `<div class="tile"><b>${esc(v)}</b><span>${l}</span></div>`;
  $("#health").innerHTML = tile(state.status || "–", "engine") + tile(f.state || "–", esc(f.source || "data feed")) + tile(`${f.failures ?? 0} / ${f.calls ?? 0}`, "failed / total API calls") +
    tile(f.seconds_since_ok ?? "–", "seconds since last good reply") + tile(w.connected ? "connected" : (w.note || "disconnected"), `WebSocket (${w.reconnects ?? 0} reconnects)`) +
    tile(`${state.duration ?? "–"} s`, `cycle ${state.cycle ?? 0}`) + tile(`${h.scanned ?? 0} / ${h.universe ?? 0}`, "instruments analysed");
  table($("#errors"), ["Time", "Error"], (h.errors || []).map((e) => [esc(e.time), `<span style="white-space:normal">${esc(e.error)}</span>`]), "No errors.");
}
const renders = { dash: renderDash, scanner: renderScanner, strategies: renderStrategies, performance: renderPerformance, trades: renderTrades, backtests: renderBacktests, system: renderSystem };
async function refresh() {
  try {
    state = await api("/api/state");
    $("#status").textContent = `${state.status} · ${state.time || ""} · cycle ${state.cycle ?? 0}`;
    $("#mock").hidden = !state.mock;
    const kill = (state.risk || {}).kill;
    $("#banner").hidden = !kill; $("#banner").textContent = kill ? "KILL SWITCH ENGAGED: " + kill + ". No new entries; open paper positions are squared off." : "";
    $("#kill").classList.toggle("on", !!kill); $("#kill").textContent = kill ? "Release kill switch" : "Kill switch";
    if (["dash", "scanner", "system"].includes(tab)) renders[tab]();
  } catch (e) { $("#status").textContent = "engine not reachable: " + e.message; }
}
$("#tabs").onclick = (e) => {
  const t = e.target.dataset.tab; if (!t) return;
  tab = t; document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("on", b.dataset.tab === t));
  document.querySelectorAll(".tab").forEach((s) => s.classList.toggle("on", s.id === t));
  Promise.resolve(renders[t]()).catch((err) => console.error(err));
};
$("#filter").oninput = renderScanner; $("#fstatus").onchange = renderScanner;
$("#kill").onclick = async () => {
  const on = (state.risk || {}).kill;
  if (on) { if (prompt('Type RELEASE to allow new paper entries again') !== "RELEASE") return; await api("/api/kill/release", { confirm: "RELEASE" }).catch((e) => alert(e.message)); }
  else if (confirm("Engage the kill switch? New entries stop and open paper positions are squared off.")) await api("/api/kill", {});
  refresh();
};
document.body.addEventListener("click", async (e) => {
  const t = e.target;
  if (t.dataset.close) { await api(`/api/positions/${t.dataset.close}/close`, {}).catch((err) => alert(err.message)); refresh(); }
  if (t.dataset.bt) showReport(t.dataset.bt);
  if (t.classList.contains("save")) {
    const box = t.closest(".strat"), body = {};
    box.querySelectorAll("input[type=number]").forEach((i) => { body[i.name] = i.value; });
    try { await api("/api/strategies/" + box.dataset.key, body); box.querySelector(".msg").textContent = "saved"; } catch (err) { box.querySelector(".msg").textContent = err.message; }
  }
});
document.body.addEventListener("change", async (e) => {
  if (e.target.classList.contains("en")) {
    const box = e.target.closest(".strat");
    try { await api("/api/strategies/" + box.dataset.key, { enabled: e.target.checked }); } catch (err) { alert(err.message); e.target.checked = !e.target.checked; }
  }
});
$("#btform").onsubmit = async (e) => {
  e.preventDefault();
  const f = new FormData(e.target), body = Object.fromEntries(f.entries()); body.walk_forward = f.has("walk_forward");
  try { await api("/api/backtests/run", body); } catch (err) { alert(err.message); }
  renderBacktests();
};
try {
  const es = new EventSource("/api/stream");          // pushed updates; the timer below is the fallback
  es.addEventListener("cycle", refresh); es.addEventListener("signal", refresh);
} catch (e) { /* polling only */ }
setInterval(refresh, 5000);
refresh();
