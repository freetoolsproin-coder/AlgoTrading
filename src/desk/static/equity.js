"use strict";
/* Intraday cash + Swing tabs. Data: /api/eq/* (the desk passes these on to python -m src.equity).
   Uses the helpers of the page's main script: $, h, fmt, rs, sgn, tone, fill, det, isNum, clock, emptyRow. */
(function(){
let E = null, eqErr = "starting", tab = "overview";
const last = {};
const sel = {intraday: {symbol: null, tf: "5m"}, swing: {symbol: null, tf: "1d"}};
const charts = {intraday: null, swing: null};
let scanFilter = "all";
const TFS = {intraday: ["1m","5m","15m","30m","1h"], swing: ["1h","4h","1d","1w"]};
const LINE_COLORS = {"EMA 9":"#e0912b","EMA 21":"#7a5cd6","EMA 20":"#e0912b","EMA 50":"#7a5cd6","VWAP":"#2f8fd8","Supertrend":"#8a93a0"};

const changed = (key, obj) => { const s = JSON.stringify(obj === undefined ? null : obj); if(last[key] === s) return false; last[key] = s; return true; };
const arrow = v => h("span","arrow " + (v===1 ? "up" : v===-1 ? "down" : "flat"), v===1 ? "▲" : v===-1 ? "▼" : "–");
const tag = (text, cls) => h("span","tag " + (cls || "plain"), text);
const hm = s => s ? String(s).slice(11,16) : "";
const dm = s => s ? String(s).slice(5,10) : "";
function toast(text, err){
  let box = $("toast"); if(!box){ box = h("div"); box.id = "toast"; document.body.append(box); }
  const t = h("div", err ? "err" : null, text); box.append(t); setTimeout(() => t.remove(), err ? 9000 : 4500);
}
async function post(path, body){
  try{
    const r = await fetch("/api/eq/" + path, {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body || {})});
    const j = await r.json().catch(() => ({}));
    if(!r.ok || j.ok === false){ toast(j.error || ("request failed: " + r.status), true); return null; }
    loadEq(); return j;
  }catch(e){ toast("the desk server is not answering", true); return null; }
}
const btn = (text, cls, fn, title) => { const b = h("button","btn " + (cls || ""), text); b.type = "button"; if(title) b.title = title; b.onclick = e => { e.stopPropagation(); fn(b); }; return b; };
const table = (heads, rows, empty) => h("table", null,
  h("thead", null, h("tr", null, heads.map(x => { const [t, c] = Array.isArray(x) ? x : [x, null]; return h("th", c, t); }))),
  h("tbody", null, rows.length ? rows : emptyRow(heads.length, empty || "nothing yet")));

/* ---------- shared control bar ---------- */
function controlBar(mod){
  const live = E.live || {}, r = E.risk || {};
  const seg = h("div","seg", ["analysis","paper","live"].map(m => {
    const b = h("button", (m === "live" ? "live " : "") + (E.mode === m ? "on" : ""), m === "analysis" ? "Analysis" : m === "paper" ? "Paper" : "Live");
    b.title = m === "analysis" ? "Signals only, no orders" : m === "paper" ? "Simulated orders, hypothetical P&L" : "Real Upstox orders (must be armed for today; each order is confirmed)";
    b.onclick = () => {
      if(E.mode === m) return;
      if(m === "live" && !confirm("Switch to LIVE trading?\n\nReal orders can then be sent to your Upstox account. Every order still waits for your confirmation here.")) return;
      post("mode", {mode: m}).then(j => j && toast("Mode: " + m.toUpperCase()));
    };
    return b; }));
  const pills = [
    h("span","pill " + (E.market_open ? "good" : ""), E.market_open ? h("span","dot") : null, E.market_open ? "Market open" : "Market closed"),
    h("span","pill " + (E.status === "ok" ? "info" : E.status === "starting" ? "warn" : "bad"), E.status === "ok" ? "Upstox data" : E.status === "starting" ? "first scan running…" : (E.status_text || E.status)),
    h("span","pill " + (E.mode === "live" ? "bad" : E.mode === "paper" ? "warn" : ""), E.mode.toUpperCase() + (E.mode === "live" ? (live.armed ? " · armed" : " · NOT armed") : "")),
    r.kill_switch ? h("span","pill bad","KILL SWITCH ON") : null];
  const kill = r.kill_switch
    ? btn("Release kill switch", "", () => confirm("Release the kill switch and allow new entries again?") && post("kill", {on:false}))
    : btn("Kill switch", "danger", () => confirm("KILL SWITCH\n\nNo new entries, pending orders are cancelled and every open position of both modules is closed" + (E.mode === "live" ? " with REAL exit orders" : "") + ". Continue?") && post("kill", {on:true}),
          "Emergency stop: close everything and block new entries");
  return h("div","card eq-bar", seg, h("div","pills grow", pills),
    h("span","small muted", E.last_scan ? `scan ${hm(E.last_scan)} · ${fmt(E.scan_seconds,0)}s` : "first scan running…"),
    btn("Scan now", "", b => { b.disabled = true; post("refresh").then(() => toast("Scan requested")); }), kill);
}
function banners(){
  const out = [], r = E.risk || {}, live = E.live || {};
  if(E.status === "starting") out.push(h("div","banner info","First scan is running", h("small", null, "Downloading candles for the 50 stocks; this takes about a minute after a start.")));
  else if(E.status !== "ok") out.push(h("div","banner bad", E.status_text || "the equity engine reports an error", h("small", null, "Signals are only ever built from real Upstox data. What you see is from the last successful scan; live prices and new signals resume when the data is available again.")));
  if(E.data_through && E.data_through < E.session) out.push(h("div","banner warn", `Candles for ${E.session} are not available from Upstox yet – charts and indicators end on ${E.data_through}`,
    h("small", null, "Upstox publishes a session's history some hours after the close; the engine asks again every 10 minutes. Signals shown for that session were generated live.")));
  if(r.kill_file) out.push(h("div","banner bad","KILL_SWITCH file present in the project folder", h("small", null, "Delete the file to allow entries again (it is shared with the other services).")));
  (r.blocked || []).filter(x => !/kill/i.test(x)).forEach(x => out.push(h("div","banner warn","Risk limit: " + x, h("small", null, "No new entries until the limit resets."))));
  if(E.mode === "live" && !live.armed) out.push(h("div","banner warn","Live mode is selected but not armed for today", h("small", null, (live.missing || []).join(" · "))));
  (live.mismatches || []).forEach(x => out.push(h("div","banner bad","Reconciliation: " + x)));
  const pend = (E.orders && E.orders.pending) || [];
  if(pend.length) out.push(h("div","banner info", `${pend.length} LIVE order${pend.length>1 ? "s" : ""} waiting for your confirmation`,
    h("small", null, "Confirming an entry also authorises its protective stop-loss order, the intraday square-off and kill-switch exits."),
    pend.map(o => h("div","row", tag(o.purpose, o.side), h("b", null, `${o.side} ${o.qty} ${o.symbol}`), `${o.order_type} ${fmt(o.price)}`,
      h("span","muted small", (o.data && o.data.reason) || ""),
      btn("Confirm – send real order", "good sm", () => confirm(`Place this LIVE order on Upstox?\n\n${o.purpose}: ${o.side} ${o.qty} ${o.symbol}  ${o.order_type} near ${fmt(o.price)}\n\nThis sends a real order with real money.`) && post(`orders/${o.id}/confirm`, {confirm:true}).then(j => j && toast(j.message || "sent"))),
      btn("Reject", "sm", () => post(`orders/${o.id}/reject`).then(j => j && toast("Rejected")))))));
  return out;
}
function riskCard(){
  const r = E.risk || {}, used = r.daily_loss_limit ? Math.max(0, Math.min(100, -100 * (r.realized_today || 0) / r.daily_loss_limit)) : 0;
  const f = (k, label, step) => { const i = h("input"); i.type = "number"; i.step = step || "any"; i.value = r[k]; i.dataset.k = k; return h("label", null, label, i); };
  const form = h("div","frm", f("capital","Capital (₹)","1000"), f("risk_per_trade_pct","Risk per trade (%)","0.05"), f("max_daily_loss_pct","Max daily loss (%)","0.1"),
    f("max_open_intraday","Max open intraday","1"), f("max_open_swing","Max open swing","1"), f("daily_trade_limit","Daily trade limit","1"),
    f("max_consecutive_losses","Stop after N losses in a row","1"), f("min_rr","Min reward:risk (planned)","0.1"), f("min_net_rr","Min reward:risk after charges","0.1"),
    f("max_position_pct","Max position (% of capital)","1"), f("slippage_bps","Slippage (bps per side)","0.5"));
  const save = btn("Save risk settings", "primary sm", () => { const body = {}; form.querySelectorAll("input").forEach(i => { if(i.value !== "") body[i.dataset.k] = Number(i.value); }); post("risk", body).then(j => j && toast("Risk settings saved")); });
  return h("div","card pad", h("h3", null, "Risk"),
    h("div","sum", h("div", null, h("span", null, "Risk / trade"), h("b", null, rs(r.risk_budget)), h("div","small muted", `${fmt(r.risk_per_trade_pct,2)}% of ${rs(r.capital)}`)),
      h("div", null, h("span", null, "Day P&L (net)"), h("b", tone(r.realized_today), rs(r.realized_today)), h("div","small muted", `limit −${rs(r.daily_loss_limit).replace("−","")} · ${fmt(used,0)}% used`)),
      h("div", null, h("span", null, "Trades today"), h("b", null, `${r.trades_today ?? 0} / ${r.daily_trade_limit}`), h("div","small muted", `${r.loss_streak ?? 0} losses in a row (stop at ${r.max_consecutive_losses})`)),
      h("div", null, h("span", null, "Min R:R"), h("b", null, `1:${fmt(r.min_rr,1)}`), h("div","small muted", `1:${fmt(r.min_net_rr,1)} after charges`))),
    h("p","note","Quantity = risk per trade ÷ (entry − stop + slippage), whole shares, capped at " + fmt(r.max_position_pct,0) + "% of capital per position (cash market, no leverage). No averaging down."),
    det("eq-risk-form", "Change risk settings", h("div", null, form, h("div", null, save))));
}

/* ---------- opportunity cards ---------- */
function compBlock(o, label){
  const max = {trend:25, volume: o.module === "swing" ? 15 : 20, momentum:20, adx:10, market:10, rel_strength: o.module === "swing" ? 20 : 15, structure:10};
  const g = h("div","comp");
  Object.entries(o.components || {}).forEach(([k, v]) => { const bar = h("i", null, h("u")); bar.firstChild.style.width = Math.max(0, Math.min(100, 100 * (v || 0) / (max[k] || 20))) + "%";
    g.append(h("span","muted", k.replace("_"," ")), bar, h("b", null, `${fmt(v,0)} / ${max[k] || ""}`)); });
  return h("div", null, g, h("p","note", label));
}
function oppCard(o, mod){
  const swing = mod === "swing", s = sel[mod];
  const card = h("div", `card opp ${o.side}${o.fresh ? "" : " dim"}${s.symbol === o.symbol ? " sel" : ""}`);
  card.onclick = () => pick(mod, o.symbol);
  const status = o.status === "TAKEN" ? tag("TAKEN","ok") : o.status === "PROPOSED" ? tag("CONFIRM?","warn") : o.status === "PENDING" ? tag("ENTRY PENDING","warn")
    : o.status === "BLOCKED" ? tag("NO TRADE","NOTRADE") : o.status === "LATE" ? tag("LATE","plain") : o.status === "ALERT" ? tag("EXIT / AVOID","SELL") : o.status === "EXPIRED" ? tag("EXPIRED","plain") : tag(o.fresh ? "ACTIVE" : o.status, o.fresh ? "ok" : "plain");
  const sym = h("b", null, o.symbol);
  card.append(h("div","opp-top", tag(o.side, o.side), sym, h("span","muted small", o.strategy_name),
    h("div","right", o.confluence > 1 ? tag(`${o.confluence} agree`,"ok") : null, o.validated ? tag("validated","pos") : tag("unvalidated","plain"), status,
      h("span","small muted", swing ? dm(o.signal_date || o.time) : hm(o.time)))));
  const lv = h("div","lv");
  const cell = (k, v, cls) => lv.append(h("div", cls, h("span", null, k), h("b", null, v)));
  cell(swing ? "Close" : "Signal px", fmt(o.price));
  if(swing || isNum(o.ltp)) cell("LTP", fmt(o.ltp));
  cell(swing ? "Entry zone" : "Entry", swing && o.zone ? `${fmt(o.zone[0])} – ${fmt(o.zone[1])}` : fmt(o.entry));
  cell("Stop-loss", fmt(o.stop), "sl"); cell("Target 1", fmt(o.t1), "tg"); cell("Target 2", fmt(o.t2), "tg");
  cell("R:R", `1:${fmt(o.rr,1)}`); cell(swing ? "Confidence" : "Strength", fmt(o.score,0));
  if(swing) cell("Hold", o.hold_text ? o.hold_text.replace(" sessions","d").replace("about ","~") : "–");
  card.append(lv, h("div","why", o.reason));
  const meta = h("div","meta");
  meta.append(h("span", null, "Volume ", h("b", o.volume_ok ? "up" : "", isNum(o.volume_x) ? `${fmt(o.volume_x,1)}x ${o.volume_ok ? "✓" : "✗"}` : "n/a")));
  if(swing){
    meta.append(h("span", null, "Trend ", h("b", null, o.trend || "–")), h("span", null, "RS score ", h("b", null, isNum(o.rs_score) ? fmt(o.rs_score,0) + "/100" : "n/a")));
    if(o.htf) meta.append(h("span", null, "1h ", arrow(o.htf.h1), " 4h ", arrow(o.htf.h4)));
    if(o.history && o.history.oos_trades) meta.append(h("span", null, "Backtest (out-of-sample): ", h("b", null, `${fmt(o.history.oos_win,0)}% of ${o.history.oos_trades}`), " – not a probability"));
  } else Object.entries(o.indicators || {}).slice(0,4).forEach(([k, v]) => meta.append(h("span", null, k.replace("_"," ") + " ", h("b", null, fmt(v)))));
  card.append(meta);
  if(o.note && o.status !== "TAKEN") card.append(h("div","note", o.note));
  card.append(det(`eq-c-${mod}-${o.symbol}-${o.side}`, swing ? "Confidence breakdown · exit conditions" : "Strength breakdown · strategies",
    h("div", null, compBlock(o, swing ? E.swing.method : E.intraday.method),
      swing && o.exits ? h("ul","checks", o.exits.map(x => h("li", null, x))) : null,
      h("div","tags", (o.strategies || []).map(x => tag(`${x.name} ${swing ? "" : hm(x.time)}`, x.validated ? "pos" : "plain"))))));
  return card;
}
function oppList(rows, mod, empty){
  return rows.length ? h("div","opps", rows.map(o => oppCard(o, mod))) : h("div","card empty", empty);
}

/* ---------- tables ---------- */
function posTable(rows, mod, closed){
  const swing = mod === "swing";
  if(closed) return table(["Stock","Side","Strategy",["Qty","n"],["Entry","n"],["Exit","n"],"Reason",["Net P&L","n"],["R","n"],"Closed"],
    rows.slice().reverse().map(p => h("tr", null, h("td","b",p.symbol), h("td", null, tag(p.side,p.side)), h("td","small",p.strategy_name || p.strategy), h("td","n",p.qty),
      h("td","n",fmt(p.entry)), h("td","n",fmt(p.exit_price)), h("td","small",(p.exit_reason || "").replace(/_/g," ")),
      h("td","n b " + (p.status === "CLOSED" ? tone(p.realized) : ""), p.status === "CLOSED" ? rs(p.realized) : "–"), h("td","n",fmt(p.r)), h("td","small",(p.closed || "").slice(5,16).replace("T"," ")))),
    swing ? "no closed swing trades in the last 14 days" : "no closed intraday trades today");
  const heads = ["Stock","Side","Status",["Qty","n"],["Entry","n"],["LTP","n"],["Stop","n"],["T1","n"],["T2","n"]];
  if(swing) heads.push(["Sessions","n"]);
  heads.push(["Unrealised","n"],["R","n"],"Strategy","");
  return table(heads, rows.map(p => {
    const waiting = p.status === "PENDING" || p.status === "PROPOSED";
    const tr = h("tr","main " + (p.side === "BUY" ? "CE" : "PE"), h("td","b",p.symbol), h("td", null, tag(p.side,p.side)),
      h("td", null, tag(waiting ? (p.status === "PENDING" ? "WAITING FOR ZONE" : "CONFIRM?") : p.status + (p.t1_done ? " · T1 ✓" : ""), waiting ? "warn" : "ok"), " ", tag(p.mode.toUpperCase(), p.mode === "live" ? "SELL" : "plain"),
        p.exit_flag ? h("div","small down", "exit flagged: " + p.exit_flag.replace(/_/g," ")) : null,
        waiting && p.zone ? h("div","small muted", `zone ${fmt(p.zone[0])} – ${fmt(p.zone[1])}`) : null),
      h("td","n", p.qty_open || p.qty), h("td","n",fmt(p.entry)), h("td","n b",fmt(p.ltp)),
      h("td","n", fmt(p.stop), isNum(p.to_stop_pct) ? h("div","small muted", fmt(p.to_stop_pct,2) + "% away") : null),
      h("td","n", fmt(p.t1), isNum(p.to_t1_pct) && !p.t1_done ? h("div","small muted", fmt(p.to_t1_pct,2) + "% to go") : null),
      h("td","n", fmt(p.t2), isNum(p.to_t2_pct) ? h("div","small muted", fmt(p.to_t2_pct,2) + "% to go") : null));
    if(swing) tr.append(h("td","n", p.sessions != null ? `${p.sessions} / ${E.swing.hold[1]}` : "–"));
    tr.append(h("td","n b " + tone(p.unrealized), rs(p.unrealized)), h("td","n",fmt(p.r)), h("td","small",p.strategy_name || p.strategy),
      h("td", null, btn(waiting ? "Cancel" : "Close", "sm", () => confirm(`${waiting ? "Cancel" : "Close"} ${p.symbol}${p.mode === "live" && !waiting ? " with a REAL exit order" : ""}?`) && post(`positions/${p.id}/close`).then(j => j && toast(j.message)))));
    tr.onclick = () => pick(mod, p.symbol);
    return tr; }), swing ? "no swing holdings or pending entries" : "no open intraday positions");
}
function stratTable(mod){
  const M = E[mod], bt = (E.backtest || {})[mod];
  const rows = (M.strategies || []).map(s => { const b = s.backtest || {}, p = s.paper || {};
    return h("tr", null, h("td", null, h("div","b",s.name), h("div","small muted",s.kind + (s.enabled ? "" : " · switched off"))),
      h("td", null, s.validated ? tag("validated","pos") : tag(b.trades != null ? "not validated" : "not tested","plain")),
      h("td","n",b.trades ?? "–"), h("td","n",isNum(b.win) ? fmt(b.win,1) + "%" : "–"), h("td","n",fmt(b.pf)), h("td","n " + tone(b.net), rs(b.net)),
      h("td","n",isNum(b.max_dd_pct) ? fmt(b.max_dd_pct,1) + "%" : "–"), h("td","n",fmt(b.sharpe)), h("td","n",fmt(b.avg_r)),
      h("td","n",b.oos_trades ?? "–"), h("td","n",fmt(b.oos_pf)), h("td","n " + tone(b.oos_net), rs(b.oos_net)), h("td","n",isNum(b.wf_positive) ? fmt(100*b.wf_positive,0) + "%" : "–"),
      h("td","n",p.trades ?? 0), h("td","n",p.trades ? fmt(100*p.wins/p.trades,0) + "%" : "–"), h("td","n b " + tone(p.net), p.trades ? rs(p.net) : "–")); });
  const job = (E.backtest || {}).job || {};
  const years = h("input"); years.type = "number"; years.step = "0.5"; years.min = "0.25"; years.max = "5"; years.value = mod === "swing" ? 3 : 2; years.style.width = "64px";
  years.className = "yrs"; years.onclick = e => e.stopPropagation();
  const run = btn(job.running ? "Backtest running…" : "Run backtest", "sm", () => post("backtest/run", {module: mod, years: Number(years.value) || undefined}).then(j => j && toast("Backtest started – real Upstox history" + (mod === "intraday" ? " (first run downloads ~1,300 requests)" : ""))));
  run.disabled = !!job.running;
  const head = h("h3", null, "Strategy performance", h("span", null, bt ? `backtest ${bt.summary.from} → ${bt.summary.to} · in-sample until ${bt.summary.in_sample_until} · run ${String(bt.created).slice(0,16).replace("T"," ")}` : "no backtest stored yet"),
    h("span","right", job.running && job.module === mod ? h("span","small muted", job.progress + " ") : job.error && job.module === mod ? h("span","small down", job.error + " ") : null, "years ", years, " ", run));
  const t = table(["Strategy","Validation",["Trades","n"],["Win","n"],["PF","n"],["Net","n"],["Max DD","n"],["Sharpe","n"],["Avg R","n"],["OOS n","n"],["OOS PF","n"],["OOS net","n"],["WF +","n"],["Paper n","n"],["Paper win","n"],["Paper net","n"]], rows);
  const notes = bt ? h("div", null, (bt.notes || []).map(n => h("p","note","• " + n)),
    h("p","note", `Validated = profitable in-sample AND out-of-sample (≥ 20 trades, profit factor ≥ 1.1 each) and in at least half of the walk-forward folds. ${E.require_validation ? "LIVE orders are proposed for validated strategies only." : ""} ${E.paper_unvalidated ? "PAPER mode also forward-tests unvalidated strategies." : "PAPER mode trades validated strategies only."}`),
    bt.walk_forward ? det(`eq-wf-${mod}`, "Walk-forward folds (trade only what looked good before each fold)", table(["Test window",["Strategies selected","n"],["Trades","n"],["Win","n"],["PF","n"],["Net","n"]],
      bt.walk_forward.map(w => h("tr", null, h("td", null, `${w.test_from} → ${w.test_to}`), h("td","n",w.selected), h("td","n",w.trades), h("td","n",isNum(w.win_rate) ? fmt(w.win_rate,1) + "%" : "–"), h("td","n",fmt(w.profit_factor)), h("td","n " + tone(w.net), rs(w.net)))))) : null,
    bt.by_stock ? det(`eq-bs-${mod}`, "Performance by stock", table(["Stock",["Trades","n"],["Win","n"],["PF","n"],["Net","n"],"Best / worst strategy"],
      bt.by_stock.map(r => { const e = Object.entries(r.by_strategy || {}).sort((a,b) => b[1][1] - a[1][1]); const b = e[0], w = e[e.length-1];
        return h("tr", null, h("td","b",r.symbol), h("td","n",r.trades), h("td","n",isNum(r.win_rate) ? fmt(r.win_rate,1) + "%" : "–"), h("td","n",fmt(r.profit_factor)), h("td","n " + tone(r.net), rs(r.net)),
          h("td","small muted", b ? `${b[0]} ${rs(b[1][1])} (${b[1][0]})` + (w && w !== b ? ` · ${w[0]} ${rs(w[1][1])} (${w[1][0]})` : "") : "")); }))) : null)
    : h("p","note","Run the backtest to measure every strategy on real history (brokerage, taxes and slippage included) with out-of-sample and walk-forward validation.");
  return h("div","blk", head, h("div","card tbl", t), notes);
}
function alertList(mod){
  const rows = (E.alerts || []).filter(a => !a.module || a.module === mod);
  return h("div","blk", h("h3", null, "Alerts", h("span", null, "today · new signals, entries, stops, targets, exits, risk limits")),
    h("div","card alist", rows.length ? rows.map(a => h("div", null, h("time", null, hm(a.time)), tag(a.kind.replace(/_/g," "), /stop|risk/.test(a.kind) ? "neg" : /target|entry/.test(a.kind) ? "pos" : a.kind === "sell_signal" ? "SELL" : a.kind === "buy_signal" ? "BUY" : "plain"), " ", a.text))
      : h("div","empty","no alerts today")));
}

/* ---------- chart ---------- */
function chartCard(mod){
  const s = sel[mod], list = (E.intraday.scanner || []).map(r => r.symbol);
  if(!s.symbol) s.symbol = ((E[mod].positions[0] || E[mod].buy[0] || E[mod].sell[0] || {}).symbol) || list[0] || null;
  const pickSym = h("select", null, list.map(x => { const o = h("option", null, x); o.value = x; o.selected = x === s.symbol; return o; }));
  pickSym.onchange = () => pick(mod, pickSym.value);
  const seg = h("div","seg", TFS[mod].map(tf => { const b = h("button", tf === s.tf ? "on" : "", tf); b.onclick = () => { s.tf = tf; last["card-" + mod] = null; renderTab(mod); }; return b; }));
  const canvas = h("canvas"); canvas.id = "eq-canvas-" + mod;
  const wrap = h("div","chart-wrap", canvas);
  const info = h("div","legend"); info.id = "eq-legend-" + mod;
  const depth = h("div"); depth.id = "eq-depth-" + mod;
  return h("div","card chartbox", h("div","chart-head", pickSym, seg, h("span","small muted", mod === "swing" ? "Daily for the setup · weekly for the trend · 1h / 4h for the entry" : "5m for signals · 15m / 1h for the trend · 1m for the entry")), wrap, info, depth);
}
function pick(mod, symbol){ sel[mod].symbol = symbol; last["card-" + mod] = null; last["opps-" + mod] = null; renderTab(mod); scrollToChart(mod); }
function scrollToChart(mod){ const c = $("eq-canvas-" + mod); if(c){ const r = c.getBoundingClientRect(); if(r.top < 110 || r.bottom > innerHeight) c.scrollIntoView({block:"center", behavior:"smooth"}); } }
async function loadChart(mod){
  const s = sel[mod]; if(!s.symbol || tab !== mod) return;
  try{
    const r = await fetch(`/api/eq/chart/${encodeURIComponent(s.symbol)}?tf=${s.tf}`, {cache:"no-store"});
    if(!r.ok) return; const d = await r.json(); if(d.symbol !== s.symbol || d.tf !== s.tf) return;
    charts[mod] = d; drawChart(mod);
  }catch(e){}
}
function css(name){ return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function drawChart(mod, hover){
  const d = charts[mod], cv = $("eq-canvas-" + mod); if(!d || !cv) return;
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight; if(!W || !H) return;
  cv.width = W * dpr; cv.height = H * dpr; const g = cv.getContext("2d"); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
  const C = d.candles || [], ink = css("--ink"), muted = css("--faint"), line = css("--line"), up = css("--ce"), dn = css("--pe");
  g.font = "11px Segoe UI, system-ui, sans-serif";
  const legend = $("eq-legend-" + mod);
  if(!C.length){ g.fillStyle = muted; g.textAlign = "center"; g.fillText(d.note || "no candles yet", W/2, H/2); if(legend) legend.replaceChildren(); return; }
  const padR = 62, padB = 20, volH = Math.round(H * 0.16), top = 8, plotH = H - padB - volH - top - 6, plotW = W - padR;
  let lo = Infinity, hi = -Infinity, vmax = 0;
  C.forEach(c => { if(c[3] != null) lo = Math.min(lo, c[3]); if(c[2] != null) hi = Math.max(hi, c[2]); vmax = Math.max(vmax, c[5] || 0); });
  const L = d.levels; if(L) [L.entry, L.stop, L.t1, L.t2].forEach(v => { if(isNum(v) && Math.abs(v - C[C.length-1][4]) / C[C.length-1][4] < 0.12){ lo = Math.min(lo, v); hi = Math.max(hi, v); } });
  const pad = (hi - lo) * 0.05 || 1; lo -= pad; hi += pad;
  const n = C.length, bw = plotW / n, X = i => (i + 0.5) * bw, Y = p => top + (hi - p) / (hi - lo) * plotH;
  g.strokeStyle = line; g.fillStyle = muted; g.lineWidth = 1; g.textAlign = "left";
  for(let k = 0; k <= 5; k++){ const p = lo + (hi - lo) * k / 5, y = Math.round(Y(p)) + .5; g.beginPath(); g.moveTo(0,y); g.lineTo(plotW,y); g.stroke(); g.fillText(fmt(p, p < 1000 ? 2 : 1), plotW + 6, y + 3); }
  const intraday = ["1m","5m","15m","30m","1h","4h"].includes(d.tf), step = Math.max(1, Math.round(n / 8)); g.textAlign = "center";
  let prevDay = "";
  for(let i = 0; i < n; i++){ const t = new Date(C[i][0] * 1000), day = t.toLocaleDateString("en-IN",{day:"2-digit",month:"short",timeZone:"Asia/Kolkata"});
    if(intraday && day !== prevDay && i){ g.strokeStyle = line; g.setLineDash([2,3]); g.beginPath(); g.moveTo(X(i) - bw/2, top); g.lineTo(X(i) - bw/2, H - padB); g.stroke(); g.setLineDash([]); }
    if(i % step === 0) g.fillText(intraday ? (day !== prevDay ? day : t.toLocaleTimeString("en-IN",{hour:"2-digit",minute:"2-digit",hour12:false,timeZone:"Asia/Kolkata"})) : day, X(i), H - 6);
    prevDay = day; }
  for(let i = 0; i < n; i++){ const [ , o, hh, ll, c, v] = C[i]; if(o == null) continue; const col = c >= o ? up : dn, x = X(i), w = Math.max(1, Math.min(11, bw * 0.68));
    g.strokeStyle = col; g.fillStyle = col; g.beginPath(); g.moveTo(x, Y(hh)); g.lineTo(x, Y(ll)); g.stroke();
    const y1 = Y(Math.max(o,c)), y2 = Y(Math.min(o,c)); g.fillRect(x - w/2, y1, w, Math.max(1, y2 - y1));
    if(vmax > 0 && v){ g.globalAlpha = .35; const vh = v / vmax * volH; g.fillRect(x - w/2, H - padB - vh, w, vh); g.globalAlpha = 1; } }
  const items = [];
  Object.entries(d.lines || {}).forEach(([name, vals]) => { const col = LINE_COLORS[name] || muted; g.strokeStyle = col; g.lineWidth = 1.3; g.beginPath(); let pen = false;
    vals.forEach((v, i) => { if(v == null || v < lo || v > hi){ pen = false; return; } const x = X(i), y = Y(v); pen ? g.lineTo(x,y) : g.moveTo(x,y); pen = true; }); g.stroke();
    const lastV = [...vals].reverse().find(v => v != null); items.push([name, col, lastV]); });
  if(L){ [["Entry", L.entry, css("--accent")], ["SL", L.stop, dn], ["T1", L.t1, up], ["T2", L.t2, up]].forEach(([name, v, col]) => { if(!isNum(v) || v < lo || v > hi) return; const y = Math.round(Y(v)) + .5;
      g.strokeStyle = col; g.setLineDash([6,4]); g.lineWidth = 1.2; g.beginPath(); g.moveTo(0,y); g.lineTo(plotW,y); g.stroke(); g.setLineDash([]);
      g.fillStyle = col; g.fillRect(plotW + 1, y - 8, padR - 2, 16); g.fillStyle = "#fff"; g.textAlign = "left"; g.fillText(`${name} ${fmt(v, v < 1000 ? 2 : 1)}`, plotW + 4, y + 4); }); }
  const lastC = C[n-1][4]; if(isNum(lastC)){ const y = Math.round(Y(lastC)) + .5; g.fillStyle = ink; g.fillRect(plotW + 1, y - 8, padR - 2, 16); g.fillStyle = css("--bg"); g.textAlign = "left"; g.fillText(fmt(lastC, lastC < 1000 ? 2 : 1), plotW + 6, y + 4); }
  let hv = null;
  if(hover != null){ const i = Math.max(0, Math.min(n - 1, Math.floor(hover / bw))); hv = C[i]; g.strokeStyle = muted; g.setLineDash([3,3]); g.beginPath(); g.moveTo(X(i), top); g.lineTo(X(i), H - padB); g.stroke(); g.setLineDash([]); }
  if(legend){ const c = hv || C[n-1], t = new Date(c[0] * 1000).toLocaleString("en-IN",{day:"2-digit",month:"short",hour:"2-digit",minute:"2-digit",hour12:false,timeZone:"Asia/Kolkata"});
    legend.replaceChildren(h("span", null, h("b", null, `${d.symbol} ${d.tf}`), ` · ${t} · O ${fmt(c[1])} H ${fmt(c[2])} L ${fmt(c[3])} C ${fmt(c[4])} · Vol ${isNum(c[5]) ? Number(c[5]).toLocaleString("en-IN") : "n/a"}`),
      ...items.map(([name, col, v]) => { const sw = h("i"); sw.style.background = col; return h("span", null, sw, `${name} ${fmt(v)}`); }),
      L ? h("span", null, `Levels: ${L.side} · ${L.source}`) : h("span", null, "no active signal levels for this stock")); }
  cv.onmousemove = e => drawChart(mod, e.offsetX < plotW ? e.offsetX : null); cv.onmouseleave = () => drawChart(mod);
  const dp = $("eq-depth-" + mod), q = d.depth;
  if(dp){ const buy = (q && q.buy) || [], sell = (q && q.sell) || [];
    dp.replaceChildren(buy.length || sell.length ? det(`eq-depth-${mod}`, `Market depth · LTP ${fmt(d.ltp)}`, h("table","depth mini-t", h("tr", null, h("th","n","Bid qty"), h("th","n","Bid"), h("th","n","Ask"), h("th","n","Ask qty")),
      [0,1,2,3,4].map(i => h("tr", null, h("td","n",buy[i] ? Number(buy[i].qty).toLocaleString("en-IN") : ""), h("td","n up",buy[i] ? fmt(buy[i].price) : ""), h("td","n down",sell[i] ? fmt(sell[i].price) : ""), h("td","n",sell[i] ? Number(sell[i].qty).toLocaleString("en-IN") : ""))))) : h("p","note","Market depth: not available (book empty or market closed).")); }
}

/* ---------- intraday tab ---------- */
function scannerTable(){
  const sc = E.scanners || {}, rows = E.intraday.scanner || [];
  const sets = {all: rows, signals: rows.filter(r => r.signal !== "NO TRADE" || r.last_signal),
    volume: (sc.high_volume || []), momentum: (sc.momentum || []), trend: (sc.strong_trend || []),
    breakout: rows.filter(r => (sc.breakout || []).some(o => o.symbol === r.symbol && o.module === "intraday"))};
  const view = (sets[scanFilter] || rows);
  const seg = h("div","seg", [["all","All 50"],["signals","With signals"],["breakout","Breakouts"],["volume","High volume"],["momentum","Momentum"],["trend","Strong trend"]].map(([k, t]) => { const b = h("button", scanFilter === k ? "on" : "", t); b.onclick = () => { scanFilter = k; last["scan"] = null; renderTab("intraday"); }; return b; }));
  const body = view.map(r => { const ls = r.last_signal;
    const tr = h("tr","main " + (r.signal === "BUY" ? "CE" : r.signal === "SELL" ? "PE" : ""), h("td", null, h("div","b",r.symbol), h("div","small muted",r.industry || "")),
      h("td","n b",fmt(r.ltp)), h("td","n " + tone(r.day_pct), isNum(r.day_pct) ? sgn(r.day_pct) + "%" : "–"),
      h("td", null, tag(r.signal, r.signal === "NO TRADE" ? "NOTRADE" : r.signal), ls && r.signal === "NO TRADE" ? h("div","small muted", `last: ${ls.side} ${hm(ls.time)}`) : null),
      h("td","n", h("span","up",fmt(r.strength_buy,0)), " / ", h("span","down",fmt(r.strength_sell,0))),
      h("td","n",fmt(r.rsi,0)), h("td","n",fmt(r.adx,0)), h("td","n " + (r.rvol_tod >= 1.5 ? "b" : ""), isNum(r.rvol_tod) ? fmt(r.rvol_tod,1) + "x" : "n/a"),
      h("td", null, arrow(r.h15), " ", arrow(r.h60)), h("td", null, r.above_vwap == null ? "n/a" : r.above_vwap ? h("span","up","above") : h("span","down","below")),
      h("td","n small", isNum(r.bid) && isNum(r.ask) ? `${fmt(r.bid)} / ${fmt(r.ask)}` : "n/a"), h("td","reason", r.error ? "data error: " + r.error : r.reason));
    tr.onclick = () => pick("intraday", r.symbol); return tr; });
  return h("div","blk", h("h3", null, "Live NIFTY 50 scanner", h("span", null, `${E.intraday.signals_today} signals today · strength = buy / sell score of the last completed 5m bar`)),
    h("div","filters", seg, h("span","hint","Click a row for its chart")),
    h("div","card tbl", table(["Stock",["LTP","n"],["Day","n"],"Signal",["Strength B / S","n"],["RSI","n"],["ADX","n"],["Rel vol","n"],"15m 1h","VWAP",["Bid / Ask","n"],"Why"], body, "no stock matches this screen right now")));
}
function tilesIntraday(){
  const m = E.market || {}, I = E.intraday, r = E.risk || {}, open = I.positions.filter(p => p.status === "OPEN" || p.status === "EXITING").length;
  const tile = (k, v, s, cls) => h("div","tile", h("div","k",k), h("div","v " + (cls || ""), v), h("div","s",s));
  return h("div","tiles", tile("Market trend", m.trend || "–", `NIFTY ${fmt(m.ltp)} ${isNum(m.day_pct) ? sgn(m.day_pct) + "%" : ""} · 15m ${m.h15 === 1 ? "up" : m.h15 === -1 ? "down" : "flat"} · 1h ${m.h60 === 1 ? "up" : m.h60 === -1 ? "down" : "flat"}`, m.trend === "BULLISH" ? "up" : m.trend === "BEARISH" ? "down" : ""),
    tile("Breadth", `${m.advancing ?? "–"} / ${m.declining ?? "–"}`, `advancing / declining · ${m.above_vwap ?? "–"} above VWAP`),
    tile("Realised P&L", rs(I.pnl.realized), "net of charges · today", tone(I.pnl.realized)), tile("Unrealised P&L", rs(I.pnl.unrealized), `${open} open position${open === 1 ? "" : "s"}`, tone(I.pnl.unrealized)),
    tile("Open / max", `${open} / ${r.max_open_intraday}`, `entries ${I.window[0]}–${I.window[1]} · square-off ${I.window[2]}`),
    tile("Trades today", `${r.trades_today ?? 0} / ${r.daily_trade_limit}`, `risk per trade ${rs(r.risk_budget)}`));
}
function renderIntraday(root){
  const I = E.intraday;
  const part = (key, data, build) => { let node = root.querySelector(`[data-p="${key}"]`); if(!node){ node = h("div"); node.dataset.p = key; root.append(node); } if(changed(key + "-intraday", data)) fill(node, build()); };
  part("head", [E.session, I.window], () => h("div","sec", h("h2", null, "Intraday cash"), h("span","sub", "NIFTY 50 · NSE cash · no options, futures or leverage · signals on completed 5-minute bars"), h("span","right", `session ${E.session}`)));
  part("bar", [E.mode, E.market_open, E.status, E.status_text, E.live, E.risk.kill_switch, E.last_scan], () => controlBar("intraday"));
  part("ban", [E.status, E.status_text, E.risk.blocked, E.risk.kill_file, E.live, E.orders.pending, E.mode, E.data_through], () => banners());
  part("tiles", [E.market, I.pnl, I.positions.length, E.risk], () => tilesIntraday());
  if(changed("opps-intraday", [I.buy, I.sell, sel.intraday.symbol])){
    let node = root.querySelector('[data-p="opps"]'); if(!node){ node = h("div"); node.dataset.p = "opps"; root.append(node); }
    fill(node, h("div","cols blk", h("div", null, h("h3", null, "Top BUY opportunities ", h("span","muted small", `${I.buy.length}`)), oppList(I.buy, "intraday", E.market_open ? "no BUY signal on the completed bars yet" : "no BUY signals in the last session")),
      h("div", null, h("h3", null, "Top SELL opportunities ", h("span","muted small", `${I.sell.length} · intraday short, squared off the same day`)), oppList(I.sell, "intraday", E.market_open ? "no SELL signal on the completed bars yet" : "no SELL signals in the last session"))));
  } else if(!root.querySelector('[data-p="opps"]')){ const node = h("div"); node.dataset.p = "opps"; root.append(node); }
  part("card", [sel.intraday, (I.scanner || []).length], () => h("div","cols wide blk", chartCard("intraday"), riskCard()));
  part("pos", I.positions, () => h("div","blk", h("h3", null, "Open positions", h("span", null, "stop → exit · target 1 → book half, stop to cost · target 2 → exit · square-off " + I.window[2])), h("div","card tbl", posTable(I.positions, "intraday"))));
  part("closed", I.closed, () => h("div","blk", h("h3", null, "Closed today"), h("div","card tbl", posTable(I.closed, "intraday", true))));
  part("scan", [I.scanner, scanFilter, E.scanners.high_volume, E.scanners.strong_trend], () => scannerTable());
  part("strat", [I.strategies, E.backtest.job, (E.backtest.intraday || {}).created], () => stratTable("intraday"));
  part("alerts", E.alerts, () => alertList("intraday"));
  part("foot", 1, () => h("p","note", "Signals are generated from real Upstox candles only. Strength is a heuristic score, not a win probability. Nothing here is investment advice; no return is guaranteed."));
}

/* ---------- swing tab ---------- */
function tilesSwing(){
  const S = E.swing, m = E.market || {}, r = E.risk || {}, open = S.positions.filter(p => p.status === "OPEN" || p.status === "EXITING"), pend = S.positions.length - open.length;
  const tile = (k, v, s, cls) => h("div","tile", h("div","k",k), h("div","v " + (cls || ""), v), h("div","s",s));
  const deployed = open.reduce((a, p) => a + (p.value || 0), 0);
  return h("div","tiles", tile("Signal date", S.signal_date ? S.signal_date.slice(5) : "–", "last completed daily bar · entries next session"),
    tile("NIFTY daily trend", m.daily_trend === 1 ? "UP" : m.daily_trend === -1 ? "DOWN" : "SIDEWAYS", `NIFTY ${fmt(m.ltp)} ${isNum(m.day_pct) ? sgn(m.day_pct) + "%" : ""}`, m.daily_trend === 1 ? "up" : m.daily_trend === -1 ? "down" : ""),
    tile("Holdings / max", `${open.length} / ${r.max_open_swing}`, `${pend} waiting for the entry zone`), tile("Deployed", rs(deployed), `of ${rs(r.capital)} capital`),
    tile("Unrealised P&L", rs(S.pnl.unrealized), "open swing positions", tone(S.pnl.unrealized)), tile("Realised today", rs(S.pnl.realized), "net of charges", tone(S.pnl.realized)));
}
function sectorTable(){
  const rows = (E.swing.sectors || []).map(s => h("tr", null, h("td", null, h("div","b",s.sector), h("div","small muted",(s.members || []).join(" · "))), h("td","n",s.stocks),
    h("td","n " + tone(s.day), isNum(s.day) ? sgn(s.day) + "%" : "n/a"), h("td","n " + tone(s.roc5), isNum(s.roc5) ? sgn(s.roc5) + "%" : "n/a"), h("td","n " + tone(s.roc20), isNum(s.roc20) ? sgn(s.roc20) + "%" : "n/a"),
    h("td","n b " + tone(s.rs20), isNum(s.rs20) ? sgn(s.rs20) + "%" : "n/a"), h("td","n",isNum(s.rank) ? fmt(s.rank,0) : "–")));
  return table(["Sector",["Stocks","n"],["Today","n"],["5 days","n"],["20 days","n"],["RS vs NIFTY (20d)","n"],["Rank","n"]], rows, "sector data appears after the first scan");
}
function watchTable(){
  const rows = (E.swing.watchlist || []).map(w => { const tr = h("tr","main", h("td","b",w.symbol), h("td","n",fmt(w.close)), h("td","n b",fmt(w.confidence_buy,0)), h("td","n",isNum(w.rs_rank) ? fmt(w.rs_rank,0) : "n/a"),
      h("td","n " + tone(w.roc5), isNum(w.roc5) ? sgn(w.roc5,1) + "%" : "n/a"), h("td","n " + tone(w.roc20), isNum(w.roc20) ? sgn(w.roc20,1) + "%" : "n/a"), h("td","n",fmt(w.rsi,0)), h("td","n",fmt(w.adx,0)),
      h("td","n",isNum(w.vol_ratio) ? fmt(w.vol_ratio,1) + "x" : "n/a"), h("td", null, arrow(w.wk_trend), " ", arrow(w.trend), " ", arrow(w.htf ? w.htf.h4 : null), " ", arrow(w.htf ? w.htf.h1 : null)),
      h("td", null, w.struct === 1 ? "HH / HL" : w.struct === -1 ? "LH / LL" : "mixed")); tr.onclick = () => pick("swing", w.symbol); return tr; });
  return table(["Stock",["Close","n"],["Confidence","n"],["RS rank","n"],["5d","n"],["20d","n"],["RSI","n"],["ADX","n"],["Volume","n"],"W · D · 4h · 1h","Structure"], rows, "the watchlist appears after the first scan");
}
function renderSwing(root){
  const S = E.swing;
  const part = (key, data, build) => { let node = root.querySelector(`[data-p="${key}"]`); if(!node){ node = h("div"); node.dataset.p = key; root.append(node); } if(changed(key + "-swing", data)) fill(node, build()); };
  part("head", [S.signal_date, S.hold], () => h("div","sec", h("h2", null, "Swing trading"), h("span","sub", `NIFTY 50 delivery · hold ${S.hold[0]}–${S.hold[1]} sessions · signals on completed daily bars, weekly trend, 1h / 4h entry timing`), h("span","right", S.signal_date ? `signals of ${S.signal_date}` : "")));
  part("bar", [E.mode, E.market_open, E.status, E.status_text, E.live, E.risk.kill_switch, E.last_scan], () => controlBar("swing"));
  part("ban", [E.status, E.status_text, E.risk.blocked, E.risk.kill_file, E.live, E.orders.pending, E.mode, E.data_through], () => banners());
  part("tiles", [E.market, S.pnl, S.positions, E.risk], () => tilesSwing());
  if(changed("opps-swing", [S.buy, S.sell, sel.swing.symbol])){
    let node = root.querySelector('[data-p="opps"]'); if(!node){ node = h("div"); node.dataset.p = "opps"; root.append(node); }
    fill(node, h("div","cols blk", h("div", null, h("h3", null, "Top swing BUY opportunities ", h("span","muted small", `${S.buy.length}`)), oppList(S.buy, "swing", "no swing BUY setup on the last completed session")),
      h("div", null, h("h3", null, "Swing SELL / exit alerts ", h("span","muted small", `${S.sell.length} · exit or avoid – the cash market has no carry-forward short`)), oppList(S.sell, "swing", "no bearish setup on the last completed session"))));
  } else if(!root.querySelector('[data-p="opps"]')){ const node = h("div"); node.dataset.p = "opps"; root.append(node); }
  part("pos", S.positions, () => h("div","blk", h("h3", null, "Portfolio holding tracker", h("span", null, "stop-loss monitoring · target tracking · sessions held · pending entries")), h("div","card tbl", posTable(S.positions, "swing"))));
  part("card", [sel.swing, (E.intraday.scanner || []).length], () => h("div","cols wide blk", chartCard("swing"), riskCard()));
  part("two", [S.sectors, S.watchlist], () => h("div","cols blk", h("div", null, h("h3", null, "Sector performance"), h("div","card tbl", sectorTable())), h("div", null, h("h3", null, "Daily swing watchlist ", h("span","muted small","top 20 by confidence")), h("div","card tbl", watchTable()))));
  part("recent", S.recent, () => h("div","blk", h("h3", null, "Earlier signals", h("span", null, "previous sessions")), h("div","card tbl", table(["Date","Stock","Side","Setup",["Close","n"],["Stop","n"],["T1","n"],["T2","n"],["Confidence","n"],"Status"],
    (S.recent || []).slice().reverse().map(s => h("tr", null, h("td","small",s.day), h("td","b",s.symbol), h("td", null, tag(s.side,s.side)), h("td","small",s.strategy_name), h("td","n",fmt(s.price)), h("td","n",fmt(s.stop)), h("td","n",fmt(s.t1)), h("td","n",fmt(s.t2)), h("td","n",fmt(s.score,0)), h("td","small",s.status + (s.note ? " · " + s.note : "")))), "no earlier signals stored"))));
  part("closed", S.closed, () => h("div","blk", h("h3", null, "Closed swing trades", h("span", null, "last 14 days")), h("div","card tbl", posTable(S.closed, "swing", true))));
  part("strat", [S.strategies, E.backtest.job, (E.backtest.swing || {}).created], () => stratTable("swing"));
  part("alerts", E.alerts, () => alertList("swing"));
  part("foot", 1, () => h("p","note","Confidence is a transparent heuristic score (components in each card), not a statistically proven win probability. Overnight gaps can exceed the stop-loss. Not investment advice; no return is guaranteed."));
}

/* ---------- wiring ---------- */
function offlineBox(title){
  return h("div", null, h("div","sec", h("h2", null, title)), h("div","card offline", h("div","badge","!"),
    h("div", null, h("div","b","The equity service is not answering"), h("div","muted small", `${eqErr}. The desk starts it automatically (python -m src.equity); the first scan after a start takes about a minute.`))));
}
function renderTab(mod){
  const root = $(mod); if(!root) return;
  if(!E){ if(changed("offline-" + mod, eqErr)) fill(root, offlineBox(mod === "swing" ? "Swing trading" : "Intraday cash")); return; }
  if(last["offline-" + mod]){ last["offline-" + mod] = null; root.replaceChildren(); Object.keys(last).forEach(k => { if(k.endsWith("-" + mod)) last[k] = null; }); }
  const before = charts[mod] ? `${sel[mod].symbol}|${sel[mod].tf}` : null;
  (mod === "swing" ? renderSwing : renderIntraday)(root);
  const now = `${sel[mod].symbol}|${sel[mod].tf}`;
  if(!$("eq-canvas-" + mod)) return;
  if(before !== now || !charts[mod] || charts[mod].symbol !== sel[mod].symbol || charts[mod].tf !== sel[mod].tf){ charts[mod] = null; loadChart(mod); } else drawChart(mod);
}
function renderOverview(){
  const box = $("eq-tiles"); if(!box) return;
  const badge = (id, n, hot) => { const e = $(id); if(e){ e.textContent = n; e.className = hot ? "hot" : ""; } };
  if(!E){ badge("n-intraday","–"); badge("n-swing","–"); if(changed("ov", eqErr)) fill(box, h("div","tile", h("div","k","NIFTY 50 cash equity"), h("div","v","–"), h("div","s","equity service not answering yet"))); return; }
  const I = E.intraday, S = E.swing, fi = I.buy.concat(I.sell).filter(o => o.fresh).length;
  badge("n-intraday", E.market_open ? fi : I.buy.length + I.sell.length, fi > 0); badge("n-swing", S.buy.length, S.buy.some(o => o.fresh));
  if(!changed("ov", [E.mode, E.market, I.pnl, S.pnl, I.buy.length, I.sell.length, S.buy.length, S.sell.length, I.positions.length, S.positions.length, E.risk.kill_switch])) return;
  const tile = (k, v, s, cls, go) => { const t = h("div","tile", h("div","k",k), h("div","v " + (cls || ""), v), h("div","s",s)); if(go){ t.style.cursor = "pointer"; t.onclick = () => { location.hash = "#" + go; }; } return t; };
  fill(box, tile("Equity mode", E.mode.toUpperCase(), E.risk.kill_switch ? "KILL SWITCH ON" : `market ${(E.market.trend || "–").toLowerCase()} · risk ${rs(E.risk.risk_budget)} / trade`, E.mode === "live" ? "down" : ""),
    tile("Intraday signals", `${I.buy.length} buy · ${I.sell.length} sell`, `${I.positions.length} open · P&L ${rs(I.pnl.total)}`, "", "intraday"),
    tile("Intraday P&L", rs(I.pnl.total), `realised ${rs(I.pnl.realized)} · unrealised ${rs(I.pnl.unrealized)}`, tone(I.pnl.total), "intraday"),
    tile("Swing setups", `${S.buy.length} buy · ${S.sell.length} exit`, `signals of ${S.signal_date || "–"}`, "", "swing"),
    tile("Swing holdings", S.positions.length, `P&L ${rs(S.pnl.total)}`, tone(S.pnl.total), "swing"));
}
async function loadEq(){
  try{
    const r = await fetch("/api/eq/state", {cache:"no-store"});
    const j = await r.json().catch(() => null);
    if(!r.ok || !j || j.offline){ E = null; eqErr = (j && j.error) || ("HTTP " + r.status); }
    else { E = j; eqErr = ""; }
  }catch(e){ E = null; eqErr = "the desk server is not answering"; }
  renderOverview();
  if(tab === "intraday" || tab === "swing") renderTab(tab);
}
let ticks = 0;
setInterval(() => { ticks++; if(document.hidden) return; const fast = tab === "intraday" || tab === "swing";
  if(fast ? ticks % 5 === 0 : ticks % 20 === 0) loadEq();
  if(fast && ticks % 20 === 0) loadChart(tab); }, 1000);
document.addEventListener("visibilitychange", () => { if(!document.hidden) loadEq(); });
addEventListener("resize", () => { if(tab === "intraday" || tab === "swing") drawChart(tab); });
addEventListener("desk-tab", e => { tab = e.detail; if(tab === "intraday" || tab === "swing"){ renderTab(tab); loadEq(); } });
tab = (document.querySelector("main > section.show") || {id:"overview"}).id;
loadEq();
})();
