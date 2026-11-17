"""
HTML dashboard for the NIFTY / SENSEX scanner. Served live by the built-in dashboard server
(http://127.0.0.1:8502, src/dashboard/app.py) and also written to output/prebreakout_dashboard.html every
cycle. Refreshes every 30 s. No extra packages needed.
"""
from __future__ import annotations

import html
from typing import Optional

STATE_STYLE = {
    "ENTRY SETUP": ("entry", "ENTRY SETUP"),
    "CONFIRMED": ("confirmed", "CONFIRMED BREAKOUT"),
    "WATCH": ("watch", "PRE-BREAKOUT WATCH"),
    "INVALIDATED": ("invalid", "INVALIDATED"),
    "MISSED": ("invalid", "MISSED - DO NOT CHASE"),
    "NO TRADE": ("notrade", "NO TRADE"),
    "IDLE": ("idle", "NO SETUP"),
}

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#14171c;--muted:#5d6672;--line:#e2e5ea;--ce:#0f8a4a;--pe:#c22b2b;
--watch:#b7791f;--watchbg:#fff6e0;--conf:#1d5fd1;--confbg:#e6efff;--entry:#0f8a4a;--entrybg:#e3f6ea;
--inv:#6b7280;--invbg:#eef0f3;--nt:#9b1c1c;--ntbg:#fde8e8;--bar:#e9ecf0}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0f1216;--card:#171b21;--ink:#e8ebef;
--muted:#9aa4b1;--line:#2a3039;--watchbg:#3a2e12;--confbg:#15264a;--entrybg:#11331f;--invbg:#252a31;--ntbg:#3d1717;--bar:#252b33}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "Segoe UI",system-ui,sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;gap:18px;flex-wrap:wrap;align-items:center}
h1{font-size:18px;margin:0}main{padding:16px 20px;max-width:1500px;margin:0 auto}
.pill{border-radius:999px;padding:2px 10px;font-size:12px;font-weight:600;border:1px solid var(--line)}
.mock{background:#7c3aed;color:#fff;border:none}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.card h2{font-size:16px;margin:0 0 4px;display:flex;justify-content:space-between;align-items:baseline}
.price{font-size:22px;font-weight:650}.muted{color:var(--muted);font-size:12px}
.state{margin:8px 0;padding:8px 10px;border-radius:8px;font-weight:650;font-size:13px}
.state small{display:block;font-weight:400;color:var(--muted);margin-top:2px}
.s-watch{background:var(--watchbg);color:var(--watch)}.s-confirmed{background:var(--confbg);color:var(--conf)}
.s-entry{background:var(--entrybg);color:var(--entry)}.s-invalid{background:var(--invbg);color:var(--inv)}
.s-notrade{background:var(--ntbg);color:var(--nt)}.s-idle{background:var(--invbg);color:var(--muted)}
.scores{display:grid;grid-template-columns:34px 1fr 40px;gap:4px 8px;align-items:center;margin:6px 0}
.track{height:9px;background:var(--bar);border-radius:6px;overflow:hidden}.fill{height:100%}
.ce{color:var(--ce)}.pe{color:var(--pe)}.fce{background:var(--ce)}.fpe{background:var(--pe)}
table{border-collapse:collapse;width:100%;font-size:12.5px}td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}.num{text-align:right;font-variant-numeric:tabular-nums}
details{margin-top:6px}summary{cursor:pointer;color:var(--muted);font-size:12px}
.k{display:grid;grid-template-columns:repeat(4,1fr);gap:4px 10px;font-size:12px;margin-top:6px}.k div span{color:var(--muted)}
.wide{grid-column:1/-1}.tag{font-size:11px;padding:1px 6px;border-radius:5px;background:var(--bar);margin-right:3px;display:inline-block;margin-bottom:2px}
.tag.pos{color:var(--ce)}.tag.neg{color:var(--pe)}.overflow{overflow-x:auto}
@media (max-width:600px){main{padding:12px 16px}.k{grid-template-columns:repeat(2,1fr)}}
"""


def _e(v) -> str:
    return html.escape("" if v is None else str(v))


def _n(v, d=1) -> str:
    try:
        return f"{float(v):,.{d}f}"
    except (TypeError, ValueError):
        return "-"


def _score_rows(st) -> str:
    out = []
    for side in ("ce", "pe"):
        s = st.get(side) or {}
        score = s.get("score") or 0
        out.append(f'<span class="{side}"><b>{side.upper()}</b></span><div class="track"><div class="fill f{side}" '
                   f'style="width:{max(0, min(100, score)):.0f}%"></div></div><span class="num">{_n(score, 0)}</span>')
    return '<div class="scores">' + "".join(out) + "</div>"


def _v2_block(st) -> str:
    v2 = st.get("v2")
    if not v2:
        return ""
    ce, pe = v2.get("ce") or {}, v2.get("pe") or {}
    cls, label = STATE_STYLE.get(v2.get("state"), ("idle", v2.get("state") or "-"))
    return (f"<div class='muted' style='margin-top:6px'>Score v2 (5m + 1m, shadow): CE <b class='ce'>{_n(ce.get('score'), 0)}</b>"
            f" · PE <b class='pe'>{_n(pe.get('score'), 0)}</b> · {_e(label)}</div>"
            f"<details><summary>v2 CE breakdown</summary>{_breakdown(ce)}</details>"
            f"<details><summary>v2 PE breakdown</summary>{_breakdown(pe)}</details>")


def _v3_block(st) -> str:
    """v3: regime, CE/PE scores with status, the 7 strategy modules, traps and the derivatives snapshot."""
    v3 = st.get("v3")
    if not v3:
        return ""
    ce, pe = v3.get("ce") or {}, v3.get("pe") or {}
    reg = v3.get("regime") or {}
    cls, label = STATE_STYLE.get(v3.get("state"), ("idle", v3.get("state") or "-"))
    mods = v3.get("modules") or {}
    rows = ""
    for m in mods.values():
        if not m.get("available", True):
            rows += f"<tr><td>{_e(m.get('name'))}</td><td colspan='3' class='muted'>{_e('; '.join(m.get('notes') or []) or 'no data')}</td></tr>"
            continue
        mce, mpe = m.get("ce") or {}, m.get("pe") or {}
        d = m.get("direction")
        rows += (f"<tr><td>{_e(m.get('name'))}</td><td class='num ce'>{_n((mce.get('confidence') or 0) * 100, 0)}</td>"
                 f"<td class='num pe'>{_n((mpe.get('confidence') or 0) * 100, 0)}</td>"
                 f"<td class='{'ce' if d == 'CE' else 'pe' if d == 'PE' else 'muted'}'>{_e(d)}</td></tr>")
    mod_table = (f"<table><tr><th>Module</th><th class='num'>CE</th><th class='num'>PE</th><th>Lean</th></tr>{rows}</table>"
                 if rows else "")

    def traps(side):
        t = side.get("traps") or []
        return "".join(f'<span class="tag neg">{_e(x.get("name"))} -{_n(x.get("points"), 0)}</span>' for x in t)

    dv = v3.get("derivatives") or {}
    deriv = ("Derivatives: " + (f"PCR(OI) {_n(dv.get('pcr_oi'), 2)} · call wall {_n(dv.get('call_wall'), 0)} · put wall "
                                f"{_n(dv.get('put_wall'), 0)} · ATM IV {_n(dv.get('atm_iv'), 1)}"
                                + (f" ({_n(dv.get('atm_iv_change_pct'), 1)}% / 30m)" if dv.get('atm_iv_change_pct') is not None else "")
                                if dv.get("available") else _e(dv.get("note") or "no chain")))
    veto = "; ".join(x for x in (ce.get("veto"), pe.get("veto")) if x)
    return (f"<div class='muted' style='margin-top:6px'>Regime: <b>{_e(reg.get('name') or '-')}</b>"
            f" ({_e(', '.join((reg.get('evidence') or [])[:2]))})</div>"
            f"<div class='muted'>Score v3 (multi-factor, shadow): CE <b class='ce'>{_n(ce.get('score'), 0)}</b> "
            f"[{_e(ce.get('status') or '-')}] · PE <b class='pe'>{_n(pe.get('score'), 0)}</b> [{_e(pe.get('status') or '-')}]"
            f" · {_e(label)}{(' · <i>' + _e(veto) + '</i>') if veto else ''}</div>"
            f"<div>{traps(ce)}{traps(pe)}</div><div class='muted'>{deriv}</div>"
            f"<details><summary>v3 strategy modules (confidence 0-100, not probabilities)</summary>{mod_table}</details>"
            f"<details><summary>v3 CE breakdown</summary>{_breakdown(ce)}</details>"
            f"<details><summary>v3 PE breakdown</summary>{_breakdown(pe)}</details>")


def _execution(ex: dict) -> str:
    if not ex or ex.get("mode") == "off":
        return "<p class='muted'>Order execution: off ([EXECUTION] mode = off) - paper book only.</p>"
    head = ("<b class='pe'>LIVE ORDERS ARMED</b>" if ex.get("armed") else
            f"simulated orders (paper broker){(' - ' + _e(ex.get('gate'))) if ex.get('gate') else ''}")
    pos_rows = "".join(f"<tr><td>{_e(p.get('symbol'))}</td><td class='num'>{_e(p.get('qty'))}/{_e(p.get('planned_qty'))}</td>"
                       f"<td class='num'>{_n(p.get('entry_avg'), 2)}</td><td class='num'>{_n(p.get('sl_trigger'), 2)}</td>"
                       f"<td class='num'>{_n(p.get('realised'), 0)}</td><td>{'closed' if p.get('closed') else 'open'}"
                       f"{(' - ' + _e(p.get('note'))) if p.get('note') else ''}</td></tr>" for p in ex.get("positions") or [])
    ord_rows = "".join(f"<tr><td>{_e(str(o.get('created', ''))[11:19])}</td><td>{_e(o.get('purpose'))}</td>"
                       f"<td>{_e((o.get('req') or {}).get('side'))} {_e((o.get('req') or {}).get('quantity'))} "
                       f"{_e((o.get('req') or {}).get('order_type'))} {_n((o.get('req') or {}).get('price'), 2)}</td>"
                       f"<td>{_e(o.get('status'))} {_e(o.get('filled'))}@{_n(o.get('avg_price'), 2)}</td>"
                       f"<td>{_e(o.get('message'))}</td></tr>" for o in (ex.get("recent") or [])[::-1])
    rej = "".join(f"<li>{_e(x)}</li>" for x in ex.get("rejections") or [])
    return (f"<p>Mode <b>{_e(ex.get('mode'))}</b> · {head} · lane {_e(ex.get('lane'))} · broker {_e(ex.get('broker'))} · "
            f"entry orders today {_e(ex.get('orders_today'))} · last reconcile {_e(str(ex.get('last_reconcile') or '-')[11:19])}"
            f"{(' · <b class=pe>BLOCKED: ' + _e(ex.get('blocked')) + '</b>') if ex.get('blocked') else ''}</p>"
            + (f"<div class='overflow'><table><tr><th>Contract</th><th class='num'>Qty</th><th class='num'>Avg</th>"
               f"<th class='num'>Stop</th><th class='num'>Realised</th><th>State</th></tr>{pos_rows}</table></div>" if pos_rows else "")
            + (f"<details><summary>Orders</summary><div class='overflow'><table><tr><th>Time</th><th>Purpose</th><th>Order</th>"
               f"<th>Status</th><th>Note</th></tr>{ord_rows}</table></div></details>" if ord_rows else "")
            + (f"<details><summary>Not sent / rejected</summary><ul>{rej}</ul></details>" if rej else ""))


def _ml_line(ml: dict) -> str:
    if not ml or not ml.get("loaded"):
        return _e((ml or {}).get("note") or "ML filter: no model yet (python -m src.scanner.ml_filter --train).")
    if ml.get("approved"):
        return (f"ML filter: approved, blocks entries below {_n((ml.get('threshold') or 0) * 100, 0)}% "
                f"(walk-forward AUC {_n(ml.get('auc'), 2)}, trained {_e(str(ml.get('trained_at'))[:10])}).")
    return (f"ML filter: advisory only, it does not block - on unseen data it was not better than chance yet "
            f"(AUC {_n(ml.get('auc'), 2)}, {_e(ml.get('samples'))} setups). It keeps learning from live setups.")


def _breakdown(s: Optional[dict]) -> str:
    if not s:
        return ""
    rows = "".join(f"<tr><td>{_e(b['family'])}{'' if b.get('available', True) else ' (n/a)'}</td>"
                   f"<td class='num'>{_n(b['points'], 1)} / {_n(b['max'], 0)}</td><td>{_e('; '.join(b.get('notes') or []))}</td></tr>"
                   for b in s.get("breakdown", []))
    sup = "".join(f'<span class="tag pos">{_e(x)}</span>' for x in s.get("supporting", []))
    con = "".join(f'<span class="tag neg">{_e(x)}</span>' for x in s.get("contradicting", []))
    return f"<table>{rows}</table><div style='margin-top:4px'>{sup}{con}</div>"


def _shortlist(rows) -> str:
    if not rows:
        return '<p class="muted">No option shortlist (no active setup, or chain unavailable).</p>'
    body = ""
    for c in rows[:5]:
        rej = "; ".join(c.get("rejected") or [])
        body += (f"<tr><td>{_e(c.get('symbol'))}</td><td class='num'>{_n(c.get('bid'))}/{_n(c.get('ask'))}</td>"
                 f"<td class='num'>{_n(c.get('spread_pct'))}%</td><td class='num'>{_n(c.get('delta'), 2)}</td>"
                 f"<td class='num'>{_n(c.get('rr'), 2)}</td><td>{_e(rej) if rej else 'OK'}</td></tr>")
    return ("<div class='overflow'><table><tr><th>Contract</th><th class='num'>Bid/Ask</th>"
            "<th class='num'>Spr</th><th class='num'>Δ</th>"
            f"<th class='num'>R:R</th><th>Filters</th></tr>{body}</table></div>")


def _index_card(st: dict) -> str:
    cls, label = STATE_STYLE.get(st.get("state"), ("idle", st.get("state") or "-"))
    lv = st.get("levels") or {}
    ind = st.get("indicators") or {}
    feed = st.get("feed") or {}
    chg = st.get("change_pct")
    chg_html = f'<span class="{"ce" if (chg or 0) >= 0 else "pe"}">{_n(chg, 2)}%</span>' if chg is not None else ""
    bias = st.get("bias") or "NEUTRAL"
    bias_cls = "ce" if bias == "CE" else "pe" if bias == "PE" else "muted"
    ce, pe = st.get("ce") or {}, st.get("pe") or {}

    def dist(s):
        if not s or s.get("level_price") is None:
            return "no level in reach"
        return f"{_e((s.get('level') or {}).get('sources'))} {_n(s.get('level_price'), 1)} ({_n(s.get('distance_atr'), 2)} ATR)"
    return f"""<section class="card">
<h2><span>{_e(st['name'])}</span><span class="muted">bias <b class="{bias_cls}">{_e(bias)}</b></span></h2>
<div><span class="price">{_n(st.get('price'), 2)}</span> {chg_html}</div>
<div class="state s-{cls}">{_e(label)}<small>{_e(st.get('state_reason'))}</small></div>
{_score_rows(st)}
<div class="muted">CE level: {dist(ce)}<br>PE level: {dist(pe)}</div>
<div class="k"><div><span>PDH</span> {_n(lv.get('pdh'))}</div><div><span>PDL</span> {_n(lv.get('pdl'))}</div>
<div><span>ORH</span> {_n(lv.get('orh'))}</div><div><span>ORL</span> {_n(lv.get('orl'))}</div>
<div><span>PWH</span> {_n(lv.get('pwh'))}</div><div><span>PWL</span> {_n(lv.get('pwl'))}</div>
<div><span>RSI</span> {_n(ind.get('rsi'), 0)}</div><div><span>ADX</span> {_n(ind.get('adx'), 0)}</div>
<div><span>ATR5</span> {_n(ind.get('atr'), 1)}</div><div><span>Fut vol</span> {_n(ind.get('f_rvol'), 2)}x</div>
<div><span>EMA20</span> {_n(ind.get('ema20'), 0)}</div><div><span>EMA50</span> {_n(ind.get('ema50'), 0)}</div></div>
<div class="muted" style="margin-top:6px">Structure: {_e((st.get('structure') or {}).get('label'))} · Expiry {_e(st.get('expiry'))}
{' (expiry day)' if st.get('expiry_day') else ''} · lot {_e(st.get('lot_size'))} · step {_e(st.get('step'))}</div>
<div class="muted">Feed: {'<b class="ce">OK</b>' if feed.get('ok') else '<b class="pe">NOT OK</b>'} {_e(feed.get('text'))}</div>
{_v2_block(st)}
{_v3_block(st)}
<details><summary>CE score breakdown</summary>{_breakdown(ce)}</details>
<details><summary>PE score breakdown</summary>{_breakdown(pe)}</details>
<details {'open' if st.get('shortlist') else ''}><summary>Option shortlist {_e(st.get('shortlist_note') or '')}</summary>{_shortlist(st.get('shortlist'))}</details>
</section>"""


def _alerts_table(alerts) -> str:
    if not alerts:
        return '<p class="muted">No alerts yet today.</p>'
    rows = ""
    for a in alerts[:40]:
        kind = a.get("kind", "")
        cls = {"entry": "s-entry", "confirmed": "s-confirmed", "high_watch": "s-watch", "strong_watch": "s-watch",
               "early_watch": "s-watch", "invalidated": "s-invalid", "missed": "s-invalid", "data": "s-notrade"}.get(kind, "")
        opt = a.get("option") or {}
        rows += (f"<tr><td>{_e(str(a.get('time', ''))[11:19])}</td><td>{_e(a.get('index'))}</td>"
                 f"<td class='{'ce' if a.get('direction') == 'CE' else 'pe'}'>{_e(a.get('direction'))}</td>"
                 f"<td><span class='pill {cls}'>{_e(a.get('stage'))}</span></td><td class='num'>{_n(a.get('score'), 0)}</td>"
                 f"<td class='num'>{_n(a.get('price'), 1)}</td><td class='num'>{_n(a.get('level'), 1)}</td>"
                 f"<td>{_e(a.get('message'))}{(' · ' + _e(opt.get('symbol'))) if opt.get('symbol') else ''}"
                 f"{(' · <i>' + _e(a.get('avoid_reason')) + '</i>') if a.get('avoid_reason') else ''}</td></tr>")
    return ("<div class='overflow'><table><tr><th>Time</th><th>Index</th><th>Dir</th><th>Stage</th><th class='num'>Score</th>"
            f"<th class='num'>Price</th><th class='num'>Level</th><th>Detail</th></tr>{rows}</table></div>")


def _paper(p: dict) -> str:
    s = p.get("summary") or {}
    rows = ""
    for t in (p.get("today") or [])[::-1]:
        rows += (f"<tr><td>{_e(str(t.get('entry_time', ''))[11:16])}</td><td>{_e(t.get('symbol'))}</td>"
                 f"<td class='num'>{_e(t.get('lots'))}x{_e(t.get('lot_size'))}</td><td class='num'>{_n(t.get('entry_price'), 2)}</td>"
                 f"<td class='num'>{_n(t.get('last_price'), 2)}</td><td class='num'>{_n(t.get('sl'), 1)} / {_n(t.get('t1'), 1)} / {_n(t.get('t2'), 1)}</td>"
                 f"<td class='num {'ce' if (t.get('net_pnl') or 0) >= 0 else 'pe'}'>{_n(t.get('net_pnl'), 0)}</td>"
                 f"<td>{_e(t.get('exit_reason') or 'open')}</td></tr>")
    table = ("<div class='overflow'><table><tr><th>In</th><th>Contract</th><th class='num'>Qty</th><th class='num'>Entry</th>"
             "<th class='num'>Last</th><th class='num'>SL / T1 / T2</th><th class='num'>Net Rs</th><th>Status</th></tr>"
             f"{rows}</table></div>") if rows else '<p class="muted">No paper trades today.</p>'
    return (f"<p>All paper trades: <b>{_e(s.get('trades'))}</b> · net Rs <b>{_n(s.get('net'), 0)}</b> · win "
            f"{_n(s.get('win_rate'), 1)}% · PF {_n(s.get('profit_factor'), 2)} (net of charges)</p>{table}")


def _ws_line(ws) -> str:
    if not ws:
        return "WebSocket: off (REST quotes only)"
    word = "connected" if ws.get("connected") else "not connected"
    extra = f" ({_e(ws.get('last_error'))})" if ws.get("last_error") and not ws.get("connected") else ""
    return f"WebSocket: {word}, {_e(ws.get('subscribed'))} instruments, reconnects {_e(ws.get('reconnects'))}{extra}"


def render_html(state: dict) -> str:
    r = state.get("risk") or {}
    mock = '<span class="pill mock">MOCK DATA</span>' if state.get("mock") else '<span class="pill">LIVE DATA</span>'
    risk_cls = "pe" if r.get("halted") else "ce"
    cards = "".join(_index_card(st) for st in (state.get("indices") or {}).values())
    th = state.get("thresholds") or {}
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="30"><title>NIFTY SENSEX Scanner</title><style>{CSS}</style></head><body>
<header><h1>NIFTY &amp; SENSEX pre-breakout CE / PE</h1>{mock}<span class="pill">{_e((state.get('trading_mode') or 'paper').upper())}</span><span class="pill">{_e(state.get('status'))}</span>
<span class="muted">updated {_e(str(state.get('generated_at', ''))[11:19])} IST · India VIX {_n(state.get('vix'), 2)}
({_n(state.get('vix_change_pct'), 1)}%)</span>
<span class="muted">Risk: day P&amp;L Rs <b class="{risk_cls}">{_n(r.get('day_pnl'), 0)}</b> of -{_n(r.get('limit'), 0)} ·
trades {_e(r.get('trades_total'))} · open {_e(r.get('open_positions'))} · {'<b class="pe">HALTED: ' + _e(r.get('halt_reason')) + '</b>' if r.get('halted') else 'entries allowed'}</span></header>
<main>
<p class="muted">{_e((state.get('notes') or ['Alerts and paper trading only - no orders are placed.'])[0])} Scores are setup-strength scores (heuristic), not probabilities.
WATCH = price near a level, not broken. CONFIRMED = a {_e(th.get('signal_minutes'))}-minute close beyond the level.
Thresholds: early {_e(th.get('early_watch'))} · high {_e(th.get('high_watch'))} · strong {_e(th.get('strong_watch'))} · entry ≥ {_e(th.get('entry_min_score'))}.</p>
<div class="grid">{cards}</div>
<div class="grid" style="margin-top:14px"><section class="card wide"><h2>Recent alerts</h2>{_alerts_table(state.get('recent_alerts'))}</section>
<section class="card wide"><h2>Paper trading</h2>{_paper(state.get('paper') or {})}
<p class="muted">{_ml_line(state.get('ml') or {})}</p></section>
{('<section class="card wide"><h2>Score v2 paper trades (shadow comparison)</h2>' + _paper(state.get('paper_v2') or {}) + '</section>') if state.get('paper_v2') else ''}
{('<section class="card wide"><h2>Score v3 paper trades (multi-factor, shadow comparison)</h2>' + _paper(state.get('paper_v3') or {}) + '</section>') if state.get('paper_v3') else ''}
<section class="card wide"><h2>Order execution</h2>{_execution(state.get('execution') or {})}</section></div>
<p class="muted">Channels: {_e(', '.join(state.get('channels') or []))} · {_ws_line(state.get('websocket'))} ·
Events: {_e((state.get('events') or {}).get('note') or ', '.join((state.get('events') or {}).get('upcoming') or []) or '-')}</p>
</main></body></html>"""
