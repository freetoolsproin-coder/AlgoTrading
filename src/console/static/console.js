/* Trading console: navigation drawer, theme, usage bars, SVG charts, status refresh. No external libraries. */
(function () {
  "use strict";
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  // ------------------------------------------------------------------ theme
  function store(k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } }
  var saved = store("tc-theme");
  if (saved) document.documentElement.setAttribute("data-theme", saved);
  $$("[data-theme-toggle]").forEach(function (b) {
    b.addEventListener("click", function () {
      var cur = document.documentElement.getAttribute("data-theme") ||
        (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      var next = cur === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      store("tc-theme", next);
    });
  });

  // ------------------------------------------------------------------ drawer
  var side = $(".side"), btn = $(".menu-btn");
  if (btn && side) {
    btn.addEventListener("click", function () {
      var open = side.classList.toggle("open");
      btn.setAttribute("aria-expanded", open ? "true" : "false");
      if (open) { var a = $("a", side); if (a) a.focus(); }
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && side.classList.contains("open")) { side.classList.remove("open"); btn.setAttribute("aria-expanded", "false"); btn.focus(); }
    });
  }

  // ------------------------------------------------------------------ usage bars
  $$(".fill[data-v]").forEach(function (el) {
    var v = parseFloat(el.getAttribute("data-v")) || 0, lim = parseFloat(el.getAttribute("data-limit")) || 0;
    var pct = lim > 0 ? Math.min(100, v / lim * 100) : 0;
    el.style.width = pct.toFixed(1) + "%";
    if (pct >= 100) el.classList.add("over"); else if (pct >= 70) el.classList.add("hot");
  });

  // ------------------------------------------------------------------ confirmations for consequential forms
  $$("form[data-confirm]").forEach(function (f) {
    f.addEventListener("submit", function (e) { if (!window.confirm(f.getAttribute("data-confirm"))) e.preventDefault(); });
  });

  $$("[data-print]").forEach(function (b) { b.addEventListener("click", function () { window.print(); }); });

  // ------------------------------------------------------------------ number formatting
  function inr(v) {
    if (v === null || v === undefined || isNaN(v)) return "—";
    var neg = v < 0, s = Math.round(Math.abs(v)).toString(), head = s.slice(0, -3), tail = s.slice(-3);
    if (head) head = head.replace(/\B(?=(\d{2})+(?!\d))/g, ",") + ",";
    return (neg ? "−" : "") + "₹" + head + tail;
  }
  var NS = "http://www.w3.org/2000/svg";
  function el(name, attrs, parent) {
    var n = document.createElementNS(NS, name);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function niceTicks(lo, hi, n) {
    if (lo === hi) { lo -= 1; hi += 1; }
    var span = hi - lo, step = Math.pow(10, Math.floor(Math.log10(span / n))), err = n / span * step;
    if (err <= .15) step *= 10; else if (err <= .35) step *= 5; else if (err <= .75) step *= 2;
    var out = [], v = Math.ceil(lo / step) * step;
    for (; v <= hi + 1e-9; v += step) out.push(v);
    return out;
  }
  function compact(v) {
    var a = Math.abs(v);
    if (a >= 1e7) return (v / 1e7).toFixed(1) + "Cr";
    if (a >= 1e5) return (v / 1e5).toFixed(1) + "L";
    if (a >= 1e3) return (v / 1e3).toFixed(0) + "k";
    return v.toFixed(0);
  }

  // kind: "line" (equity, cumulative) or "bars" (daily net)
  function draw(host, points, kind) {
    host.textContent = "";
    var tip = host.nextElementSibling && host.nextElementSibling.classList.contains("chart-tip") ? host.nextElementSibling : null;
    if (!points.length) { host.innerHTML = '<div class="empty"><b>Nothing to chart yet</b>Closed trades appear here as they are recorded.</div>'; return; }
    var W = host.clientWidth || 600, H = host.clientHeight || 220, L = 52, R = 10, T = 10, B = 24;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": host.getAttribute("aria-label") || "chart" }, host);
    var vals = points.map(function (p) { return p.v; });
    var lo = Math.min(0, Math.min.apply(null, vals)), hi = Math.max(0, Math.max.apply(null, vals));
    var ticks = niceTicks(lo, hi, 4); lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]);
    var x = function (i) { return L + (points.length === 1 ? (W - L - R) / 2 : i * (W - L - R) / (points.length - 1)); };
    var y = function (v) { return T + (hi - v) / (hi - lo || 1) * (H - T - B); };
    ticks.forEach(function (t) {
      el("line", { x1: L, x2: W - R, y1: y(t), y2: y(t), "class": t === 0 ? "zero" : "grid" }, svg);
      var tx = el("text", { x: L - 6, y: y(t) + 4, "text-anchor": "end" }, svg); tx.textContent = compact(t);
    });
    var first = el("text", { x: L, y: H - 6 }, svg); first.textContent = (points[0].label || "").slice(0, 10);
    var last = el("text", { x: W - R, y: H - 6, "text-anchor": "end" }, svg); last.textContent = (points[points.length - 1].label || "").slice(0, 10);
    if (kind === "bars") {
      var bw = Math.max(1, Math.min(24, (W - L - R) / points.length * .7));
      points.forEach(function (p, i) {
        var cx = points.length === 1 ? (L + W - R) / 2 : L + (i + .5) * (W - L - R) / points.length;
        var r = el("rect", { x: cx - bw / 2, width: bw, y: Math.min(y(p.v), y(0)), height: Math.max(1, Math.abs(y(p.v) - y(0))),
          "class": p.v >= 0 ? "bar-pos" : "bar-neg" }, svg);
        var t = el("title", {}, r); t.textContent = p.label + ": " + inr(p.v);
      });
    } else {
      var d = points.map(function (p, i) { return (i ? "L" : "M") + x(i).toFixed(1) + " " + y(p.v).toFixed(1); }).join(" ");
      el("path", { d: d + " L" + x(points.length - 1).toFixed(1) + " " + y(0) + " L" + x(0).toFixed(1) + " " + y(0) + "Z", "class": "area" }, svg);
      el("path", { d: d, "class": "line" }, svg);
      var cross = el("line", { y1: T, y2: H - B, "class": "axis", visibility: "hidden" }, svg);
      svg.addEventListener("mousemove", function (e) {
        var rect = svg.getBoundingClientRect(), px = (e.clientX - rect.left) / rect.width * W;
        var i = Math.max(0, Math.min(points.length - 1, Math.round((px - L) / ((W - L - R) / Math.max(1, points.length - 1)))));
        cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
        if (tip) tip.textContent = (points[i].label || "") + "  cumulative net " + inr(points[i].v) + "  (trade " + (i + 1) + ")";
      });
      svg.addEventListener("mouseleave", function () { cross.setAttribute("visibility", "hidden"); if (tip) tip.textContent = ""; });
    }
  }

  var dataEl = $("#chart-data");
  if (dataEl) {
    var data = {};
    try { data = JSON.parse(dataEl.textContent || "{}"); } catch (e) { data = {}; }
    var render = function () {
      var eq = $("#chart-equity"), dl = $("#chart-daily");
      if (eq) draw(eq, (data.equity || []).map(function (p) { return { v: p.v, label: (p.t || "").replace("T", " ").slice(0, 16) }; }), "line");
      if (dl) draw(dl, (data.daily || []).map(function (p) { return { v: p.net, label: p.day }; }), "bars");
    };
    render();
    var to; window.addEventListener("resize", function () { clearTimeout(to); to = setTimeout(render, 150); });
  }


  // ------------------------------------------------------------------ calibration chart (ML lab)
  function drawCalibration(host, rows) {
    host.textContent = "";
    if (!rows.length) return;
    var W = host.clientWidth || 500, H = host.clientHeight || 220, L = 40, R = 10, T = 10, B = 24;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": host.getAttribute("aria-label") }, host);
    var hi = Math.max.apply(null, rows.map(function (r) { return Math.max(r.predicted, r.actual); })) * 1.15 || 1;
    var y = function (v) { return T + (1 - v / hi) * (H - T - B); };
    niceTicks(0, hi, 4).forEach(function (t) {
      el("line", { x1: L, x2: W - R, y1: y(t), y2: y(t), "class": t === 0 ? "zero" : "grid" }, svg);
      var tx = el("text", { x: L - 6, y: y(t) + 4, "text-anchor": "end" }, svg); tx.textContent = Math.round(t * 100) + "%";
    });
    var slot = (W - L - R) / rows.length;
    rows.forEach(function (r, i) {
      var cx = L + (i + .5) * slot, bw = Math.min(40, slot * .5);
      var bar = el("rect", { x: cx - bw / 2, width: bw, y: y(r.actual), height: Math.max(1, y(0) - y(r.actual)), "class": "bar-act" }, svg);
      el("title", {}, bar).textContent = "actual " + (r.actual * 100).toFixed(1) + "% of " + r.n + " setups";
      var d = el("circle", { cx: cx, cy: y(r.predicted), r: 4.5, "class": "dot-pred" }, svg);
      el("title", {}, d).textContent = "predicted " + (r.predicted * 100).toFixed(1) + "%";
      var tx = el("text", { x: cx, y: H - 6, "text-anchor": "middle" }, svg); tx.textContent = "Q" + (i + 1);
    });
    var tip = host.nextElementSibling;
    if (tip && tip.classList.contains("chart-tip")) tip.textContent = "Bars: actual hit rate per predicted quintile. Dots: what the model predicted. Close = well calibrated.";
  }
  var calHost = $("#chart-calibration");
  if (calHost && dataEl) {
    var calData = {};
    try { calData = JSON.parse(dataEl.textContent || "{}"); } catch (e) { calData = {}; }
    drawCalibration(calHost, calData.calibration || []);
  }

  // ------------------------------------------------------------------ assistant: safe markdown -> DOM (no innerHTML)
  function inline(parent, text) {
    var parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
    parts.forEach(function (p) {
      if (!p) return;
      if (p.slice(0, 2) === "**" && p.slice(-2) === "**") { var b = document.createElement("strong"); b.textContent = p.slice(2, -2); parent.appendChild(b); }
      else if (p[0] === "`" && p.slice(-1) === "`") { var c = document.createElement("code"); c.textContent = p.slice(1, -1); parent.appendChild(c); }
      else parent.appendChild(document.createTextNode(p));
    });
  }
  function renderMarkdown(target, text) {
    target.textContent = "";
    var lines = String(text || "").split(/\r?\n/), list = null, table = null;
    function flush() { list = null; table = null; }
    lines.forEach(function (raw) {
      var line = raw.replace(/\s+$/, "");
      if (/^\s*\|/.test(line)) {
        var cells = line.trim().replace(/^\||\|$/g, "").split("|").map(function (c) { return c.trim(); });
        if (cells.every(function (c) { return /^:?-{2,}:?$/.test(c); })) return;
        if (!table) { var wrap = document.createElement("div"); wrap.className = "table-wrap"; table = document.createElement("table"); wrap.appendChild(table); target.appendChild(wrap); list = null;
          var tr = document.createElement("tr"); cells.forEach(function (c) { var th = document.createElement("th"); inline(th, c); tr.appendChild(th); }); var th_ = document.createElement("thead"); th_.appendChild(tr); table.appendChild(th_); table.appendChild(document.createElement("tbody")); return; }
        var row = document.createElement("tr");
        cells.forEach(function (c) { var td = document.createElement("td"); if (/^[−\-+₹\d.,%\s]+$/.test(c)) td.className = "n"; inline(td, c); row.appendChild(td); });
        table.tBodies[0].appendChild(row); return;
      }
      var m = line.match(/^\s*(?:[-*•]|\d+[.)])\s+(.*)$/);
      if (m) {
        if (!list) { list = document.createElement(/^\s*\d/.test(line) ? "ol" : "ul"); target.appendChild(list); table = null; }
        var li = document.createElement("li"); inline(li, m[1]); list.appendChild(li); return;
      }
      if (!line.trim()) { flush(); return; }
      flush();
      var h = line.match(/^#{1,4}\s+(.*)$/);
      var p = document.createElement("p");
      if (h) { var st = document.createElement("strong"); inline(st, h[1]); p.appendChild(st); } else inline(p, line);
      target.appendChild(p);
    });
  }
  $$(".msg-assistant[data-md]").forEach(function (n) { renderMarkdown(n, n.textContent); });

  var chatForm = $("#chat-form");
  if (chatForm) {
    var log = $("#chat-log"), q = $("#chat-q"), send = $("#chat-send"), status = $("#chat-status");
    var csrfInput = $('input[name="csrf"]');
    var csrfToken = csrfInput ? csrfInput.value : "";
    function add(role, text, tools) {
      var empty = $("#chat-empty"); if (empty) empty.remove();
      var li = document.createElement("li"); li.className = "msg msg-" + role;
      if (role === "assistant") renderMarkdown(li, text); else li.textContent = text;
      if (tools && tools.length) {
        var t = document.createElement("span"); t.className = "tools";
        t.textContent = "Looked up: " + tools.map(function (x) { var a = x.input || {}; return x.name.replace(/^get_/, "").replace(/_/g, " ") + (a.source ? " (" + a.source + (a.book ? ", " + a.book : "") + ")" : ""); }).join("; ");
        li.appendChild(t);
      }
      log.appendChild(li); li.scrollIntoView({ block: "nearest" });
    }
    function ask(text) {
      text = (text || "").trim(); if (!text) return;
      add("user", text); q.value = ""; send.disabled = true; status.textContent = "Looking at your data…";
      fetch(chatForm.getAttribute("data-endpoint"), { method: "POST", credentials: "same-origin",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": csrfToken, Accept: "application/json" },
        body: JSON.stringify({ question: text }) })
        .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, status: r.status, j: j }; }); })
        .then(function (res) {
          if (res.status === 401) { window.location = "/login"; return; }
          if (!res.ok) { add("error", res.j.error || "Something went wrong."); status.textContent = ""; return; }
          add("assistant", res.j.answer, res.j.tools);
          var u = res.j.usage || {}; status.textContent = (u.input_tokens || 0) + " in / " + (u.output_tokens || 0) + " out tokens";
        })
        .catch(function () { add("error", "The console did not answer. Check that it is still running."); status.textContent = ""; })
        .then(function () { send.disabled = false; q.focus(); });
    }
    chatForm.addEventListener("submit", function (e) { e.preventDefault(); ask(q.value); });
    q.addEventListener("keydown", function (e) { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(q.value); } });
    $$("[data-q]").forEach(function (b) { b.addEventListener("click", function () { ask(b.getAttribute("data-q")); }); });
    $("#chat-clear").addEventListener("click", function () {
      fetch(chatForm.getAttribute("data-clear"), { method: "POST", credentials: "same-origin", headers: { "X-CSRF-Token": csrfToken } })
        .then(function () { window.location.reload(); });
    });
  }

  // ------------------------------------------------------------------ status strip refresh
  var strip = $(".strip[data-poll]");
  function setItem(name, text, cls) {
    var it = $('[data-k="' + name + '"]', strip); if (!it) return;
    var v = $(".v", it), d = $(".dot", it);
    if (v) v.textContent = text;
    if (d) d.className = "dot " + (cls || "");
  }
  function poll() {
    if (document.hidden) return;
    fetch("/api/status", { credentials: "same-origin", headers: { Accept: "application/json" } })
      .then(function (r) { if (r.status === 401) { window.location = "/login"; return null; } return r.ok ? r.json() : null; })
      .then(function (s) {
        if (!s) return;
        setItem("engine", s.engine + " (" + s.engine_age + ")", s.engine === "running" ? "ok" : s.engine === "stale" ? "warn" : "fail");
        setItem("session", s.session, s.market_open ? "ok" : "");
        setItem("kill", s.kill ? "ENGAGED" : "released", s.kill ? "fail" : "ok");
        setItem("live", s.armed ? "ARMED today" : "not armed", s.armed ? "fail" : "ok");
      })
      .catch(function () { setItem("engine", "console unreachable", "fail"); });
  }
  if (strip) setInterval(poll, 20000);
})();
