(() => {
  const D = window.DATA;
  const $ = (s) => document.querySelector(s);
  const NS = "http://www.w3.org/2000/svg";
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const SERIES = ["--muted", "--mark", "--mark-2"];

  function h(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }
  function s(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  // ---- tooltip (mouse hover; tap on touch) ----
  const tip = $("#tip");
  let tipTimer;
  function place(e) {
    const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    const y = e.clientY - tip.offsetHeight - 12;
    tip.style.left = Math.max(8, x) + "px";
    tip.style.top = (y < 8 ? e.clientY + 18 : y) + "px";
  }
  function show(e, html) {
    tip.innerHTML = html;
    tip.hidden = false;
    place(e);
    clearTimeout(tipTimer);
    if (e.pointerType === "touch") tipTimer = setTimeout(() => (tip.hidden = true), 2200);
  }
  function hide() { tip.hidden = true; }
  function bind(node, html) {
    node.addEventListener("pointerenter", (e) => show(e, typeof html === "function" ? html(e) : html));
    node.addEventListener("pointermove", (e) => { if (e.pointerType !== "touch") place(e); });
    node.addEventListener("pointerleave", (e) => { if (e.pointerType !== "touch") hide(); });
  }

  // data table for screen readers; wrapped in a div because a <table> ignores width: 1px
  function srTable(el, caption, head, rows) {
    const t = h("div", "sr");
    t.innerHTML = `<table><caption>${caption}</caption><tr>${head.map((x) => `<th>${x}</th>`).join("")}</tr>` +
      rows.map((r) => `<tr>${r.map((x) => `<td>${x}</td>`).join("")}</tr>`).join("") + "</table>";
    el.appendChild(t);
  }

  function legend(el, items) {
    const l = h("div", "legend");
    l.innerHTML = items.map(([color, label, n]) =>
      `<span><i style="background:${color}"></i>${label}${n != null ? `<b>${n}</b>` : ""}</span>`).join("");
    el.appendChild(l);
  }

  // ---- punch card: one COBOL line, IBM 029 Hollerith code ----
  function hollerith(ch) {
    if (ch === " ") return [];
    if (/[0-9]/.test(ch)) return [+ch];
    const c = ch.charCodeAt(0);
    if (ch >= "A" && ch <= "I") return [12, c - 64];
    if (ch >= "J" && ch <= "R") return [11, c - 73];
    if (ch >= "S" && ch <= "Z") return [0, c - 81];
    return { ".": [12, 3, 8], "-": [11], "(": [12, 5, 8], ")": [11, 5, 8], ",": [0, 3, 8], "=": [6, 8],
      "+": [12, 6, 8], "*": [11, 4, 8], "/": [0, 1], "'": [5, 8], ":": [2, 8] }[ch] || [];
  }
  function punchcard() {
    const svg = $("#punchcard");
    if (!svg) return;
    const W = 738, H = 325, X0 = 22, CW = (W - 44) / 80, Y0 = 38, RH = 23.2;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("role", "img");
    const line = ("000140 PROCEDURE DIVISION USING LINKED-ITEMS.".padEnd(72, " ") + "COBFORGE").slice(0, 80);
    const defs = s("defs", {}, svg);
    const g = s("linearGradient", { id: "pc", x1: 0, y1: 0, x2: 0, y2: 1 }, defs);
    s("stop", { offset: "0", "stop-color": "#f4ebd3" }, g);
    s("stop", { offset: "1", "stop-color": "#e3d6b4" }, g);
    s("path", { d: `M24,0 H${W - 10} Q${W},0 ${W},10 V${H - 10} Q${W},${H} ${W - 10},${H} H10 Q0,${H} 0,${H - 10} V24 Z`, fill: "url(#pc)" }, svg);
    const colX = (c) => X0 + CW * c + CW / 2;
    const rowY = (r) => Y0 + RH * r; // r: 0 = row 12, 1 = row 11, 2..11 = digits 0..9
    // field dividers + labels, as on a COBOL coding card
    for (const c of [6, 7, 72]) s("line", { x1: X0 + CW * c, x2: X0 + CW * c, y1: 26, y2: H - 18, stroke: "#c9b98f", "stroke-width": 0.8 }, svg);
    const label = (txt, c0, c1) => {
      const t = s("text", { x: (colX(c0) + colX(c1)) / 2, y: H - 7, "text-anchor": "middle", "font-size": 7, "font-family": "IBM Plex Mono, monospace", fill: "#9c8c66", "letter-spacing": 1 }, svg);
      t.textContent = txt;
    };
    label("SEQUENCE", 0, 5); label("COBOL STATEMENT", 7, 71); label("IDENT", 72, 79);
    for (let c = 0; c < 80; c++) {
      for (let d = 0; d <= 9; d++) {
        const t = s("text", { x: colX(c), y: rowY(d + 2) + 3, "text-anchor": "middle", "font-size": 6.4, "font-family": "IBM Plex Mono, monospace", fill: "#a8987a" }, svg);
        t.textContent = d;
      }
    }
    [...line].forEach((ch, c) => {
      const delay = 300 + c * 22 + "ms";
      if (ch !== " ") {
        const t = s("text", { class: "ch", x: colX(c), y: 19, "text-anchor": "middle", "font-size": 9.5, "font-weight": 600, "font-family": "IBM Plex Mono, monospace", fill: "#3a3122" }, svg);
        t.textContent = ch;
        t.style.transitionDelay = delay;
      }
      for (const p of hollerith(ch)) {
        const r = p === 12 ? 0 : p === 11 ? 1 : p + 2;
        const hole = s("rect", { class: "hole", x: colX(c) - 2.7, y: rowY(r) - 6.5, width: 5.4, height: 12.5, rx: 1, fill: "#05070c" }, svg);
        hole.style.transitionDelay = delay;
      }
    });
    s("title", {}, svg).textContent = "Punch card: " + line.trim();
    requestAnimationFrame(() => setTimeout(() => svg.classList.add("go"), 60));
  }

  // ---- horizontal bar list ----
  function bars(el, items, { max, fmt, hl = () => false, label = (it) => it[0], tipText }) {
    const wrap = h("div", "bars");
    items.forEach((it, i) => {
      const v = it[1];
      const row = h("div", "row" + (hl(it) ? " hl" : ""));
      row.innerHTML = `<span class="lab">${label(it)}</span><span class="track"><i style="--w:${(100 * v) / max}%;--d:${i * 70}ms"></i></span><span class="val">${fmt(v)}</span>`;
      bind(row, tipText ? tipText(it, i) : `${label(it)} · ${fmt(v)}`);
      wrap.appendChild(row);
    });
    el.appendChild(wrap);
  }

  function funnel() {
    const el = $("#funnel");
    const f = D.funnel;
    bars(el, f, {
      max: f[0][1], fmt: (v) => v.toLocaleString("en-US"), hl: (it) => it === f[f.length - 1],
      tipText: (it, i) => `${it[0]} · ${it[1].toLocaleString("en-US")}` + (i ? ` (${Math.round((100 * it[1]) / f[i - 1][1])}% of previous)` : ""),
    });
    srTable(el, "Training data funnel", ["Stage", "Count"], f);
  }

  function teacher() {
    const el = $("#teacher");
    bars(el, D.teacher, {
      max: 100, fmt: (v) => v.toFixed(1) + "%", hl: (it) => it[2],
      label: (it) => it[0] + (it[2] ? " <small>· teacher</small>" : ""),
      tipText: (it) => `${it[0]} · ${it[1].toFixed(1)}% on 30 problems${it[2] ? " · chosen as teacher" : ""}`,
    });
    srTable(el, "Teacher check, pass@1 on 30 problems", ["Model", "pass@1 %"], D.teacher.map((t) => [t[0], t[1]]));
  }

  // same-harness leaderboard: strict pass@1 (solid) plus the extra from COBOLEval's section-order fix (faint)
  function harnessChart() {
    const el = $("#harness");
    if (!el || !D.harness.length) return;
    const items = [...D.harness].sort((a, b) => b.pass - a.pass);
    legend(el, [[css("--mark"), "our fine-tune"], [css("--muted"), "strict, as base and SFT are graded"],
      ["rgba(255,255,255,0.16)", "extra with COBOLEval's section-order fix"]]);
    const wrap = h("div", "bars");
    items.forEach((it, i) => {
      const row = h("div", "row" + (it.hl ? " hl" : ""));
      const sub = it.size + (it.think ? ` · thinking ${it.think}` : "");
      row.innerHTML = `<span class="lab">${it.label} <small>· ${sub}</small></span><span class="track">` +
        (it.fixed > it.pass ? `<i class="ghost" style="--w:${it.fixed}%;--d:${i * 70}ms"></i>` : "") +
        `<i style="--w:${it.pass}%;--d:${i * 70}ms"></i></span><span class="val">${it.pass.toFixed(1)}%</span>`;
      const pf = (p) => (p < 0.001 ? p.toExponential(1) : p.toFixed(2));
      const vs = it.p == null ? "" : ` · vs SFT ${it.diff >= 0 ? "+" : ""}${it.diff.toFixed(1)} pts, p = ${pf(it.p)}` +
        (it.fixed > it.pass && it.p_fix != null ? ` (with fix: ${it.diff_fix >= 0 ? "+" : ""}${it.diff_fix.toFixed(1)}, p = ${pf(it.p_fix)})` : "");
      bind(row, `${it.label} · ${it.pass.toFixed(1)}%` + (it.fixed > it.pass ? ` (${it.fixed.toFixed(1)}% with section fix)` : "") +
        ` · $${it.cost.toFixed(4)}/problem${vs}`);
      wrap.appendChild(row);
    });
    el.appendChild(wrap);
    srTable(el, "Same-harness COBOLEval pass@1", ["Model", "Size", "pass@1 % strict", "pass@1 % with section fix", "$/problem", "p vs SFT"],
      items.map((x) => [x.label, x.size, x.pass.toFixed(1), x.fixed.toFixed(1), x.cost.toFixed(4), x.p == null ? "" : x.p.toPrecision(2)]));
  }

  function published() {
    const el = $("#published");
    const items = [...D.published].sort((a, b) => b[1] - a[1]);
    bars(el, items, {
      max: 60, fmt: (v) => v.toFixed(1) + "%", hl: (it) => it[2] === "ours",
      tipText: (it) => `${it[0]} · ${it[1].toFixed(2)}% · ${it[2] === "ours" ? "this repo" : it[2] === "paper" ? "COBOL-Coder paper" : it[2]}`,
    });
    srTable(el, "Published COBOLEval pass@1", ["System", "pass@1 %", "Source"], items);
  }

  // ---- outcomes: 100% stacked bars per system ----
  const OUT = [["pass", "s-pass", "--good", "✓ pass"], ["wrong answer", "s-wrong", "--warn", "≠ wrong answer"],
    ["compile error", "s-compile", "--crit", "✕ compile error"], ["blocked", "s-blocked", null, "⊘ blocked"]];
  function outcomes() {
    const el = $("#outcomes");
    const used = OUT.filter(([k]) => D.systems.some((x) => x.outcomes[k]));
    legend(el, used.map(([, , v, label]) => [v ? css(v) : "#6b6b66", label]));
    D.systems.forEach((sys, si) => {
      const row = h("div", "stack-row");
      row.appendChild(h("div", "lab", `${sys.label}<small>pass@1 ${(100 * sys.pass_rate).toFixed(1)}% · compiles ${(100 * sys.compiled).toFixed(0)}%</small>`));
      const st = h("div", "stack");
      used.forEach(([k, cls, , label], i) => {
        const n = sys.outcomes[k] || 0;
        if (!n) return;
        const seg = h("i", cls, n / sys.n >= 0.06 ? n : "");
        seg.style.flex = `${n} 1 0`;
        seg.style.setProperty("--d", `${si * 150 + i * 90}ms`);
        bind(seg, `${sys.label} · ${label}: ${n} (${((100 * n) / sys.n).toFixed(1)}%)`);
        st.appendChild(seg);
      });
      row.appendChild(st);
      el.appendChild(row);
    });
    srTable(el, "Outcomes per system", ["System", ...used.map((u) => u[0])], D.systems.map((x) => [x.label, ...used.map(([k]) => x.outcomes[k] || 0)]));
  }

  // ---- waffle: every problem, base vs SFT ----
  const WAF = [["new", "w-new", "solved only after SFT", "var(--good)"], ["both", "w-both", "solved by both", "rgba(12,163,12,0.42)"],
    ["lost", "w-lost", "solved only by base", "var(--crit)"], ["none", "w-none", "solved by neither", "rgba(255,255,255,0.14)"]];
  function waffle() {
    const el = $("#waffle");
    const count = (k) => D.waffle.filter((c) => c.state === k).length;
    legend(el, WAF.map(([k, , label, color]) => [color, label, count(k)]));
    const grid = h("div", "waffle");
    let i = 0;
    for (const [k, cls, label] of WAF) {
      for (const c of D.waffle.filter((x) => x.state === k)) {
        const cell = h("i", cls);
        cell.style.setProperty("--d", `${i++ * 5}ms`);
        bind(cell, `${c.id} · ${label}`);
        grid.appendChild(cell);
      }
    }
    el.appendChild(grid);
    srTable(el, "Per-problem outcome, base vs SFT", ["Outcome", "Problems"], WAF.map(([k, , label]) => [label, count(k)]));
  }

  // ---- compile errors by type, grouped per system ----
  function errors() {
    const el = $("#errors");
    const names = Object.keys(D.errors);
    const kinds = Object.keys(D.errors[names[0]]).sort((a, b) => (a === "other") - (b === "other") || D.errors[names[0]][b] - D.errors[names[0]][a]);
    for (const n of names.slice(1)) for (const k of Object.keys(D.errors[n])) if (!kinds.includes(k)) kinds.push(k);
    const max = Math.max(...names.flatMap((n) => Object.values(D.errors[n]))) / 0.88; // leave room for value labels
    const total = (n) => Object.values(D.errors[n]).reduce((a, b) => a + b, 0);
    legend(el, names.map((n, i) => [css(SERIES[i]), n, total(n)]));
    kinds.forEach((k, ki) => {
      const g = h("div", "grp");
      g.appendChild(h("span", "lab", k));
      const pair = h("div", "pair");
      names.forEach((n, i) => {
        const v = D.errors[n][k] || 0;
        const d = h("div", "", `<i style="--w:${(100 * v) / max}%;--d:${ki * 80 + i * 60}ms;background:${css(SERIES[i])}"></i><span style="--w:${(100 * v) / max}%">${v}</span>`);
        bind(d, `${n} · ${k}: ${v}`);
        pair.appendChild(d);
      });
      g.appendChild(pair);
      el.appendChild(g);
    });
    srTable(el, "Compile errors by type", ["Type", ...names], kinds.map((k) => [k, ...names.map((n) => D.errors[n][k] || 0)]));
  }

  // ---- line chart with crosshair ----
  function lineChart(el, pts, { fmtY, yMax, markers = [], name }) {
    let drawn = false;
    function draw() {
      el.querySelector("svg")?.remove();
      const W = el.clientWidth;
      if (!W) return;
      const Hh = W < 520 ? 190 : 230, m = { l: 44, r: 14, t: 14, b: 26 };
      const svg = s("svg", { width: W, height: Hh, viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": name });
      const x0 = pts[0][0], x1 = pts[pts.length - 1][0];
      const X = (x) => m.l + ((x - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
      const Y = (y) => Hh - m.b - (y / yMax) * (Hh - m.t - m.b);
      for (let i = 0; i <= 4; i++) {
        const v = (yMax * i) / 4;
        s("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: css("--grid") }, svg);
        s("text", { x: m.l - 8, y: Y(v) + 3.5, "text-anchor": "end" }, svg).textContent = fmtY(v);
      }
      const span = x1 - x0, stepX = span > 60 ? 15 : span > 20 ? 5 : 1;
      for (let x = Math.ceil(x0 / stepX) * stepX; x <= x1; x += stepX) {
        s("text", { x: X(x), y: Hh - 6, "text-anchor": "middle" }, svg).textContent = x;
      }
      for (const [mx, label] of markers) {
        s("line", { x1: X(mx), x2: X(mx), y1: m.t, y2: Hh - m.b, stroke: "rgba(255,255,255,0.22)" }, svg);
        s("text", { x: X(mx) + 6, y: m.t + 10, fill: css("--ink-dim") }, svg).textContent = label;
      }
      const d = pts.map(([x, y], i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join("");
      const path = s("path", { d, fill: "none", stroke: css("--mark"), "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
      const cross = s("line", { y1: m.t, y2: Hh - m.b, stroke: "rgba(255,255,255,0.35)", visibility: "hidden" }, svg);
      const dot = s("circle", { r: 4.5, fill: css("--mark"), stroke: "#14161d", "stroke-width": 2, visibility: "hidden" }, svg);
      const hit = s("rect", { x: m.l, y: 0, width: W - m.l - m.r, height: Hh, fill: "transparent" }, svg);
      const near = (e) => {
        const r = svg.getBoundingClientRect();
        const px = e.clientX - r.left;
        return pts.reduce((b, p) => (Math.abs(X(p[0]) - px) < Math.abs(X(b[0]) - px) ? p : b));
      };
      hit.addEventListener("pointermove", (e) => {
        const p = near(e);
        cross.setAttribute("x1", X(p[0])); cross.setAttribute("x2", X(p[0]));
        dot.setAttribute("cx", X(p[0])); dot.setAttribute("cy", Y(p[1]));
        cross.setAttribute("visibility", "visible"); dot.setAttribute("visibility", "visible");
        show(e, `step ${p[0]} · ${fmtY(p[1], true)}`);
      });
      hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); hide(); });
      el.prepend(svg);
      if (!drawn && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
        const len = path.getTotalLength();
        path.style.strokeDasharray = len;
        path.style.strokeDashoffset = len;
        const go = () => { path.style.transition = "stroke-dashoffset 1.6s cubic-bezier(.3,.6,.2,1)"; path.style.strokeDashoffset = 0; };
        const scene = el.closest(".scene");
        if (scene.classList.contains("in")) requestAnimationFrame(go);
        else scene.addEventListener("revealed", () => requestAnimationFrame(go), { once: true });
      }
      drawn = true;
    }
    draw();
    let last = el.clientWidth;
    new ResizeObserver(() => { if (el.clientWidth !== last) { last = el.clientWidth; draw(); } }).observe(el);
    srTable(el, name, ["step", "value"], pts);
  }

  function loss() {
    const pts = D.loss;
    const top = Math.ceil(Math.max(...pts.map((p) => p[1])) * 20) / 20;
    const half = Math.ceil(pts[pts.length - 1][0] / 2);
    lineChart($("#loss"), pts, { yMax: top, fmtY: (v, full) => (full ? "loss " : "") + v.toFixed(full ? 3 : 2), markers: [[half + 0.5, "epoch 2"]], name: "SFT training loss per step" });
  }

  function reward() {
    const g = D.grpo;
    if (!g.main) return;
    const pts = g.runs[g.main];
    if (pts.length < 3) return;
    $("#grpo-fig").hidden = false;
    lineChart($("#reward"), pts, { yMax: 1, fmtY: (v, full) => (full ? "reward " : "") + v.toFixed(2), name: "GRPO mean rollout reward per step" });
  }

  // ---- reveal on scroll ----
  function reveal() {
    const scenes = document.querySelectorAll(".scene");
    const mark = (x) => { x.classList.add("in"); x.dispatchEvent(new Event("revealed")); };
    if (!("IntersectionObserver" in window)) return scenes.forEach(mark);
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { mark(e.target); io.unobserve(e.target); } }), { threshold: 0.12 });
    scenes.forEach((x) => io.observe(x));
  }

  punchcard();
  funnel(); teacher(); outcomes(); waffle(); errors(); harnessChart(); published();
  reveal();
  loss(); reward();
})();
