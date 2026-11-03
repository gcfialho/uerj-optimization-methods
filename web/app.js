// Optimization lab. Python (nelder_mead, tank_optimizer and web/lab.py) runs in the browser
// through Pyodide; this file only draws controls and Plotly charts around it.
import { STRINGS } from "./i18n.js";

const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/";
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");
const METHODS = ["SD", "Newton", "DFP"];
const EXAMPLES = [
  { D0: 0.45, L0: 0.55, method: "DFP" },
  { D0: 0.22, L0: 1.82, method: "Newton" },
  { D0: 0.98, L0: 0.95, method: "SD" },
  { D0: 1.1, L0: 1.5, method: "DFP" },
  { D0: 1.1, L0: 0.6, method: "Newton" },
];
const COEF_RANGES = {
  reflection: [0.5, 3, 0.05, "alpha"],
  expansion: [1.05, 4, 0.05, "gamma"],
  contraction: [0.1, 0.9, 0.05, "rho"],
  shrink: [0.1, 0.9, 0.05, "sigma"],
};
const DEFAULT_COEFS = { reflection: 1, expansion: 2, contraction: 0.5, shrink: 0.5 };

const state = {
  lang: initialLang(),
  tab: location.hash === "#nm" ? "nm" : "tank",
  tank: { D0: 0.45, L0: 0.55, method: "compare", tol: 1e-6, maxIter: 200, result: null, error: null },
  nm: { fn: "rosenbrock", start: null, coefs: { ...DEFAULT_COEFS }, result: null, error: null, k: 0, playing: false },
};

let lab = null; // Pyodide proxy of web/lab.py
let nmFunctions = null;
const controlsEl = document.querySelector("[data-controls]");
const resultsEl = document.querySelector("[data-results]");
const statusEl = document.querySelector("[data-status]");

/* ------------------------------------------------------------------ helpers */

function initialLang() {
  const q = new URLSearchParams(location.search).get("lang");
  if (q === "pt" || q === "en") return q;
  try {
    const saved = localStorage.getItem("optlab-lang");
    if (saved === "pt" || saved === "en") return saved;
  } catch {}
  return (navigator.language || "pt").toLowerCase().startsWith("pt") ? "pt" : "en";
}
const T = () => STRINGS[state.lang];
const locale = () => (state.lang === "pt" ? "pt-BR" : "en-US");
const fmt = (n, d = 2) => (n == null || !Number.isFinite(n) ? "–" : new Intl.NumberFormat(locale(), { minimumFractionDigits: d, maximumFractionDigits: d }).format(n));
const sci = (n) => (n == null || !Number.isFinite(n) ? "–" : n.toExponential(2).replace("e", "×10^").replace(/\^(-?)\+?(\d+)/, "^$1$2"));
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const nextFrame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
const methodColor = (m) => css(m === "SD" ? "--sd" : m === "Newton" ? "--newton" : "--dfp");

function status(text, { busy = false, error = false } = {}) {
  statusEl.classList.toggle("is-error", error);
  statusEl.innerHTML = `${busy ? '<span class="spinner" aria-hidden="true"></span>' : ""}<span>${esc(text)}</span>`;
}

function call(name, payload) {
  const out = JSON.parse(lab.call(name, payload ? JSON.stringify(payload) : ""));
  if (out && out.error) {
    if (out.trace) console.error(out.trace);
    throw new Error(out.error);
  }
  return out;
}

function baseLayout(extra = {}) {
  const ink = css("--ink"), muted = css("--muted"), line = css("--line");
  return {
    margin: { l: 56, r: 16, t: 10, b: 46 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Bricolage Grotesque, system-ui, sans-serif", color: ink, size: 12 },
    xaxis: { gridcolor: line, zerolinecolor: line, linecolor: line, tickfont: { color: muted } },
    yaxis: { gridcolor: line, zerolinecolor: line, linecolor: line, tickfont: { color: muted } },
    showlegend: false,
    hovermode: "closest",
    ...extra,
  };
}
const plotConfig = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"] };

// A calm sequential scale for contour fills (low cost light, high cost darker).
function contourScale() {
  return darkQuery.matches
    ? [[0, "#123a35"], [0.35, "#1b2a36"], [1, "#151c25"]]
    : [[0, "#cfeee6"], [0.35, "#eef3f6"], [1, "#ffffff"]];
}

// Reveal a path point by point over about one second (all at once with reduced motion).
const animations = {};
function revealPaths(divId, traceIndices, fullX, fullY) {
  cancelAnimationFrame(animations[divId]);
  if (reduceMotion) return Plotly.restyle(divId, { x: fullX, y: fullY }, traceIndices);
  const longest = Math.max(...fullX.map((a) => a.length));
  const frames = Math.min(70, Math.max(20, longest));
  let f = 0;
  const tick = () => {
    f += 1;
    const t = f / frames;
    Plotly.restyle(
      divId,
      { x: fullX.map((a) => a.slice(0, Math.max(1, Math.ceil(a.length * t)))), y: fullY.map((a) => a.slice(0, Math.max(1, Math.ceil(a.length * t)))) },
      traceIndices,
    );
    if (f < frames) animations[divId] = requestAnimationFrame(tick);
  };
  animations[divId] = requestAnimationFrame(tick);
}

/* ------------------------------------------------------------------ tank */

function tankControls() {
  const t = T().tank, s = state.tank;
  const seg = [...METHODS.map((m) => [m, m]), ["compare", t.compare]]
    .map(([v, label]) => `<button type="button" data-action="tank-method" data-value="${v}" aria-pressed="${s.method === v}">${esc(label)}</button>`)
    .join("");
  return `
    <div class="card"><p class="note">${esc(t.intro)}</p></div>
    <form class="card" data-form="tank">
      <h2>${esc(t.start)}</h2>
      <div class="row2">
        <label class="field"><span class="label">${esc(t.D)}</span><input type="number" name="D0" min="0.05" max="1.5" step="0.01" value="${s.D0}" required></label>
        <label class="field"><span class="label">${esc(t.L)}</span><input type="number" name="L0" min="0.05" max="2.5" step="0.01" value="${s.L0}" required></label>
      </div>
      <p class="note" style="margin-top:.4rem">${esc(t.startHint)}</p>
      <div class="field" style="margin-top:1rem"><span class="label">${esc(t.method)}</span><div class="segmented" role="group">${seg}</div></div>
      <div class="row2" style="margin-top:.8rem">
        <label class="field"><span class="label">${esc(t.tol)}</span>
          <select name="tol">${[1e-4, 1e-6, 1e-8].map((v) => `<option value="${v}" ${v === s.tol ? "selected" : ""}>${v.toExponential(0)}</option>`).join("")}</select></label>
        <label class="field"><span class="label">${esc(t.maxIter)}</span><input type="number" name="maxIter" min="1" max="2000" step="1" value="${s.maxIter}"></label>
      </div>
      <button type="submit" class="btn primary" style="margin-top:1rem" ${lab ? "" : "disabled"}>${esc(T().run)}</button>
    </form>
    <div class="card">
      <h2>${esc(t.examples)}</h2>
      <div class="examples">${EXAMPLES.map(
        (e, i) => `<button type="button" data-action="tank-example" data-i="${i}" ${lab ? "" : "disabled"}><span>${esc(t.ex[i])}</span><span class="meta">${e.method} · ${e.D0}, ${e.L0}</span></button>`,
      ).join("")}</div>
    </div>`;
}

function tankResults() {
  const t = T().tank, s = state.tank;
  if (s.error) return `<div class="error-box">${esc(s.error)}</div>`;
  if (!s.result) return "";
  const methods = Object.keys(s.result.results);
  const legend = methods.map((m) => `<span><i class="swatch" style="background:${methodColor(m)}"></i>${m}</span>`).join("");
  const showNote = methods.includes("DFP") && !s.result.results.DFP.converged;
  return `
    <div class="card plot-card">
      <div class="plot-head"><h2>${esc(t.contour)}</h2><button type="button" class="btn" data-action="tank-replay">${esc(T().replay)}</button></div>
      <div class="legend-inline">${legend}<span><i class="swatch" style="background:var(--accent);opacity:.35;height:.6rem"></i>${esc(t.feasible)}</span></div>
      <div id="tank-contour" class="plot"></div>
    </div>
    <div class="grid2">
      <div class="card plot-card"><div class="plot-head"><h2>${esc(t.convergence)}</h2></div><div id="tank-conv" class="plot small"></div></div>
      <div class="card"><h2>${esc(t.table)}</h2><div class="table-wrap">${tankTable(methods)}</div></div>
    </div>
    ${showNote ? `<div class="card"><p class="note">${esc(t.note)}</p></div>` : ""}`;
}

function tankTable(methods) {
  const t = T().tank, R = state.tank.result.results;
  const yn = (v) => `<span class="${v ? "ok" : "bad"}">${v ? t.yes : t.no}</span>`;
  const rows = [
    [t.rows.cost, (r) => `US$ ${fmt(r.final_cost)}`],
    [t.rows.D, (r) => fmt(r.final_D, 4)],
    [t.rows.L, (r) => fmt(r.final_L, 4)],
    [t.rows.volume, (r) => fmt(r.final_volume, 4)],
    [t.rows.feasible, (r) => yn(r.feasible)],
    [t.rows.iterations, (r) => fmt(r.iterations, 0)],
    [t.rows.evals, (r) => `${r.num_eval} / ${r.num_grad} / ${r.num_hess}`],
    [t.rows.stages, (r) => {
      const pen = r.stages.filter((s) => s.kind === "penalty").length;
      return `${pen} + ${r.stages.length - pen}`;
    }],
    [t.rows.converged, (r) => yn(r.converged)],
  ];
  return `<table><thead><tr><th></th>${methods.map((m) => `<th><span class="method-dot" style="background:${methodColor(m)}"></span>${m}</th>`).join("")}</tr></thead>
    <tbody>${rows.map(([label, f]) => `<tr><td>${esc(label)}</td>${methods.map((m) => `<td>${f(R[m])}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

function feasiblePolygon(contour) {
  const { v_min, v_max, d_max, l_max } = contour.limits;
  const lAt = (V, D) => Math.min((4 * V) / (Math.PI * D * D), l_max);
  const ds = [];
  for (let i = 0; i <= 200; i++) ds.push(0.1 + (i / 200) * (Math.min(d_max, 1.2) - 0.1));
  const D = ds.filter((d) => lAt(v_min, d) < l_max);
  return { x: [...D, ...[...D].reverse()], y: [...D.map((d) => lAt(v_max, d)), ...[...D].reverse().map((d) => lAt(v_min, d))] };
}

function drawTank() {
  const s = state.tank;
  if (!s.result || !document.getElementById("tank-contour")) return;
  const t = T().tank;
  const c = s.result.contour;
  const methods = Object.keys(s.result.results);
  const poly = feasiblePolygon(c);
  const accent = css("--accent");
  const traces = [
    { type: "contour", x: c.d, y: c.l, z: c.cost, colorscale: contourScale(), showscale: false, ncontours: 22, line: { color: css("--edge"), width: 0.6 }, hovertemplate: "D %{x:.3f} m<br>L %{y:.3f} m<br>US$ %{z:,.0f}<extra></extra>" },
    { type: "scatter", x: poly.x, y: poly.y, fill: "toself", fillcolor: accent + "26", line: { color: accent, width: 1.2, dash: "dot" }, hoverinfo: "skip", mode: "lines" },
    { type: "scatter", x: [s.D0], y: [s.L0], mode: "markers", marker: { size: 11, color: css("--surface"), line: { color: css("--ink"), width: 2.5 } }, hovertemplate: "start<extra></extra>" },
    ...methods.map((m) => ({ type: "scatter", mode: "lines+markers", x: [], y: [], name: m, line: { color: methodColor(m), width: 2.5 }, marker: { size: 4, color: methodColor(m) }, hovertemplate: `${m}<br>D %{x:.4f}<br>L %{y:.4f}<extra></extra>` })),
  ];
  const layout = baseLayout({
    xaxis: { ...baseLayout().xaxis, title: { text: "D (m)" }, range: [0.1, 1.2] },
    yaxis: { ...baseLayout().yaxis, title: { text: "L (m)" }, range: [0.1, 2.2] },
  });
  Plotly.react("tank-contour", traces, layout, plotConfig).then((div) => {
    div.removeAllListeners?.("plotly_click");
    div.on("plotly_click", (ev) => {
      const p = ev.points?.[0];
      if (!p) return;
      s.D0 = Math.round(p.x * 100) / 100;
      s.L0 = Math.round(p.y * 100) / 100;
      runTank();
    });
  });
  const idx = methods.map((_, i) => i + 3);
  revealPaths("tank-contour", idx, methods.map((m) => s.result.results[m].history.map((p) => p[0])), methods.map((m) => s.result.results[m].history.map((p) => p[1])));

  const conv = methods.map((m) => ({
    type: "scatter", mode: "lines", name: m,
    x: s.result.results[m].errors.map((_, i) => i), y: s.result.results[m].errors,
    line: { color: methodColor(m), width: 2 }, hovertemplate: `${m}: %{y:.2e}<extra></extra>`,
  }));
  Plotly.react("tank-conv", conv, baseLayout({
    xaxis: { ...baseLayout().xaxis, title: { text: t.iteration } },
    yaxis: { ...baseLayout().yaxis, type: "log", exponentformat: "power" },
  }), plotConfig);
}

async function runTank() {
  if (!lab) return;
  const s = state.tank;
  status(T().loading.running, { busy: true });
  await nextFrame();
  try {
    const payload = { D0: s.D0, L0: s.L0, tol: s.tol, max_iter: s.maxIter };
    if (s.method === "compare") s.result = call("tank_compare", payload);
    else {
      const one = call("tank_run", { ...payload, method: s.method });
      s.result = { results: { [s.method]: one }, contour: one.contour };
    }
    s.error = null;
  } catch (err) {
    s.error = err.message;
  }
  status(T().loading.ready);
  render();
}

/* ------------------------------------------------------------------ Nelder-Mead */

function nmControls() {
  const t = T().nm, s = state.nm;
  const fnSeg = Object.keys(t.fns)
    .map((k) => `<button type="button" data-action="nm-fn" data-value="${k}" aria-pressed="${s.fn === k}">${esc(t.fns[k])}</button>`)
    .join("");
  const coefs = Object.entries(COEF_RANGES)
    .map(([key, [min, max, step, label]]) => `
      <label class="field"><span class="range-head"><span class="label">${esc(t[label])}</span><b data-coef-value="${key}">${fmt(s.coefs[key], 2)}</b></span>
        <input type="range" min="${min}" max="${max}" step="${step}" value="${s.coefs[key]}" data-coef="${key}" ${lab ? "" : "disabled"}></label>`)
    .join("");
  const startLabel = s.start ? t.startAt(fmt(s.start[0], 2), fmt(s.start[1], 2)) : t.startDefault;
  return `
    <div class="card"><p class="note">${esc(t.intro)}</p></div>
    <div class="card">
      <h2>${esc(t.fn)}</h2>
      <div class="segmented" role="group">${fnSeg}</div>
      <div class="field" style="margin-top:1rem"><span class="label">${esc(t.start)}</span>
        <span>${esc(startLabel)}</span>
        <span class="note">${esc(t.startHint)}</span>
        ${s.start ? `<button type="button" class="btn" data-action="nm-reset" style="justify-self:start;margin-top:.3rem">${esc(t.reset)}</button>` : ""}
      </div>
    </div>
    <div class="card"><h2>${esc(t.coefs)}</h2>${coefs}
      <button type="button" class="btn" data-action="nm-defaults" style="margin-top:.8rem">1 · 2 · 0,5 · 0,5</button></div>`;
}

function nmResults() {
  const t = T().nm, s = state.nm;
  if (s.error) return `<div class="error-box">${esc(s.error)}</div>`;
  if (!s.result) return "";
  const r = s.result;
  const total = Math.max(1, r.nit);
  const ops = Object.entries(r.counts)
    .map(([op, n]) => `<div class="op"><span>${esc(t.opNames[op])}</span><span class="bar"><i style="transform:scaleX(${(n / total).toFixed(3)})"></i></span><span class="n">${n}</span></div>`)
    .join("");
  return `
    <div class="card plot-card">
      <div class="plot-head"><h2>${esc(t.contour)}</h2></div>
      <div class="legend-inline"><span><i class="swatch" style="background:var(--accent);height:.6rem;opacity:.5"></i>simplex</span><span><i class="swatch" style="background:var(--flag)"></i>${esc(t.minimizer)}</span></div>
      <div id="nm-contour" class="plot"></div>
      <div class="player">
        <button type="button" class="btn" data-action="nm-play">${esc(s.playing ? t.pause : t.play)}</button>
        <input type="range" min="0" max="${r.nit}" step="1" value="${s.k}" data-nm-k aria-label="${esc(t.step)}">
      </div>
      <p class="step-label" data-nm-label></p>
    </div>
    <div class="grid2">
      <div class="card plot-card"><div class="plot-head"><h2>${esc(t.convergence)}</h2></div><div id="nm-conv" class="plot small"></div></div>
      <div class="card">
        <dl class="stats">
          <div><dt>${esc(t.stats.nit)}</dt><dd>${r.nit}</dd></div>
          <div><dt>${esc(t.stats.nfev)}</dt><dd>${r.nfev}</dd></div>
          <div><dt>${esc(t.stats.x)}</dt><dd>(${fmt(r.x[0], 5)}, ${fmt(r.x[1], 5)})</dd></div>
          <div><dt>${esc(t.stats.fun)}</dt><dd>${sci(r.fun)}</dd></div>
          <div><dt>${esc(t.stats.error)}</dt><dd>${sci(r.distance)}</dd></div>
          <div><dt>${esc(t.stats.status)}</dt><dd class="${r.converged ? "ok" : "bad"}">${esc(r.message)}</dd></div>
        </dl>
        <h2 style="margin-top:1rem">${esc(t.ops)}</h2>
        <div class="ops">${ops}</div>
      </div>
    </div>`;
}

const closed = (simplex) => [...simplex, simplex[0]];

function drawNm() {
  const s = state.nm;
  if (!s.result || !document.getElementById("nm-contour")) return;
  const g = nmFunctions[s.fn];
  const r = s.result;
  const accent = css("--accent");
  const z = g.z.map((row) => row.map((v) => Math.log10(1 + v)));
  const first = closed(r.simplices[0]);
  const traces = [
    { type: "contour", x: g.x, y: g.y, z, colorscale: contourScale(), showscale: false, ncontours: 26, line: { color: css("--edge"), width: 0.6 }, hovertemplate: "x %{x:.3f}<br>y %{y:.3f}<extra></extra>" },
    { type: "scatter", mode: "lines", x: first.map((p) => p[0]), y: first.map((p) => p[1]), line: { color: css("--muted"), width: 1.2, dash: "dot" }, hoverinfo: "skip" },
    { type: "scatter", mode: "lines", x: [], y: [], line: { color: accent, width: 1.5 }, opacity: 0.6, hoverinfo: "skip" },
    { type: "scatter", mode: "lines", x: [], y: [], fill: "toself", fillcolor: accent + "40", line: { color: accent, width: 2.5 }, hoverinfo: "skip" },
    { type: "scatter", mode: "markers", x: g.minimizers.map((p) => p[0]), y: g.minimizers.map((p) => p[1]), marker: { symbol: "star", size: 13, color: css("--flag"), line: { color: css("--surface"), width: 1 } }, hovertemplate: `${T().nm.minimizer}<extra></extra>` },
  ];
  const layout = baseLayout({
    xaxis: { ...baseLayout().xaxis, range: g.bounds[0], title: { text: "x" } },
    yaxis: { ...baseLayout().yaxis, range: g.bounds[1], title: { text: "y" } },
  });
  Plotly.react("nm-contour", traces, layout, plotConfig).then((div) => {
    div.removeAllListeners?.("plotly_click");
    div.on("plotly_click", (ev) => {
      const p = ev.points?.[0];
      if (!p) return;
      s.start = [Math.round(p.x * 100) / 100, Math.round(p.y * 100) / 100];
      runNm({ play: true });
    });
  });
  Plotly.react("nm-conv", [
    { type: "scatter", mode: "lines", x: r.best.map((_, i) => i), y: r.best.map((v) => (v != null && v > 0 ? v : null)), line: { color: accent, width: 2 }, hovertemplate: "%{x}: %{y:.3e}<extra></extra>" },
  ], baseLayout({
    xaxis: { ...baseLayout().xaxis, title: { text: T().tank.iteration } },
    yaxis: { ...baseLayout().yaxis, type: "log", exponentformat: "power" },
  }), plotConfig);
  showStep(s.k);
}

function showStep(k) {
  const s = state.nm, r = s.result;
  if (!r || !document.getElementById("nm-contour")) return;
  k = Math.max(0, Math.min(k, r.nit));
  s.k = k;
  const tri = closed(r.simplices[k]);
  const trail = r.simplices.slice(0, k + 1).map((sim) => sim[0]);
  Plotly.restyle("nm-contour", { x: [trail.map((p) => p[0]), tri.map((p) => p[0])], y: [trail.map((p) => p[1]), tri.map((p) => p[1])] }, [2, 3]);
  Plotly.relayout("nm-conv", { shapes: [{ type: "line", xref: "x", yref: "paper", x0: k, x1: k, y0: 0, y1: 1, line: { color: css("--flag"), width: 1.5, dash: "dot" } }] });
  const slider = document.querySelector("[data-nm-k]");
  if (slider && Number(slider.value) !== k) slider.value = k;
  const label = document.querySelector("[data-nm-label]");
  if (label) {
    const op = T().nm.opNames[r.steps[k]];
    label.innerHTML = esc(T().nm.at(k, r.nit, "")) + `<b>${esc(op)}</b>`;
  }
}

let playFrame = 0;
function play() {
  const s = state.nm;
  cancelAnimationFrame(playFrame);
  if (!s.result) return;
  if (reduceMotion) {
    s.playing = false;
    return showStep(s.result.nit);
  }
  if (s.k >= s.result.nit) s.k = 0;
  s.playing = true;
  const perStep = Math.max(28, Math.min(160, 4500 / Math.max(1, s.result.nit)));
  let last = performance.now();
  const tick = (now) => {
    if (!s.playing) return;
    if (now - last >= perStep) {
      last = now;
      showStep(s.k + 1);
      if (s.k >= s.result.nit) {
        s.playing = false;
        updatePlayButton();
        return;
      }
    }
    playFrame = requestAnimationFrame(tick);
  };
  playFrame = requestAnimationFrame(tick);
  updatePlayButton();
}
function updatePlayButton() {
  const b = document.querySelector('[data-action="nm-play"]');
  if (b) b.textContent = state.nm.playing ? T().nm.pause : T().nm.play;
}

async function runNm({ play: autoplay = true } = {}) {
  if (!lab) return;
  const s = state.nm;
  cancelAnimationFrame(playFrame);
  s.playing = false;
  try {
    if (!nmFunctions) nmFunctions = call("nm_functions");
    s.result = call("nm_run", { function: s.fn, start: s.start, coefficients: s.coefs, max_iter: 400 });
    s.error = null;
    s.k = 0;
  } catch (err) {
    s.error = err.message;
    s.result = null;
  }
  render();
  if (autoplay && s.result) play();
}

/* ------------------------------------------------------------------ render */

let lastKey = "";
function render() {
  const key = state.tab;
  controlsEl.innerHTML = state.tab === "tank" ? tankControls() : nmControls();
  resultsEl.innerHTML = state.tab === "tank" ? tankResults() : nmResults();
  if (key !== lastKey && !reduceMotion) {
    for (const el of [controlsEl, resultsEl]) {
      el.classList.remove("view-enter");
      void el.offsetWidth;
      el.classList.add("view-enter");
    }
  }
  lastKey = key;
  updateTabs();
  if (window.Plotly) state.tab === "tank" ? drawTank() : drawNm();
}

function updateTabs() {
  const tabs = [...document.querySelectorAll("[data-tab]")];
  tabs.forEach((tab) => {
    const on = tab.dataset.tab === state.tab;
    tab.setAttribute("aria-selected", String(on));
    tab.tabIndex = on ? 0 : -1;
  });
  const active = tabs.find((t) => t.dataset.tab === state.tab);
  const ind = document.querySelector(".tab-indicator");
  if (active && ind) ind.style.transform = `translateX(${active.offsetLeft + 8}px) scaleX(${active.offsetWidth - 16})`;
}

function applyStaticText() {
  const t = T();
  document.documentElement.lang = state.lang === "pt" ? "pt-BR" : "en";
  document.title = `${t.title} · ${state.lang === "pt" ? "Optimization lab" : "Laboratório de otimização"}`;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const v = el.dataset.i18n.split(".").reduce((o, k) => o?.[k], t);
    if (typeof v === "string") el.textContent = v;
  });
  document.querySelectorAll("[data-i18n-label]").forEach((el) => el.setAttribute("aria-label", t[el.dataset.i18nLabel]));
  document.querySelectorAll("[data-lang]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === state.lang)));
}

/* ------------------------------------------------------------------ events */

document.querySelector(".lang").addEventListener("click", (e) => {
  const b = e.target.closest("[data-lang]");
  if (!b || b.dataset.lang === state.lang) return;
  state.lang = b.dataset.lang;
  try {
    localStorage.setItem("optlab-lang", state.lang);
  } catch {}
  applyStaticText();
  if (lab) status(T().loading.ready);
  render();
});

document.querySelector(".tabs").addEventListener("click", async (e) => {
  const tab = e.target.closest("[data-tab]");
  if (!tab || tab.dataset.tab === state.tab) return;
  state.tab = tab.dataset.tab;
  history.replaceState(null, "", state.tab === "nm" ? "#nm" : "#tank");
  render();
  if (state.tab === "nm" && !state.nm.result && lab) runNm();
});

document.addEventListener("submit", (e) => {
  const form = e.target.closest('[data-form="tank"]');
  if (!form) return;
  e.preventDefault();
  const fd = new FormData(form);
  Object.assign(state.tank, { D0: Number(fd.get("D0")), L0: Number(fd.get("L0")), tol: Number(fd.get("tol")), maxIter: Number(fd.get("maxIter")) });
  runTank();
});

document.addEventListener("click", (e) => {
  const el = e.target.closest("[data-action]");
  if (!el) return;
  const a = el.dataset.action;
  if (a === "tank-method") {
    state.tank.method = el.dataset.value;
    return runTank();
  }
  if (a === "tank-example") {
    const ex = EXAMPLES[Number(el.dataset.i)];
    Object.assign(state.tank, { D0: ex.D0, L0: ex.L0, method: ex.method });
    return runTank();
  }
  if (a === "tank-replay") return drawTank();
  if (a === "nm-fn") {
    state.nm.fn = el.dataset.value;
    state.nm.start = null;
    return runNm();
  }
  if (a === "nm-reset") {
    state.nm.start = null;
    return runNm();
  }
  if (a === "nm-defaults") {
    state.nm.coefs = { ...DEFAULT_COEFS };
    return runNm();
  }
  if (a === "nm-play") {
    if (state.nm.playing) {
      state.nm.playing = false;
      cancelAnimationFrame(playFrame);
      return updatePlayButton();
    }
    return play();
  }
});

let coefTimer = 0;
document.addEventListener("input", (e) => {
  const el = e.target;
  if (el.dataset.coef) {
    const key = el.dataset.coef;
    state.nm.coefs[key] = Number(el.value);
    document.querySelector(`[data-coef-value="${key}"]`).textContent = fmt(state.nm.coefs[key], 2);
    clearTimeout(coefTimer);
    coefTimer = setTimeout(() => runNm({ play: true }), 180);
  } else if (el.matches("[data-nm-k]")) {
    state.nm.playing = false;
    cancelAnimationFrame(playFrame);
    updatePlayButton();
    showStep(Number(el.value));
  }
});

window.addEventListener("resize", updateTabs);
darkQuery.addEventListener("change", () => render());

/* ------------------------------------------------------------------ boot */

async function boot() {
  applyStaticText();
  render();
  try {
    status(T().loading.python, { busy: true });
    const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
    status(T().loading.numpy, { busy: true });
    await pyodide.loadPackage("numpy");
    status(T().loading.code, { busy: true });
    const files = await (await fetch("py/manifest.json")).json();
    const root = "/home/pyodide/lab";
    await Promise.all(
      files.map(async (f) => {
        const text = await (await fetch(`py/${f}`)).text();
        const path = `${root}/${f}`;
        pyodide.FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
        pyodide.FS.writeFile(path, text);
      }),
    );
    pyodide.runPython(`import sys; sys.path.insert(0, "${root}")`);
    lab = pyodide.pyimport("lab");
    status(T().loading.ready);
    if (state.tab === "tank") await runTank();
    else await runNm();
  } catch (err) {
    console.error(err);
    status(T().loading.failed(err.message), { error: true });
  }
}
boot();
