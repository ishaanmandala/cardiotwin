// CardioTwin dashboard - no framework, no external dependencies.
const $ = (sel, el = document) => el.querySelector(sel);
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const state = {
  models: [],
  demos: [],
  activeDemo: null,
  files: [],
  result: null,
  playing: !reducedMotion,
  rafId: null,
};

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (v, nd = 0) => (v === null || v === undefined || Number.isNaN(v) ? "-" : Number(v).toFixed(nd));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) { /* not JSON */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return res.json();
}

function setStatus(text, kind = "") {
  const el = $("#status");
  el.className = `status ${kind}`;
  el.innerHTML = kind === "busy" ? `<span class="spinner" aria-hidden="true"></span>${esc(text)}` : esc(text);
}

// ------------------------------------------------------------------- setup
async function init() {
  setupTabs();
  setupUpload();
  $("#anim-toggle").addEventListener("click", toggleAnimation);
  $("#saliency-toggle").addEventListener("change", () => state.result && drawECG(state.result));
  $("#use-demo-demographics").addEventListener("change", () => state.activeDemo && runDemo(state.activeDemo));
  $("#model-select").addEventListener("change", () => state.activeDemo && runDemo(state.activeDemo));
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => state.result && (drawECG(state.result), renderMeasurements(state.result)), 120);
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => state.result && render(state.result));
  if (reducedMotion) $("#anim-toggle").textContent = "Animate";

  try {
    const { models } = await api("/api/models");
    state.models = models;
    if (!models.length) {
      $("#empty").hidden = false;
      setStatus("");
      return;
    }
    const sel = $("#model-select");
    for (const m of models) sel.add(new Option(m.title, m.name));
    $("#attribution").textContent = models.map((m) => m.license_note).filter(Boolean)[0] || "";
    const { demos } = await api("/api/demos");
    state.demos = demos;
    renderDemoList();
    if (demos.length) runDemo(demos[Math.min(2, demos.length - 1)].id);
    else switchTab("upload");
  } catch (err) {
    setStatus(`Could not reach the CardioTwin server: ${err.message}`, "error");
  }
}

function setupTabs() {
  $("#tab-demo").addEventListener("click", () => switchTab("demo"));
  $("#tab-upload").addEventListener("click", () => switchTab("upload"));
}

function switchTab(which) {
  const demo = which === "demo";
  $("#tab-demo").setAttribute("aria-selected", String(demo));
  $("#tab-upload").setAttribute("aria-selected", String(!demo));
  $("#panel-demo").hidden = !demo;
  $("#panel-upload").hidden = demo;
}

function renderDemoList() {
  const list = $("#demo-list");
  list.innerHTML = "";
  for (const d of state.demos) {
    const b = document.createElement("button");
    b.className = "demo-btn";
    b.type = "button";
    b.dataset.id = d.id;
    b.setAttribute("aria-pressed", "false");
    const sex = d.sex === "M" ? "male" : d.sex === "F" ? "female" : "sex unknown";
    b.innerHTML = `<span class="t">${esc(d.label)}</span><span class="s">${d.age ? Math.round(d.age) + " y" : "age unknown"} &middot; ${sex}</span><span class="s">${esc(d.rhythm || "")}</span>`;
    b.addEventListener("click", () => runDemo(d.id));
    list.appendChild(b);
  }
}

function setupUpload() {
  const input = $("#file-input");
  const dz = $("#dropzone");
  const update = () => {
    $("#dz-files").textContent = state.files.map((f) => f.name).join(", ");
    $("#upload-btn").disabled = !state.files.length;
  };
  input.addEventListener("change", () => { state.files = [...input.files]; update(); });
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => { state.files = [...e.dataTransfer.files]; update(); });
  $("#upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!state.files.length) return;
    const fd = new FormData();
    state.files.forEach((f) => fd.append("files", f, f.name));
    const age = $("#age").value, sex = $("#sex").value, fs = $("#fs").value, model = $("#model-select").value;
    if (age) fd.append("age", age);
    if (sex) fd.append("sex", sex);
    if (fs) fd.append("fs", fs);
    if (model) fd.append("model", model);
    state.activeDemo = null;
    markDemo(null);
    setStatus("Reading the ECG and estimating the heart...", "busy");
    try {
      render(await api("/api/predict", { method: "POST", body: fd }));
      setStatus(`Analyzed ${state.files.map((f) => f.name).join(", ")}`);
    } catch (err) {
      setStatus(err.message, "error");
    }
  });
}

function markDemo(id) {
  document.querySelectorAll(".demo-btn").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.id === id)));
}

async function runDemo(id) {
  state.activeDemo = id;
  markDemo(id);
  setStatus("Reading the ECG and estimating the heart...", "busy");
  try {
    const body = {
      demo_id: id,
      model: $("#model-select").value || null,
      use_demographics: $("#use-demo-demographics").checked,
    };
    const res = await api("/api/predict/demo", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    render(res);
    const d = state.demos.find((x) => x.id === id);
    setStatus(`${d ? d.label : id}: ${res.model.title}`);
  } catch (err) {
    setStatus(err.message, "error");
  }
}

// ------------------------------------------------------------------ render
function render(res) {
  state.result = res;
  $("#results").hidden = false;
  renderWarnings(res);
  renderChips(res);
  drawECG(res);
  renderMeasurements(res);
  renderRisk(res);
  renderFuture(res);
  renderAbout(res);
  startHeart(res);
}

function renderWarnings(res) {
  const el = $("#warnings");
  const w = res.warnings || [];
  el.hidden = !w.length;
  el.innerHTML = w.length ? `<strong>Please note</strong><ul>${w.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "";
}

function renderChips(res) {
  const f = res.features || {};
  const chips = [];
  if (f.heart_rate) chips.push(`Heart rate <b>${fmt(f.heart_rate)} bpm</b>`);
  if (f.qrs_ms) chips.push(`QRS width <b>${fmt(f.qrs_ms)} ms</b>`);
  if (f.rr_cv !== null && f.rr_cv !== undefined) chips.push(`Rhythm <b>${f.rr_cv > 0.12 ? "irregular" : "regular"}</b>`);
  chips.push(`Leads <b>${esc(res.ecg.leads.join(", "))}</b>`);
  chips.push(`Input <b>${fmt(res.input.fs)} Hz &middot; ${fmt(res.input.duration_s, 1)} s</b>`);
  $("#ecg-chips").innerHTML = chips.map((c) => `<span class="chip">${c}</span>`).join("");
  const seg = res.explain && res.explain.segments;
  const derived = (res.input.derived_leads || []).length ? ` Leads ${res.input.derived_leads.join(", ")} were derived from the uploaded leads.` : "";
  $("#ecg-caption").textContent = seg
    ? `Shaded: where the ejection-fraction estimate came from (integrated gradients). QRS ${fmt(seg.QRS * 100)}%, T wave ${fmt(seg.T * 100)}%, P wave ${fmt(seg.P * 100)}% of the attribution.${derived} 25 mm/s, 10 mm/mV.`
    : `25 mm/s, 10 mm/mV.${derived}`;
}

// ---------------------------------------------------------------- ECG paper
function drawECG(res) {
  const canvas = $("#ecg-canvas");
  const { signal, fs, leads } = res.ecg;
  const sal = $("#saliency-toggle").checked && res.explain ? res.explain.saliency : null;
  const n = signal[0].length;
  const dur = n / fs;
  const cssW = canvas.parentElement.clientWidth;
  const pxPerMm = cssW / (dur * 25);
  const rowMv = leads.length > 6 ? 2.0 : 3.2;
  const rowH = rowMv * 10 * pxPerMm;
  const cssH = Math.round(rowH * leads.length);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.height = `${cssH}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = css("--paper");
  ctx.fillRect(0, 0, cssW, cssH);

  if (sal) {
    ctx.fillStyle = css("--saliency");
    leads.forEach((_, i) => {
      const row = sal[i];
      let start = -1;
      for (let k = 0; k <= n; k++) {
        const on = k < n && row[k] > 0.35;
        if (on && start < 0) start = k;
        if (!on && start >= 0) {
          ctx.fillRect((start / fs) * 25 * pxPerMm, i * rowH, ((k - start) / fs) * 25 * pxPerMm, rowH);
          start = -1;
        }
      }
    });
  }
  const line = (x0, y0, x1, y1) => { ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); };
  for (const [color, step, width] of [[css("--paper-minor"), 1, 0.6], [css("--paper-major"), 5, 1]]) {
    if (step === 1 && pxPerMm < 2.2) continue; // too dense to be useful
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    for (let mm = 0; mm * pxPerMm <= cssW + 0.5; mm += step) line(mm * pxPerMm, 0, mm * pxPerMm, cssH);
    for (let mm = 0; mm * pxPerMm <= cssH + 0.5; mm += step) line(0, mm * pxPerMm, cssW, mm * pxPerMm);
  }
  ctx.strokeStyle = css("--trace");
  ctx.lineWidth = 1.25;
  ctx.lineJoin = "round";
  leads.forEach((name, i) => {
    const row = signal[i];
    const sorted = [...row].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const base = (i + 0.5) * rowH;
    ctx.beginPath();
    for (let k = 0; k < n; k++) {
      const x = (k / fs) * 25 * pxPerMm;
      const y = base - (row[k] - median) * 10 * pxPerMm;
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = css("--ink-2");
    ctx.font = "600 12px system-ui, sans-serif";
    ctx.fillText(name, 8, i * rowH + 16);
  });
  canvas.setAttribute("aria-label", `ECG tracing, ${leads.length} leads, ${dur.toFixed(0)} seconds`);
}

// ------------------------------------------------------------- measurements
function sevIcon(sev) {
  return sev === 0 ? "&#10003;" : sev === 1 ? "!" : sev >= 2 ? "!!" : "?";
}

function rangeBar(item, width) {
  const [lo, hi] = item.display_range;
  const x = (v) => 6 + ((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * (width - 12);
  const parts = [`<rect class="track" x="6" y="12" width="${width - 12}" height="6" rx="3"/>`];
  if (item.normal_range) {
    const a = x(item.normal_range[0]), b = x(item.normal_range[1]);
    parts.push(`<rect class="band" x="${a}" y="9" width="${Math.max(2, b - a)}" height="12" rx="3"/>`);
  }
  if (item.low90 !== undefined && item.high90 !== undefined) {
    parts.push(`<line class="int" x1="${x(item.low90)}" y1="15" x2="${x(item.high90)}" y2="15"/>`);
  }
  if (item.truth !== undefined) {
    const t = x(item.truth);
    parts.push(`<path class="truth" d="M${t} 8 L${t + 6} 15 L${t} 22 L${t - 6} 15 Z"><title>Actual echo: ${fmt(item.truth, 1)} ${esc(item.unit)}</title></path>`);
  }
  parts.push(`<circle class="dot" cx="${x(item.value)}" cy="15" r="6"><title>CardioTwin: ${fmt(item.value, 1)} ${esc(item.unit)}</title></circle>`);
  parts.push(`<text class="tick" x="6" y="33">${lo}</text><text class="tick" x="${width - 6}" y="33" text-anchor="end">${hi}</text>`);
  return `<svg viewBox="0 0 ${width} 34" width="${width}" height="34" aria-hidden="true">${parts.join("")}</svg>`;
}

function renderMeasurements(res) {
  const list = $("#measure-list");
  list.innerHTML = "";
  for (const it of res.profile.current) {
    const li = document.createElement("li");
    li.className = "measure";
    const unit = it.unit === "%" ? "%" : ` ${it.unit}`;
    const nd = it.unit === "g" ? 0 : (Math.abs(it.value) < 20 ? 1 : 0);
    const truth = it.truth !== undefined ? `<div class="m-truth"><i class="lg-truth" aria-hidden="true"></i> echo ${fmt(it.truth, nd)}${esc(unit)}</div>` : "";
    const range = it.normal_range ? `normal ${fmt(it.normal_range[0], nd)}&ndash;${fmt(it.normal_range[1], nd)}${esc(unit)}` : "";
    const pAbn = it.p_abnormal !== null && it.p_abnormal !== undefined ? ` &middot; ${fmt(it.p_abnormal * 100)}% chance abnormal` : "";
    li.innerHTML = `
      <div class="m-info">
        <div class="m-label">${esc(it.label)}</div>
        <div class="sev sev-${it.severity}"><i aria-hidden="true">${sevIcon(it.severity)}</i>${esc(it.category)}</div>
        <div class="m-desc">${esc(it.description)}</div>
      </div>
      <div class="m-barcol"><div class="m-bar"></div><div class="m-range">${range}${pAbn}</div></div>
      <div class="m-val">
        <div><span class="m-num">${fmt(it.value, nd)}</span><span class="m-unit">${esc(it.unit)}</span></div>
        <div class="m-range">90%: ${fmt(it.low90, nd)}&ndash;${fmt(it.high90, nd)}</div>
        ${truth}
      </div>`;
    list.appendChild(li);
    const bar = li.querySelector(".m-bar");
    bar.innerHTML = rangeBar(it, Math.max(140, Math.floor(bar.clientWidth || 220)));
  }
}

function renderRisk(res) {
  const r = res.profile.structural_risk;
  const icon = r.level === "LOW" ? "&#10003;" : r.level === "MODERATE" ? "!" : "!!";
  const drivers = r.drivers && r.drivers.length
    ? `<div class="drivers">Most likely abnormal:<ul>${r.drivers.slice(0, 5).map((d) => `<li>${esc(d)}</li>`).join("")}</ul></div>`
    : `<div class="drivers">No measurement is likely to be outside its normal range.</div>`;
  $("#risk-body").innerHTML = `
    <div class="level level-${r.level}"><span class="level-icon" aria-hidden="true">${icon}</span><span class="level-text">${esc(r.level)}</span></div>
    ${drivers}
    <p class="note">Based on the probability that each estimated measurement lies outside the ASE normal range for ${res.profile.sex === "F" ? "women" : res.profile.sex === "M" ? "men" : "adults (sex unknown: widest range used)"}.</p>`;
}

function statusColor(level) {
  return level === "HIGH" ? css("--critical") : level === "MODERATE" ? css("--warning") : css("--good");
}

function renderFuture(res) {
  const fut = res.profile.future;
  const el = $("#future-body");
  if (!fut) {
    el.innerHTML = `<p class="drivers">This model does not forecast the future. Future-heart estimates need longitudinal training data (Levels 12-13: a later echo or follow-up events).</p>`;
    return;
  }
  let html = "";
  if (fut.risk) {
    const r = fut.risk;
    const pct = Math.round(r.probability * 100);
    html += `
      <div class="hero">${pct}%</div>
      <div class="hero-label">estimated risk of <strong>${esc(r.label.toLowerCase())}</strong> within ${fmt(r.horizon_years)} years</div>
      <div class="meter" role="meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}" aria-label="Future risk"><span style="width:${Math.max(2, pct)}%;background:${statusColor(r.level)}"></span></div>
      <div class="meter-scale"><span>0%</span><span>100%</span></div>
      <div class="sev sev-${r.level === "HIGH" ? 3 : r.level === "MODERATE" ? 1 : 0}" style="margin-top:6px"><i aria-hidden="true">${sevIcon(r.level === "HIGH" ? 3 : r.level === "MODERATE" ? 1 : 0)}</i>${esc(r.level)} risk</div>
      ${r.percentile !== null && r.percentile !== undefined ? `<p class="drivers" style="margin-top:4px">Higher than ${fmt(r.percentile)}% of patients in the reference group.</p>` : ""}
      <p class="note">Only meaningful for people like the training population: ${esc(r.population || "see model card")}.</p>`;
  }
  if (fut.measurements && fut.measurements.length) {
    html += `<div class="future-meas">${fut.measurements.map((m) => m.kind === "binary"
      ? `<div><strong>${esc(m.label)}:</strong> ${fmt(m.value * 100)}%</div>`
      : `<div><strong>${esc(m.label)}:</strong> ${fmt(m.value, 1)} ${esc(m.unit)} <span class="muted">(${esc(m.category)})</span></div>`).join("")}</div>`;
  }
  el.innerHTML = html;
}

function renderAbout(res) {
  const m = state.models.find((x) => x.name === res.model.name) || {};
  const tm = m.test_metrics || {};
  const labels = Object.fromEntries(res.profile.current.map((c) => [c.target, c]));
  const rows = Object.entries(tm).filter(([t]) => labels[t]).map(([t, v]) => {
    const unit = labels[t].unit;
    return `<tr><td>${esc(labels[t].label)}</td><td class="num">${fmt(v.mae, 1)} ${esc(unit)}</td><td class="num">${fmt(v.pearson_r, 2)}</td></tr>`;
  }).join("");
  const ef = tm.lvef || {};
  const fut = m.future;
  const cidx = fut && fut.test_c_index ? `<li>Future risk ranking (C-index): ${fmt(fut.test_c_index[0], 2)} (95% CI ${fmt(fut.test_c_index[1], 2)}-${fmt(fut.test_c_index[2], 2)})</li>` : "";
  $("#about-body").innerHTML = `
    <div>
      <h3>${esc(m.title || res.model.title)}</h3>
      <p>${esc(m.description || "")}</p>
      <p style="margin-top:8px">${esc(m.dataset || "")}</p>
      <ul style="margin-top:8px">
        <li>Ensemble of ${esc(m.n_members ?? "?")} networks; uses ${m.uses_demographics ? "the ECG plus age and sex" : "the ECG only"}.</li>
        ${ef.auroc_ef_le_40 ? `<li>Detecting EF &le; 40% on held-out patients: AUROC ${fmt(ef.auroc_ef_le_40, 2)}</li>` : ""}
        ${cidx}
      </ul>
    </div>
    <div>
      <h3>Accuracy on held-out patients ${m.cohort ? `(n = ${m.cohort.patients})` : ""}</h3>
      <table class="metrics"><thead><tr><th>Measurement</th><th style="text-align:right">Typical error (MAE)</th><th style="text-align:right">r</th></tr></thead><tbody>${rows}</tbody></table>
    </div>
    <div>
      <h3>Read this before trusting a number</h3>
      <ul>
        <li>Each estimate comes with a 90% interval; wide intervals mean the ECG did not pin the value down.</li>
        <li>The model learned from ${esc(m.dataset_description ? "the population described on the left" : "its training data")}; people who differ from it (e.g. healthy hearts, children, other ECG machines) may get poor estimates.</li>
        <li>An echocardiogram is the measurement. CardioTwin is a research estimate of it.</li>
      </ul>
    </div>`;
}

// ------------------------------------------------------------------- heart
function heartDims(res) {
  const v = Object.fromEntries(res.profile.current.map((c) => [c.target, c.value]));
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const ef = clamp(v.lvef ?? 60, 8, 80);
  const edd = clamp(v.lvedd ?? 48, 34, 90);
  let esd = v.lvesd ?? edd * Math.cbrt(1 - ef / 100);
  esd = clamp(esd, 15, edd - 2);
  return {
    ef, edd, esd,
    ivs: clamp(v.ivsd ?? 9, 5, 24),
    pw: clamp(v.lvpwd ?? 9, 5, 24),
    la: clamp(v.la_diam ?? 35, 22, 75),
    rv: clamp(v.rv_diam ?? 26, 14, 50),
    hr: clamp((res.features && res.features.heart_rate) || 70, 40, 150),
  };
}

function lvPaths(x0, y0, D, ivs, pw, s) {
  const a = (D * s) / 2;
  const L = D * s * (1.85 - 0.006 * (D - 48));
  const h1 = 0.32 * L, ry = L - h1;
  const ts = ivs * s, tp = pw * s, ta = ((ivs + pw) / 2) * s * 0.8;
  const cavity = `M${x0 - a},${y0} L${x0 - a},${y0 + h1} A${a},${ry} 0 0 0 ${x0 + a},${y0 + h1} L${x0 + a},${y0} Z`;
  const outer = `M${x0 - a - ts},${y0 - 3} L${x0 - a - ts},${y0 + h1} A${a + (ts + tp) / 2},${ry + ta} 0 0 0 ${x0 + a + tp},${y0 + h1} L${x0 + a + tp},${y0 - 3} Z`;
  return { cavity, outer, a, L, h1, ry, ts, tp };
}

function drawHeartFrame(d, c, atr, valveOpen) {
  const svg = $("#heart-svg");
  const s = 1.5;
  const x0 = 192, y0 = 160;
  const D = d.edd - (d.edd - d.esd) * c;
  const thick = 1 + 0.45 * c * (d.ef / 60);
  const lv = lvPaths(x0, y0, D, d.ivs * thick, d.pw * thick, s);
  const lvDia = lvPaths(x0, y0, d.edd, d.ivs, d.pw, s);
  const ghost = lvPaths(x0, y0, 48, 9, 9, s);
  // Right ventricle wraps the septum on the left.
  const Sx = x0 - lvDia.a - lvDia.ts, Sy = y0 + 2, L = lvDia.h1 + lvDia.ry;
  const w = d.rv * s * 0.8 * (1 - 0.18 * c);
  const rv = `M${Sx + 2},${Sy} C${Sx - w * 0.6},${Sy - 8} ${Sx - w * 1.05},${Sy + 10} ${Sx - w},${Sy + 42} C${Sx - w * 0.95},${Sy + L * 0.55} ${Sx - w * 0.4},${Sy + L * 0.8} ${Sx + 2},${Sy + L * 0.82} Z`;
  // Atria sit on the valve plane; they shrink when they contract and fill during ventricular systole.
  const fill = 1 - 0.08 * atr + 0.04 * c;
  const laRx = d.la * s * 0.5 * fill, laRy = laRx * 0.86;
  const laCx = x0 + lv.a * 0.2, laCy = y0 - laRy * 0.78;
  const raRx = 19 * s * fill, raRy = raRx * 0.9;
  const raCx = Sx - w * 0.45, raCy = y0 - raRy * 0.7;
  const gR = 35 * s * 0.5;
  const leaf = 16 * valveOpen;
  svg.innerHTML = `
    <title id="heart-desc">Schematic heart drawn from the estimates: LV diameter ${fmt(d.edd)} mm, ejection fraction ${fmt(d.ef)}%, walls ${fmt(d.ivs)}/${fmt(d.pw)} mm, left atrium ${fmt(d.la)} mm</title>
    <ellipse class="atrium ra" cx="${raCx}" cy="${raCy}" rx="${raRx}" ry="${raRy}"/>
    <ellipse class="atrium" cx="${laCx}" cy="${laCy}" rx="${laRx}" ry="${laRy}"/>
    <path class="rv" d="${rv}"/>
    <path class="myo" d="${lv.outer}"/>
    <path class="blood" d="${lv.cavity}"/>
    <line class="valve" x1="${x0 - lv.a}" y1="${y0}" x2="${x0 - lv.a * 0.2}" y2="${y0 + leaf}"/>
    <line class="valve" x1="${x0 + lv.a}" y1="${y0}" x2="${x0 + lv.a * 0.2}" y2="${y0 + leaf}"/>
    <path class="ghost" d="${ghost.outer}"/>
    <ellipse class="ghost" cx="${x0 + ghost.a * 0.2}" cy="${y0 - gR * 0.86 * 0.78}" rx="${gR}" ry="${gR * 0.86}"/>
    <text class="lbl-in" x="${x0}" y="${y0 + lv.h1 + lv.ry * 0.3}" text-anchor="middle">LV</text>
    <text class="lbl-in" x="${laCx}" y="${laCy - laRy * 0.15}" text-anchor="middle">LA</text>
    <text class="lbl-in" x="${raCx}" y="${raCy - raRy * 0.1}" text-anchor="middle">RA</text>
    <text class="lbl-in" x="${Sx - w * 0.5}" y="${Sy + L * 0.42}" text-anchor="middle">RV</text>
    <text class="lbl-sub" x="160" y="352" text-anchor="middle">EF ${fmt(d.ef)}% &middot; LV ${fmt(D)} mm &middot; ${fmt(d.hr)} beats/min</text>`;
}

function cycle(p) {
  const easeOut = (t) => 1 - Math.pow(1 - t, 3);
  const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  let c = 0;
  if (p < 0.32) c = easeOut(p / 0.32);
  else if (p < 0.46) c = 1 - easeInOut((p - 0.32) / 0.14);
  const atr = p > 0.82 && p < 0.98 ? Math.sin(((p - 0.82) / 0.16) * Math.PI) : 0;
  const valveOpen = p > 0.46 ? Math.min(1, (p - 0.46) / 0.06) : 0;
  return { c, atr, valveOpen };
}

function startHeart(res) {
  const d = heartDims(res);
  cancelAnimationFrame(state.rafId);
  if (!state.playing) {
    drawHeartFrame(d, 0, 0, 1);
    return;
  }
  const period = 60000 / d.hr;
  const t0 = performance.now();
  const tick = (now) => {
    const p = ((now - t0) % period) / period;
    const { c, atr, valveOpen } = cycle(p);
    drawHeartFrame(d, c, atr, valveOpen);
    state.rafId = requestAnimationFrame(tick);
  };
  state.rafId = requestAnimationFrame(tick);
}

function toggleAnimation() {
  state.playing = !state.playing;
  const btn = $("#anim-toggle");
  btn.textContent = state.playing ? "Pause" : "Animate";
  btn.setAttribute("aria-pressed", String(state.playing));
  if (state.result) startHeart(state.result);
}

init();
