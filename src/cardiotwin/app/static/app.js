// CardioTwin dashboard - no framework, no external dependencies.
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const state = {
  models: [],
  demos: [],
  activeDemo: null,
  files: [],
  result: null,
  playing: !reducedMotion,
  loopToken: 0,
  medical: false,
  tab: "overview",
};

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (v, nd = 0) => (v === null || v === undefined || Number.isNaN(v) ? "-" : Number(v).toFixed(nd));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const store = {
  get: (k, d) => { try { return localStorage.getItem(k) ?? d; } catch (_) { return d; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch (_) { /* storage blocked */ } },
};

// --------------------------------------------------------------------------------------
// Plain-language dictionary: what each measurement is, in everyday words.
// --------------------------------------------------------------------------------------
const PLAIN = {
  lvef: {
    name: "Pumping strength", subject: "the heart's pump",
    adj: ["working normally", "slightly weak", "weak", "very weak"],
    words: ["Normal", "Slightly weak", "Weak", "Very weak"],
    what: "How much of the blood inside the main pumping chamber is pushed out with each heartbeat.",
    why: "A weak pump is the most common sign of heart failure: the body gets less blood than it needs.",
    analogy: "Think of squeezing a water balloon. A healthy heart pushes out more than half of what's inside, every single beat.",
  },
  lvedd: {
    name: "Main chamber size", subject: "the main pumping chamber",
    adj: ["a normal size", "slightly enlarged", "enlarged", "very enlarged"],
    words: ["Normal size", "Slightly large", "Large", "Very large"],
    what: "How wide the main pumping chamber (left ventricle) is when it is relaxed and full of blood.",
    why: "A heart that has worked too hard for a long time often stretches and gets bigger, which makes it pump less efficiently.",
    analogy: "Like a balloon that has been blown up too many times: it gets bigger and floppier.",
  },
  lvesd: {
    name: "Chamber size after a squeeze", subject: "the chamber after each squeeze",
    adj: ["normal", "bigger than normal", "bigger than normal", "much bigger than normal"],
    words: ["Normal", "Larger than normal", "Larger than normal", "Much larger"],
    what: "How wide the main chamber still is at the very end of a squeeze.",
    why: "If it stays wide even after squeezing, the heart isn't emptying well.",
  },
  ivsd: {
    name: "Middle wall thickness", subject: "the middle heart wall",
    adj: ["normal", "slightly thick", "thick", "very thick"],
    words: ["Normal", "Slightly thick", "Thick", "Very thick"],
    what: "Thickness of the muscle wall between the left and right sides of the heart (the septum).",
    why: "Heart walls thicken when they push against high pressure for years, for example with high blood pressure.",
    analogy: "Like any muscle that lifts heavy weights every day, it bulks up.",
  },
  lvpwd: {
    name: "Back wall thickness", subject: "the back heart wall",
    adj: ["normal", "slightly thick", "thick", "very thick"],
    words: ["Normal", "Slightly thick", "Thick", "Very thick"],
    what: "Thickness of the back wall of the main pumping chamber.",
    why: "Together with the middle wall, it shows whether the heart muscle has thickened.",
  },
  la_diam: {
    name: "Upper-left chamber size", subject: "the upper-left chamber",
    adj: ["a normal size", "slightly enlarged", "enlarged", "very enlarged"],
    words: ["Normal size", "Slightly large", "Large", "Very large"],
    what: "Size of the left atrium: the chamber that collects blood coming back from the lungs.",
    why: "It enlarges when pressure in the heart is high for a long time, and an enlarged atrium makes irregular heartbeats (atrial fibrillation) more likely.",
  },
  rv_diam: {
    name: "Right pumping chamber", subject: "the right pumping chamber",
    adj: ["a normal size", "bigger than normal", "bigger than normal", "much bigger than normal"],
    words: ["Normal size", "Larger than normal", "Larger than normal", "Much larger"],
    what: "Size of the right ventricle: the chamber that pumps blood to the lungs.",
    why: "It can enlarge with lung disease, valve problems or when the left side of the heart is failing.",
  },
  lv_mass: {
    name: "Heart muscle weight", subject: "the heart muscle",
    adj: ["a normal weight", "slightly heavy", "heavy", "very heavy"],
    words: ["Normal", "Slightly heavy", "Heavy", "Very heavy"],
    what: "Estimated weight of the main chamber's muscle, calculated from its size and wall thickness.",
    why: "Extra muscle mass (hypertrophy) usually means the heart has been overworking for a long time.",
  },
  aortic_root: {
    name: "Main artery width", subject: "the base of the main artery",
    adj: ["normal", "wider than normal", "wider than normal", "much wider than normal"],
    words: ["Normal", "Wider than normal", "Wider than normal", "Much wider"],
    what: "Width of the aorta, the body's main artery, where it leaves the heart.",
    why: "A widened aorta needs monitoring because it can weaken over time.",
  },
  stroke_volume: { name: "Blood per beat", subject: "the blood pumped per beat", adj: ["normal", "low", "low", "very low"],
    words: ["Normal", "Low", "Low", "Very low"], what: "How many millilitres of blood the heart pumps out with each beat.", why: "Too little means the body may not get enough blood." },
  future_lvef: { name: "Pumping strength later", subject: "the future pump strength", adj: ["normal", "slightly weak", "weak", "very weak"],
    words: ["Normal", "Slightly weak", "Weak", "Very weak"], what: "The AI's guess for pumping strength at a heart scan 6-12 months later.", why: "Shows whether the heart may get worse." },
  delta_lvef: { name: "Change in pumping strength", subject: "the pump", adj: ["stable", "slightly declining", "declining", "strongly declining"],
    words: ["Stable", "Slight drop", "Drop", "Big drop"], what: "Expected change in pumping strength over the next 6-12 months (negative = weaker).", why: "Spotting a heart that is getting weaker early is the main goal of the future-heart research." },
  ef_decline: { name: "Chance of a big drop", subject: "the pump", adj: [], words: [], what: "Chance that pumping strength falls by at least 10 points to below 50% at the next scan.", why: "An early warning of worsening." },
};

const ICONS = {
  pump: '<svg viewBox="0 0 24 24"><path d="M12 21s-8-5-8-11a4.5 4.5 0 0 1 8-2.8A4.5 4.5 0 0 1 20 10c0 6-8 11-8 11z"/></svg>',
  size: '<svg viewBox="0 0 24 24"><path d="M4 4h6v2H7.4l3.3 3.3-1.4 1.4L6 7.4V10H4zm16 16h-6v-2h2.6l-3.3-3.3 1.4-1.4 3.3 3.3V14h2z"/></svg>',
  wall: '<svg viewBox="0 0 24 24"><path d="M3 5h18v4H3zm0 5h8v4H3zm10 0h8v4h-8zM3 15h18v4H3z"/></svg>',
  future: '<svg viewBox="0 0 24 24"><path d="M7 2h2v2h6V2h2v2h3v18H4V4h3zm11 8H6v10h12zm-9 2h2v2H9zm4 0h2v2h-2zm-4 4h2v2H9z"/></svg>',
};

const GLOSSARY = [
  ["ECG (electrocardiogram)", "A recording of the heart's electrical signal, picked up by stickers on the skin. Quick, cheap and available almost everywhere."],
  ["Echocardiogram (echo)", "An ultrasound scan of the heart. It shows the heart's size, walls, valves and how well it pumps. This is what CardioTwin tries to estimate."],
  ["Ejection fraction (EF)", "The share of blood the main pumping chamber pushes out per beat. Healthy is about 52-72%; 40% or less is called 'reduced'."],
  ["Left ventricle", "The main pumping chamber. It sends blood to the whole body, so it is the strongest part of the heart."],
  ["Left atrium", "The upper-left chamber. It receives blood from the lungs and passes it to the left ventricle."],
  ["Right ventricle", "The chamber that pumps blood to the lungs to pick up oxygen."],
  ["Septum", "The muscle wall between the left and right sides of the heart."],
  ["Heart failure", "When the heart can't pump as well as the body needs. Despite the name, the heart has not stopped."],
  ["Atrial fibrillation (AF)", "A common irregular heartbeat, where the upper chambers quiver instead of beating in rhythm."],
  ["Digital twin", "A computer model of a specific person's heart, here estimated from their ECG."],
  ["Likely range", "The range the AI is 90% sure contains the true value. A wide range means it is unsure."],
  ["Neural network", "A type of AI that learns patterns from examples: here, hundreds of people who had both an ECG and a heart scan."],
];

const TARGET_ICON = { lvef: "pump", lvedd: "size", la_diam: "size", lvesd: "size", rv_diam: "size", ivsd: "wall", lvpwd: "wall", lv_mass: "wall" };

function shortTitle(title) {
  // "CardioTwin v0.1 (MUSIC, X/Y/Z leads) - ECG + age + sex (CardioTwin)" -> "ECG + age + sex"
  const parts = String(title || "").split(" - ");
  return (parts.length > 1 ? parts[parts.length - 1] : title).replace(/\s*\(CardioTwin\)\s*$/, "");
}

function plain(t) { return PLAIN[t] || { name: t, subject: t, adj: [], words: [], what: "", why: "" }; }

function displayName(item) {
  return state.medical ? item.label : (PLAIN[item.target]?.name || item.label);
}

function secondaryName(item) {
  return state.medical ? (PLAIN[item.target]?.name || "") : item.label;
}

function verdictWord(item) {
  if (state.medical) return item.category;
  const c = item.category || "";
  if (c.startsWith("Below normal")) return "Smaller than typical";
  if (c.startsWith("Above normal")) return "Above typical";
  if (item.severity < 0) return "Unknown";
  return plain(item.target).words[item.severity] || c;
}

function sevIcon(sev) {
  return sev === 0 ? "&#10003;" : sev === 1 ? "!" : sev >= 2 ? "!!" : "?";
}

function verdictChip(item) {
  const sev = item.severity ?? -1;
  return `<span class="verdict sev-${sev}"><i aria-hidden="true">${sevIcon(sev)}</i>${esc(verdictWord(item))}</span>`;
}

function confidence(item) {
  if (item.low90 === undefined || !item.display_range) return "";
  const [lo, hi] = item.display_range;
  const ratio = (item.high90 - item.low90) / (hi - lo);
  const level = ratio < 0.3 ? 3 : ratio < 0.5 ? 2 : 1;
  const word = level === 3 ? "Fairly sure" : level === 2 ? "Somewhat sure" : "Not very sure";
  return `<span class="confidence" title="Based on how wide the likely range is">${word}<b>${[1, 2, 3].map((k) => `<i class="${k <= level ? "on" : ""}"></i>`).join("")}</b></span>`;
}

function nd(item) {
  return item.unit === "g" ? 0 : Math.abs(item.value) < 20 ? 1 : 0;
}

function unitText(item) {
  return item.unit === "%" ? "%" : item.unit === "probability" ? "" : ` ${item.unit}`;
}

function sevColor(sev) {
  return [css("--good"), css("--warning"), css("--serious"), css("--critical")][clamp(sev, 0, 3)] || css("--muted");
}

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

// --------------------------------------------------------------------------------------
// Setup
// --------------------------------------------------------------------------------------
async function init() {
  setupTheme();
  setupTabs();
  setupUpload();
  state.medical = store.get("cardiotwin-medical", "0") === "1";
  $("#medical-toggle").checked = state.medical;
  $("#medical-toggle").addEventListener("change", (e) => {
    state.medical = e.target.checked;
    store.set("cardiotwin-medical", state.medical ? "1" : "0");
    if (state.result) render(state.result);
  });
  $("#anim-toggle").addEventListener("click", toggleAnimation);
  if (reducedMotion) $("#anim-toggle").textContent = "Animate";
  $("#saliency-toggle").addEventListener("change", () => state.result && drawECG(state.result));
  $("#use-demo-demographics").addEventListener("change", () => state.activeDemo && runDemo(state.activeDemo));
  $("#model-select").addEventListener("change", () => state.activeDemo && runDemo(state.activeDemo));
  $("#download-json").addEventListener("click", downloadJSON);
  document.addEventListener("click", (e) => {
    const go = e.target.closest("[data-goto]");
    if (go) showTab(go.dataset.goto, true);
  });
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => state.result && state.tab === "ecg" && drawECG(state.result), 120);
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => state.result && render(state.result));

  try {
    const { models } = await api("/api/models");
    state.models = models;
    if (!models.length) {
      $("#empty").hidden = false;
      return;
    }
    const sel = $("#model-select");
    for (const m of models) sel.add(new Option(shortTitle(m.title), m.name));
    $("#attribution").textContent = models.map((m) => m.license_note).filter(Boolean)[0] || "";
    const { demos } = await api("/api/demos");
    state.demos = demos;
    renderPeople();
    if (demos.length) runDemo(demos[Math.min(2, demos.length - 1)].id);
    else openUpload();
  } catch (err) {
    setStatus(`Could not reach the CardioTwin server: ${err.message}`, "error");
  }
}

function setupTheme() {
  const order = ["auto", "light", "dark"];
  const icons = {
    auto: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm0 18V4a8 8 0 0 1 0 16z"/></svg>',
    light: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4.5"/><path d="M11 1h2v4h-2zm0 18h2v4h-2zM1 11h4v2H1zm18 0h4v2h-4zM4.2 5.6l1.4-1.4 2.8 2.8-1.4 1.4zm11.4 11.4 1.4-1.4 2.8 2.8-1.4 1.4zM4.2 18.4l2.8-2.8 1.4 1.4-2.8 2.8zM15.6 7l2.8-2.8 1.4 1.4L17 8.4z"/></svg>',
    dark: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 15.5A8.5 8.5 0 0 1 8.5 4a8.5 8.5 0 1 0 11.5 11.5z"/></svg>',
  };
  let theme = store.get("cardiotwin-theme", "auto");
  const apply = () => {
    if (theme === "auto") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
    const btn = $("#theme-toggle");
    btn.innerHTML = icons[theme];
    btn.title = `Theme: ${theme}`;
    btn.setAttribute("aria-label", `Colour theme: ${theme}. Click to change.`);
    if (state.result) render(state.result);
  };
  $("#theme-toggle").addEventListener("click", () => {
    theme = order[(order.indexOf(theme) + 1) % order.length];
    store.set("cardiotwin-theme", theme);
    apply();
  });
  apply();
}

function setupTabs() {
  const tabs = $$(".tabs [role=tab]");
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => showTab(t.dataset.tab, true));
    t.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
      next.focus();
      showTab(next.dataset.tab, true);
    });
  });
  const fromHash = location.hash.replace("#", "");
  if (tabs.some((t) => t.dataset.tab === fromHash)) state.tab = fromHash;
  window.addEventListener("hashchange", () => {
    const h = location.hash.replace("#", "");
    if (tabs.some((t) => t.dataset.tab === h) && h !== state.tab) showTab(h);
  });
}

function showTab(name, userAction = false) {
  state.tab = name;
  $$(".tabs [role=tab]").forEach((t) => {
    const on = t.dataset.tab === name;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
  });
  $$(".panel").forEach((p) => { p.hidden = p.id !== `panel-${name}`; });
  if (userAction) {
    history.replaceState(null, "", `#${name}`);
    const tabsEl = $(".tabs");
    if (tabsEl.getBoundingClientRect().top < 0 || tabsEl.getBoundingClientRect().top > window.innerHeight * 0.6) {
      tabsEl.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "start" });
    }
  }
  if (name === "ecg" && state.result) drawECG(state.result);
}

function renderPeople() {
  const list = $("#demo-list");
  list.innerHTML = "";
  const rhythmWords = { sinus: "regular rhythm", "atrial fibrillation": "irregular rhythm (AF)", paced: "pacemaker", "atrial flutter": "atrial flutter" };
  state.demos.forEach((d, i) => {
    const b = document.createElement("button");
    b.className = "person";
    b.type = "button";
    b.dataset.id = d.id;
    b.setAttribute("aria-pressed", "false");
    const sex = d.sex === "M" ? "male" : d.sex === "F" ? "female" : "sex unknown";
    b.innerHTML = `<span class="avatar" aria-hidden="true">${i + 1}</span>
      <span class="t">Person ${i + 1}</span>
      <span class="s">${d.age ? Math.round(d.age) : "?"} y &middot; ${sex}<br>${esc(rhythmWords[d.rhythm] || d.rhythm || "")}</span>`;
    b.addEventListener("click", () => runDemo(d.id));
    list.appendChild(b);
  });
  const up = document.createElement("button");
  up.className = "person upload";
  up.type = "button";
  up.innerHTML = `<span class="avatar" aria-hidden="true">+</span><span class="t">Upload an ECG</span><span class="s">CSV, WFDB, NPZ, JSON</span>`;
  up.addEventListener("click", openUpload);
  list.appendChild(up);
}

function markPerson(id) {
  $$(".person[data-id]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.id === id)));
}

function openUpload() {
  const dlg = $("#upload-dialog");
  if (typeof dlg.showModal === "function") dlg.showModal();
  else dlg.setAttribute("open", "");
}

function closeUpload() {
  const dlg = $("#upload-dialog");
  if (typeof dlg.close === "function") dlg.close();
  else dlg.removeAttribute("open");
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
  $("#upload-close").addEventListener("click", closeUpload);
  $("#upload-cancel").addEventListener("click", closeUpload);
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
    closeUpload();
    state.activeDemo = null;
    markPerson(null);
    setStatus("Reading the ECG and building the digital twin...", "busy");
    try {
      render(await api("/api/predict", { method: "POST", body: fd }));
      setStatus(`Your upload: ${state.files.map((f) => f.name).join(", ")}`);
    } catch (err) {
      setStatus(err.message, "error");
    }
  });
}

async function runDemo(id) {
  state.activeDemo = id;
  markPerson(id);
  setStatus("Reading the ECG and building the digital twin...", "busy");
  try {
    const body = { demo_id: id, model: $("#model-select").value || null, use_demographics: $("#use-demo-demographics").checked };
    const res = await api("/api/predict/demo", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    render(res);
    const idx = state.demos.findIndex((x) => x.id === id);
    setStatus(`Showing person ${idx + 1} · ${res.model.uses_demographics ? "AI used the ECG + age and sex" : "AI used the ECG only"}`);
  } catch (err) {
    setStatus(err.message, "error");
  }
}

// --------------------------------------------------------------------------------------
// Rendering
// --------------------------------------------------------------------------------------
function render(res) {
  state.result = res;
  $("#results").hidden = false;
  renderWarnings(res);
  renderOverview(res);
  renderMeasurements(res);
  renderECGPanel(res);
  renderFuture(res);
  renderAbout(res);
  showTab(state.tab);
  startHeart(res);
}

function items(res) { return Object.fromEntries(res.profile.current.map((c) => [c.target, c])); }

function renderWarnings(res) {
  const el = $("#warnings");
  const w = res.warnings || [];
  el.hidden = !w.length;
  el.innerHTML = w.length ? `<strong>Please note</strong><ul>${w.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "";
}

// ---------------------------------------------------------------- overview
function headline(res) {
  const weight = { lvef: 1.5, lvedd: 0.5, la_diam: 0.3 };
  const abnormal = res.profile.current.filter((c) => c.severity >= 1)
    .sort((a, b) => (b.severity + (weight[b.target] || 0)) - (a.severity + (weight[a.target] || 0))
      || (b.p_abnormal ?? 0) - (a.p_abnormal ?? 0));
  if (state.medical) {
    if (!abnormal.length) return "No structural abnormality is estimated.";
    return `Estimated: ${abnormal.slice(0, 3).map((c) => `${c.category.toLowerCase()} ${c.label}`).join("; ")}.`;
  }
  if (!abnormal.length) return "This heart looks typical in the AI's estimate.";
  const phrases = abnormal.slice(0, 2).map((c) => `${plain(c.target).subject} may be ${plain(c.target).adj[c.severity] || c.category.toLowerCase()}`);
  return `CardioTwin thinks ${phrases.join(", and ")}.`;
}

function renderOverview(res) {
  const it = items(res);
  $("#ov-headline").textContent = headline(res);
  const risk = res.profile.structural_risk;
  $("#ov-eyebrow").textContent = `The big picture · overall: ${riskWord(risk.level)}`;
  const points = [];
  const bullet = (sev, html) => points.push(`<li><span class="bullet" style="background:${sevColor(sev)}">${sevIcon(sev)}</span><span>${html}</span></li>`);
  if (it.lvef) {
    const v = it.lvef;
    bullet(v.severity, state.medical
      ? `LVEF ${fmt(v.value)}% (90% interval ${fmt(v.low90)}&ndash;${fmt(v.high90)}%): ${esc(v.category.toLowerCase())}.`
      : `The heart pushes out about <strong>${fmt(v.value)}%</strong> of its blood with each beat. Healthy hearts push out about 52&ndash;72%.`);
  }
  if (it.lvedd) {
    const v = it.lvedd;
    const typical = v.normal_range ? (v.normal_range[0] + v.normal_range[1]) / 2 : 50;
    const pct = Math.round((v.value / typical - 1) * 100);
    bullet(v.severity, state.medical
      ? `LVEDD ${fmt(v.value)} mm: ${esc(v.category.toLowerCase())}.`
      : `The main pumping chamber is about <strong>${Math.abs(pct)}% ${pct >= 0 ? "bigger" : "smaller"}</strong> than in a typical heart.`);
  }
  const walls = ["ivsd", "lvpwd"].filter((t) => it[t]);
  if (walls.length) {
    const avg = walls.reduce((s, t) => s + it[t].value, 0) / walls.length;
    const worst = walls.map((t) => it[t]).sort((a, b) => b.severity - a.severity)[0];
    bullet(worst.severity, state.medical
      ? `Wall thickness ${walls.map((t) => `${it[t].label.split(" ")[0]} ${fmt(it[t].value, 1)} mm`).join(", ")}.`
      : `The heart walls are about <strong>${fmt(avg, 1)} mm</strong> thick (${esc(verdictWord(worst).toLowerCase())}; typical is 6&ndash;10 mm).`);
  }
  const fr = res.profile.future && res.profile.future.risk;
  if (fr) {
    const n = Math.round(fr.probability * 100);
    bullet(fr.level === "HIGH" ? 3 : fr.level === "MODERATE" ? 1 : 0, state.medical
      ? `${fmt(fr.horizon_years)}-year risk of ${esc(fr.label.toLowerCase())}: ${n}% (${esc(fr.level.toLowerCase())}).`
      : `About <strong>${n} in 100</strong> people like this have a serious heart-failure event within ${fmt(fr.horizon_years)} years.`);
  }
  $("#ov-points").innerHTML = points.join("");

  const reality = $("#ov-reality");
  if (it.lvef && it.lvef.truth !== undefined) {
    const diff = Math.round(it.lvef.value - it.lvef.truth);
    reality.hidden = false;
    reality.innerHTML = `<span class="r-icon" aria-hidden="true">&#10003;</span><span><strong>Reality check:</strong> the real heart scan measured a pumping strength of <strong>${fmt(it.lvef.truth)}%</strong>. The AI said ${fmt(it.lvef.value)}% &mdash; ${diff === 0 ? "spot on" : `${Math.abs(diff)} points too ${diff > 0 ? "high" : "low"}`}.</span>`;
  } else {
    reality.hidden = true;
  }
  renderTiles(res, it);
}

function riskWord(level) {
  return { LOW: "looks mostly typical", MODERATE: "some changes possible", HIGH: "changes likely", UNKNOWN: "unknown" }[level] || level;
}

function gaugeSVG(item) {
  // Semicircle 0..80% with ASE zones: <30, 30-40, 41-51, 52+.
  const [lo, hi] = [0, 80];
  const cx = 110, cy = 100, r = 80;
  const ang = (v) => Math.PI * (1 - (clamp(v, lo, hi) - lo) / (hi - lo));
  const pt = (v, rad = r) => [cx + rad * Math.cos(ang(v)), cy - rad * Math.sin(ang(v))];
  const arc = (a, b) => {
    const [x0, y0] = pt(a), [x1, y1] = pt(b);
    return `M${x0},${y0} A${r},${r} 0 0 1 ${x1},${y1}`;
  };
  const zones = [[0, 29.5, css("--critical")], [30, 40.5, css("--serious")], [41, 51.5, css("--warning")], [52, 80, css("--good")]];
  const [nx, ny] = pt(item.value, r - 22);
  let truth = "";
  if (item.truth !== undefined) {
    const [tx, ty] = pt(item.truth, r + 13);
    truth = `<circle cx="${tx}" cy="${ty}" r="5" fill="${css("--truth")}" stroke="${css("--card")}" stroke-width="2"><title>Real heart scan: ${fmt(item.truth)}%</title></circle>`;
  }
  return `<svg class="gauge" viewBox="0 0 220 122" aria-hidden="true">
    ${zones.map(([a, b, c]) => `<path d="${arc(a, b)}" stroke="${c}" stroke-width="14" fill="none" opacity="0.9"/>`).join("")}
    <line x1="${cx}" y1="${cy}" x2="${nx}" y2="${ny}" stroke="${css("--ink")}" stroke-width="4" stroke-linecap="round"/>
    <circle cx="${cx}" cy="${cy}" r="7" fill="${css("--ink")}"/>
    ${truth}
    <text x="${cx - r}" y="${cy + 16}" text-anchor="middle">0%</text>
    <text x="${cx + r}" y="${cy + 16}" text-anchor="middle">80%</text>
  </svg>`;
}

function renderTiles(res, it) {
  const tiles = [];
  if (it.lvef) {
    const v = it.lvef;
    tiles.push({ goto: "heart", icon: "pump", title: state.medical ? "LV ejection fraction" : "Pumping strength",
      sub: state.medical ? "LVEF" : "share of blood pumped per beat",
      visual: gaugeSVG(v), big: `<span data-count="${v.value}">${fmt(v.value)}</span><small>%</small>`, chip: verdictChip(v),
      note: confidence(v) });
  }
  if (it.lvedd) {
    const v = it.lvedd;
    const typical = v.normal_range ? (v.normal_range[0] + v.normal_range[1]) / 2 : 50;
    const scale = 70 / Math.max(typical, v.value);
    const pct = Math.round((v.value / typical - 1) * 100);
    tiles.push({ goto: "heart", icon: "size", title: state.medical ? "LV end-diastolic diameter" : "Heart size",
      sub: state.medical ? "LVEDD" : "main chamber, relaxed",
      visual: `<div class="size-compare"><figure><div class="ball typ" style="width:${typical * scale}px;height:${typical * scale}px"></div><figcaption>typical</figcaption></figure>
               <figure><div class="ball this" style="width:${v.value * scale}px;height:${v.value * scale}px"></div><figcaption>this heart</figcaption></figure></div>`,
      big: state.medical ? `<span data-count="${v.value}">${fmt(v.value)}</span><small>mm</small>` : `${pct >= 0 ? "+" : "&minus;"}<span data-count="${Math.abs(pct)}">${Math.abs(pct)}</span><small>% vs typical</small>`,
      chip: verdictChip(v), note: confidence(v) });
  }
  const walls = ["ivsd", "lvpwd"].filter((t) => it[t]).map((t) => it[t]);
  if (walls.length) {
    const avg = walls.reduce((s, w) => s + w.value, 0) / walls.length;
    const worst = [...walls].sort((a, b) => b.severity - a.severity)[0];
    const ring = (mm, cls, color) => `<circle cx="60" cy="56" r="${34 - mm * 0.9}" fill="none" stroke="${color}" stroke-width="${mm * 1.9}" class="${cls}"/>`;
    tiles.push({ goto: "heart", icon: "wall", title: state.medical ? "Wall thickness" : "Heart wall thickness",
      sub: state.medical ? "IVSd / LVPWd (mean)" : "how thick the muscle is",
      visual: `<svg viewBox="0 0 120 112" width="120" aria-hidden="true">${ring(avg, "", css("--myo"))}<circle cx="60" cy="56" r="${34 - avg * 1.85}" fill="${css("--blood")}"/><circle cx="60" cy="56" r="${34 - 9 * 1.85}" fill="none" stroke="${css("--ghost")}" stroke-dasharray="4 3" stroke-width="1.5"><title>Inner edge of a typical 9 mm wall</title></circle></svg>`,
      big: `<span data-count="${avg}" data-nd="1">${fmt(avg, 1)}</span><small>mm</small>`, chip: verdictChip(worst),
      note: `<span class="confidence">typical: 6&ndash;10 mm</span>` });
  }
  const fr = res.profile.future && res.profile.future.risk;
  if (fr) {
    const n = Math.round(fr.probability * 100);
    const color = fr.level === "HIGH" ? css("--critical") : fr.level === "MODERATE" ? css("--warning") : css("--good");
    tiles.push({ goto: "future", icon: "future", title: state.medical ? "Event risk" : "Future outlook",
      sub: `${fmt(fr.horizon_years)}-year heart-failure risk`,
      visual: `<div class="mini-people" aria-hidden="true">${Array.from({ length: 100 }, (_, k) => `<i style="${k < n ? `background:${color}` : ""}"></i>`).join("")}</div>`,
      big: `<span data-count="${n}">${n}</span><small>in 100</small>`,
      chip: `<span class="verdict sev-${fr.level === "HIGH" ? 3 : fr.level === "MODERATE" ? 1 : 0}"><i aria-hidden="true">${sevIcon(fr.level === "HIGH" ? 3 : fr.level === "MODERATE" ? 1 : 0)}</i>${state.medical ? esc(fr.level) + " risk" : { HIGH: "Higher risk", MODERATE: "Medium risk", LOW: "Lower risk" }[fr.level]}</span>`,
      note: "" });
  }
  $("#ov-tiles").innerHTML = tiles.map((t) => `
    <article class="card tile" data-goto="${t.goto}" tabindex="0" role="button" aria-label="${esc(t.title)}: open details">
      <div class="tile-head"><span class="tile-icon" aria-hidden="true">${ICONS[t.icon]}</span><div><div class="tile-title">${esc(t.title)}</div><div class="tile-sub">${esc(t.sub)}</div></div></div>
      <div class="tile-visual">${t.visual}</div>
      <div class="tile-big">${t.big}</div>
      ${t.chip}
      ${t.note ? `<div>${t.note}</div>` : ""}
    </article>`).join("");
  $$("#ov-tiles .tile").forEach((el) => el.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); showTab(el.dataset.goto, true); }
  }));
  countUp($("#ov-tiles"));
}

function countUp(root) {
  if (reducedMotion) return;
  $$("[data-count]", root).forEach((el) => {
    const to = Number(el.dataset.count), d = Number(el.dataset.nd || 0);
    const t0 = performance.now(), dur = 700;
    const step = (now) => {
      const p = Math.min(1, (now - t0) / dur);
      el.textContent = (to * (1 - Math.pow(1 - p, 3))).toFixed(d);
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
}

// ------------------------------------------------------------ heart tab
function rangeBar(item) {
  const [lo, hi] = item.display_range;
  const x = (v) => ((clamp(v, lo, hi) - lo) / (hi - lo)) * 100;
  const parts = ['<div class="track"></div>'];
  if (item.normal_range) {
    const a = x(item.normal_range[0]), b = x(item.normal_range[1]);
    parts.push(`<div class="band" style="left:${a}%;width:${Math.max(1, b - a)}%" title="Healthy range"></div>`);
  }
  if (item.low90 !== undefined) {
    const a = x(item.low90), b = x(item.high90);
    parts.push(`<div class="int" style="left:${a}%;width:${Math.max(0.5, b - a)}%"></div>`);
  }
  if (item.truth !== undefined) parts.push(`<div class="truth" style="left:${x(item.truth)}%" title="Real heart scan: ${fmt(item.truth, nd(item))}${esc(unitText(item))}"></div>`);
  parts.push(`<div class="dot" style="left:${x(item.value)}%" title="AI estimate: ${fmt(item.value, nd(item))}${esc(unitText(item))}"></div>`);
  parts.push(`<span class="tick">${lo}</span><span class="tick r">${hi}</span>`);
  return `<div class="rbar" aria-hidden="true">${parts.join("")}</div>`;
}

function renderMeasurements(res) {
  $("#measure-grid").innerHTML = res.profile.current.map((it) => {
    const p = plain(it.target);
    const d = nd(it);
    const u = unitText(it);
    const healthy = it.normal_range ? `healthy: ${fmt(it.normal_range[0], d)}&ndash;${fmt(it.normal_range[1], d)}${esc(u)}` : "";
    const truth = it.truth !== undefined
      ? `<div class="mcard-truth"><i class="lg-truth"></i>${state.medical ? "Echo" : "Real heart scan"}: <strong>${fmt(it.truth, d)}${esc(u)}</strong></div>` : "";
    const pa = it.p_abnormal !== null && it.p_abnormal !== undefined
      ? `<span class="small muted">${fmt(it.p_abnormal * 100)}% chance it's outside the healthy range</span>` : "";
    const why = [p.why, p.analogy].filter(Boolean).map((s) => `<p>${esc(s)}</p>`).join("");
    return `<article class="card mcard">
      <div class="mcard-head">
        <div><div class="mcard-name">${esc(displayName(it))}</div><div class="mcard-med">${esc(secondaryName(it))}</div></div>
        <div class="mcard-val"><span class="mcard-num">${fmt(it.value, d)}</span><span class="mcard-unit">${esc(u)}</span>
          <div class="mcard-range">likely ${fmt(it.low90, d)}&ndash;${fmt(it.high90, d)}</div></div>
      </div>
      ${rangeBar(it)}
      <div class="mcard-row">${verdictChip(it)}${confidence(it)}</div>
      <p class="mcard-text">${esc(state.medical ? it.description : (p.what || it.description))} <span class="muted">${healthy}</span></p>
      ${truth}
      ${pa}
      ${why ? `<details><summary>Why does this matter?</summary>${why}</details>` : ""}
    </article>`;
  }).join("");
}

// ------------------------------------------------------------------ ECG tab
function renderECGPanel(res) {
  const f = res.features || {};
  const facts = [];
  if (f.heart_rate) {
    const hr = f.heart_rate;
    const note = hr < 60 ? "Slower than the typical 60-100 at rest (common in fit people or with some medicines)."
      : hr > 100 ? "Faster than the typical 60-100 at rest." : "In the typical resting range (60-100).";
    facts.push(["Heart rate", `${fmt(hr)} <small>beats/min</small>`, note]);
  }
  if (f.qrs_ms) {
    facts.push(["QRS width", `${fmt(f.qrs_ms)} <small>ms</small>`, f.qrs_ms > 120
      ? "Wide: the electrical wave spreads slowly through the main chambers (e.g. a bundle-branch block)."
      : "Normal speed of the electrical wave through the main chambers (under 120 ms)."]);
  }
  if (f.rr_cv !== null && f.rr_cv !== undefined) {
    facts.push(["Rhythm", f.rr_cv > 0.12 ? "Irregular" : "Regular", f.rr_cv > 0.12
      ? "Beats are unevenly spaced, as in atrial fibrillation." : "Beats arrive at an even pace."]);
  }
  facts.push(["Leads", esc(res.ecg.leads.join(" · ")), `Recorded at ${fmt(res.input.fs)} Hz for ${fmt(res.input.duration_s, 1)} s.`]);
  $("#ecg-facts").innerHTML = facts.map(([k, v, n]) => `<div class="fact"><div class="k">${k}</div><div class="v">${v}</div><div class="n">${n}</div></div>`).join("");
  const derived = (res.input.derived_leads || []).length ? ` Leads ${res.input.derived_leads.join(", ")} were calculated from the uploaded leads.` : "";
  $("#ecg-caption").textContent = `Drawn like ECG paper: each small square is 0.04 s wide and 0.1 mV tall.${derived}`;
  const seg = res.explain && res.explain.segments;
  if (seg) {
    const parts = [["QRS spike", seg.QRS, css("--beat-qrs")], ["T wave", seg.T, css("--beat-t")], ["P wave", seg.P, css("--beat-p")], ["In between", seg.other, css("--line")]];
    $("#attention-bar").innerHTML = parts.map(([, v, c]) => `<span style="width:${(v * 100).toFixed(1)}%;background:${c}"></span>`).join("");
    $("#attention-legend").innerHTML = parts.map(([n, v, c]) => `<div><span><i style="background:${c}"></i>${n}</span><strong>${fmt(v * 100)}%</strong></div>`).join("");
  } else {
    $("#attention-bar").innerHTML = "";
    $("#attention-legend").innerHTML = '<div class="muted">Not available for this model.</div>';
  }
}

function drawECG(res) {
  const canvas = $("#ecg-canvas");
  if (!canvas.parentElement.clientWidth) return; // hidden tab: draw when shown
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
    if (step === 1 && pxPerMm < 2.2) continue;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    for (let mm = 0; mm * pxPerMm <= cssW + 0.5; mm += step) line(mm * pxPerMm, 0, mm * pxPerMm, cssH);
    for (let mm = 0; mm * pxPerMm <= cssH + 0.5; mm += step) line(0, mm * pxPerMm, cssW, mm * pxPerMm);
  }
  ctx.strokeStyle = css("--trace");
  ctx.lineWidth = 1.4;
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
    ctx.font = "700 13px system-ui, sans-serif";
    ctx.fillText(name, 10, i * rowH + 18);
  });
  canvas.setAttribute("aria-label", `ECG tracing, ${leads.length} leads, ${dur.toFixed(0)} seconds`);
}

// ---------------------------------------------------------------- future tab
const PERSON_SVG = (fill) => `<svg viewBox="0 0 20 28" aria-hidden="true"><circle cx="10" cy="6" r="5" class="person-icon" ${fill ? `style="fill:${fill}"` : ""}/><path d="M2 27v-7a8 8 0 0 1 16 0v7z" class="person-icon" ${fill ? `style="fill:${fill}"` : ""}/></svg>`;

function renderFuture(res) {
  const fut = res.profile.future;
  const el = $("#future-body");
  if (!fut) {
    el.innerHTML = `<div class="card"><h2>No forecast for this model</h2><p class="muted" style="margin-top:8px">Predicting the future needs training data that follows people over time (Levels 12-13): a second heart scan months later, or what happened to them. The bundled MUSIC models have this; models you train with <code>cardiotwin train</code> do not.</p></div>`;
    return;
  }
  let html = "";
  if (fut.risk) {
    const r = fut.risk;
    const n = Math.round(r.probability * 100);
    const color = r.level === "HIGH" ? css("--critical") : r.level === "MODERATE" ? css("--warning") : css("--good");
    const sev = r.level === "HIGH" ? 3 : r.level === "MODERATE" ? 1 : 0;
    const pct = r.percentile;
    html += `<div class="future-grid">
      <div class="card future-hero">
        <div class="eyebrow">Future heart &middot; next ${fmt(r.horizon_years)} years</div>
        <div class="future-big">${n}%</div>
        <p class="future-sentence">${state.medical
          ? `Estimated ${fmt(r.horizon_years)}-year risk of ${esc(r.label.toLowerCase())}.`
          : `About <strong>${n} out of 100</strong> people with an ECG like this one have a serious heart-failure event (death from heart failure or a heart transplant) within ${fmt(r.horizon_years)} years.`}</p>
        <span class="verdict sev-${sev}"><i aria-hidden="true">${sevIcon(sev)}</i>${state.medical ? esc(r.level) + " risk" : { HIGH: "Higher than most", MODERATE: "In the middle", LOW: "Lower than most" }[r.level]}</span>
        ${pct !== null && pct !== undefined ? `<div class="pctl"><p class="small" style="margin-bottom:8px">Compared with everyone in the study, this risk is higher than <strong>${fmt(pct)}%</strong> of them.</p>
          <div class="pctl-track"><div class="pctl-marker" style="left:${clamp(pct, 0, 100)}%"></div></div>
          <div class="pctl-scale"><span>lowest risk</span><span>highest risk</span></div></div>` : ""}
        <p class="muted small">Only meaningful for people like the study group: ${esc(r.population || "see the model card")}. It is an average over many people, not a prediction for one person, and treatment can change it.</p>
      </div>
      <div class="card">
        <h3>What ${n}% looks like</h3>
        <p class="muted small" style="margin-bottom:12px">Each figure is one person with a similar ECG. Coloured figures had an event within ${fmt(r.horizon_years)} years.</p>
        <div class="people" role="img" aria-label="${n} out of 100 figures coloured">${Array.from({ length: 100 }, (_, k) => PERSON_SVG(k < n ? color : null)).join("")}</div>
      </div>
    </div>`;
  }
  if (fut.measurements && fut.measurements.length) {
    html += `<div class="future-cards">${fut.measurements.map((m) => {
      const p = plain(m.target);
      const value = m.kind === "binary" ? `${fmt(m.value * 100)}%` : `${fmt(m.value, 1)}${esc(unitText(m))}`;
      return `<article class="card mcard"><div class="mcard-name">${esc(state.medical ? m.label : p.name)}</div>
        <div class="mcard-num">${value}</div>${m.kind === "binary" ? "" : verdictChip(m)}
        <p class="mcard-text">${esc(p.what || m.description)}</p></article>`;
    }).join("")}</div>`;
  }
  el.innerHTML = html;
}

// ----------------------------------------------------------------- about tab
function renderAbout(res) {
  const m = state.models.find((x) => x.name === res.model.name) || {};
  const tm = m.test_metrics || {};
  const rows = [];
  if (tm.lvef && tm.lvef.mae !== undefined) {
    rows.push([`&plusmn;${fmt(tm.lvef.mae)}`, "points, on average", `Pumping-strength estimates were this far from the real heart scan, on average, for ${m.cohort ? m.cohort.patients : "the"} people the AI had never seen.`]);
  }
  if (tm.lvef && tm.lvef.auroc_ef_le_40) {
    rows.push([`${fmt(tm.lvef.auroc_ef_le_40 * 100)}%`, "spots the weaker pump", "Shown one person with a weak pump (40% or less) and one without, it picks the right one this often. 50% would be a coin flip."]);
  }
  if (m.future && m.future.test_c_index) {
    rows.push([`${fmt(m.future.test_c_index[0] * 100)}%`, "ranks future risk", "Shown two people, it correctly guesses who had a heart-failure event first this often (50% = coin flip)."]);
  }
  const table = Object.entries(tm).filter(([t]) => t in items(res)).map(([t, v]) => {
    const it = items(res)[t];
    return `<tr><td>${esc(displayName(it))}</td><td class="num">${fmt(v.mae, 1)}${esc(unitText(it))}</td><td class="num">${fmt(v.pearson_r, 2)}</td></tr>`;
  }).join("");
  $("#accuracy-card").innerHTML = `
    <h3>How accurate is it?</h3>
    <p class="muted small" style="margin-bottom:6px">Model: ${esc(shortTitle(m.title || res.model.title))} &middot; ${m.n_members || "?"} networks &middot; ${m.uses_demographics ? "uses the ECG plus age and sex" : "uses the ECG only"}</p>
    ${rows.map(([num, title, text]) => `<div class="acc-item"><div class="acc-num">${num}</div><div class="acc-title">${title}</div><div class="acc-text">${text}</div></div>`).join("")}
    <details style="margin-top:10px"><summary class="small" style="cursor:pointer;font-weight:700">Show all the numbers</summary>
      <table class="metrics"><thead><tr><th>Measurement</th><th style="text-align:right">Average miss</th><th style="text-align:right">Correlation r</th></tr></thead><tbody>${table}</tbody></table>
      <p class="muted small" style="margin-top:6px">${esc(m.dataset || "")}. Correlation r runs from 0 (no relationship) to 1 (perfect).</p>
    </details>`;
  $("#glossary").innerHTML = GLOSSARY.map(([k, v]) => `<details><summary>${esc(k)}</summary><p>${esc(v)}</p></details>`).join("");
}

function downloadJSON() {
  if (!state.result) return;
  const { ecg, explain, ...rest } = state.result;
  const payload = { ...rest, explain: explain ? { target: explain.target, segments: explain.segments } : null, exported_at: new Date().toISOString() };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `cardiotwin-estimate-${state.activeDemo || "upload"}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// --------------------------------------------------------------------- heart
function heartDims(res) {
  const v = Object.fromEntries(res.profile.current.map((c) => [c.target, c.value]));
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
    raw: { la: v.la_diam, edd: v.lvedd, ef: v.lvef,
           walls: v.ivsd !== undefined && v.lvpwd !== undefined ? (v.ivsd + v.lvpwd) / 2 : v.ivsd ?? v.lvpwd },
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

function heartMarkup(d, c, atr, valveOpen, callouts) {
  const s = 1.5;
  const x0 = callouts ? 178 : 192, y0 = 160;
  const D = d.edd - (d.edd - d.esd) * c;
  const thick = 1 + 0.45 * c * (d.ef / 60);
  const lv = lvPaths(x0, y0, D, d.ivs * thick, d.pw * thick, s);
  const lvDia = lvPaths(x0, y0, d.edd, d.ivs, d.pw, s);
  const ghost = lvPaths(x0, y0, 48, 9, 9, s);
  const Sx = x0 - lvDia.a - lvDia.ts, Sy = y0 + 2, L = lvDia.h1 + lvDia.ry;
  const w = d.rv * s * 0.8 * (1 - 0.18 * c);
  const rv = `M${Sx + 2},${Sy} C${Sx - w * 0.6},${Sy - 8} ${Sx - w * 1.05},${Sy + 10} ${Sx - w},${Sy + 42} C${Sx - w * 0.95},${Sy + L * 0.55} ${Sx - w * 0.4},${Sy + L * 0.8} ${Sx + 2},${Sy + L * 0.82} Z`;
  const fill = 1 - 0.08 * atr + 0.04 * c;
  const laRx = d.la * s * 0.5 * fill, laRy = laRx * 0.86;
  const laCx = x0 + lv.a * 0.2, laCy = y0 - laRy * 0.78;
  const raRx = 19 * s * fill, raRy = raRx * 0.9;
  const raCx = Sx - w * 0.45, raCy = y0 - raRy * 0.7;
  const gR = 35 * s * 0.5;
  const leaf = 16 * valveOpen;
  const names = state.medical ? ["LV", "LA", "RA", "RV"] : ["Main", "Upper L", "Upper R", "Right"];
  let extra = "";
  if (callouts) {
    const cx = 318;
    const call = (ax, ay, ty, name, val) => `<polyline class="callout-line" fill="none" points="${ax},${ay} ${cx - 8},${ty} ${cx},${ty}"/>
      <circle cx="${ax}" cy="${ay}" r="3" fill="${css("--ink")}"/>
      <text class="callout-name" x="${cx + 4}" y="${ty - 3}">${name}</text><text class="callout-val" x="${cx + 4}" y="${ty + 12}">${val}</text>`;
    const r = d.raw;
    const val = (v, unit, digits = 0) => (v === undefined || v === null ? "not estimated" : `${fmt(v, digits)}${unit}`);
    extra = [
      call(laCx + laRx * 0.7, laCy - laRy * 0.5, 40, state.medical ? "Left atrium" : "Upper-left chamber", val(r.la, " mm")),
      call(x0 + lv.a * 0.6, y0 + lv.h1 * 0.6, 150, state.medical ? "LV diameter (live)" : "Main chamber width",
           r.edd === undefined ? "not estimated" : `${fmt(D)} mm`),
      call(x0 + lv.a + lv.tp * 0.5, y0 + lv.h1 + lv.ry * 0.5, 235, "Wall thickness", val(r.walls, " mm", 1)),
      call(x0, y0 + lv.h1 + lv.ry * 0.75, 320, state.medical ? "Ejection fraction" : "Pumping strength", val(r.ef, "%")),
    ].join("");
  }
  return `
    <ellipse class="atrium ra" cx="${raCx}" cy="${raCy}" rx="${raRx}" ry="${raRy}"/>
    <ellipse class="atrium" cx="${laCx}" cy="${laCy}" rx="${laRx}" ry="${laRy}"/>
    <path class="rv" d="${rv}"/>
    <path class="myo" d="${lv.outer}"/>
    <path class="blood" d="${lv.cavity}"/>
    <line class="valve" x1="${x0 - lv.a}" y1="${y0}" x2="${x0 - lv.a * 0.2}" y2="${y0 + leaf}"/>
    <line class="valve" x1="${x0 + lv.a}" y1="${y0}" x2="${x0 + lv.a * 0.2}" y2="${y0 + leaf}"/>
    <path class="ghost" d="${ghost.outer}"/>
    <ellipse class="ghost" cx="${x0 + ghost.a * 0.2}" cy="${y0 - gR * 0.86 * 0.78}" rx="${gR}" ry="${gR * 0.86}"/>
    <text class="lbl-in" x="${x0}" y="${y0 + lv.h1 + lv.ry * 0.3}" text-anchor="middle">${names[0]}</text>
    <text class="lbl-in" x="${laCx}" y="${laCy - laRy * 0.15}" text-anchor="middle">${names[1]}</text>
    <text class="lbl-in" x="${raCx}" y="${raCy - raRy * 0.1}" text-anchor="middle" style="font-size:10px">${names[2]}</text>
    <text class="lbl-in" x="${Sx - w * 0.5}" y="${Sy + L * 0.42}" text-anchor="middle" style="font-size:10px">${names[3]}</text>
    ${callouts ? "" : `<text class="lbl-sub" x="160" y="352" text-anchor="middle">${fmt(d.hr)} beats per minute</text>`}
    ${extra}`;
}

function drawHearts(d, c, atr, valveOpen) {
  $$(".heart-svg").forEach((svg) => {
    if (!svg.getBoundingClientRect().width) return; // hidden
    svg.innerHTML = heartMarkup(d, c, atr, valveOpen, svg.dataset.callouts === "1");
  });
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
  const token = ++state.loopToken;
  if (!state.playing) {
    // Static drawing; re-drawn periodically so hearts on newly shown tabs appear.
    const idle = () => {
      if (token !== state.loopToken) return;
      drawHearts(d, 0, 0, 1);
      setTimeout(idle, 400);
    };
    idle();
    return;
  }
  const period = 60000 / d.hr;
  const t0 = performance.now();
  const tick = (now) => {
    if (token !== state.loopToken) return;
    const { c, atr, valveOpen } = cycle(((now - t0) % period) / period);
    drawHearts(d, c, atr, valveOpen);
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function toggleAnimation() {
  state.playing = !state.playing;
  const btn = $("#anim-toggle");
  btn.textContent = state.playing ? "Pause" : "Animate";
  btn.setAttribute("aria-pressed", String(state.playing));
  if (state.result) startHeart(state.result);
}

init();
