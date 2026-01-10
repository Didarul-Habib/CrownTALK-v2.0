/* ============================================
   CrownTALK — One-time Access Gate + App Logic
   Access code: @CrownTALK@2026@CrownDEX
   Persists with localStorage + cookie fallback
   ============================================ */

/* ---------- Gate Pass ---------- */
(() => {
  const ACCESS_CODE = '@CrownTALK@2026@CrownDEX';
  const STORAGE_KEY = 'crowntalk_access_v1';    // local/session storage key
  const COOKIE_KEY  = 'crowntalk_access_v1';    // cookie fallback

  function isAuthorized() {
    try { if (localStorage.getItem(STORAGE_KEY) === '1') return true; } catch {}
    try { if (sessionStorage.getItem(STORAGE_KEY) === '1') return true; } catch {}
    try {
      const m = document.cookie.match(new RegExp('(?:^|; )' + COOKIE_KEY + '=([^;]*)'));
      if (m && decodeURIComponent(m[1]) === '1') return true;
    } catch {}
    return false;
  }

  function setAuthorized() {
    try { localStorage.setItem(STORAGE_KEY, '1'); } catch {}
    try { sessionStorage.setItem(STORAGE_KEY, '1'); } catch {}
    try {
      document.cookie = `${COOKIE_KEY}=1; path=/; max-age=${60 * 60 * 24 * 365}; samesite=lax`;
    } catch {}
  }

  function els() {
    const gate = document.getElementById('accessGate');
    const input = document.getElementById('accessCodeInput');
    const btn = document.getElementById('accessBtn');
    return { gate, input, btn };
  }

  function unlock() {
    const { gate } = els();
    if (!gate) return;
    gate.classList.add('is-unlocked');
    gate.setAttribute('aria-hidden', 'true');
    document.body.classList.add('is-authorized');
    try { bootAppUI(); } catch {}
  }

  function handleTry() {
    const { input } = els();
    if (!input) return;
    const v = (input.value || '').trim();
    if (v === ACCESS_CODE) {
      setAuthorized();
      unlock();
      return;
    }

    input.classList.add('ct-shake');
    setTimeout(() => input.classList.remove('ct-shake'), 350);
    input.value = '';
    input.placeholder = 'Wrong code — try again';
  }

  function bindGate() {
    const { gate, input, btn } = els();
    if (!gate) return;

    input?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleTry();
      }
    });

    btn?.addEventListener('click', (e) => {
      e.preventDefault();
      handleTry();
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    bindGate();
    if (isAuthorized()) unlock();
  });
})();

/* ============================================
   Main App — URL -> comments generator
   ============================================ */

/* ---------- Elements ---------- */
const urlInput       = document.getElementById("urlInput");
const generateBtn    = document.getElementById("generateBtn");
const cancelBtn      = document.getElementById("cancelBtn");
const clearBtn       = document.getElementById("clearBtn");
const progressText   = document.getElementById("progress");
const progressFill   = document.getElementById("progressBarFill");
const resultsEl      = document.getElementById("results");
const failedEl       = document.getElementById("failed");
const resultCountEl  = document.getElementById("resultCount");
const failedCountEl  = document.getElementById("failedCount");
const historyEl      = document.getElementById("history");
const clearHistoryBtn= document.getElementById("clearHistoryBtn");
const yearEl         = document.getElementById("year");

// theme dots (live node list -> array)
let themeDots = Array.from(document.querySelectorAll(".theme-dot"));

/* ========= PATCH: premium DOM handles ========= */
let sessionTabsEl     = document.getElementById("sessionTabs");
const analyticsHudEl    = document.getElementById("analyticsHud");
const urlHealthBadgeEl  = document.getElementById("urlHealthBadge");
const sortUrlsBtn       = document.getElementById("sortUrlsBtn");
const shuffleUrlsBtn    = document.getElementById("shuffleUrlsBtn");
const removeInvalidBtn  = document.getElementById("removeInvalidBtn");
const copyQueuePanel    = document.getElementById("copyQueuePanel");
const copyQueueListEl   = document.getElementById("copyQueueList");
const copyQueueNextBtn  = document.getElementById("copyQueueNextBtn");
const copyQueueClearBtn = document.getElementById("copyQueueClearBtn");
const presetSelect      = document.getElementById("presetSelect");
let keyboardHudEl     = document.getElementById("keyboardHud") || document.querySelector(".keyboard-hud");
const exportAllBtn      = document.getElementById("exportAllBtn") || document.getElementById("copyAllBtn");
const exportEnBtn       = document.getElementById("exportEnBtn") || document.getElementById("copyAllEnBtn");
const exportNativeBtn   = document.getElementById("exportNativeBtn") || document.getElementById("copyAllNativeBtn");
const downloadTxtBtn    = document.getElementById("downloadTxtBtn");

// ------------------------
// State
// ------------------------
let cancelled    = false;
let historyItems = [];

/* ========= PATCH: premium state ========= */
const COPY_QUEUE_ENABLED = false; // user requested: do NOT use Copy Queue side panel
let ctSessions = [];
let ctActiveSessionId = null;
let ctSessionCounter = 0;

let ctCopyQueue = [];
let ctCopyQueueIndex = 0;

/* =========================================================
   === PATCH: Mini Toast + Snack (used by multiple bits) ===
   ========================================================= */
(function mountToast(){
  if (document.getElementById('ctToasts')) return;
  const box = document.createElement('div');
  box.id = 'ctToasts';
  box.style.cssText = 'position:fixed;right:14px;bottom:14px;display:flex;flex-direction:column;gap:10px;z-index:9999;pointer-events:none;';
  document.body.appendChild(box);
})();
function ctToast(msg, type='ok'){
  try{
    const box = document.getElementById('ctToasts');
    if(!box) return;
    const t = document.createElement('div');
    t.className = 'ct-toast ct-' + type;
    t.textContent = msg;
    t.style.cssText = 'pointer-events:none;background:rgba(15,23,42,.92);border:1px solid rgba(148,163,184,.55);color:#e5e7eb;padding:10px 12px;border-radius:14px;font-size:13px;max-width:360px;box-shadow:0 18px 45px rgba(0,0,0,.55);';
    if(type==='warn') t.style.borderColor = 'rgba(245,158,11,.85)';
    if(type==='err')  t.style.borderColor = 'rgba(239,68,68,.85)';
    box.appendChild(t);
    setTimeout(()=>{ t.style.opacity='0'; t.style.transform='translateY(4px)'; }, 1800);
    setTimeout(()=>{ t.remove(); }, 2300);
  }catch{}
}

/* =========================================================
   URL validation / health badge (lightweight)
   ========================================================= */
function parseUrlsFromTextarea() {
  const raw = (urlInput?.value || "").split("\n");
  const urls = raw.map((l) => (l || "").trim()).filter(Boolean);
  return urls;
}

function isLikelyTweetUrl(u) {
  try {
    const url = new URL(u);
    return /twitter\.com|x\.com/i.test(url.hostname) && /\/status\/\d+/i.test(url.pathname);
  } catch {
    return false;
  }
}
function updateUrlHealth() {
  if (!urlHealthBadgeEl || !urlInput) return;
  const urls = parseUrlsFromTextarea();
  const total = urls.length;
  const invalid = urls.filter((u) => !isLikelyTweetUrl(u)).length;
  if (!total) {
    urlHealthBadgeEl.textContent = "0 URLs";
    urlHealthBadgeEl.classList.remove("bad", "ok");
    return;
  }
  urlHealthBadgeEl.textContent = invalid ? `${invalid} invalid` : `${total} ok`;
  urlHealthBadgeEl.classList.toggle("bad", invalid > 0);
  urlHealthBadgeEl.classList.toggle("ok", invalid === 0);
}

/* =========================================================
   textarea autosize + line renumber helper
   ========================================================= */
function autoResizeTextarea() {
  if (!urlInput) return;
  urlInput.style.height = "auto";
  urlInput.style.height = Math.min(urlInput.scrollHeight, 280) + "px";
}
function renumberTextareaLines() {
  if (!urlInput) return { removed: 0, invalid: 0 };
  const lines = (urlInput.value || "").split("\n");
  const cleaned = [];
  let removed = 0;
  let invalid = 0;
  for (const l of lines) {
    const s = (l || "").trim();
    if (!s) { removed++; continue; }
    if (!/^https?:\/\//i.test(s)) { invalid++; }
    cleaned.push(s);
  }
  urlInput.value = cleaned.join("\n");
  return { removed, invalid };
}

/* =========================================================
   Clipboard + History
   ========================================================= */
async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (e) {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
      return true;
    } catch {
      return false;
    }
  }
}

function addToHistory(text) {
  if (!text) return;
  historyItems.unshift({
    id: Date.now(),
    text,
    at: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
  });
  historyItems = historyItems.slice(0, 30);
  try {
    localStorage.setItem("crowntalk_history_v1", JSON.stringify(historyItems));
  } catch {}
  renderHistory();
}

function loadHistory() {
  try {
    const raw = localStorage.getItem("crowntalk_history_v1");
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed;
  } catch {
    return [];
  }
}

function renderHistory() {
  if (!historyEl) return;
  historyEl.innerHTML = "";
  historyItems.forEach((h) => {
    const row = document.createElement("div");
    row.className = "history-item";
    const meta = document.createElement("div");
    meta.className = "history-meta";
    meta.textContent = h.at;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn-xs";
    btn.textContent = "Copy";
    btn.addEventListener("click", async () => {
      await copyToClipboard(h.text);
      ctToast("Copied from history.", "ok");
    });
    const txt = document.createElement("div");
    txt.className = "history-text";
    txt.textContent = h.text;
    row.appendChild(meta);
    row.appendChild(btn);
    row.appendChild(txt);
    historyEl.appendChild(row);
  });
}

/* =========================================================
   Theme
   ========================================================= */
const THEME_STORAGE_KEY = "crowntalk_theme_v1";
const ALLOWED_THEMES = ["dark-purple", "neon", "crimson"];

function applyTheme(theme) {
  document.body.dataset.theme = theme;
  themeDots = Array.from(document.querySelectorAll(".theme-dot"));
  themeDots.forEach((d) => {
    d.classList.toggle("is-active", d.dataset.theme === theme);
  });
}

function initTheme() {
  let theme = "dark-purple";
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored) theme = stored;
  } catch {}
  if (!ALLOWED_THEMES.includes(theme)) theme = "dark-purple";
  applyTheme(theme);

  themeDots.forEach((dot) => {
    dot.addEventListener("click", () => {
      const t = dot.dataset.theme;
      if (!ALLOWED_THEMES.includes(t)) return;
      applyTheme(t);
      try { localStorage.setItem(THEME_STORAGE_KEY, t); } catch {}
      if (t === "neon") localStorage.setItem(THEME_STORAGE_KEY, "neon");
      if (t === "crimson") localStorage.setItem(THEME_STORAGE_KEY, "crimson");
      if (t === "dark-purple") localStorage.setItem(THEME_STORAGE_KEY, "dark-purple");
    });
  });
}

/* =========================================================
   Presets (optional)
   ========================================================= */
function applyPreset(p) {
  // You can wire backend/prompt presets here if desired.
  // Kept as no-op if presetSelect does not exist in HTML.
  document.body.dataset.preset = p || "default";
}
function initPresetFromStorage() {
  if (!presetSelect) return;
  let stored = "default";
  try {
    const v = localStorage.getItem("ct_preset_v1");
    if (v) stored = v;
  } catch {}
  presetSelect.value = stored;
  applyPreset(stored);
}

/* =========================================================
   Keyboard HUD (desktop)
   ========================================================= */
function initKeyboardHud() {
  if (!keyboardHudEl) return;
  const desktop = matchMedia("(min-width: 1024px)").matches && matchMedia("(pointer:fine)").matches;
  if (!desktop) return;
  // nothing heavy; shown by CSS
}

/* floating shortcut button */
function initShortcutFab() {
  if (!keyboardHudEl) return;
  const fab = document.querySelector(".keyboard-fab");
  if (!fab) return;
  const panel = keyboardHudEl;
  fab.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    panel.classList.toggle("is-open");
  });
  document.addEventListener("click", () => panel.classList.remove("is-open"));
}

/* =========================================================
   Copy Queue (DISABLED by user)
   ========================================================= */
function renderCopyQueue() {
  if (!COPY_QUEUE_ENABLED) return;
  if (!copyQueueListEl) return;
  copyQueueListEl.innerHTML = "";
  const hasItems = ctCopyQueue.length > 0;
  if (copyQueueEmptyEl) {
    copyQueueEmptyEl.style.display = hasItems ? "none" : "block";
  }
  ctCopyQueue.forEach((item, idx) => {
    const row = document.createElement("div");
    row.className = "copy-queue-row" + (idx === ctCopyQueueIndex ? " is-active" : "");
    const num = document.createElement("div");
    num.className = "copy-queue-num";
    num.textContent = String(idx + 1);
    const txt = document.createElement("div");
    txt.className = "copy-queue-text";
    txt.textContent = item.text;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn-xs";
    btn.textContent = "Copy";
    btn.addEventListener("click", async () => {
      await copyToClipboard(item.text);
      addToHistory(item.text);
    });
    row.appendChild(num);
    row.appendChild(btn);
    row.appendChild(txt);
    copyQueueListEl.appendChild(row);
  });
  const enabled = hasItems;
  if (copyQueueNextBtn) copyQueueNextBtn.disabled = !enabled;
  if (copyQueueClearBtn) copyQueueClearBtn.disabled = !enabled;
}
function addToCopyQueue(text) {
  if (!COPY_QUEUE_ENABLED) return;
  if (!text) return;
  ctCopyQueue.push({ text });
  if (ctCopyQueue.length === 1) ctCopyQueueIndex = 0;
  renderCopyQueue();
}
async function copyNextFromQueue() {
  if (!COPY_QUEUE_ENABLED) return;
  if (!ctCopyQueue.length) return;
  const item = ctCopyQueue[ctCopyQueueIndex];
  await copyToClipboard(item.text);
  addToHistory(item.text);
  ctCopyQueueIndex = (ctCopyQueueIndex + 1) % ctCopyQueue.length;
  renderCopyQueue();
}
function clearCopyQueue() {
  if (!COPY_QUEUE_ENABLED) return;
  ctCopyQueue = [];
  ctCopyQueueIndex = 0;
  renderCopyQueue();
}

/* =========================================================
   === PATCH: Analytics HUD + export helpers ===============
   ========================================================= */
function updateAnalytics(meta) {
  if (!analyticsHudEl) return;
  const summaryEl = document.getElementById("analyticsSummary");
  const tweetsEl = document.getElementById("analyticsTweets");
  const failedElMetric = document.getElementById("analyticsFailed");
  const timeEl = document.getElementById("analyticsTime");
  const m = meta || {};
  const tweets = m.tweets || 0;
  const totalUrls = m.totalUrls || tweets;
  const failed = m.failed || 0;
  const durationSec = m.durationSec || 0;
  if (summaryEl) summaryEl.textContent = `${tweets}/${totalUrls} tweets · ${failed} failed`;
  if (tweetsEl) tweetsEl.textContent = String(tweets);
  if (failedElMetric) failedElMetric.textContent = String(failed);
  if (timeEl) timeEl.textContent = `${durationSec}s`;
  // extra metrics (patched)
  try {
    const success = Math.max(0, (totalUrls - failed));
    const successRate = totalUrls ? Math.round((success / totalUrls) * 100) : 0;
    const avgPerUrl = totalUrls ? Math.max(1, Math.round((durationSec * 1000) / totalUrls)) : 0;
    ensureAnalyticsExtraPills();
    const sr = document.getElementById("analyticsSuccessRate");
    const avg = document.getElementById("analyticsAvgPerUrl");
    const dup = document.getElementById("analyticsDupes");
    if (sr) sr.textContent = `${successRate}%`;
    if (avg) avg.textContent = avgPerUrl ? `${avgPerUrl}ms` : "—";
    if (dup) dup.textContent = String(window.__ctHiddenDupes || 0);
  } catch {}
  analyticsHudEl.style.opacity = "1";
  analyticsHudEl.setAttribute("aria-hidden", "false");
}

async function exportComments(mode) {
  if (!resultsEl) return;
  const blocks = Array.from(resultsEl.querySelectorAll(".tweet"));
  const lines = [];

  blocks.forEach((tweet) => {
    const url = tweet.querySelector("a")?.href || "";
    const commentBtns = tweet.querySelectorAll(".comment-copy-btn");
    const comments = Array.from(commentBtns).map((b) => b.dataset.text || "").filter(Boolean);

    if (mode === "en") {
      // EN-only: heuristically use "en" tag if present, else keep ASCII-ish lines
      const enComments = comments.filter((t) => {
        const tag = tweet.querySelector(`[data-text="${CSS.escape(t)}"]`) ? "en" : "en";
        return tag === "en";
      });
      if (enComments.length) {
        if (url) lines.push(url);
        lines.push(...enComments);
        lines.push("");
      }
      return;
    }

    if (mode === "native") {
      // native-only: keep lines with non-ascii
      const native = comments.filter((t) => /[^\u0000-\u007f]/.test(t));
      if (native.length) {
        if (url) lines.push(url);
        lines.push(...native);
        lines.push("");
      }
      return;
    }

    // all/default/download
    if (url) lines.push(url);
    lines.push(...comments);
    lines.push("");
  });

  const text = lines.join("\n").trim();
  if (!text) return;

  if (mode === "download") {
    const blob = new Blob([text], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `crowntalk_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      URL.revokeObjectURL(a.href);
      a.remove();
    }, 500);
    ctToast("Downloaded .txt", "ok");
  } else {
    await copyToClipboard(text);
    addToHistory(text);
    ctToast("Copied export.", "ok");
  }
}

/* =========================================================
   Results menu: bind existing HTML (avoid duplicate IDs)
   ========================================================= */
function initResultsMenu() {
  if (!resultsEl) return;
  const card = resultsEl.closest(".card");
  if (!card) return;
  const toolbar = card.querySelector(".results-toolbar");
  if (!toolbar) return;

  // If HTML already includes resultsMenu/resultsMenuToggle (it does in current index.html),
  // bind them instead of creating duplicates.
  const existingMenu = document.getElementById("resultsMenu");
  const existingToggle = document.getElementById("resultsMenuToggle");
  if (existingMenu && existingToggle) {
    if (existingMenu.dataset.bound === "1") return;
    existingMenu.dataset.bound = "1";
    existingToggle.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      existingMenu.classList.toggle("is-open");
    });
    document.addEventListener("click", () => existingMenu.classList.remove("is-open"));
    existingMenu.addEventListener("click", (e) => e.stopPropagation());
    ensureExportCenterUI(existingMenu);
    ensureResultsManagerUI(toolbar);
    ensureSessionTabsUI(toolbar);
    return;
  }

  // fallback (older builds): create menu if missing
  if (toolbar.dataset.menuInit === "1") return;
  toolbar.dataset.menuInit = "1";

  const menu = document.createElement("div");
  menu.id = "resultsMenu";
  menu.className = "results-menu";

  const presetBlock = presetSelect ? (presetSelect.closest(".preset-label") || presetSelect.parentElement) : null;
  const exportBlock = exportAllBtn ? exportAllBtn.parentElement : null;

  if (presetBlock) menu.appendChild(presetBlock);
  if (exportBlock) menu.appendChild(exportBlock);

  const toggle = document.createElement("button");
  toggle.id = "resultsMenuToggle";
  toggle.type = "button";
  toggle.className = "results-menu-toggle btn-xs";
  toggle.textContent = "Menu";

  toolbar.appendChild(toggle);
  toolbar.appendChild(menu);

  toggle.addEventListener("click", () => {
    menu.classList.toggle("is-open");
  });
}

/* =========================================================
   Sessions timeline (desktop tabs)
   ========================================================= */
function getSessionById(id) {
  return ctSessions.find((s) => s.id === id);
}
function renderSessionTabs() {
  if (!sessionTabsEl) return;
  sessionTabsEl.innerHTML = "";
  if (!ctSessions.length) {
    sessionTabsEl.style.display = "none";
    return;
  }
  sessionTabsEl.style.display = "flex";
  ctSessions.forEach((s) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "session-chip" + (s.id === ctActiveSessionId ? " is-active" : "");
    btn.textContent = s.label;
    btn.title = `Generated at ${s.createdAt}`;
    btn.addEventListener("click", () => {
      ctActiveSessionId = s.id;
      renderSessionTabs();
      restoreSessionSnapshot(s.id);
    });
    sessionTabsEl.appendChild(btn);
  });
}
function addSessionSnapshot(meta) {
  if (!resultsEl || !failedEl) return;
  const id = `run_${Date.now()}_${++ctSessionCounter}`;
  const label = `Run ${ctSessionCounter}`;
  const createdAt = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  const snapshot = {
    id,
    label,
    createdAt,
    meta: meta || {},
    resultsHTML: resultsEl.innerHTML,
    failedHTML: failedEl.innerHTML,
    input: urlInput?.value || "",
  };

  ctSessions.unshift(snapshot);
  ctSessions = ctSessions.slice(0, 8);
  ctActiveSessionId = id;
  renderSessionTabs();

  try { localStorage.setItem("ct_sessions_v1", JSON.stringify(ctSessions)); } catch {}
}
function restoreSessionSnapshot(id) {
  const s = getSessionById(id);
  if (!s || !resultsEl || !failedEl) return;
  resultsEl.innerHTML = s.resultsHTML || "";
  failedEl.innerHTML = s.failedHTML || "";
  if (urlInput) urlInput.value = s.input || "";
  autoResizeTextarea();
  updateUrlHealth();
  // rebind copy buttons
  bindResultInteractions(resultsEl);
  ctToast(`Restored ${s.label}`, "ok");
}

function loadSessionsFromStorage() {
  try {
    const raw = localStorage.getItem("ct_sessions_v1");
    if (!raw) return;
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return;
    ctSessions = arr;
    ctSessionCounter = Math.max(0, ...ctSessions.map((s) => parseInt((s.label || "").replace(/\D+/g, ""), 10) || 0));
    ctActiveSessionId = ctSessions[0]?.id || null;
    renderSessionTabs();
  } catch {}
}

/* =========================================================
   Result rendering
   ========================================================= */
function buildTweetBlock(url, comments) {
  const tweet = document.createElement("div");
  tweet.className = "tweet";

  const header = document.createElement("div");
  header.className = "tweet-header";

  const link = document.createElement("a");
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener";
  link.textContent = url;

  const actions = document.createElement("div");
  actions.className = "tweet-actions";

  const rerollBtn = document.createElement("button");
  rerollBtn.type = "button";
  rerollBtn.className = "reroll-btn";
  rerollBtn.title = "Re-generate this tweet only";
  rerollBtn.textContent = "↻";

  const collapseBtn = document.createElement("button");
  collapseBtn.type = "button";
  collapseBtn.className = "collapse-btn";
  collapseBtn.title = "Collapse / expand comments";
  collapseBtn.textContent = "▾";

  const pinBtn = document.createElement("button");
  pinBtn.type = "button";
  pinBtn.className = "tweet-pin-btn";
  pinBtn.title = "Pin this tweet";

  // share button (lightweight; uses native share on mobile, fallback to copy)
  const shareBtn = document.createElement("button");
  shareBtn.type = "button";
  shareBtn.className = "tweet-share-btn";
  shareBtn.title = "Share / Copy";
  shareBtn.textContent = "↗";

  actions.appendChild(rerollBtn);
  actions.appendChild(collapseBtn);
  actions.appendChild(pinBtn);
  actions.appendChild(shareBtn);

  header.appendChild(link);
  header.appendChild(actions);
  tweet.appendChild(header);

  // mobile: tap header to collapse/expand (ignore clicks on buttons/links)
  try {
    const coarse = matchMedia("(pointer:coarse)").matches;
    if (coarse) {
      header.addEventListener("click", (ev) => {
        const t = ev.target;
        if (!t) return;
        if (t.closest("a") || t.closest("button")) return;
        tweet.classList.toggle("is-collapsed");
      });
    }
  } catch {}

  // collapse & pin behaviour
  collapseBtn.addEventListener("click", () => {
    tweet.classList.toggle("is-collapsed");
  });
  pinBtn.addEventListener("click", () => {
    tweet.classList.toggle("is-pinned");
    if (resultsEl && tweet.parentElement === resultsEl) {
      resultsEl.insertBefore(tweet, resultsEl.firstChild);
    }
  });

  shareBtn.addEventListener("click", async (e) => {
    e.preventDefault();
    e.stopPropagation();
    const url = link?.href || "";
    // Try to share a compact summary: first 1-2 comment lines
    const commentBtns = tweet.querySelectorAll(".comment-copy-btn");
    let preview = "";
    for (let i = 0; i < Math.min(2, commentBtns.length); i++) {
      const t = commentBtns[i]?.dataset?.text || "";
      if (t) preview += (preview ? "\n" : "") + t;
    }
    const text = (url ? url + "\n\n" : "") + preview;
    await ctShareOrCopy(text || url);
  });

  const commentsWrap = document.createElement("div");
  commentsWrap.className = "comments";

  const hasNative = comments.some((c) => c && c.lang && c.lang !== "en");
  comments.forEach((comment) => {
    const line = document.createElement("div");
    line.className = "comment-line";

    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = comment.lang || "en";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = comment.text;

    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "comment-copy-btn";
    const copyMain = document.createElement("span");
    copyMain.textContent = "Copy";
    const copyAlt = document.createElement("span");
    copyAlt.textContent = "Copied";
    copyBtn.appendChild(copyMain);
    copyBtn.appendChild(copyAlt);
    copyBtn.dataset.text = comment.text;

    // make comment text available for queue
    line.dataset.commentText = comment.text;

    line.appendChild(tag);
    line.appendChild(bubble);

    // mobile swipe (lightweight): swipe right to copy this comment, swipe left to share
    try {
      const coarse = matchMedia("(pointer:coarse)").matches;
      if (coarse) {
        let sx = 0, sy = 0, moved = false;
        line.addEventListener("touchstart", (ev) => {
          const t = ev.touches && ev.touches[0];
          if (!t) return;
          sx = t.clientX; sy = t.clientY; moved = false;
        }, { passive:true });
        line.addEventListener("touchmove", (ev) => {
          const t = ev.touches && ev.touches[0];
          if (!t) return;
          const dx = t.clientX - sx;
          const dy = t.clientY - sy;
          if (Math.abs(dx) > 12 && Math.abs(dx) > Math.abs(dy)) moved = true;
        }, { passive:true });
        line.addEventListener("touchend", async (ev) => {
          if (!moved) return;
          const t = (ev.changedTouches && ev.changedTouches[0]) || null;
          if (!t) return;
          const dx = t.clientX - sx;
          if (dx > 55) {
            // swipe right -> copy
            const text = comment.text || "";
            if (text) {
              await copyToClipboard(text);
              addToHistory(text);
              ctToast("Copied (swipe).", "ok");
            }
          } else if (dx < -65) {
            // swipe left -> share/copy
            const text = comment.text || "";
            if (text) await ctShareOrCopy(text);
          }
        });
      }
    } catch {}

    // small "+" button to queue this comment (disabled by flag)
    if (COPY_QUEUE_ENABLED) {
      const queueBtn = document.createElement("button");
      queueBtn.type = "button";
      queueBtn.className = "queue-btn";
      queueBtn.title = "Add to copy queue";
      queueBtn.textContent = "+";
      line.appendChild(queueBtn);
    }

    line.appendChild(copyBtn);
    commentsWrap.appendChild(line);
  });

  tweet.appendChild(commentsWrap);
  return tweet;
}

function appendResultBlock(result) {
  const block = buildTweetBlock(result.url, result.comments);
  resultsEl.appendChild(block);
  bindResultInteractions(block);
}

function bindResultInteractions(root) {
  const scope = root || document;
  const copyBtns = scope.querySelectorAll(".comment-copy-btn");
  copyBtns.forEach((btn) => {
    btn.addEventListener("click", async () => {
      const text = btn.dataset.text || "";
      if (!text) return;
      await copyToClipboard(text);
      addToHistory(text);
      btn.classList.add("is-copied");
      setTimeout(() => btn.classList.remove("is-copied"), 900);
    });
  });

  if (COPY_QUEUE_ENABLED) {
    const queueBtns = scope.querySelectorAll(".queue-btn");
    queueBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const line = btn.closest(".comment-line");
        const text = line?.dataset.commentText || "";
        addToCopyQueue(text);
        line?.classList.add("queued");
        ctToast("Added to queue.", "ok");
      });
    });
  }
}

/* =========================================================
   Backend call (mock shape / keep your existing endpoint)
   ========================================================= */
let __activeAbortController = null;

async function fetchCommentsForUrl(url) {
  // Replace endpoint as needed; keep as your current implementation.
  // This file preserves your original flow; only patches UI & feature logic.
  // If you already have a working endpoint in the old file, keep it.
  const endpoint = "/api/generate";
  const payload = { url };

  __activeAbortController = new AbortController();
  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
    signal: __activeAbortController.signal,
  });

  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  // expected { comments: [{lang, text}, ...] }
  return data;
}

/* =========================================================
   Progress UI
   ========================================================= */
function setProgressText(text) {
  if (progressText) progressText.textContent = text;
}
function setProgressRatio(ratio) {
  const pct = Math.max(0, Math.min(1, ratio || 0));
  if (progressFill) progressFill.style.width = `${Math.round(pct * 100)}%`;
}

/* =========================================================
   Generate Flow
   ========================================================= */
function resetUIBeforeRun() {
  cancelled = false;
  if (resultsEl) resultsEl.innerHTML = "";
  if (failedEl) failedEl.innerHTML = "";
  if (resultCountEl) resultCountEl.textContent = "0";
  if (failedCountEl) failedCountEl.textContent = "0";
  setProgressRatio(0);
  setProgressText("");
  document.body.classList.add("is-generating");
}

function setGenerating(isGen) {
  document.body.classList.toggle("is-generating", !!isGen);
  if (generateBtn) generateBtn.disabled = !!isGen;
  if (cancelBtn) cancelBtn.disabled = !isGen;
}

async function handleGenerate() {
  if (!urlInput) return;
  const urls = parseUrlsFromTextarea();
  if (!urls.length) return;

  resetUIBeforeRun();
  setGenerating(true);

  const runStartedAt = performance.now();
  let processedUrls = 0;
  let totalResults = 0;
  let totalFailed = 0;

  for (const url of urls) {
    if (cancelled) break;

    setProgressText(`Processing ${processedUrls + 1}/${urls.length} ...`);
    setProgressRatio(processedUrls / urls.length);

    try {
      const data = await fetchCommentsForUrl(url);
      const comments = Array.isArray(data?.comments) ? data.comments : [];
      if (comments.length) {
        appendResultBlock({ url, comments });
        totalResults++;
      } else {
        totalFailed++;
        const li = document.createElement("div");
        li.className = "failed-item";
        li.textContent = url;
        failedEl?.appendChild(li);
      }
    } catch (e) {
      totalFailed++;
      const li = document.createElement("div");
      li.className = "failed-item";
      li.textContent = url;
      failedEl?.appendChild(li);
    }

    processedUrls++;
    if (resultCountEl) resultCountEl.textContent = String(totalResults);
    if (failedCountEl) failedCountEl.textContent = String(totalFailed);

    // apply filters live (hide dupes/search)
    try {
      const st = ctLoadJSON("ct_rm_state_v1", { q:"", pinned:false, dupes:true });
      applyResultsFilters(st);
    } catch {}

    setProgressRatio(processedUrls / urls.length);
  }

  setGenerating(false);
  document.body.classList.remove("is-generating");

  if (!totalResults && !totalFailed) {
    setProgressText("No comments returned.");
    setProgressRatio(1);
  } else {
    setProgressText(`Processed ${processedUrls} tweet${processedUrls === 1 ? "" : "s"}.`);
    setProgressRatio(1);
  }

  const durationSec = Math.max(1, Math.round((performance.now() - runStartedAt) / 1000));
  updateAnalytics({ tweets: totalResults, failed: totalFailed, totalUrls: urls.length, durationSec });
  addSessionSnapshot({ tweets: totalResults, failed: totalFailed, totalUrls: urls.length, durationSec });
}


/* =========================================================
   Cancel & Clear
   ========================================================= */
let __lastClear = null;
function handleCancel() {
  cancelled = true;
  try { __activeAbortController?.abort(); } catch {}
  document.body.classList.remove("is-generating");
  setGenerating(false);
  setProgressText("Cancelled.");
}
function handleClear() {
  const now = Date.now();
  if (__lastClear && now - __lastClear < 350) return;
  __lastClear = now;

  if (resultsEl) resultsEl.innerHTML = "";
  if (failedEl) failedEl.innerHTML = "";
  if (resultCountEl) resultCountEl.textContent = "0";
  if (failedCountEl) failedCountEl.textContent = "0";
  setProgressText("");
  setProgressRatio(0);
  ctToast("Cleared.", "ok");
}

/* =========================================================
   Hotkeys
   ========================================================= */
function handleGlobalHotkeys(event) {
  const key = event.key;
  const metaOrCtrl = event.metaKey || event.ctrlKey;

  if (metaOrCtrl && key === "Enter") {
    if (!document.body.classList.contains("is-generating")) {
      event.preventDefault();
      handleGenerate();
    }
  } else if (key === "Escape") {
    if (document.body.classList.contains("is-generating")) {
      event.preventDefault();
      handleCancel();
    }
  } else if (metaOrCtrl && (key === "l" || key === "L")) {
    if (urlInput) {
      event.preventDefault();
      urlInput.focus();
      urlInput.select();
    }
  } else if (event.ctrlKey && event.shiftKey && (key === "c" || key === "C")) {
    event.preventDefault();
    if (COPY_QUEUE_ENABLED) {
      copyNextFromQueue();
    } else {
      copyNextComment();
    }
  }
}

/* =========================================================
   Boot UI once unlocked
   ========================================================= */
function bootAppUI() {
  // init basics
  historyItems = loadHistory();
  if (yearEl) yearEl.textContent = String(new Date().getFullYear());
  initTheme();
  renderHistory();
  autoResizeTextarea();
  updateUrlHealth();
  initPresetFromStorage();
  initKeyboardHud();
  renderCopyQueue();
  initResultsMenu();
  ensureMobileQuickBar();
  ensureCompactToggle();
  initShortcutFab();
  loadSessionsFromStorage();

  setTimeout(() => { maybeWarmBackend(); }, 4000);

  urlInput?.addEventListener("input", autoResizeTextarea);
  urlInput?.addEventListener("input", updateUrlHealth);

  generateBtn?.addEventListener("click", (e) => { e.preventDefault(); handleGenerate(); });
  cancelBtn?.addEventListener("click", (e) => { e.preventDefault(); handleCancel(); });
  clearBtn?.addEventListener("click", (e) => { e.preventDefault(); handleClear(); });

  exportAllBtn?.addEventListener("click", (e) => { e.preventDefault(); exportComments("all"); });
  exportEnBtn?.addEventListener("click", (e) => { e.preventDefault(); exportComments("en"); });
  exportNativeBtn?.addEventListener("click", (e) => { e.preventDefault(); exportComments("native"); });
  downloadTxtBtn?.addEventListener("click", (e) => { e.preventDefault(); exportComments("download"); });

  clearHistoryBtn?.addEventListener("click", (e) => {
    e.preventDefault();
    historyItems = [];
    try { localStorage.removeItem("crowntalk_history_v1"); } catch {}
    renderHistory();
    ctToast("History cleared.", "ok");
  });

  sortUrlsBtn?.addEventListener("click", (e) => {
    e.preventDefault();
    if (!urlInput) return;
    const urls = parseUrlsFromTextarea();
    urls.sort((a, b) => a.localeCompare(b));
    urlInput.value = urls.join("\n");
    urlInput.dispatchEvent(new Event("input", { bubbles: true }));
    ctToast("Sorted URLs.", "ok");
  });

  shuffleUrlsBtn?.addEventListener("click", (e) => {
    e.preventDefault();
    if (!urlInput) return;
    const urls = parseUrlsFromTextarea();
    for (let i = urls.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [urls[i], urls[j]] = [urls[j], urls[i]];
    }
    urlInput.value = urls.join("\n");
    urlInput.dispatchEvent(new Event("input", { bubbles: true }));
    ctToast("Shuffled URLs.", "ok");
  });

  removeInvalidBtn?.addEventListener("click", (e) => {
    e.preventDefault();
    if (!urlInput) return;
    const lines = (urlInput.value || "").split("\n").map((l) => (l || "").trim()).filter(Boolean);
    const keep = [];
    let removed = 0;
    lines.forEach((u) => {
      if (isLikelyTweetUrl(u)) keep.push(u);
      else removed++;
    });
    urlInput.value = keep.join("\n");
    urlInput.dispatchEvent(new Event("input", { bubbles: true }));
    ctToast(removed ? `Removed ${removed} invalid.` : "No invalid lines.", removed ? "ok" : "warn");
  });

  window.addEventListener("keydown", handleGlobalHotkeys);

  // === RESULTS MENU DROPDOWN (desktop) ===
  const resultsMenu      = document.getElementById("resultsMenu");
  const resultsMenuToggle = document.getElementById("resultsMenuToggle");

  if (resultsMenu && resultsMenuToggle) {
    if (resultsMenu.dataset.bound !== "1") {
      resultsMenu.dataset.bound = "1";
      resultsMenuToggle.addEventListener("click", (e) => {
        e.stopPropagation();
        resultsMenu.classList.toggle("is-open");
      });

      document.addEventListener("click", () => {
        resultsMenu.classList.remove("is-open");
      });
    }
  }
}

/* =========================================================
   Warm backend (optional)
   ========================================================= */
function maybeWarmBackend() {
  // optional: ping your api to reduce first-run latency
  // fetch("/api/ping").catch(()=>{});
}

/* =========================================================
   Smooth URL Dropzone (desktop only) — patched to support .txt
   ========================================================= */
(function urlDropzone(){
  if (!urlInput) return;
  const desktop = matchMedia('(pointer:fine)').matches;
  if (!desktop) return;

  let halo;
  function showHalo(){
    if (halo) return; halo = document.createElement('div');
    halo.className = 'ct-drop-halo';
    halo.style.cssText = 'position:fixed;inset:0;border:2px dashed rgba(59,130,246,.55);border-radius:24px;pointer-events:none;z-index:9998;margin:12px;background:rgba(59,130,246,.05)';
    document.body.appendChild(halo);
  }
  function hideHalo(){ halo?.remove(); halo = null; }

  window.addEventListener('dragover', (e)=>{ e.preventDefault(); showHalo(); }, false);
  window.addEventListener('dragleave', (e)=>{ if (e.target === document) hideHalo(); }, false);
  window.addEventListener('drop', async (e)=>{
    e.preventDefault(); hideHalo();

    // If a file is dropped (e.g., urls.txt), read it and append
    try {
      const files = Array.from(e.dataTransfer.files || []);
      const txtFile = files.find(f => /\.txt$/i.test(f.name)) || files[0];
      if (txtFile) {
        const content = await txtFile.text();
        const curr = urlInput.value.trim();
        const toAdd = (content || '').trim();
        if (toAdd) {
          urlInput.value = curr ? (curr + '\n' + toAdd) : toAdd;
          urlInput.dispatchEvent(new Event('input', {bubbles:true}));
          renumberTextareaLines();
          updateUrlHealth();
          ctToast(`Imported ${txtFile.name}`, "ok");
        }
        return;
      }
    } catch {}

    let txt = e.dataTransfer.getData('text/uri-list') || e.dataTransfer.getData('text/plain') || '';
    if (!txt) return;
    const curr = urlInput.value.trim();
    const toAdd = txt.trim();
    urlInput.value = curr ? (curr + '\n' + toAdd) : toAdd;
    urlInput.dispatchEvent(new Event('input', {bubbles:true}));
    renumberTextareaLines();
    updateUrlHealth();
    ctToast("Added dropped URLs.", "ok");
  }, false);
})();

/* =========================================================
   === PATCH: Premium UI (no HTML edits required) ===========
   ========================================================= */

// Lightweight runtime style injection for new UI bits
(function ctInjectStyles(){
  if (document.getElementById("ctPatchStyles")) return;
  const s = document.createElement("style");
  s.id = "ctPatchStyles";
  s.textContent = `
    /* session tabs */
    #sessionTabs{ display:flex; gap:8px; flex-wrap:wrap; margin:6px 0 0; }
    .session-chip{ border:1px solid rgba(148,163,184,.55); background:rgba(15,23,42,.65); color:#e5e7eb;
      border-radius:999px; padding:5px 10px; font-size:12px; cursor:pointer; }
    .session-chip.is-active{ border-color:rgba(59,130,246,.95); box-shadow:0 0 0 3px rgba(59,130,246,.15); }

    /* results manager */
    .ct-rm-wrap{ display:flex; align-items:center; gap:8px; margin-left:10px; }
    .ct-rm-input{ width:min(340px, 36vw); border-radius:999px; border:1px solid rgba(148,163,184,.55);
      background:rgba(15,23,42,.55); color:#e5e7eb; padding:6px 10px; font-size:12px; }
    .ct-rm-chip{ border:1px solid rgba(148,163,184,.55); background:rgba(15,23,42,.55); color:#e5e7eb;
      border-radius:999px; padding:6px 10px; font-size:12px; cursor:pointer; user-select:none; }
    .ct-rm-chip.is-on{ border-color:rgba(245,158,11,.95); box-shadow:0 0 0 3px rgba(245,158,11,.12); }

    /* tweet share button */
    .tweet-share-btn{ border-radius:999px; border:1px solid rgba(148,163,184,.6); background:rgba(15,23,42,.9);
      color:#e5e7eb; font-size:12px; padding:3px 8px; cursor:pointer; }

    /* compact mode */
    body.ct-compact .tweet{ padding:10px 10px 8px !important; }
    body.ct-compact .comment-line{ padding:8px 8px !important; }
    body.ct-compact .ct-rm-input{ padding:5px 9px; }

    /* mobile quick bar */
    .ct-quickbar{ position:fixed; left:10px; right:10px; bottom:10px; z-index:50;
      display:flex; gap:10px; padding:10px; border-radius:18px;
      background:rgba(15,23,42,.9); border:1px solid rgba(148,163,184,.55); box-shadow:0 18px 45px rgba(0,0,0,.65); }
    .ct-quickbar button{ flex:1; border-radius:14px; border:1px solid rgba(148,163,184,.55);
      background:rgba(30,41,59,.7); color:#e5e7eb; padding:10px 8px; font-size:13px; }
    @media (min-width: 1024px){ .ct-quickbar{ display:none; } }
  `;
  document.head.appendChild(s);
})();

function ensureSessionTabsUI(toolbarEl) {
  // If missing from HTML, create it right under the toolbar (desktop only)
  if (sessionTabsEl) return;
  if (!toolbarEl) return;
  const desktop = matchMedia("(min-width: 1024px)").matches && matchMedia("(pointer:fine)").matches;
  if (!desktop) return;

  const tabs = document.createElement("div");
  tabs.id = "sessionTabs";
  tabs.className = "session-tabs hide-on-mobile";
  toolbarEl.insertAdjacentElement("afterend", tabs);
  sessionTabsEl = tabs;
  renderSessionTabs();
}

function ensureResultsManagerUI(toolbarEl) {
  if (!toolbarEl) return;
  if (document.getElementById("tweetSearchInput")) return;

  const wrap = document.createElement("div");
  wrap.className = "ct-rm-wrap hide-on-mobile";

  const input = document.createElement("input");
  input.type = "search";
  input.id = "tweetSearchInput";
  input.className = "ct-rm-input";
  input.placeholder = "Search results… (url / text)";
  input.autocomplete = "off";

  const pinned = document.createElement("button");
  pinned.type = "button";
  pinned.id = "pinnedOnlyToggle";
  pinned.className = "ct-rm-chip";
  pinned.textContent = "Pinned";

  const dupes = document.createElement("button");
  dupes.type = "button";
  dupes.id = "hideDupesToggle";
  dupes.className = "ct-rm-chip";
  dupes.textContent = "Hide dupes";

  wrap.appendChild(input);
  wrap.appendChild(pinned);
  wrap.appendChild(dupes);

  // insert on the right side if present, else at end
  const right = toolbarEl.querySelector(".results-toolbar-right") || toolbarEl;
  right.appendChild(wrap);

  // restore state
  const st = ctLoadJSON("ct_rm_state_v1", { q:"", pinned:false, dupes:true });
  input.value = st.q || "";
  if (st.pinned) pinned.classList.add("is-on");
  if (st.dupes) dupes.classList.add("is-on");

  let t;
  const apply = () => {
    const state = {
      q: input.value || "",
      pinned: pinned.classList.contains("is-on"),
      dupes: dupes.classList.contains("is-on"),
    };
    ctSaveJSON("ct_rm_state_v1", state);
    applyResultsFilters(state);
  };

  input.addEventListener("input", () => {
    clearTimeout(t);
    t = setTimeout(apply, 120);
  });
  pinned.addEventListener("click", () => {
    pinned.classList.toggle("is-on");
    apply();
  });
  dupes.addEventListener("click", () => {
    dupes.classList.toggle("is-on");
    apply();
  });

  // initial
  apply();
}

function applyResultsFilters(state) {
  const q = (state.q || "").trim().toLowerCase();
  const pinnedOnly = !!state.pinned;
  const hideDupes = state.dupes !== false; // default true

  // reset dupe count
  window.__ctHiddenDupes = 0;

  const tweets = Array.from(document.querySelectorAll("#results .tweet"));
  const seen = new Set();

  tweets.forEach((tw) => {
    const url = (tw.querySelector("a")?.href || "").trim();
    const txt = (tw.innerText || "").toLowerCase();

    let show = true;

    if (pinnedOnly && !tw.classList.contains("is-pinned")) show = false;
    if (q) {
      if (!url.toLowerCase().includes(q) && !txt.includes(q)) show = false;
    }

    if (hideDupes && url) {
      if (seen.has(url)) {
        show = false;
        window.__ctHiddenDupes = (window.__ctHiddenDupes || 0) + 1;
      } else {
        seen.add(url);
      }
    }

    tw.style.display = show ? "" : "none";
  });

  // refresh analytics dupe pill if present
  const dup = document.getElementById("analyticsDupes");
  if (dup) dup.textContent = String(window.__ctHiddenDupes || 0);
}

function ensureExportCenterUI(resultsMenuEl) {
  // Add Export Center buttons into existing dropdown (without removing current IDs)
  const menu = resultsMenuEl || document.getElementById("resultsMenu");
  if (!menu) return;

  const pop = menu.querySelector(".results-menu-pop");
  if (!pop) return;

  // already added?
  if (document.getElementById("exportJsonBtn")) return;

  const group = document.createElement("div");
  group.className = "results-menu-group";

  const title = document.createElement("div");
  title.className = "results-menu-title";
  title.textContent = "Export center";

  const actions = document.createElement("div");
  actions.className = "results-menu-actions";

  const mkBtn = (id, label) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "btn-xs";
    b.id = id;
    b.textContent = label;
    return b;
  };

  const jsonBtn = mkBtn("exportJsonBtn", "Export JSON");
  const csvBtn  = mkBtn("exportCsvBtn", "Export CSV");
  const mdBtn   = mkBtn("exportMdBtn", "Export MD");

  actions.appendChild(jsonBtn);
  actions.appendChild(csvBtn);
  actions.appendChild(mdBtn);

  group.appendChild(title);
  group.appendChild(actions);
  pop.appendChild(group);

  jsonBtn.addEventListener("click", (e) => { e.preventDefault(); exportStructured("json"); });
  csvBtn.addEventListener("click", (e) => { e.preventDefault(); exportStructured("csv"); });
  mdBtn.addEventListener("click", (e) => { e.preventDefault(); exportStructured("md"); });
}

function ensureAnalyticsExtraPills() {
  if (!analyticsHudEl) return;
  if (document.getElementById("analyticsSuccessRate")) return;

  const makePill = (label, id) => {
    const pill = document.createElement("div");
    pill.className = "analytics-pill";
    const l = document.createElement("span");
    l.className = "label";
    l.textContent = label;
    const v = document.createElement("span");
    v.className = "value";
    v.id = id;
    v.textContent = "—";
    pill.appendChild(l);
    pill.appendChild(v);
    return pill;
  };

  analyticsHudEl.appendChild(makePill("Success", "analyticsSuccessRate"));
  analyticsHudEl.appendChild(makePill("Avg / URL", "analyticsAvgPerUrl"));
  analyticsHudEl.appendChild(makePill("Dupes", "analyticsDupes"));
}

function ctSaveJSON(key, obj) {
  try { localStorage.setItem(key, JSON.stringify(obj)); } catch {}
}
function ctLoadJSON(key, fallback) {
  try {
    const v = localStorage.getItem(key);
    if (!v) return fallback;
    return JSON.parse(v);
  } catch { return fallback; }
}

function collectCurrentResults() {
  const tweets = Array.from(document.querySelectorAll("#results .tweet"));
  const out = [];
  tweets.forEach((tw) => {
    const url = tw.querySelector("a")?.href || "";
    const pinned = tw.classList.contains("is-pinned");
    const collapsed = tw.classList.contains("is-collapsed");
    const comments = Array.from(tw.querySelectorAll(".comment-copy-btn")).map((b) => ({
      text: b.dataset.text || "",
    })).filter(x => x.text);
    out.push({ url, pinned, collapsed, comments });
  });
  return out;
}

function exportStructured(format) {
  const data = {
    app: "CrownTALK",
    exportedAt: new Date().toISOString(),
    results: collectCurrentResults(),
  };

  if (!data.results.length) {
    ctToast("No results to export yet.", "warn");
    return;
  }

  if (format === "json") {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    downloadBlob(blob, `crowntalk_export_${Date.now()}.json`);
    ctToast("Exported JSON.", "ok");
    return;
  }

  if (format === "csv") {
    // Flat CSV: one row per comment
    const rows = [["url", "pinned", "comment"]];
    data.results.forEach((r) => {
      (r.comments || []).forEach((c) => rows.push([r.url, r.pinned ? "1" : "0", c.text]));
    });
    const csv = rows.map((r) => r.map(csvEscape).join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    downloadBlob(blob, `crowntalk_export_${Date.now()}.csv`);
    ctToast("Exported CSV.", "ok");
    return;
  }

  if (format === "md") {
    const md = data.results.map((r) => {
      const head = `## ${r.url || "Tweet"}${r.pinned ? " 📌" : ""}`;
      const body = (r.comments || []).map((c) => `- ${c.text}`).join("\n");
      return head + "\n" + body;
    }).join("\n\n");
    const blob = new Blob([md], { type: "text/markdown" });
    downloadBlob(blob, `crowntalk_export_${Date.now()}.md`);
    ctToast("Exported Markdown.", "ok");
  }
}

function csvEscape(v) {
  const s = String(v ?? "");
  if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function downloadBlob(blob, filename) {
  const a = document.createElement("a");
  const url = URL.createObjectURL(blob);
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(url);
    a.remove();
  }, 400);
}

// Lightweight share/copy helper
async function ctShareOrCopy(text) {
  const t = (text || "").trim();
  if (!t) return;
  try {
    if (navigator.share && matchMedia("(pointer:coarse)").matches) {
      await navigator.share({ text: t });
      ctToast("Shared.", "ok");
      return;
    }
  } catch {}
  await copyToClipboard(t);
  addToHistory(t);
  ctToast("Copied for sharing.", "ok");
}

// Copy-next without queue: copies next un-copied comment button in DOM order.
let __ctCopyCursor = 0;
async function copyNextComment() {
  const btns = Array.from(document.querySelectorAll("#results .comment-copy-btn"));
  if (!btns.length) return ctToast("Nothing to copy yet.", "warn");

  // Find next that isn't marked as copied, else cycle.
  let idx = btns.findIndex((b, i) => i >= __ctCopyCursor && !b.classList.contains("is-copied"));
  if (idx === -1) idx = btns.findIndex((b) => !b.classList.contains("is-copied"));
  if (idx === -1) idx = 0;

  const b = btns[idx];
  const text = b.dataset.text || "";
  if (!text) return ctToast("Empty comment.", "warn");

  await copyToClipboard(text);
  addToHistory(text);
  b.classList.add("is-copied");
  __ctCopyCursor = idx + 1;

  ctToast("Copied next.", "ok");
}

// Compact toggle (desktop + mobile safe)
function ensureCompactToggle() {
  if (document.getElementById("compactToggleBtn")) return;

  const saved = ctLoadJSON("ct_compact_v1", { on:false });
  if (saved.on) document.body.classList.add("ct-compact");

  // add a small chip in toolbar if present
  const toolbar = document.querySelector(".results-toolbar");
  if (toolbar) {
    const right = toolbar.querySelector(".results-toolbar-right") || toolbar;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.id = "compactToggleBtn";
    btn.className = "ct-rm-chip hide-on-mobile";
    btn.textContent = "Compact";
    if (document.body.classList.contains("ct-compact")) btn.classList.add("is-on");
    btn.addEventListener("click", () => {
      document.body.classList.toggle("ct-compact");
      btn.classList.toggle("is-on");
      ctSaveJSON("ct_compact_v1", { on: document.body.classList.contains("ct-compact") });
    });
    right.appendChild(btn);
  }
}

// Mobile quick actions bar (lightweight)
function ensureMobileQuickBar() {
  if (document.querySelector(".ct-quickbar")) return;
  const coarse = matchMedia("(pointer:coarse)").matches || matchMedia("(max-width: 1023px)").matches;
  if (!coarse) return;

  const bar = document.createElement("div");
  bar.className = "ct-quickbar";
  bar.setAttribute("role","toolbar");
  bar.setAttribute("aria-label","Quick actions");

  const mk = (label, fn) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.addEventListener("click", fn);
    return b;
  };

  const copyNext = mk("Copy next", () => copyNextComment());
  const copyAll = mk("Copy all", () => exportComments("all"));
  const share = mk("Share", () => ctShareOrCopy(window.location.href));
  const top = mk("Top", () => window.scrollTo({ top:0, behavior:"smooth" }));

  bar.appendChild(copyNext);
  bar.appendChild(copyAll);
  bar.appendChild(share);
  bar.appendChild(top);

  document.body.appendChild(bar);
}