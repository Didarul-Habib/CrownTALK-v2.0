/* ============================================
   CrownTALK — One-time Access Gate + App Logic
   Access code: @CrownTALK@2026@CrownDEX
   Persists with localStorage + cookie fallback
   ============================================ */

/* ---------- Gate Pass ---------- */
(() => {
  
// Backend base URL: prefer injected config, then same-origin, else fall back to hosted API.
const API_BASE = (() => {
  try {
    if (window.CT_API_BASE && String(window.CT_API_BASE).trim()) return String(window.CT_API_BASE).trim().replace(/\/$/, "");
    // When served from a domain, same-origin is best for local/prod parity.
    if (location && location.origin && location.origin !== "null") return location.origin.replace(/\/$/, "");
  } catch {}
  return "https://crowntalk.onrender.com";
})();

const ACCESS_CODE = '@CrownTALK@2026@CrownDEX';
  const STORAGE_KEY = 'crowntalk_access_v1';    // local/session storage key
  const COOKIE_KEY  = 'crowntalk_access_v1';    // cookie fallback

  function getStoredToken() {
    let token = "";
    try { token = localStorage.getItem(STORAGE_KEY) || token; } catch {}
    try { token = token || sessionStorage.getItem(STORAGE_KEY) || ""; } catch {}
    if (!token) {
      try {
        const cookie = document.cookie.split(";").find((p) =>
          p.trim().startsWith(COOKIE_KEY + "=")
        );
        if (cookie) token = cookie.split("=")[1];
      } catch {}
    }
    return token || "";
  }

  function isAuthorized() {
    const token = getStoredToken();

    // No token at all → not authorized
    if (!token) return false;

    const isHashToken = /^[a-f0-9]{64}$/i.test(token);
    const isAccessCodeToken = token === ACCESS_CODE;

    // Accept either a real backend-issued hash token
    // OR the raw access code we store as a fallback.
    if (isHashToken || isAccessCodeToken) {
      return true;
    }

    // Anything else is a legacy / bad token → wipe it.
    try { localStorage.removeItem(STORAGE_KEY); } catch {}
    try { sessionStorage.removeItem(STORAGE_KEY); } catch {}
    try {
      document.cookie = `${COOKIE_KEY}=; max-age=0; path=/; samesite=lax`;
    } catch {}

    return false;
  }


  function markAuthorized(token) {
    if (!token) return;
    try { localStorage.setItem(STORAGE_KEY, token); } catch {}
    try { sessionStorage.setItem(STORAGE_KEY, token); } catch {}
    try {
      document.cookie = `${COOKIE_KEY}=${token}; max-age=${365*24*3600}; path=/; samesite=lax`;
    } catch {}
    window.__CROWNTALK_AUTH_TOKEN = token;
  }

  function exposeTokenHelpers() {
    const token = getStoredToken();
    window.__CROWNTALK_AUTH_TOKEN = token;
    window.CROWNTALK = window.CROWNTALK || {};
    window.CROWNTALK.getAccessToken = getStoredToken;
    window.CROWNTALK.TOKEN_HEADER = "X-Crowntalk-Token";
  }

  function els() {
    return {
      gate:  document.getElementById('adminGate'),
      input: document.getElementById('password'),
    };
  }

  function showGate() {
    const { gate, input } = els();
    if (!gate) return;
    gate.hidden = false;
    gate.style.display = 'grid';
    document.body.style.overflow = 'hidden';
    if (input) {
      input.value = '';
      setTimeout(() => input.focus(), 0);
    }
  }

  function hideGate() {
    const { gate } = els();
    if (!gate) return;
    gate.hidden = true;
    gate.style.display = 'none';
    document.body.style.overflow = '';
  }

  async function tryAuth() {
    const { input } = els();
    if (!input) return;
    const val = (input.value || '').trim();
    if (!val) return;

    // Prefer backend verification so the server can enforce the gate too.
    try {
       const res = await fetch(`${API_BASE}/verify_access`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: val }),
      });
      let data = {};
      try { data = await res.json(); } catch {}
      if (res.ok && data && (data.ok || data.token)) {
        const token = (data && data.token) ? String(data.token) : val;
        markAuthorized(token);
        exposeTokenHelpers();
        hideGate();
        bootAppUI();
        return;
      }
    } catch (err) {
      console.warn('verify_access failed, falling back to local ACCESS_CODE', err);
    }

// Fallback: allow the hard-coded ACCESS_CODE if backend is not reachable yet.
// Hash and store a SHA-256 hex so the token looks like a backend token.
if (val === ACCESS_CODE) {
  async function sha256Hex(str) {
    if (typeof crypto !== 'undefined' && crypto.subtle && typeof TextEncoder !== 'undefined') {
      const enc = new TextEncoder().encode(str);
      const hash = await crypto.subtle.digest('SHA-256', enc);
      return Array.from(new Uint8Array(hash)).map(b => b.toString(16).padStart(2, '0')).join('');
    }
    return null;
  }

  try {
    const hashed = await sha256Hex(ACCESS_CODE);
    if (hashed) {
      markAuthorized(hashed);
    } else {
      // Environment doesn't support SubtleCrypto — store raw code temporarily.
      markAuthorized(ACCESS_CODE);
      console.warn('SubtleCrypto not available — stored raw ACCESS_CODE temporarily.');
    }
  } catch (err) {
    // Hash failed for some reason — fall back to storing raw code (temporary).
    markAuthorized(ACCESS_CODE);
    console.warn('Hashing failed — stored raw ACCESS_CODE temporarily.', err);
  }

  exposeTokenHelpers();
  hideGate();
  bootAppUI();
  return;
}
    input.classList.add('ct-shake');
    setTimeout(() => input.classList.remove('ct-shake'), 350);
    input.value = '';
    input.placeholder = 'Wrong code — try again';
  }


  function bindGate() {
    const { gate, input } = els();
    if (!gate) return;

    const form = gate.querySelector('form');
    if (form) {
      form.addEventListener('submit', (e) => {
        e.preventDefault();
        tryAuth();
      });
    }

    const button = gate.querySelector('[data-ct-auth-submit]');
    if (button) button.addEventListener('click', tryAuth);

    if (input) {
      input.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
          event.preventDefault();
          tryAuth();
        }
      });
    }

    const lockIcon = document.getElementById('lockFloatingIcon');
    if (lockIcon) {
      lockIcon.addEventListener('click', tryAuth);
    }
  }

  function init() {
  exposeTokenHelpers();
  // Defer auth check slightly so in-flight async writes (hash -> storage) can finish on mobile.
  setTimeout(() => {
    if (isAuthorized()) {
      hideGate();
      bootAppUI();
    } else {
      showGate();
      bindGate();
    }
  }, 150);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

/* ========= App code ========= */

// ------------------------
// Backend endpoints
// ------------------------
const backendBase = (() => {
  try {
    if (window.CT_API_BASE && String(window.CT_API_BASE).trim()) return String(window.CT_API_BASE).trim().replace(/\/$/, "");
    if (window.CT_API && String(window.CT_API).trim()) return String(window.CT_API).trim().replace(/\/$/, "");
  } catch {}
  return "https://crowntalk.onrender.com";
})();
const commentURL  = `${backendBase}/comment`;
const rerollURL   = `${backendBase}/reroll`;

// ------------------------
// DOM elements
// ------------------------
const urlInput        = document.getElementById("urlInput");
const generateBtn     = document.getElementById("generateBtn");
const cancelBtn       = document.getElementById("cancelBtn");
const clearBtn        = document.getElementById("clearBtn");
const progressEl      = document.getElementById("progress");
const progressPctEl   = document.getElementById("progressPct");
const progressBarFill = document.getElementById("progressBarFill");
const resultsEl       = document.getElementById("results");
const failedEl        = document.getElementById("failed");
const resultCountEl   = document.getElementById("resultCount");
const failedCountEl   = document.getElementById("failedCount");
const historyEl       = document.getElementById("history");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const yearEl          = document.getElementById("year");
const welcomeOverlayEl  = document.getElementById("welcomeOverlay");
const welcomeDismissBtn = document.getElementById("welcomeDismissBtn");
const retryFailedBtn  = document.getElementById("retryFailedBtn");

let themeDots = Array.from(document.querySelectorAll(".theme-dot"));

const sessionTabsEl         = document.getElementById("sessionTabs");
const analyticsHudEl        = document.getElementById("analyticsHud");
const urlHealthBadgeEl      = document.getElementById("urlHealthBadge");
const urlHealthMeterFillEl  = document.getElementById("urlHealthMeterFill");
const urlQueueEl            = document.getElementById("urlQueue");
const sortUrlsBtn           = document.getElementById("sortUrlsBtn");
const shuffleUrlsBtn        = document.getElementById("shuffleUrlsBtn");
const removeInvalidBtn      = document.getElementById("removeInvalidBtn");
const copyQueuePanel        = document.getElementById("copyQueuePanel");
const copyQueueListEl       = document.getElementById("copyQueueList");
const copyQueueEmptyEl      = document.getElementById("copyQueueEmpty");
const copyQueueNextBtn      = document.getElementById("copyQueueNextBtn");
const copyQueueClearBtn     = document.getElementById("copyQueueClearBtn");
const presetSelect          = document.getElementById("presetSelect");
const keyboardHudEl         = document.getElementById("keyboardHud");
const exportAllBtn          = document.getElementById("exportAllBtn");
const exportEnBtn           = document.getElementById("exportEnBtn");
const exportNativeBtn       = document.getElementById("exportNativeBtn");
const downloadTxtBtn        = document.getElementById("downloadTxtBtn");
const langEnToggle          = document.getElementById("langEnToggle");
const langNativeToggle      = document.getElementById("langNativeToggle");
const nativeLangSelect     = document.getElementById("nativeLangSelect");
const runCountEl            = document.getElementById("runCount");
const safeModeToggle        = document.getElementById("safeModeToggle");
const holoCheckEl = document.getElementById("holo-check");
const holoCardEl  = document.querySelector(".card-holo");


// ------------------------
// State
// ------------------------
let cancelled    = false;
let historyItems = [];
let failedUrlList = [];
let ctSessions = [];
let ctActiveSessionId = null;
let urlQueueState = [];
let ctSessionCounter = 0;
let ctCopyQueue = [];
let ctCopyQueueIndex = 0;
let langPrefs = { useEn: true, useNative: true };
let safeModeOn = false;
let runCounter = 0;

// Premium feature bridge (no-op if premium script missing)
window.CROWN_PREMIUM = window.CROWN_PREMIUM || {};
window.CROWN_PREMIUM.hooks = window.CROWN_PREMIUM.hooks || {
  onAnalytics: [],
  onQueueRender: [],
  onResultAppend: [],
  onRunStart: [],
  onRunFinish: [],
  onProgress: [],
  onItemStage: []
};

function ctPremiumEmit(ev, payload) {
  const hooks = (window.CROWN_PREMIUM && window.CROWN_PREMIUM.hooks && window.CROWN_PREMIUM.hooks[ev]) || [];
  for (const fn of hooks) {
    try { fn(payload); } catch (e) { /* ignore */ }
  }
}


 /* =========================================================
   === Mini Toast + Snack (used by multiple bits) ===
   ========================================================= */
(function mountToast(){
  if (document.getElementById('ctToasts')) return;
  const box = document.createElement('div');
  box.id = 'ctToasts';
  document.body.appendChild(box);
})();
function ctToast(msg, kind='ok', ms=1800){
  const host = document.getElementById('ctToasts');
  if (!host) return alert(msg);
  const el = document.createElement('div');
  el.className = 'ct-toast';
  el.dataset.kind = kind;
  el.textContent = msg;
  host.appendChild(el);
  const t = setTimeout(()=>{ el.classList.add('ct-leave'); setTimeout(()=>el.remove(), 240); }, ms);
  el.addEventListener('click', ()=>{ clearTimeout(t); el.classList.add('ct-leave'); setTimeout(()=>el.remove(), 160); });
}

/* =========================================================
   Backend helpers (warmup + timeout fetch)
   ========================================================= */
const PER_URL_TIMEOUT_MS = 60000; 

function warmBackendOnce() {
  try {
    fetch(backendBase + "/ping", {
      method: "GET",
      cache: "no-store",
      mode: "cors",
      keepalive: true,
    }).catch(() => {});
  } catch (err) {
    console.warn("warmBackendOnce error", err);
  }
}
let lastWarmAt = 0;
function maybeWarmBackend() {
  const now = Date.now();
  const FIVE_MIN = 5 * 60 * 1000;
  if (now - lastWarmAt > FIVE_MIN) {
    lastWarmAt = now;
    warmBackendOnce();
  }
}


/* ================================
    Abort-aware fetch timeout
   ================================ */
let __activeAbortController = // ------------------------
// Utilities
// ------------------------

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Extract full URLs (with scheme)
const CT_URL_EXTRACT_RE =
  /(https?:\/\/(?:www\.)?(?:x\.com|twitter\.com|mobile\.twitter\.com|m\.twitter\.com)\/(?:i\/status\/\d+|[A-Za-z0-9_]{1,15}\/status\/\d+))/gi;

// Extract URLs WITHOUT scheme (x.com/... or twitter.com/...)
const CT_URL_EXTRACT_NOSCHEME_RE =
  /((?:^|[^a-z0-9_])(?:(?:x|twitter)\.com)\/(?:i\/status\/\d+|[A-Za-z0-9_]{1,15}\/status\/\d+))/gi;

function parseURLs(raw) {
  if (!raw) return [];

  // If user pasted URLs back-to-back with no spaces/newlines, extract them.
  // NOTE: This can create "duplicates" like:
  //   http://x.com/...   +   x.com/... -> https://x.com/...
  // We fix this by forcing HTTPS + canonical hostname later.
  try {
    const hits = [];
    const m1 = raw.match(CT_URL_EXTRACT_RE) || [];
    m1.forEach((u) => hits.push(u));

    const m2 = raw.match(CT_URL_EXTRACT_NOSCHEME_RE) || [];
    m2.forEach((u) => hits.push(String(u).replace(/^[^a-z0-9]+/i, "")));

    const uniqHits = Array.from(new Set(hits)).filter(Boolean);

    // Only rewrite the raw input if it looks like multiple URLs were pasted together
    if (uniqHits.length >= 2) {
      raw = uniqHits.join("\n");
    }
  } catch {}

  // Strip numbering & blank lines first
  const lines = raw
    .split(/\r?\n/)
    .map((line) => line.replace(/^\s*(?:\d+[\.)]\s*)?/, "").trim())
    .filter(Boolean);

  const norm = lines.map((line) => {
    let candidate = line;

    // If user omitted scheme but host looks like x/twitter.com → prepend https://
    if (/^(?:x|twitter)\.com\//i.test(candidate)) {
      candidate = "https://" + candidate;
    } else if (/^(?:\/\/)(?:www\.)?(?:x|twitter)\.com\//i.test(candidate)) {
      // things like //x.com/...
      candidate = "https:" + candidate;
    }

    try {
      const u = new URL(candidate);

      // ✅ Canonicalize hostname to x.com
      // twitter.com / mobile.twitter.com / m.twitter.com → x.com
      if (/^(?:www\.)?(?:mobile\.)?twitter\.com$/i.test(u.hostname) || /^(?:www\.)?m\.twitter\.com$/i.test(u.hostname)) {
        u.hostname = "x.com";
      }
      // www.x.com → x.com
      if (/^www\.x\.com$/i.test(u.hostname)) {
        u.hostname = "x.com";
      }

      // ✅ Force HTTPS (this is what fixes your auto-duplicate: http vs https)
      u.protocol = "https:";

      // Remove query + hash + trailing slash
      u.search = "";
      u.hash = "";
      u.pathname = u.pathname.replace(/\/+$/, "");

      return u.toString();
    } catch {
      // If URL parsing fails, return the cleaned line (no extra transforms)
      return line;
    }
  });

  // De-dupe while preserving order
  const seen = new Set();
  const unique = [];
  for (const v of norm) {
    if (!seen.has(v)) {
      seen.add(v);
      unique.push(v);
    }
  }
  return unique;
       }null;

async function fetchWithTimeout(url, options = {}, timeoutMs = 45000) {
  if (typeof AbortController === "undefined") {
    return fetch(url, options);
  }
  const controller = new AbortController();
  __activeAbortController = controller;

  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    return res;
  } finally {
    clearTimeout(id);
    if (__activeAbortController === controller) __activeAbortController = null;
  }
}


function setProgressText(text) {
  if (progressEl) progressEl.textContent = text || "";
}
function setProgressRatio(ratio) {
  if (!progressBarFill) return;
  const clamped = Math.max(0, Math.min(1, Number.isFinite(ratio) ? ratio : 0));
  // drive only via CSS variables (progress bar CSS uses --ct-progress-frac)
  document.documentElement.style.setProperty('--ct-progress-frac', String(clamped));
  document.documentElement.style.setProperty('--ct-progress-pct', String(Math.round(clamped*100)));
  try {
    const pct = Math.round(clamped * 100);
    if (progressPctEl) progressPctEl.dataset.pct = `${pct}%`;
  } catch {}
}
function resetProgress() {
  setProgressText("");
  setProgressRatio(0);
}
function resetResults() {
  if (resultsEl) resultsEl.innerHTML = "";
  if (failedEl) failedEl.innerHTML = "";
  if (resultCountEl) resultCountEl.textContent = "0 tweets";
  if (failedCountEl) failedCountEl.textContent = "0";
  failedUrlList = [];
}
async function copyToClipboard(text) {
  if (!text) return;
  if (navigator.clipboard?.writeText) {
    try { await navigator.clipboard.writeText(text); return; }
    catch (err) { console.warn("navigator.clipboard failed, using fallback", err); }
  }
  const helper = document.createElement("span");
  helper.textContent = text;
  helper.style.position = "fixed";
  helper.style.left = "-9999px";
  helper.style.top = "0";
  helper.style.whiteSpace = "pre";
  document.body.appendChild(helper);
  const selection = window.getSelection();
  const range = document.createRange();
  range.selectNodeContents(helper);
  selection.removeAllRanges();
  selection.addRange(range);
  try { document.execCommand("copy"); } catch (err) { console.error("execCommand copy failed", err); }
  selection.removeAllRanges();
  document.body.removeChild(helper);
}
function formatTweetCount(count) {
  const n = Number(count) || 0;
  return `${n} tweet${n === 1 ? "" : "s"}`;
}
function autoResizeTextarea() {
  if (!urlInput) return;
  urlInput.style.height = "auto";
  const base = 180;
  const newHeight = Math.max(base, urlInput.scrollHeight);
  urlInput.style.height = newHeight + "px";
}

/* =========================================================
   === numbering + dedupe + gentle renumber UI ===
   ========================================================= */
function renumberTextareaAndDedupe(showToasts=true) {
  if (!urlInput) return { changed:false, removed:0 };
  const original = urlInput.value;
  // parseURLs already strips numbering + dedupes.
  const uniq = parseURLs(original);
  // write back with clean 1.,2.,3. numbers so your UI shows correct sequence
  const enumerated = uniq.map((u, i) => `${i+1}. ${u}`).join('\n');
  const removed = original.split(/\r?\n/).filter(Boolean).length - uniq.length;
  const changed = enumerated !== original;
  if (changed) {
    urlInput.value = enumerated;
    urlInput.dispatchEvent(new Event('input', {bubbles:true}));
    if (showToasts && removed > 0) {
      ctToast(`Removed ${removed} duplicate URL${removed>1?'s':''}.`, 'ok');
    }
  }
  return { changed, removed };
}

/* =========================================================
   === URL health + sorting / cleaning ===
   ========================================================= */
function analyzeUrlLines(raw) {
  const lines = (raw || "").split(/\r?\n/);
  const entries = [];
  const seen = new Set();
  let valid = 0,
    invalid = 0,
    duplicates = 0;

  const urlRegex =
    /^(?:https?:\/\/)?(?:www\.)?(?:x|twitter)\.com\/[^/]+\/status\/\d+/i;

  for (let line of lines) {
    const cleaned = line.replace(/^\s*(?:\d+[\.)]\s*)?/, "").trim();
    if (!cleaned) continue;

    const isValid = urlRegex.test(cleaned);
    const isDup = seen.has(cleaned);
    if (!isDup) seen.add(cleaned);
    else duplicates++;

    if (isValid) valid++;
    else invalid++;

    entries.push({ raw: line, url: cleaned, isValid, isDup });
  }

  return { entries, valid, invalid, duplicates, total: entries.length };
}

function updateUrlHealth() {
  if (!urlInput || !urlHealthBadgeEl) return;
  const info = analyzeUrlLines(urlInput.value || "");
  const badge = urlHealthBadgeEl;
  const meterFill = urlHealthMeterFillEl;
  const meter = meterFill ? meterFill.parentElement : null;

  if (!info.total) {
    badge.textContent = "No URLs yet";
    badge.dataset.status = "empty";
    if (meter) {
      meter.dataset.status = "empty";
      if (meterFill) meterFill.style.width = "0%";
    }
    return;
  }

  badge.textContent = `${info.valid} valid · ${info.invalid} invalid · ${info.duplicates} dupes`;

  let status = "ok";
  if (info.invalid || info.duplicates) status = "warn";
  if (info.invalid >= info.valid && info.total > 0) status = "bad";
  badge.dataset.status = status;

  if (meter) {
    const ratio = info.total ? info.valid / info.total : 0;
    const clamped = Math.max(0, Math.min(1, ratio));
    meter.dataset.status = status;
    if (meterFill) {
      meterFill.style.width = `${Math.round(clamped * 100)}%`;
    }
  }
}
function sortUrlsAscending() {
  if (!urlInput) return;
  const urls = parseURLs(urlInput.value || "");
  urls.sort((a, b) => a.localeCompare(b));
  urlInput.value = urls.map((u, i) => `${i + 1}. ${u}`).join("\n");
  urlInput.dispatchEvent(new Event("input", { bubbles: true }));
}
function shuffleUrlsOrder() {
  if (!urlInput) return;
  const urls = parseURLs(urlInput.value || "");
  for (let i = urls.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [urls[i], urls[j]] = [urls[j], urls[i]];
  }
  urlInput.value = urls.map((u, i) => `${i + 1}. ${u}`).join("\n");
  urlInput.dispatchEvent(new Event("input", { bubbles: true }));
}
function removeInvalidUrls() {
  if (!urlInput) return;
  const info = analyzeUrlLines(urlInput.value || "");
  const validEntries = info.entries.filter((e) => e.isValid);
  if (!validEntries.length) {
    ctToast("No valid X URLs found.", "warn");
    return;
  }
  urlInput.value = validEntries.map((entry, idx) => `${idx + 1}. ${entry.url}`).join("\n");
  urlInput.dispatchEvent(new Event("input", { bubbles: true }));
  if (info.invalid) {
    ctToast(`Removed ${info.invalid} invalid line${info.invalid > 1 ? "s" : ""}.`, "ok");
  }
}

/* =========================================================
   === Sessions timeline (desktop tabs) ==============
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
  const createdAt = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const m = meta || {};
  const tweets = m.tweets || 0;
  const totalUrls = m.totalUrls || tweets;
  const failed = m.failed || 0;

  let quality = "perfect";
  if (failed >= 3) quality = "rough";
  else if (failed >= 1) quality = "smooth";

  const label = `Run ${ctSessionCounter} • ${tweets}/${totalUrls} • ${quality}`;

  const snapshot = {
    resultsHTML: resultsEl.innerHTML,
    failedHTML: failedEl.innerHTML,
    rc: resultCountEl ? resultCountEl.textContent : "",
    fc: failedCountEl ? failedCountEl.textContent : "",
    meta: m,
  };
  ctSessions.push({ id, label, createdAt, snapshot });
  ctActiveSessionId = id;
  renderSessionTabs();
}
function restoreSessionSnapshot(id) {
  const s = getSessionById(id);
  if (!s || !s.snapshot) return;
  if (resultsEl) resultsEl.innerHTML = s.snapshot.resultsHTML || "";
  if (failedEl) failedEl.innerHTML = s.snapshot.failedHTML || "";
  if (resultCountEl) resultCountEl.textContent = s.snapshot.rc || "0 tweets";
  if (failedCountEl) failedCountEl.textContent = s.snapshot.fc || "0";
}

/* =========================================================
   ===Copy queue===
   ========================================================= */
function renderCopyQueue() {
  if (!copyQueueListEl) return;
  copyQueueListEl.innerHTML = "";
  const hasItems = ctCopyQueue.length > 0;
  if (copyQueueEmptyEl) {
    copyQueueEmptyEl.style.display = hasItems ? "none" : "block";
  }
  ctCopyQueue.forEach((item, idx) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "copy-queue-item" + (idx === ctCopyQueueIndex ? " is-active" : "");
    row.textContent = item.text;
    row.title = "Click to copy";
    row.addEventListener("click", async () => {
      await copyToClipboard(item.text);
      addToHistory(item.text);
    });
    copyQueueListEl.appendChild(row);
  });
  const enabled = hasItems;
  if (copyQueueNextBtn) copyQueueNextBtn.disabled = !enabled;
  if (copyQueueClearBtn) copyQueueClearBtn.disabled = !enabled;
}
function addToCopyQueue(text) {
  if (!text) return;
  ctCopyQueue.push({ text });
  if (ctCopyQueue.length === 1) ctCopyQueueIndex = 0;
  renderCopyQueue();
}
async function copyNextFromQueue() {
  if (!ctCopyQueue.length) return;
  const item = ctCopyQueue[ctCopyQueueIndex];
  await copyToClipboard(item.text);
  addToHistory(item.text);
  ctCopyQueueIndex = (ctCopyQueueIndex + 1) % ctCopyQueue.length;
  renderCopyQueue();
}
function clearCopyQueue() {
  ctCopyQueue = [];
  ctCopyQueueIndex = 0;
  renderCopyQueue();
}

/* =========================================================
   URL Queue chips (per-batch visual)
   ========================================================= */
function resetUrlQueue() {
  urlQueueState = [];
  renderUrlQueue();
}

function renderUrlQueue() {
  if (!urlQueueEl) return;
  urlQueueEl.innerHTML = "";

  if (!urlQueueState || !urlQueueState.length) {
    urlQueueEl.setAttribute("aria-hidden", "true");
    urlQueueEl.classList.add("is-empty");
    return;
  }

  urlQueueEl.removeAttribute("aria-hidden");
  urlQueueEl.classList.remove("is-empty");

  urlQueueState.forEach((item, idx) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "url-chip";

    if (item.status === "processing") chip.classList.add("is-processing");
    else if (item.status === "done") chip.classList.add("is-done");
    else if (item.status === "failed") chip.classList.add("is-failed");

    const indexSpan = document.createElement("span");
    indexSpan.className = "url-chip-index";
    indexSpan.textContent = `${idx + 1}.`;

    const labelSpan = document.createElement("span");
    labelSpan.className = "url-chip-label";
    const short = item.shortLabel || item.url.replace(/^https?:\/\//, "");
    labelSpan.textContent = short.slice(0, 40);

    chip.appendChild(indexSpan);
    chip.appendChild(labelSpan);

    chip.addEventListener("click", () => {
      if (!resultsEl) return;
      const sel = `.tweet[data-url="${item.url}"]`;
      const el = resultsEl.querySelector(sel);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "start" });
        el.classList.add("ct-highlight");
        setTimeout(() => el.classList.remove("ct-highlight"), 900);
      }
    });

    urlQueueEl.appendChild(chip);
  });

  ctPremiumEmit("onQueueRender", { urlQueueState: urlQueueState.slice() });
}

/* ============================================================ Analytics HUD + export helpers ===============
   ========================================================= */
function updateAnalytics(meta) {
  const summaryEl       = document.getElementById("analyticsSummary");
  const tweetsEl        = document.getElementById("analyticsTweets");
  const failedElMetric  = document.getElementById("analyticsFailed");
  const timeEl          = document.getElementById("analyticsTime");
  const sessionSummaryEl = document.getElementById("sessionSummary");

  const m          = meta || {};
  const tweets     = m.tweets || 0;
  const totalUrls  = m.totalUrls || tweets;
  const failed     = m.failed || 0;
  const durationSec = m.durationSec || 0;
  const comments   = m.comments || (tweets * 2); // fallback guess

  if (summaryEl)      summaryEl.textContent = `${tweets}/${totalUrls} tweets · ${failed} failed`;
  if (tweetsEl)       tweetsEl.textContent = String(tweets);
  if (failedElMetric) failedElMetric.textContent = String(failed);
  if (timeEl)         timeEl.textContent = `${durationSec}s`;

  if (sessionSummaryEl) {
    const tLabel = `${tweets} tweet${tweets === 1 ? "" : "s"}`;
    const cLabel = `${comments} comment${comments === 1 ? "" : "s"}`;
    const fLabel = `${failed} failed`;
    sessionSummaryEl.textContent = `${tLabel} • ${cLabel} • ${fLabel}`;
  }

  if (analyticsHudEl) {
    analyticsHudEl.style.opacity = "1";
    analyticsHudEl.setAttribute("aria-hidden", "false");
  }
    ctPremiumEmit("onAnalytics", meta || {});
}
// ------------------------
// Language filter helpers
// ------------------------
function languageFilterAllows(lang) {
  const code = lang || "native";
  if (!langPrefs) return true;
  if (code === "en") return !!langPrefs.useEn;
  return !!langPrefs.useNative;
}

function applyLangFilterToDom() {
  if (!resultsEl) return;
  const nodes = resultsEl.querySelectorAll(".comment-line");
  nodes.forEach((line) => {
    const lang = line.dataset.lang || "native";
    const show = languageFilterAllows(lang);
    line.style.display = show ? "" : "none";
  });
}

function loadLangPrefsFromStorage() {
  try {
    const en = localStorage.getItem("ct_lang_en_v1");
    const native = localStorage.getItem("ct_lang_native_v1");
    if (en === "0") langPrefs.useEn = false;
    if (native === "0") langPrefs.useNative = false;
  } catch {}
}

function persistLangPrefs() {
  try {
    localStorage.setItem("ct_lang_en_v1", langPrefs.useEn ? "1" : "0");
    localStorage.setItem("ct_lang_native_v1", langPrefs.useNative ? "1" : "0");
  } catch {}
}

function syncLangUIFromPrefs() {
  if (langEnToggle && langEnToggle.parentElement) {
    langEnToggle.checked = !!langPrefs.useEn;
    langEnToggle.parentElement.setAttribute("data-active", langPrefs.useEn ? "true" : "false");
  }
  toggleNativeLangSelect();

  if (langNativeToggle && langNativeToggle.parentElement) {
    langNativeToggle.checked = !!langPrefs.useNative;
    langNativeToggle.parentElement.setAttribute("data-active", langPrefs.useNative ? "true" : "false");
  }
}

function syncLangPrefsFromUI() {
  if (langEnToggle) langPrefs.useEn = !!langEnToggle.checked;
  if (langNativeToggle) langPrefs.useNative = !!langNativeToggle.checked;
  syncLangUIFromPrefs();
  persistLangPrefs();
  applyLangFilterToDom();
}

function getLanguagePreferenceArray() {
  const langs = [];
  if (langPrefs.useEn) langs.push("en");
  if (langPrefs.useNative) langs.push("native");
  if (!langs.length) langs.push("en");
  return langs;
}

function initLanguageToggles() {
  loadLangPrefsFromStorage();
  syncLangUIFromPrefs();
  applyLangFilterToDom();
  if (langEnToggle) langEnToggle.addEventListener("change", syncLangPrefsFromUI);
  if (langNativeToggle) langNativeToggle.addEventListener("change", () => { syncLangPrefsFromUI(); toggleNativeLangSelect(); });
  if (nativeLangSelect) nativeLangSelect.addEventListener("change", () => { /* no-op; used in payload */ });
}


/* ------------------------
   Run counter + Safe mode
   ------------------------ */
function initRunCounter() {
  try {
    const raw = localStorage.getItem("ct_run_counter_v1");
    const n = raw ? parseInt(raw, 10) : 0;
    if (!Number.isNaN(n) && n >= 0) runCounter = n;
  } catch {}
  if (runCountEl) {
    const initial = runCounter && runCounter > 0 ? runCounter : 1;
    runCountEl.textContent = String(initial);
  }
}

function bumpRunCounter() {
  runCounter = (runCounter || 0) + 1;
  if (runCountEl) {
    runCountEl.textContent = String(runCounter);
  }
  try {
    localStorage.setItem("ct_run_counter_v1", String(runCounter));
  } catch {}
}

function loadSafeModeFromStorage() {
  try {
    const val = localStorage.getItem("ct_safe_mode_v1");
    safeModeOn = val === "1";
  } catch {}
}

function persistSafeMode() {
  try {
    localStorage.setItem("ct_safe_mode_v1", safeModeOn ? "1" : "0");
  } catch {}
}

function applySafeMode() {
  const html = document.documentElement;
  const body = document.body;
  if (!html || !body) return;

  if (safeModeOn) {
    html.classList.add("ct-safe");
    body.classList.add("low-motion");
  } else {
    html.classList.remove("ct-safe");
    body.classList.remove("low-motion");
  }

  if (safeModeToggle) {
    safeModeToggle.setAttribute("aria-pressed", safeModeOn ? "true" : "false");
    safeModeToggle.textContent = safeModeOn ? "Low motion: ON" : "Low motion: OFF";
    safeModeToggle.style.display = "inline-block";
  }
}

function initSafeModeToggle() {
  loadSafeModeFromStorage();
  applySafeMode();
  if (safeModeToggle) {
    safeModeToggle.addEventListener("click", () => {
      safeModeOn = !safeModeOn;
      persistSafeMode();
      applySafeMode();
      ctToast(safeModeOn ? "Low-motion mode enabled" : "Low-motion mode disabled", "ok");
    });
  }
}
async function exportComments(mode) {
  if (!resultsEl) return;
  const lines = [];
  const nodes = resultsEl.querySelectorAll(".comment-line");
  nodes.forEach((line) => {
    const bubble = line.querySelector(".comment-text");
    if (!bubble) return;
    const text = bubble.textContent.trim();
    if (!text) return;
    const lang = line.dataset.lang || "native";
    if (mode === "en" && lang !== "en") return;
    if (mode === "native" && lang === "en") return;
    if ((mode === "all" || mode === "download") && !languageFilterAllows(lang)) return;
    lines.push(text);
  });
  if (!lines.length) {
    ctToast("No comments to export yet.", "warn");
    return;
  }
  const joined = lines.join("\n");
  if (mode === "download") {
    const blob = new Blob([joined], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "crowntalk-comments.txt";
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      URL.revokeObjectURL(url);
      a.remove();
    }, 0);
    ctToast(`Downloaded ${lines.length} comment${lines.length > 1 ? "s" : ""} as .txt`, "ok");
  } else {
    await copyToClipboard(joined);
    ctToast(`Copied ${lines.length} comment${lines.length > 1 ? "s" : ""}.`, "ok");
  }
}

/* =========================================================
   === Presets & keyboard HUD / hotkeys =============
   ========================================================= */
function applyPreset(presetName) {
  const p = presetName || (presetSelect && presetSelect.value) || "default";
  document.documentElement.setAttribute("data-ct-preset", p);
  try {
    localStorage.setItem("ct_preset_v1", p);
  } catch {}
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
function initKeyboardHud() {
  if (!keyboardHudEl) return;
  const desktop = matchMedia("(min-width: 1024px)").matches && matchMedia("(pointer:fine)").matches;
  if (!desktop) {
    keyboardHudEl.style.display = "none";
    return;
  }
  // we'll control visibility via .is-open + floating fab
  keyboardHudEl.style.display = "none";
}
function handleGlobalHotkeys(event) {
  const key = event.key;
  const metaOrCtrl = event.metaKey || event.ctrlKey;
  if (metaOrCtrl && key === "Enter") {
    event.preventDefault();
    if (!document.body.classList.contains("is-generating")) {
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
    copyNextFromQueue();
  }
}

/* floating shortcut button */
function initShortcutFab() {
  if (!keyboardHudEl) return;
  const desktop = matchMedia("(min-width: 1024px)").matches && matchMedia("(pointer:fine)").matches;
  if (!desktop) return;
  if (document.getElementById("shortcutFab")) return;

  const fab = document.createElement("button");
  fab.id = "shortcutFab";
  fab.type = "button";
  fab.className = "shortcut-fab";
  fab.title = "Keyboard shortcuts";
  fab.textContent = "⌨";
  document.body.appendChild(fab);

  fab.addEventListener("click", () => {
    const open = !keyboardHudEl.classList.contains("is-open");
    keyboardHudEl.classList.toggle("is-open", open);
    keyboardHudEl.style.display = open ? "flex" : "none";
  });
}

/* =========================================================
                        History
   ========================================================= */
function addToHistory(text) {
  if (!text) return;
  const timestamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  historyItems.push({ text, timestamp });
  renderHistory();
}
function renderHistory() {
  if (!historyEl) return;
  historyEl.innerHTML = "";
  if (!historyItems.length) {
    historyEl.textContent = "Copied comments will show up here.";
    return;
  }
  [...historyItems].reverse().forEach((item) => {
    const entry = document.createElement("div");
    entry.className = "history-item";
    const textSpan = document.createElement("div");
    textSpan.className = "history-text";
    textSpan.textContent = item.text;
    const right = document.createElement("div");
    right.style.display = "flex";
    right.style.flexDirection = "column";
    right.style.alignItems = "flex-end";
    right.style.gap = "4px";
    const meta = document.createElement("div");
    meta.className = "history-meta";
    meta.textContent = item.timestamp;
    const btn = document.createElement("button");
    btn.className = "history-copy-btn";
    const main = document.createElement("span");
    main.innerHTML = `
      <svg width="12" height="12" fill="#0E418F" xmlns="http://www.w3.org/2000/svg"
           shape-rendering="geometricPrecision" text-rendering="geometricPrecision"
           image-rendering="optimizeQuality" fill-rule="evenodd" clip-rule="evenodd"
           viewBox="0 0 467 512.22">
        <path fill-rule="nonzero"
          d="M131.07 372.11c.37 1 .57 2.08.57 3.2 0 1.13-.2 2.21-.57 3.21v75.91c0 10.74 4.41 20.53 11.5 27.62s16.87 11.49 27.62 11.49h239.02c10.75 0 20.53-4.4 27.62-11.49s11.49-16.88 11.49-27.62V152.42c0-10.55-4.21-20.15-11.02-27.18l-.47-.43c-7.09-7.09-16.87-11.5-27.62-11.5H170.19c-10.75 0-20.53 4.41-27.62 11.5s-11.5 16.87-11.5 27.61v219.69zm-18.67 12.54H57.23c-15.82 0-30.1-6.58-40.45-17.11C6.41 356.97 0 342.4 0 326.52V57.79c0-15.86 6.5-30.3 16.97-40.78l.04-.04C27.51 6.49 41.94 0 57.79 0h243.63c15.87 0 30.3 6.51 40.77 16.98l.03.03c10.48 10.48 16.99 24.93 16.99 40.78v36.85h50c15.9 0 30.36 6.5 40.82 16.96l.54.58c10.15 10.44 16.43 24.66 16.43 40.24v302.01c0 15.9-6.5 30.36-16.96 40.82-10.47 10.47-24.93 16.97-40.83 16.97H170.19c-15.9 0-30.35-6.5-40.82-16.97-10.47-10.46-16.97-24.92-16.97-40.82v-69.78zM340.54 94.64V57.79c0-10.74-4.41-20.53-11.5-27.63-7.09-7.08-16.86-11.48-27.62-11.48H57.79c-10.78 0-20.56 4.38-27.62 11.45l-.04.04c-7.06 7.06-11.45 16.84-11.45 27.62v268.73c0 10.86 4.34 20.79 11.38 27.97 6.95 7.07 16.54 11.49 27.17 11.49h55.17V152.42c0-15.9 6.5-30.35 16.97-40.82 10.47-10.47 24.92-16.96 40.82-16.96h170.35z">
        </path>
      </svg>
      Copy
    `;
    const alt = document.createElement("span");
    alt.textContent = "Copied";
    btn.appendChild(main);
    btn.appendChild(alt);
    btn.dataset.text = item.text;
    right.appendChild(meta);
    right.appendChild(btn);
    entry.appendChild(textSpan);
    entry.appendChild(right);
    historyEl.appendChild(entry);
  });
}

/* =========================================================
                    Rendering helpers
   ========================================================= */
function buildTweetBlock(result) {
  const url = (result && result.url) || "";
  const comments = Array.isArray(result && result.comments) ? result.comments : [];

  const tweet = document.createElement("div");
  tweet.className = "tweet";
  tweet.dataset.url = url;

  const header = document.createElement("div");
  header.className = "tweet-header";

  // left side: link + optional “research assisted” badge
  const headerLeft = document.createElement("div");
  headerLeft.className = "tweet-header-left";

  const link = document.createElement("a");
  link.className = "tweet-link";
  link.href = url || "#";
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = url || "(no url)";
  link.title = "Open tweet (tap to open)";

  headerLeft.appendChild(link);

  // NEW: research-assisted badge if backend says so
  if (result && result.used_research) {
    const badge = document.createElement("span");
    badge.className = "tweet-badge";
    badge.textContent = "Research assisted";
    headerLeft.appendChild(badge);
  }

  const actions = document.createElement("div");
  actions.className = "tweet-actions";

  const rerollBtn = document.createElement("button");
  rerollBtn.className = "reroll-btn";
  rerollBtn.textContent = "Reroll";

  /* collapse + pin buttons */
  const collapseBtn = document.createElement("button");
  collapseBtn.type = "button";
  collapseBtn.className = "tweet-collapse-btn";
  collapseBtn.title = "Collapse / expand comments";
  collapseBtn.textContent = "▾";

  const pinBtn = document.createElement("button");
  pinBtn.type = "button";
  pinBtn.className = "tweet-pin-btn";
  pinBtn.title = "Pin this tweet";

  actions.appendChild(rerollBtn);
  actions.appendChild(collapseBtn);
  actions.appendChild(pinBtn);

  header.appendChild(headerLeft);
  header.appendChild(actions);
  tweet.appendChild(header);

  // ----- comments list -----
  const commentsWrap = document.createElement("div");
  commentsWrap.className = "comments";

  const hasNative = comments.some((c) => c && c.lang && c.lang !== "en");
  const multilingual = hasNative;

  comments.forEach((comment, idx) => {
    if (!comment || !comment.text) return;

    const lang = comment.lang || "native";

    const line = document.createElement("div");
    line.className = "comment-line";
    line.dataset.lang = lang;
    if (!languageFilterAllows(lang)) {
      line.style.display = "none";
    }

    const tag = document.createElement("span");
    tag.className = "comment-tag";
    tag.textContent = multilingual
      ? (lang === "en" ? "EN" : (lang || "native").toUpperCase())
      : `EN #${idx + 1}`;

    const bubble = document.createElement("span");
    bubble.className = "comment-text";
    bubble.textContent = comment.text;

    const copyBtn = document.createElement("button");
    let copyLabel = "Copy";
    if (multilingual) {
      if (comment.lang === "en") {
        copyBtn.className = "copy-btn-en";
        copyLabel = "Copy EN";
      } else {
        copyBtn.className = "copy-btn";
      }
    } else {
      copyBtn.className = "copy-btn";
    }

    const copyMain = document.createElement("span");
    copyMain.innerHTML = `
      <svg width="12" height="12" fill="#0E418F" xmlns="http://www.w3.org/2000/svg"
           shape-rendering="geometricPrecision" text-rendering="geometricPrecision"
           image-rendering="optimizeQuality" fill-rule="evenodd" clip-rule="evenodd"
           viewBox="0 0 467 512.22">
        <path fill-rule="nonzero"
          d="M131.07 372.11c.37 1 .57 2.08.57 3.2 0 1.13-.2 2.21-.57 3.21v75.91c0 10.74 4.41 20.53 11.5 27.62s16.87 11.49 27.62 11.49h239.02c10.75 0 20.53-4.4 27.62-11.49s11.49-16.88 11.49-27.62V152.42c0-10.55-4.21-20.15-11.02-27.18l-.47-.43c-7.09-7.09-16.87-11.5-27.62-11.5H170.19c-10.75 0-20.53 4.41-27.62 11.5s-11.5 16.87-11.5 27.61v219.69zm-18.67 12.54H57.23c-15.82 0-30.1-6.58-40.45-17.11C6.41 356.97 0 342.4 0 326.52V57.79c0-15.86 6.5-30.3 16.97-40.78l.04-.04C27.51 6.49 41.94 0 57.79 0h243.63c15.87 0 30.3 6.51 40.77 16.98l.03.03c10.48 10.48 16.99 24.93 16.99 40.78v36.85h50c15.9 0 30.36 6.5 40.82 16.96l.54.58c10.15 10.44 16.43 24.66 16.43 40.24v302.01c0 15.9-6.5 30.36-16.96 40.82-10.47 10.47-24.93 16.97-40.83 16.97H170.19c-15.9 0-30.35-6.5-40.82-16.97-10.47-10.46-16.97-24.92-16.97-40.82v-69.78zM340.54 94.64V57.79c0-10.74-4.41-20.53-11.5-27.63-7.09-7.08-16.86-11.48-27.62-11.48H57.79c-10.78 0-20.56 4.38-27.62 11.45l-.04.04c-7.06 7.06-11.45 16.84-11.45 27.62v268.73c0 10.86 4.34 20.79 11.38 27.97 6.95 7.07 16.54 11.49 27.17 11.49h55.17V152.42c0-15.9 6.5-30.35 16.97-40.82 10.47-10.47 24.92-16.96 40.82-16.96h170.35z">
        </path>
      </svg>
      ${copyLabel}
    `;
    const copyAlt = document.createElement("span");
    copyAlt.textContent = "Copied";
    copyBtn.appendChild(copyMain);
    copyBtn.appendChild(copyAlt);
    copyBtn.dataset.text = comment.text;

    // make comment text available for queue
    line.dataset.commentText = comment.text;

    line.appendChild(tag);
    line.appendChild(bubble);

    // small "+" button to queue this comment
    const queueBtn = document.createElement("button");
    queueBtn.type = "button";
    queueBtn.className = "queue-btn";
    queueBtn.title = "Add to copy queue";
    queueBtn.textContent = "+";
    line.appendChild(queueBtn);

    line.appendChild(copyBtn);
    commentsWrap.appendChild(line);
  });

  tweet.appendChild(commentsWrap);
  return tweet;
}

function appendResultBlock(result) {
  const block = buildTweetBlock(result);
  resultsEl.appendChild(block);

  ctPremiumEmit("onResultAppend", { result, element: block });
}
function updateTweetBlock(tweetEl, result) {
  if (!tweetEl) return;
  tweetEl.dataset.url = result.url || tweetEl.dataset.url || "";
  const oldComments = tweetEl.querySelector(".comments");
  if (oldComments) oldComments.remove();
  const newComments = buildTweetBlock(result).querySelector(".comments");
  if (newComments) tweetEl.appendChild(newComments);
  tweetEl.style.transition = "box-shadow 0.25s ease, transform 0.25s ease";
  const oldShadow = tweetEl.style.boxShadow;
  const oldTransform = tweetEl.style.transform;
  tweetEl.style.boxShadow = "0 0 0 2px rgba(56,189,248,0.9)";
  tweetEl.style.transform = "translateY(-1px)";
  setTimeout(() => {
    tweetEl.style.boxShadow = oldShadow;
    tweetEl.style.transform = oldTransform;
  }, 420);
}
function appendFailedItem(failure) {
  const wrapper = document.createElement("div");
  wrapper.className = "failed-item";
  const urlSpan = document.createElement("div");
  urlSpan.className = "failed-url";
  urlSpan.textContent = failure.url || "(unknown URL)";
  const reasonSpan = document.createElement("div");
  reasonSpan.className = "failed-reason";
  reasonSpan.textContent = failure.reason || "Unknown error";
  wrapper.appendChild(urlSpan);
  wrapper.appendChild(reasonSpan);
  failedEl.appendChild(wrapper);
}
function showSkeletons(count) {
  resultsEl.innerHTML = "";
  const num = Math.min(Math.max(count, 1), 6);
  for (let i = 0; i < num; i++) {
    const sk = document.createElement("div");
    sk.className = "tweet-skeleton";
    for (let j = 0; j < 3; j++) {
      const line = document.createElement("div");
      line.className = "tweet-skeleton-line";
      sk.appendChild(line);
    }
    resultsEl.appendChild(sk);
  }
}

/* =========================================================
   =============      Batch ETA Whisper ====================
   ========================================================= */
(function mountETA(){
  if (!progressEl || document.getElementById('progressEta')) return;
  const et = document.createElement('div');
  et.id = 'progressEta';
  et.className = 'progress-eta';
  progressEl.parentElement?.appendChild(et);
})();
let __ctStartTs = 0;
function __updateETAFromText(txt){
  const et = document.getElementById('progressEta');
  if (!et) return;
  const m = /Processed\s+(\d+)\s*\/\s*(\d+)/i.exec(txt || '');
  if (!m) { et.textContent = ''; return; }
  const done = parseInt(m[1],10), total = parseInt(m[2],10);
  if (!total || done<0) { et.textContent=''; return; }
  const now = performance.now();
  if (!__ctStartTs) __ctStartTs = now;
  const elapsed = Math.max(1, (now-__ctStartTs)/1000);
  const rate = done/elapsed; // items per sec
  const left = Math.max(0, total - done);
  const eta = rate>0 ? Math.round(left/rate) : 0;
  if (done === total) { et.textContent = 'Done'; __ctStartTs = 0; }
  else et.textContent = `~${eta}s left`;
}
const __origSetProgressText = setProgressText;
setProgressText = function patchedSetProgressText(t){
  __updateETAFromText(t || '');
  return __origSetProgressText.apply(this, arguments);
};

/* =========================================================
   Hologram engine state helper
   ========================================================= */
function setEngineStatus(state) {
  // state: "idle" | "running" | "error"
  if (!holoCardEl || !holoCheckEl) return;

  // Remember state for CSS tweaks
  holoCardEl.setAttribute("data-state", state);

  // Use checkbox as the main ON/OFF switch for the animation
  holoCheckEl.checked = state === "running";
}


/* =========================================================
                         Generate flow
   ========================================================= */
async function handleGenerate() {
  const raw = urlInput.value;

  // preflight renumber + dedupe BEFORE parse
  renumberTextareaAndDedupe(true);

  const urls = parseURLs(raw);
  if (!urls.length) {
    alert("Please paste at least one tweet URL.");
    return;
  }

  // NEW: clear previous failed URLs for this run
  failedUrlList = [];

  maybeWarmBackend();

  cancelled = false;
  document.body.classList.add("is-generating");
  setEngineStatus("running");

  // Ultra-Lite mode if phone/low-motion
  if (matchMedia("(pointer:coarse)").matches || matchMedia("(prefers-reduced-motion: reduce)").matches) {
    document.documentElement.classList.add("ultralite-on");
  }

  generateBtn.disabled = true;
  cancelBtn.disabled   = false;
  resetResults();
  resetProgress();
  resetUrlQueue();

  // PREMIUM: notify run start
  ctPremiumEmit("onRunStart", { urls, preset: presetSelect?.value || "" });

  // build visual queue for this batch
  urlQueueState = urls.map((u) => ({
    url: u,
    status: "queued",
    shortLabel: u.replace(/^https?:\/\//, "")
  }));
  renderUrlQueue();

  // PREMIUM: initial queue snapshot
  ctPremiumEmit("onQueueRender", { urlQueueState: urlQueueState.slice() });

  setProgressText(`Processing ${urls.length} URL${urls.length === 1 ? "" : "s"}…`);
  setProgressRatio(0);

  let totalResults    = 0;
  let totalFailed     = 0;
  let processedUrls   = 0;
  let totalComments   = 0;
  let clearedSkeletons = false;

  showSkeletons(Math.min(urls.length, 4));
  const runStartedAt = performance.now();

  // Fire-and-forget warmup
  maybeWarmBackend();

  // Mark all as processing at start (progress updates will come via Premium WS if enabled)
  for (let i = 0; i < urlQueueState.length; i++) {
    urlQueueState[i].status = "processing";
  }
  renderUrlQueue();

  const headers = {
    "Content-Type": "application/json",
  };
  if (window.CROWNTALK?.getAccessToken && window.CROWNTALK.TOKEN_HEADER) {
    const t = window.CROWNTALK.getAccessToken();
    if (t) headers[window.CROWNTALK.TOKEN_HEADER] = t;
  }

  const payload = {
    urls, // ✅ send the whole batch in ONE request
    lang_en: !!(langEnToggle && langEnToggle.checked),
    lang_native: !!(langNativeToggle && langNativeToggle.checked),
    native_lang: (nativeLangSelect && nativeLangSelect.value) || "",
    safe_mode: !!(safeModeToggle && safeModeToggle.checked),
    preset: presetSelect?.value || "",
  };

  const requestOptions = {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  };

  // Timeout scales with batch size (but caps so it doesn't get absurd)
  const RUN_TIMEOUT_MS = Math.min(12 * 60 * 1000, Math.max(PER_URL_TIMEOUT_MS, urls.length * PER_URL_TIMEOUT_MS));

  try {
    let res;
    try {
      res = await fetchWithTimeout(commentURL, requestOptions, RUN_TIMEOUT_MS);
    } catch (firstErr) {
      console.warn("Generate attempt failed, warming backend then retrying once…", firstErr);
      setProgressText("Waking CrownTALK engine… retrying once");
      warmBackendOnce();
      res = await fetchWithTimeout(commentURL, requestOptions, RUN_TIMEOUT_MS);
    }

    let data = {};
    try { data = await res.json(); } catch {}

    if (!clearedSkeletons) {
      resultsEl.innerHTML = "";
      failedEl.innerHTML  = "";
      clearedSkeletons = true;
    }

    if (!res.ok) {
      // Whole-run failure
      const failure = {
        url: "(batch)",
        reason: data?.error || `Backend error: ${res.status}`,
        code: data?.code || `http_${res.status}`,
      };
      appendFailedItem(failure);
      totalFailed += 1;
      failedCountEl.textContent = String(totalFailed);

      // Mark all as failed in queue
      for (let i = 0; i < urlQueueState.length; i++) {
        urlQueueState[i].status = "failed";
      }
      renderUrlQueue();
    } else {
      const results = Array.isArray(data.results) ? data.results : [];
      const failed  = Array.isArray(data.failed)  ? data.failed  : [];
      try { window.__CT_RESULTS = results; window.__CT_FAILED = failed; } catch {}

      // Render results
      for (const item of results) {
        appendResultBlock(item);
        totalResults += 1;
        if (item && Array.isArray(item.comments)) {
          totalComments += item.comments.length;
        }
      }

      // Render failed
      for (const f of failed) {
        appendFailedItem(f);
        totalFailed += 1;
        if (f && f.url && !failedUrlList.includes(f.url)) failedUrlList.push(f.url);
      }

      resultCountEl.textContent = formatTweetCount(totalResults);
      failedCountEl.textContent = String(totalFailed);

      // Update queue statuses based on response
      const failedSet = new Set((failed || []).map((x) => String(x?.url || "")));
      const okSet = new Set((results || []).map((x) => String(x?.url || "")));

      for (let i = 0; i < urlQueueState.length; i++) {
        const u = urlQueueState[i]?.url || "";
        // Some URLs may be normalized by backend; treat "contains tweet id" match as ok
        const isFailed = failedSet.has(u);
        const isOk = okSet.has(u);
        urlQueueState[i].status = isFailed ? "failed" : (isOk ? "done" : "done");
      }
      renderUrlQueue();
    }
  } catch (err) {
    if (!clearedSkeletons) {
      resultsEl.innerHTML = "";
      failedEl.innerHTML  = "";
      clearedSkeletons = true;
    }
    const failure = {
      url: "(batch)",
      reason: String(err),
      code: "client_timeout_or_network",
    };
    appendFailedItem(failure);
    totalFailed += 1;
    failedCountEl.textContent = String(totalFailed);

    for (let i = 0; i < urlQueueState.length; i++) urlQueueState[i].status = "failed";
    renderUrlQueue();
  }

  // Done
  setProgressRatio(1);
  setProgressText("Done");


  if (cancelled) return;

  document.body.classList.remove("is-generating");
  document.documentElement.classList.remove("ultralite-on");
  generateBtn.disabled = false;
  cancelBtn.disabled   = true;

  if (!totalResults && totalFailed) {
    setProgressText("All URLs failed to process.");
    setProgressRatio(1);
  } else if (!totalResults && !totalFailed) {
    setProgressText("No comments returned.");
    setProgressRatio(1);
  } else {
    setProgressText(`Processed ${processedUrls} tweet${processedUrls === 1 ? "" : "s"}.`);
    setProgressRatio(1);
  }

  if (!totalResults && totalFailed) {
    setEngineStatus("error");
  } else {
    // normal finish
    setEngineStatus("idle");
  }

  applyLangFilterToDom();
  bumpRunCounter();
  const durationSec = Math.max(1, Math.round((performance.now() - runStartedAt) / 1000));
  const meta = {
    tweets: totalResults,
    failed: totalFailed,
    totalUrls: urls.length,
    durationSec,
    comments: totalComments,
  };

  updateAnalytics(meta);
  addSessionSnapshot(meta);

  // PREMIUM: notify run finish
  ctPremiumEmit("onRunFinish", meta);
}

// ------------------------
// Cancel & Clear
// ------------------------
let __lastClear = null;
function handleCancel() {
  cancelled = true;

  try { __activeAbortController?.abort(); } catch {}

  document.body.classList.remove("is-generating");
  document.documentElement.classList.remove('ultralite-on');
  generateBtn.disabled = false;
  cancelBtn.disabled   = true;
  setProgressText("Cancelled.");
  setProgressRatio(0);
  resetUrlQueue();
  setEngineStatus("idle");
}
function handleClear() {
  __lastClear = {
    input: urlInput.value,
    results: resultsEl.innerHTML,
    failed: failedEl.innerHTML,
    rc: resultCountEl.textContent,
    fc: failedCountEl.textContent
  };
  if (!document.getElementById('ctUndo')) {
    const u = document.createElement('div');
    u.id = 'ctUndo';
    u.className = 'ct-snack';
    u.innerHTML = `<span>Cleared.</span><button type="button" class="ct-snack-btn">Undo</button>`;
    document.body.appendChild(u);
    u.querySelector('button').addEventListener('click', ()=> {
      if (!__lastClear) return;
      urlInput.value = __lastClear.input;
      resultsEl.innerHTML = __lastClear.results;
      failedEl.innerHTML = __lastClear.failed;
      resultCountEl.textContent = __lastClear.rc;
      failedCountEl.textContent = __lastClear.fc;
      failedUrlList = [];
      failedEl.querySelectorAll(".failed-url").forEach((node) => {
        const text = (node.textContent || "").trim();
        if (text && !failedUrlList.includes(text)) {
          failedUrlList.push(text);
        }
      });
      autoResizeTextarea();
      u.classList.remove('show');
      ctToast('Restore complete', 'ok');
    });
  }
  setTimeout(()=>document.getElementById('ctUndo')?.classList.add('show'), 10);
  urlInput.value = "";
  resetResults();
  resetProgress();
  resetUrlQueue();
  autoResizeTextarea();
}
function hideUndoSnackSoon(){
  const s = document.getElementById('ctUndo');
  if (!s) return;
  setTimeout(()=> s.classList.remove('show'), 4000);
}

function handleRetryFailed() {
  if (!failedUrlList.length) {
    if (typeof ctToast === "function") {
      ctToast("No failed URLs to retry yet.", "info");
    } else {
      alert("No failed URLs to retry yet.");
    }
    return;
  }

  // Put only failed URLs back into the textarea (numbered)
  const lines = failedUrlList.map((u, idx) => `${idx + 1}. ${u}`);
  urlInput.value = lines.join("\n");
  autoResizeTextarea();
  hideUndoSnackSoon();

  // Run the normal generate flow on just these URLs
  handleGenerate();
}

// ------------------------
// Reroll
// ------------------------
async function handleReroll(tweetEl) {
  const url = tweetEl?.dataset.url;
  if (!url) return;
  const button = tweetEl.querySelector(".reroll-btn");
  if (!button) return;

  // 🔹 Smooth focus + highlight when rerolling this tweet
  tweetEl.scrollIntoView({ behavior: "smooth", block: "start" });
  tweetEl.classList.add("ct-highlight");
  setTimeout(() => tweetEl.classList.remove("ct-highlight"), 900);

  const oldLabel = button.textContent;
  button.disabled = true;
  button.textContent = "Rerolling…";

  const commentsWrap = tweetEl.querySelector(".comments");
  if (commentsWrap) {
    commentsWrap.innerHTML = "";
    const sk1 = document.createElement("div"); sk1.className = "tweet-skeleton-line";
    const sk2 = document.createElement("div"); sk2.className = "tweet-skeleton-line";
    commentsWrap.appendChild(sk1); commentsWrap.appendChild(sk2);
  }

  try {
    const headers = { "Content-Type": "application/json" };
    try {
      const token = (window.CROWNTALK && typeof window.CROWNTALK.getAccessToken === "function"
        ? window.CROWNTALK.getAccessToken()
        : (window.__CROWNTALK_AUTH_TOKEN || ""));
      if (token) {
        const headerName = (window.CROWNTALK && window.CROWNTALK.TOKEN_HEADER) || "X-Crowntalk-Token";
        headers[headerName] = token;
      }
    } catch {}

    const payload = { url, lang_en: !!(langEnToggle && langEnToggle.checked), lang_native: !!(langNativeToggle && langNativeToggle.checked), native_lang: (nativeLangSelect && nativeLangSelect.value) || "" };
    try {
      const langs = getLanguagePreferenceArray();
      if (Array.isArray(langs) && langs.length) {
        payload.languages = langs;
      }
    } catch {}

    const res = await fetch(rerollURL, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`Reroll failed: ${res.status}`);

    const data = await res.json();
    if (data && Array.isArray(data.comments)) {
      updateTweetBlock(tweetEl, { url: data.url || url, comments: data.comments });
    } else {
      setProgressText("Reroll failed for this tweet.");
    }
  } catch (err) {
    console.error("Reroll error", err);
    setProgressText("Network error during reroll.");
  } finally {
    button.disabled = false;
    button.textContent = oldLabel;
  }
}
// ------------------------
// Theme
// ------------------------
const THEME_STORAGE_KEY = "crowntalk_theme";
const ALLOWED_THEMES = ["white","dark-purple","gold","blue","black","emerald","crimson"];
function sanitizeThemeDots() {
  themeDots.forEach((dot) => {
    if (!dot?.dataset) return;
    const t = (dot.dataset.theme || "").trim();
    if (t === "texture") {
      dot.parentElement && dot.parentElement.removeChild(dot);
    } else if (!ALLOWED_THEMES.includes(t)) {
      dot.dataset.theme = "crimson";
    }
  });
  themeDots = Array.from(document.querySelectorAll(".theme-dot"));
}
function applyTheme(themeName) {
  const html = document.documentElement;
  const t = ALLOWED_THEMES.includes(themeName) ? themeName : "dark-purple";
  html.setAttribute("data-theme", t);
  themeDots.forEach((dot) =>
    dot.classList.toggle("is-active", dot.dataset.theme === t)
  );
  try {
    localStorage.setItem(THEME_STORAGE_KEY, t);
  } catch {}
}
function initTheme() {
  sanitizeThemeDots();
  let theme = "dark-purple";
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored) {
      theme = stored === "neon" ? "crimson" : stored;
      if (stored === "neon") localStorage.setItem(THEME_STORAGE_KEY, "crimson");
    }
    if (!ALLOWED_THEMES.includes(theme)) theme = "dark-purple";
  } catch {}
  applyTheme(theme);
}

/* ---------- Results menu: wrap preset + export into dropdown ---------- */
function initResultsMenu() {
  if (!resultsEl) return;

  const card = resultsEl.closest(".card");
  if (!card) return;

  const toolbar = card.querySelector(".results-toolbar");
  if (!toolbar) return;

  if (toolbar.dataset.menuInit === "1") return;

  // NEW: if the HTML already contains a menu + toggle,
  // we respect that and do not rebuild or move anything.
  const existingMenu   = card.querySelector("#resultsMenu");
  const existingToggle = card.querySelector("#resultsMenuToggle");
  if (existingMenu && existingToggle) {
    toolbar.dataset.menuInit = "1";
    return;
  }

  // Legacy fallback (kept for backwards compatibility):
  // only runs if there is no static markup.
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
// ------------------------
// Daily welcome card (Bangladesh day, reset 6AM)
// ------------------------
const WELCOME_DAY_KEY = "crowntalk_welcome_seen_v1";

function getBangladeshNow() {
  const now = new Date();
  const utcMs = now.getTime() + now.getTimezoneOffset() * 60000;
  const BD_OFFSET_HOURS = 6; // Asia/Dhaka, UTC+6, no DST
  return new Date(utcMs + BD_OFFSET_HOURS * 60 * 60 * 1000);
}

function getBangladeshDayKey() {
  const bd = getBangladeshNow();
  let y = bd.getFullYear();
  let m = bd.getMonth();
  let d = bd.getDate();

  // Treat 00:00–05:59 as "previous day" so reset happens at 6AM
  if (bd.getHours() < 6) {
    const prev = new Date(bd.getTime() - 24 * 60 * 60 * 1000);
    y = prev.getFullYear();
    m = prev.getMonth();
    d = prev.getDate();
  }

  return `${y}-${String(m + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
}

function shouldShowWelcomeCard() {
  const key = getBangladeshDayKey();
  try {
    const stored = localStorage.getItem(WELCOME_DAY_KEY) || sessionStorage.getItem(WELCOME_DAY_KEY);
    return stored !== key;
  } catch (e) {
    // if storage is blocked, just show it
    return true;
  }
}

function markWelcomeSeen() {
  const key = getBangladeshDayKey();
  try { localStorage.setItem(WELCOME_DAY_KEY, key); } catch (e) {}
  try { sessionStorage.setItem(WELCOME_DAY_KEY, key); } catch (e) {}
}

function initWelcomeCard() {
  if (!welcomeOverlayEl || !welcomeDismissBtn) return;

  if (!shouldShowWelcomeCard()) {
    welcomeOverlayEl.classList.remove("is-visible");
    return;
  }

  welcomeOverlayEl.classList.add("is-visible");

  welcomeDismissBtn.addEventListener("click", () => {
    welcomeOverlayEl.classList.remove("is-visible");
    markWelcomeSeen();
  });

  // Click on dark background also dismisses
  welcomeOverlayEl.addEventListener("click", (e) => {
    if (e.target === welcomeOverlayEl) {
      welcomeOverlayEl.classList.remove("is-visible");
      markWelcomeSeen();
    }
  });
}

/* ---------- Boot UI once unlocked ---------- */
function bootAppUI() {
  if (yearEl) yearEl.textContent = String(new Date().getFullYear());
  initTheme();
  renderHistory();
  autoResizeTextarea();
  updateUrlHealth();
  initLanguageToggles();
  initRunCounter();
  initSafeModeToggle();
  initPresetFromStorage();
  initKeyboardHud();
  renderCopyQueue();
  initResultsMenu();
  initShortcutFab();
  renderCopyQueue();
  initResultsMenu();
  initShortcutFab();
  initWelcomeCard();
  setEngineStatus("idle");

  setTimeout(() => { maybeWarmBackend(); }, 4000);

  urlInput?.addEventListener("input", autoResizeTextarea);
  urlInput?.addEventListener("input", updateUrlHealth);

  urlInput?.addEventListener('blur', ()=> renumberTextareaAndDedupe(false));

  generateBtn?.addEventListener("click", () => {
    if (!document.body.classList.contains("is-generating")) handleGenerate();
  });

  generateBtn?.addEventListener('click', () => { renumberTextareaAndDedupe(true); hideUndoSnackSoon(); }, true);

  cancelBtn?.addEventListener("click", handleCancel);
  clearBtn?.addEventListener("click", handleClear);
  retryFailedBtn?.addEventListener("click", (e) => {
    e.preventDefault();
    handleRetryFailed();
  });

  clearHistoryBtn?.addEventListener("click", () => {
    historyItems = [];
    renderHistory();
  });

  resultsEl?.addEventListener("click", async (event) => {
    const copyBtn  = event.target.closest(".copy-btn, .copy-btn-en");
    const rerollBtn = event.target.closest(".reroll-btn");
    const queueBtn = event.target.closest(".queue-btn");

    if (copyBtn) {
      const text = copyBtn.dataset.text || "";
      if (!text) return;
      await copyToClipboard(text);
      addToHistory(text);
      const line = copyBtn.closest(".comment-line");
      if (line) line.classList.add("copied");
      copyBtn.classList.add("is-copied");
      const old = copyBtn.textContent;
      copyBtn.textContent = "Copied";
      copyBtn.disabled = true;
      setTimeout(() => {
        copyBtn.textContent = old;
        copyBtn.disabled = false;
      }, 700);
    }

    if (queueBtn) {
      const line = queueBtn.closest(".comment-line");
      const text = (line && line.dataset.commentText) || line?.querySelector(".comment-text")?.textContent || "";
      if (text) {
        addToCopyQueue(text);
        if (line) line.classList.add("queued");
      }
    }

    if (rerollBtn) {
      const tweetEl = rerollBtn.closest(".tweet");
      if (tweetEl) handleReroll(tweetEl);
    }
  });

  historyEl?.addEventListener("click", async (event) => {
    const btn = event.target.closest(".history-copy-btn");
    if (!btn) return;
    const text = btn.dataset.text || "";
    if (!text) return;
    await copyToClipboard(text);
    const old = btn.textContent;
    btn.textContent = "Copied";
    btn.disabled = true;
    setTimeout(() => {
      btn.textContent = old;
      btn.disabled = false;
    }, 700);
  });

  themeDots.forEach((dot) => {
    dot.addEventListener("click", () => {
      const t = dot.dataset.theme;
      if (t) applyTheme(t);
    });
  });

  sortUrlsBtn?.addEventListener("click", (e) => { e.preventDefault(); sortUrlsAscending(); });
  shuffleUrlsBtn?.addEventListener("click", (e) => { e.preventDefault(); shuffleUrlsOrder(); });
  removeInvalidBtn?.addEventListener("click", (e) => { e.preventDefault(); removeInvalidUrls(); });

  copyQueueNextBtn?.addEventListener("click", (e) => { e.preventDefault(); copyNextFromQueue(); });
  copyQueueClearBtn?.addEventListener("click", (e) => { e.preventDefault(); clearCopyQueue(); });

  presetSelect?.addEventListener("change", () => applyPreset(presetSelect.value));

  exportAllBtn?.addEventListener("click", (e) => { e.preventDefault(); exportComments("all"); });
  exportEnBtn?.addEventListener("click", (e) => { e.preventDefault(); exportComments("en"); });
  exportNativeBtn?.addEventListener("click", (e) => { e.preventDefault(); exportComments("native"); });
  downloadTxtBtn?.addEventListener("click", (e) => { e.preventDefault(); exportComments("download"); });

  window.addEventListener("keydown", handleGlobalHotkeys);
  // === RESULTS MENU DROPDOWN (desktop) ===
  const resultsMenu      = document.getElementById("resultsMenu");
  const resultsMenuToggle = document.getElementById("resultsMenuToggle");

  if (resultsMenu && resultsMenuToggle) {
    resultsMenuToggle.addEventListener("click", (e) => {
      e.stopPropagation();
      resultsMenu.classList.toggle("is-open");
    });

    document.addEventListener("click", () => {
      resultsMenu.classList.remove("is-open");
    });
  }
}

/* =========================================================
   Keep the backend warm
   ========================================================= */
(function keepAliveWhileVisible() {
  const PING_MS = 4 * 60 * 1000;
  let timer = null;
  function schedule() {
    clearInterval(timer);
    if (document.visibilityState === "visible") {
      timer = setInterval(() => {
        fetch("https://crowntalk.onrender.com/ping", {
          method: "GET",
          cache: "no-store",
          mode: "no-cors",
          keepalive: true,
        }).catch(() => {});
      }, PING_MS);
    }
  }
  document.addEventListener("visibilitychange", schedule);
  window.addEventListener("pagehide", () => clearInterval(timer));
  schedule();
})();

/* =========================================================
   CrownTALK — Progress helper (indeterminate + determinate)
========================================================= */
(function () {
  const BODY = document.body;
  const FILL = document.getElementById('progressBarFill');
  let finishTimer = null;

  function clamp01(x){ return Math.max(0, Math.min(1, x)); }
  function setGenerating(on){
    BODY.classList.toggle('is-generating', !!on);
    if (on) {
      if (FILL && (!FILL.style.width || FILL.style.width === '0%')) FILL.style.width = '8%';
    }
  }
  function flashDone(ms = 420){
    BODY.classList.add('ct-progress-done');
    clearTimeout(finishTimer);
    finishTimer = setTimeout(() => BODY.classList.remove('ct-progress-done'), ms);
  }

  window.ctProgress = {
    start(initialPct = 0){
      setGenerating(true);
      if (FILL && typeof initialPct === 'number') {
        FILL.style.width = (clamp01(initialPct / 100) * 100).toFixed(2) + '%';
      }
    },
    step(pct){
      if (!FILL || typeof pct !== 'number') return;
      FILL.style.width = (clamp01(pct / 100) * 100).toFixed(2) + '%';
      document.documentElement.style.setProperty('--ct-progress-pct', String(Math.round(Math.max(0,Math.min(100,pct)))));
      document.documentElement.style.setProperty('--ct-progress-frac', String(Math.max(0,Math.min(1,pct/100))));
    },
    done({ flash = true, resetWidth = true } = {}){
      setGenerating(false);
      if (flash) flashDone();
      if (resetWidth && FILL) FILL.style.width = '0%';
      document.documentElement.style.setProperty('--ct-progress-pct','0');
      document.documentElement.style.setProperty('--ct-progress-frac','0');
    },
    cancel(){
      setGenerating(false);
      if (FILL) FILL.style.width = '0%';
      document.documentElement.style.setProperty('--ct-progress-pct','0');
      document.documentElement.style.setProperty('--ct-progress-frac','0');
    }
  };
})();

/* =========================================================
   ==========  Smooth URL Dropzone (desktop only) ==========
   ========================================================= */
(function urlDropzone(){
  if (!urlInput) return;
  const desktop = matchMedia('(pointer:fine)').matches;
  if (!desktop) return;

  let halo;
  function showHalo(){
    if (halo) return; halo = document.createElement('div');
    halo.className = 'ct-drop-halo';
    document.body.appendChild(halo);
  }
  function hideHalo(){ halo?.remove(); halo = null; }

  window.addEventListener('dragover', (e)=>{ e.preventDefault(); showHalo(); }, false);
  window.addEventListener('dragleave', (e)=>{ if (e.target === document) hideHalo(); }, false);
  window.addEventListener('drop', (e)=>{
    e.preventDefault(); hideHalo();
    let txt = e.dataTransfer.getData('text/uri-list') || e.dataTransfer.getData('text/plain') || '';
    if (!txt) return;
    const curr = urlInput.value.trim();
    const toAdd = txt.trim();
    urlInput.value = curr ? (curr + '\n' + toAdd) : toAdd;
    urlInput.dispatchEvent(new Event('input', {bubbles:true}));
    const { removed } = renumberTextareaAndDedupe(true);
    ctToast(removed>0 ? 'Link added · duplicates removed' : 'Link added', 'ok');
  }, false);
})();

/* =========================================================
   =============  Crash Guard restore banner ===============
   ========================================================= */
(function crashGuardSave(){
  try {
    window.addEventListener('beforeunload', ()=>{
      try {
        if (!urlInput) return;
        const snap = {
          input: urlInput.value || "",
          resultsHTML: resultsEl ? resultsEl.innerHTML : "",
          failedHTML: failedEl ? failedEl.innerHTML : "",
        };
        if (!snap.input && !snap.resultsHTML && !snap.failedHTML) {
          localStorage.removeItem('ct_crash_snapshot_v1');
          return;
        }
        localStorage.setItem('ct_crash_snapshot_v1', JSON.stringify(snap));
      } catch {}
    });
  } catch {}
})();

(function crashGuardRestore(){
  try {
    const raw = localStorage.getItem('ct_crash_snapshot_v1');
    if (!raw) return;
    const snap = JSON.parse(raw);
    if (!snap || !snap.input) return;
    const bar = document.createElement('div');
    bar.className = 'ct-restore';
    bar.innerHTML = `
      <span>Recovered previous session</span>
      <button type="button" class="ct-restore-btn">Restore</button>
      <button type="button" class="ct-restore-skip">Dismiss</button>`;
    document.body.appendChild(bar);
    bar.querySelector('.ct-restore-btn').addEventListener('click', ()=>{
      if (urlInput) { urlInput.value = snap.input; urlInput.dispatchEvent(new Event('input',{bubbles:true})); }
      if (resultsEl) resultsEl.innerHTML = snap.resultsHTML || '';
      if (failedEl)  failedEl.innerHTML  = snap.failedHTML || '';
      ctToast('Restored', 'ok');
      bar.remove();
      localStorage.removeItem('ct_crash_snapshot_v1');
    });
    bar.querySelector('.ct-restore-skip').addEventListener('click', ()=>{
      bar.remove();
      localStorage.removeItem('ct_crash_snapshot_v1');
    });
  } catch {}
})();

function toggleNativeLangSelect() {
  if (!nativeLangSelect) return;
  const on = !!(langNativeToggle && langNativeToggle.checked);
  nativeLangSelect.style.display = on ? "inline-flex" : "none";
  nativeLangSelect.disabled = !on;
}


