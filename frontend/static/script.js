/* ============================================================
   CrownTALK — Gate + App (stable revert)
   - Gate required; code saved once per browser
   - Sends X-CT-Key on every request
   - Works with #adminGate, #password, and the lock SVG
   ============================================================ */

/* ---------- Storage helpers ---------- */
const AUTH_FLAG   = 'crowntalk_access_v1';
const KEY_STORAGE = 'crowntalk_key_v1';
const COOKIE_AUTH = 'crowntalk_access_v1';
const COOKIE_KEY  = 'crowntalk_key_v1';

function saveKey(key) {
  try { localStorage.setItem(KEY_STORAGE, key); } catch {}
  try { sessionStorage.setItem(KEY_STORAGE, key); } catch {}
  try { document.cookie = `${COOKIE_KEY}=${encodeURIComponent(key)}; max-age=${365*24*3600}; path=/; samesite=lax`; } catch {}
}
function getKeyFromCookie() {
  try {
    const m = document.cookie.match(new RegExp(`(?:^|; )${COOKIE_KEY}=([^;]*)`));
    return m ? decodeURIComponent(m[1]) : '';
  } catch { return ''; }
}
function getKey() {
  try { const v = localStorage.getItem(KEY_STORAGE); if (v) return v; } catch {}
  try { const v = sessionStorage.getItem(KEY_STORAGE); if (v) return v; } catch {}
  return getKeyFromCookie();
}
function markAuthorized() {
  try { localStorage.setItem(AUTH_FLAG, '1'); } catch {}
  try { sessionStorage.setItem(AUTH_FLAG, '1'); } catch {}
  try { document.cookie = `${COOKIE_AUTH}=1; max-age=${365*24*3600}; path=/; samesite=lax`; } catch {}
}
function isAuthorized() {
  const k = getKey();
  if (k && k.length > 0) return true;
  try { if (localStorage.getItem(AUTH_FLAG) === '1') return true; } catch {}
  try { if (sessionStorage.getItem(AUTH_FLAG) === '1') return true; } catch {}
  try { if (document.cookie.includes(`${COOKIE_AUTH}=1`)) return true; } catch {}
  return false;
}

/* ---------- Gate UI helpers ---------- */
function gateEls() {
  const gate = document.getElementById('adminGate');
  const input = gate ? (gate.querySelector('#password') || gate.querySelector('input[type="password"]') || gate.querySelector('input')) : null;
  const lockIcon = gate ? gate.querySelector('svg') : null;
  const form = gate ? gate.querySelector('form') : null;
  return { gate, input, lockIcon, form };
}
function showGate(msg) {
  const { gate, input } = gateEls();
  if (!gate) return;
  gate.hidden = false;
  gate.style.display = 'grid';
  document.body.style.overflow = 'hidden';
  if (input) {
    if (msg) input.placeholder = msg;
    input.value = '';
    setTimeout(() => input.focus(), 0);
  }
}
function hideGate() {
  const { gate } = gateEls();
  if (!gate) return;
  gate.hidden = true;
  gate.style.display = 'none';
  document.body.style.overflow = '';
}
function tryAuth() {
  const { input } = gateEls();
  if (!input) { hideGate(); bootAppUI(); return; }
  const val = (input.value || '').trim();
  if (!val) {
    input.classList.add('ct-shake');
    setTimeout(() => input.classList.remove('ct-shake'), 350);
    return;
  }
  saveKey(val);
  markAuthorized();
  hideGate();
  bootAppUI();
}
function bindGate() {
  const { gate, input, lockIcon, form } = gateEls();
  if (!gate) return;
  if (input) {
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); tryAuth(); }
    });
  }
  if (lockIcon) {
    lockIcon.style.cursor = 'pointer';
    lockIcon.addEventListener('click', (e) => { e.preventDefault(); tryAuth(); });
  }
  if (form) {
    form.addEventListener('submit', (e) => { e.preventDefault(); tryAuth(); });
  }
}

/* ---------- Expose helper for other code ---------- */
window.CTGate = {
  getKey,
  requireKey: () => showGate('➤ ENTER ACCESS KEY'),
  show403: () => showGate('Access denied. Enter valid key'),
  hide: hideGate,
  submit: tryAuth
};

/* ============================================================
   App code (unchanged UI; now always sends X-CT-Key)
   ============================================================ */
const backendBase = "https://crowntalk.onrender.com";
const commentURL  = `${backendBase}/comment`;
const rerollURL   = `${backendBase}/reroll`;

const urlInput        = document.getElementById("urlInput");
const generateBtn     = document.getElementById("generateBtn");
const cancelBtn       = document.getElementById("cancelBtn");
const clearBtn        = document.getElementById("clearBtn");
const progressEl      = document.getElementById("progress");
const progressBarFill = document.getElementById("progressBarFill");
const resultsEl       = document.getElementById("results");
const failedEl        = document.getElementById("failed");
const resultCountEl   = document.getElementById("resultCount");
const failedCountEl   = document.getElementById("failedCount");
const historyEl       = document.getElementById("history");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const yearEl          = document.getElementById("year");

let themeDots = Array.from(document.querySelectorAll(".theme-dot"));
let cancelled = false;
let historyItems = [];

function warmBackendOnce(){ try{ fetch(backendBase + "/", {method:"GET",cache:"no-store",mode:"no-cors",keepalive:true}).catch(()=>{});}catch{} }
let lastWarmAt=0;
function maybeWarmBackend(){ const now=Date.now(), FIVE=5*60*1000; if(now-lastWarmAt>FIVE){ lastWarmAt=now; warmBackendOnce(); } }

async function fetchWithTimeout(url, options={}, timeoutMs=45000){
  if (typeof AbortController === "undefined") return fetch(url, options);
  const controller = new AbortController(); const id=setTimeout(()=>controller.abort(), timeoutMs);
  try { const res = await fetch(url, {...options, signal: controller.signal}); clearTimeout(id); return res; }
  catch (e){ clearTimeout(id); throw e; }
}

function parseURLs(raw){ if(!raw) return []; return raw.split(/\r?\n/).map(l=>l.trim()).filter(Boolean).map(l=>l.replace(/^\s*\d+\.\s*/,"").trim()); }
function setProgressText(t){ if(progressEl) progressEl.textContent=t||""; }
function setProgressRatio(r){ if(progressBarFill) progressBarFill.style.width=(Math.max(0,Math.min(1,+r||0))*100).toFixed(2)+"%"; }
function resetProgress(){ setProgressText(""); setProgressRatio(0); }
function resetResults(){ if(resultsEl) resultsEl.innerHTML=""; if(failedEl) failedEl.innerHTML=""; if(resultCountEl) resultCountEl.textContent="0"; if(failedCountEl) failedCountEl.textContent="0"; }
async function copyToClipboard(text){
  if (!text) return;
  if (navigator.clipboard?.writeText){ try{ await navigator.clipboard.writeText(text); return; }catch{} }
  const helper=document.createElement("span"); helper.textContent=text; helper.style.position="fixed"; helper.style.left="-9999px"; helper.style.top="0"; helper.style.whiteSpace="pre";
  document.body.appendChild(helper);
  const sel=window.getSelection(), range=document.createRange(); range.selectNodeContents(helper); sel.removeAllRanges(); sel.addRange(range);
  try{ document.execCommand("copy"); }catch{} sel.removeAllRanges(); document.body.removeChild(helper);
}
function formatTweetCount(n){ n=Number(n)||0; return `${n}`; }
function autoResizeTextarea(){ if(!urlInput) return; urlInput.style.height="auto"; urlInput.style.height=Math.max(180,urlInput.scrollHeight)+"px"; }

function addToHistory(text){ if(!text) return; const ts=new Date().toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"}); historyItems.push({text, timestamp:ts}); renderHistory(); }
function renderHistory(){
  if (!historyEl) return; historyEl.innerHTML="";
  if (!historyItems.length){ historyEl.textContent="Copied comments will show up here."; return; }
  [...historyItems].reverse().forEach((item)=>{
    const entry=document.createElement("div"); entry.className="history-item";
    const textSpan=document.createElement("div"); textSpan.className="history-text"; textSpan.textContent=item.text;
    const right=document.createElement("div"); right.style.display="flex"; right.style.flexDirection="column"; right.style.alignItems="flex-end"; right.style.gap="4px";
    const meta=document.createElement("div"); meta.className="history-meta"; meta.textContent=item.timestamp;
    const btn=document.createElement("button"); btn.className="history-copy-btn";
    const main=document.createElement("span"); main.innerHTML=`<svg width="12" height="12" fill="#0E418F" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 467 512.22"><path fill-rule="nonzero" d="M131.07 372.11c.37 1 .57 2.08.57 3.2 0 1.13-.2 2.21-.57 3.21v75.91c0 10.74 4.41 20.53 11.5 27.62s16.87 11.49 27.62 11.49h239.02c10.75 0 20.53-4.4 27.62-11.49s11.49-16.88 11.49-27.62V152.42c0-10.55-4.21-20.15-11.02-27.18l-.47-.43c-7.09-7.09-16.87-11.5-27.62-11.5H170.19c-10.75 0-20.53 4.41-27.62 11.5s-11.5 16.87-11.5 27.61v219.69zm-18.67 12.54H57.23c-15.82 0-30.1-6.58-40.45-17.11C6.41 356.97 0 342.4 0 326.52V57.79c0-15.86 6.5-30.3 16.97-40.78l.04-.04C27.51 6.49 41.94 0 57.79 0h243.63c15.87 0 30.3 6.51 40.77 16.98l.03.03c10.48 10.48 16.99 24.93 16.99 40.78v36.85h50c15.9 0 30.36 6.5 40.82 16.96l.54.58c10.15 10.44 16.43 24.66 16.43 40.24v302.01c0 15.9-6.5 30.36-16.96 40.82-10.47-10.47-24.93 16.97-40.83 16.97H170.19c-15.9 0-30.35-6.5-40.82-16.97-10.47-10.46-16.97-24.92-16.97-40.82v-69.78z"></path></svg> Copy`;
    const alt=document.createElement("span"); alt.textContent="Copied";
    btn.appendChild(main); btn.appendChild(alt); btn.dataset.text=item.text;
    right.appendChild(meta); right.appendChild(btn);
    entry.appendChild(textSpan); entry.appendChild(right);
    historyEl.appendChild(entry);
  });
}

function buildTweetBlock(result){
  const url=result.url||""; const comments=Array.isArray(result.comments)?result.comments:[];
  const tweet=document.createElement("div"); tweet.className="tweet"; tweet.dataset.url=url;

  const header=document.createElement("div"); header.className="tweet-header";
  const link=document.createElement("a"); link.className="tweet-link"; link.href=url||"#"; link.target="_blank"; link.rel="noopener noreferrer"; link.textContent=url||"(no url)"; link.title="Open tweet (tap to open)";
  const actions=document.createElement("div"); actions.className="tweet-actions";
  const rerollBtn=document.createElement("button"); rerollBtn.className="reroll-btn"; rerollBtn.textContent="Reroll";
  actions.appendChild(rerollBtn); header.appendChild(link); header.appendChild(actions); tweet.appendChild(header);

  const commentsWrap=document.createElement("div"); commentsWrap.className="comments";
  const hasNative=comments.some((c)=>c&&c.lang&&c.lang!=="en"); const multilingual=hasNative;

  comments.forEach((comment,idx)=>{
    if (!comment||!comment.text) return;
    const line=document.createElement("div"); line.className="comment-line"; if (comment.lang) line.dataset.lang=comment.lang;
    const tag=document.createElement("span"); tag.className="comment-tag";
    tag.textContent = multilingual ? (comment.lang==="en"?"EN":(comment.lang||"native").toUpperCase()) : `EN ${idx+1}`;
    const bubble=document.createElement("span"); bubble.className="comment-text"; bubble.textContent=comment.text;

    const copyBtn=document.createElement("button"); let copyLabel="Copy";
    if (multilingual){ if (comment.lang==="en"){ copyBtn.className="copy-btn-en"; copyLabel="Copy EN"; } else { copyBtn.className="copy-btn"; } } else { copyBtn.className="copy-btn"; }

    const copyMain=document.createElement("span"); copyMain.innerHTML=`<svg width="12" height="12" fill="#0E418F" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 467 512.22"><path fill-rule="nonzero" d="M131.07 372.11c.37 1 .57 2.08.57 3.2 0 1.13-.2 2.21-.57 3.21v75.91c0 10.74 4.41 20.53 11.5 27.62s16.87 11.49 27.62 11.49h239.02c10.75 0 20.53-4.4 27.62-11.49s11.49-16.88 11.49-27.62V152.42c0-10.55-4.21-20.15-11.02-27.18l-.47-.43c-7.09-7.09-16.87-11.5-27.62-11.5H170.19c-10.75 0-20.53 4.41-27.62 11.5s-11.5 16.87-11.5 27.61v219.69zm-18.67 12.54H57.23c-15.82 0-30.1-6.58-40.45-17.11C6.41 356.97 0 342.4 0 326.52V57.79c0-15.86 6.5-30.3 16.97-40.78l.04-.04C27.51 6.49 41.94 0 57.79 0h243.63c15.87 0 30.3 6.51 40.77 16.98l.03.03c10.48 10.48 16.99 24.93 16.99 40.78v36.85h50c15.9 0 30.36 6.5 40.82 16.96l.54.58c10.15 10.44 16.43 24.66 16.43 40.24v302.01c0 15.9-6.5 30.36-16.96 40.82-10.47-10.47-24.93 16.97-40.83 16.97H170.19c-15.9 0-30.35-6.5-40.82-16.97-10.47-10.46-16.97-24.92-16.97-40.82v-69.78z"></path></svg> ${copyLabel}`;
    const copyAlt=document.createElement("span"); copyAlt.textContent="Copied";
    copyBtn.appendChild(copyMain); copyBtn.appendChild(copyAlt); copyBtn.dataset.text=comment.text;

    line.appendChild(tag); line.appendChild(bubble); line.appendChild(copyBtn);
    commentsWrap.appendChild(line);
  });

  tweet.appendChild(commentsWrap);
  return tweet;
}

function appendResultBlock(result){ resultsEl.appendChild(buildTweetBlock(result)); }
function updateTweetBlock(tweetEl,result){
  if (!tweetEl) return;
  tweetEl.dataset.url = result.url || tweetEl.dataset.url || "";
  const old = tweetEl.querySelector(".comments"); if (old) old.remove();
  const neu = buildTweetBlock(result).querySelector(".comments"); if (neu) tweetEl.appendChild(neu);
  tweetEl.style.transition="box-shadow .25s ease, transform .25s ease";
  const s=tweetEl.style.boxShadow, t=tweetEl.style.transform;
  tweetEl.style.boxShadow="0 0 0 2px rgba(56,189,248,.9)"; tweetEl.style.transform="translateY(-1px)";
  setTimeout(()=>{ tweetEl.style.boxShadow=s; tweetEl.style.transform=t; },420);
}
function appendFailedItem(f){
  const el=document.createElement("div"); el.className="failed-item";
  const u=document.createElement("div"); u.className="failed-url"; u.textContent=f.url||"(unknown URL)";
  const r=document.createElement("div"); r.className="failed-reason"; r.textContent=f.error||f.reason||"Unknown error";
  el.appendChild(u); el.appendChild(r); failedEl.appendChild(el);
}
function showSkeletons(count){
  resultsEl.innerHTML=""; const num=Math.min(Math.max(count,1),6);
  for (let i=0;i<num;i++){ const sk=document.createElement("div"); sk.className="tweet-skeleton";
    for (let j=0;j<3;j++){ const line=document.createElement("div"); line.className="tweet-skeleton-line"; sk.appendChild(line); }
    resultsEl.appendChild(sk);
  }
}

async function handleGenerate(){
  const raw=urlInput.value; const urls=parseURLs(raw);
  if (!urls.length){ alert("Please paste at least one tweet URL."); return; }

  const key = getKey(); // gate stored key
  const headers = { "Content-Type":"application/json", "X-CT-Key": key || "" };

  maybeWarmBackend();
  cancelled=false; document.body.classList.add("is-generating");
  generateBtn.disabled=true; cancelBtn.disabled=false; resetResults(); resetProgress();
  setProgressText(`Processing ${urls.length} URL${urls.length===1?"":"s"}…`); setProgressRatio(0.03);
  showSkeletons(urls.length);

  try{
    let res;
    try { res = await fetchWithTimeout(commentURL, { method:"POST", headers, body: JSON.stringify({ urls }) }, 45000); }
    catch(e){ setProgressText("Waking CrownTALK engine… retrying once."); warmBackendOnce();
              res = await fetchWithTimeout(commentURL, { method:"POST", headers, body: JSON.stringify({ urls }) }, 45000); }

    if (res.status === 403){ window.CTGate.show403(); document.body.classList.remove("is-generating"); generateBtn.disabled=false; cancelBtn.disabled=true; setProgressText("Access denied. Enter valid key."); setProgressRatio(0); return; }
    if (!res.ok) throw new Error(`Backend error: ${res.status}`);

    const data = await res.json();
    if (cancelled) return;

    const results = Array.isArray(data.results)?data.results:[];
    const failed  = Array.isArray(data.failed)?data.failed:[];

    resultsEl.innerHTML=""; failedEl.innerHTML="";
    let processed=0, total = results.length || urls.length;

    let delay=50;
    results.forEach((item)=>{
      setTimeout(()=>{
        if (cancelled) return;
        appendResultBlock(item); processed += 1;
        const ratio = total ? processed/total : 1;
        setProgressRatio(ratio);
        setProgressText(`Processed ${processed}/${total}…`);
        resultCountEl.textContent = formatTweetCount(processed);

        if (processed === total){
          setProgressText(`Processed ${processed}.`);
          document.body.classList.remove("is-generating");
          generateBtn.disabled=false; cancelBtn.disabled=true;
        }
      }, delay);
      delay += 120;
    });

    failed.forEach((f)=>appendFailedItem(f));
    failedCountEl.textContent = String(failed.length);

    if (!results.length){
      document.body.classList.remove("is-generating");
      generateBtn.disabled=false; cancelBtn.disabled=true;
      setProgressText(failed.length ? "All URLs failed to process." : "No comments returned.");
      setProgressRatio(1);
    }
  }catch(err){
    console.error("Generate error", err);
    document.body.classList.remove("is-generating");
    generateBtn.disabled=false; cancelBtn.disabled=true;
    setProgressText("Error contacting CrownTALK backend. Please try again.");
    setProgressRatio(0);
  }
}

function handleCancel(){ cancelled=true; document.body.classList.remove("is-generating"); generateBtn.disabled=false; cancelBtn.disabled=true; setProgressText("Cancelled."); setProgressRatio(0); }
function handleClear(){ urlInput.value=""; resetResults(); resetProgress(); autoResizeTextarea(); }

async function handleReroll(tweetEl){
  const url=tweetEl?.dataset.url; if (!url) return;

  const button=tweetEl.querySelector(".reroll-btn"); if (!button) return;
  const oldLabel=button.textContent; button.disabled=true; button.textContent="Rerolling…";

  const commentsWrap=tweetEl.querySelector(".comments");
  if (commentsWrap){
    commentsWrap.innerHTML=""; const sk1=document.createElement("div"); sk1.className="tweet-skeleton-line";
    const sk2=document.createElement("div"); sk2.className="tweet-skeleton-line"; commentsWrap.appendChild(sk1); commentsWrap.appendChild(sk2);
  }

  try{
    const headers = { "Content-Type":"application/json", "X-CT-Key": getKey() || "" };
    const res = await fetch(rerollURL, { method:"POST", headers, body: JSON.stringify({ url }) });
    if (res.status === 403){ window.CTGate.show403(); return; }
    if (!res.ok) throw new Error(`Reroll failed: ${res.status}`);

    const data = await res.json();
    if (data && Array.isArray(data.comments)){
      updateTweetBlock(tweetEl, { url: data.url || url, comments: data.comments });
    } else {
      setProgressText("Reroll failed for this tweet.");
    }
  }catch(err){
    console.error("Reroll error", err);
    setProgressText("Network error during reroll.");
  }finally{
    button.disabled=false; button.textContent=oldLabel;
  }
}

/* ---------- Theme ---------- */
const THEME_STORAGE_KEY="crowntalk_theme";
const ALLOWED_THEMES=["white","dark-purple","gold","blue","black","emerald","crimson"];
function sanitizeThemeDots(){
  themeDots.forEach((dot)=>{
    if (!dot?.dataset) return;
    const t=(dot.dataset.theme||"").trim();
    if (t==="texture"){ dot.parentElement && dot.parentElement.removeChild(dot); }
    else if (!ALLOWED_THEMES.includes(t)){ dot.dataset.theme="crimson"; }
  });
  themeDots = Array.from(document.querySelectorAll(".theme-dot"));
}
function applyTheme(themeName){
  const html=document.documentElement;
  const t=ALLOWED_THEMES.includes(themeName)?themeName:"dark-purple";
  html.setAttribute("data-theme", t);
  themeDots.forEach((dot)=>dot.classList.toggle("is-active", dot.dataset.theme===t));
  try{ localStorage.setItem(THEME_STORAGE_KEY, t); }catch{}
}
function initTheme(){
  sanitizeThemeDots();
  let theme="dark-purple";
  try{
    const stored=localStorage.getItem(THEME_STORAGE_KEY);
    if (stored){ theme = stored==="neon" ? "crimson" : stored; if (stored==="neon") localStorage.setItem(THEME_STORAGE_KEY, "crimson"); }
    if (!ALLOWED_THEMES.includes(theme)) theme="dark-purple";
  }catch{}
  applyTheme(theme);
}

/* ---------- Boot sequence ---------- */
function bootAppUI(){
  if (yearEl) yearEl.textContent=String(new Date().getFullYear());
  initTheme(); renderHistory(); autoResizeTextarea();

  setTimeout(()=>{ maybeWarmBackend(); }, 4000);

  urlInput?.addEventListener("input", autoResizeTextarea);
  generateBtn?.addEventListener("click", ()=>{ if (!document.body.classList.contains("is-generating")) handleGenerate(); });
  cancelBtn?.addEventListener("click", handleCancel);
  clearBtn?.addEventListener("click", handleClear);

  clearHistoryBtn?.addEventListener("click", ()=>{ historyItems=[]; renderHistory(); });

  resultsEl?.addEventListener("click", async (event)=>{
    const copyBtn  = event.target.closest(".copy-btn, .copy-btn-en");
    const rerollBtn = event.target.closest(".reroll-btn");

    if (copyBtn){
      const text = copyBtn.dataset.text || ""; if (!text) return;
      await copyToClipboard(text); addToHistory(text);

      const line = copyBtn.closest(".comment-line"); if (line) line.classList.add("copied");
      copyBtn.classList.add("is-copied");
      const old=copyBtn.textContent; copyBtn.textContent="Copied"; copyBtn.disabled=true;
      setTimeout(()=>{ copyBtn.textContent=old; copyBtn.disabled=false; }, 700);
    }

    if (rerollBtn){
      const tweetEl = rerollBtn.closest(".tweet");
      if (tweetEl) handleReroll(tweetEl);
    }
  });

  historyEl?.addEventListener("click", async (event)=>{
    const btn=event.target.closest(".history-copy-btn"); if (!btn) return;
    const text=btn.dataset.text || ""; if (!text) return;
    await copyToClipboard(text); const old=btn.textContent; btn.textContent="Copied"; btn.disabled=true;
    setTimeout(()=>{ btn.textContent=old; btn.disabled=false; }, 700);
  });

  themeDots.forEach((dot)=>{ dot.addEventListener("click", ()=>{ const t=dot.dataset.theme; if (t) applyTheme(t); }); });
}

/* Show gate first; unlock if already authorized */
function bootWithGate(){
  if (isAuthorized()){ hideGate(); bootAppUI(); }
  else { showGate('➤ ENTER ACCESS KEY'); bindGate(); }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', bootWithGate);
} else {
  bootWithGate();
}

/* Keep backend warm */
(function keepAliveWhileVisible(){
  const PING_MS = 4 * 60 * 1000;
  let timer=null;
  function schedule(){
    clearInterval(timer);
    if (document.visibilityState === "visible"){
      timer = setInterval(()=>{
        fetch("https://crowntalk.onrender.com/ping", { method:"GET", cache:"no-store", mode:"no-cors", keepalive:true }).catch(()=>{});
      }, PING_MS);
    }
  }
  document.addEventListener("visibilitychange", schedule);
  window.addEventListener("pagehide", ()=>clearInterval(timer));
  schedule();
})();
