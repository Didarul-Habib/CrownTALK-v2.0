/* CrownTALK Upgrades v1 (no-build drop-in)
   - Desktop-only Command Palette (Ctrl+K / ⌘K)
   - Cache per URL set (localStorage)
   - Offline + Cached bottom-right pills
   - Mobile paste bar + URL chips
   - Custom Theme Studio (data-theme="custom")
*/
(function(){
  "use strict";

  var LS_PREFIX = "ct_upgrades_v1:";
  var CACHE_KEY = LS_PREFIX + "cache_v1";
  var LAST_URLS_KEY = LS_PREFIX + "last_urls";
  var CUSTOM_THEME_KEY = LS_PREFIX + "custom_theme";

  function nowISO(){ try { return new Date().toISOString(); } catch(e){ return ""; } }

  function safeJsonParse(str){
    try { return JSON.parse(str); } catch(e){ return null; }
  }

  function stableStringify(obj){
    // stable-ish stringify for cache key
    try{
      return JSON.stringify(obj, Object.keys(obj).sort());
    }catch(e){
      return JSON.stringify(obj);
    }
  }

  function hashStr(s){
    // tiny non-crypto hash
    var h=2166136261;
    for (var i=0;i<s.length;i++){
      h ^= s.charCodeAt(i);
      h = (h * 16777619) >>> 0;
    }
    return ("00000000"+h.toString(16)).slice(-8);
  }

  function normalizeUrls(urls){
    if(!urls || !urls.length) return [];
    var out = [];
    for(var i=0;i<urls.length;i++){
      var u = (urls[i]||"").trim();
      if(!u) continue;
      out.push(u);
    }
    out.sort();
    return out;
  }

  function getCache(){
    var raw = localStorage.getItem(CACHE_KEY);
    var data = safeJsonParse(raw);
    if(!data || typeof data !== "object") return { items: [], v: 1 };
    if(!Array.isArray(data.items)) data.items = [];
    return data;
  }

  function setCache(data){
    try{ localStorage.setItem(CACHE_KEY, JSON.stringify(data)); }catch(e){}
  }

  function cacheGet(key){
    var data = getCache();
    for (var i=0;i<data.items.length;i++){
      if(data.items[i] && data.items[i].key === key){
        return data.items[i].value;
      }
    }
    return null;
  }

  function cacheSet(key, value){
    var data = getCache();
    // remove existing
    var next = [];
    for (var i=0;i<data.items.length;i++){
      if(data.items[i] && data.items[i].key !== key) next.push(data.items[i]);
    }
    next.unshift({ key: key, value: value, t: Date.now() });
    // LRU cap
    next = next.slice(0, 10);
    data.items = next;
    data.v = 1;
    setCache(data);
  }

  function makeCacheKeyFromPayload(payload){
    if(!payload || typeof payload !== "object") return null;
    var norm = {
      urls: normalizeUrls(payload.urls || []),
      preset: payload.preset || "",
      safe_mode: !!payload.safe_mode,
      lang_en: !!payload.lang_en,
      lang_native: !!payload.lang_native,
      native_lang: payload.native_lang || ""
    };
    var s = stableStringify(norm);
    return "k:" + hashStr(s);
  }

  // ====== Status Pills (Offline + Cached) ======
  var pillHost, pillOffline, pillCached, cachedTimer;

  function ensurePills(){
    if(pillHost) return;
    pillHost = document.createElement("div");
    pillHost.className = "ct-status-fab";
    pillHost.setAttribute("aria-hidden", "true");

    pillOffline = document.createElement("div");
    pillOffline.className = "ct-pill offline";
    pillOffline.innerHTML = '<span class="dot"></span><span>Offline</span>';

    pillCached = document.createElement("div");
    pillCached.className = "ct-pill cached";
    pillCached.innerHTML = '<span class="dot"></span><span>Cached</span>';

    pillHost.appendChild(pillOffline);
    pillHost.appendChild(pillCached);
    document.body.appendChild(pillHost);
  }

  function setOffline(on){
    ensurePills();
    if(on){ pillOffline.classList.add("is-on"); }
    else{ pillOffline.classList.remove("is-on"); }
  }

  function flashCached(){
    ensurePills();
    pillCached.classList.add("is-on");
    if(cachedTimer) clearTimeout(cachedTimer);
    cachedTimer = setTimeout(function(){
      pillCached.classList.remove("is-on");
    }, 1600);
  }

  function syncOnline(){
    setOffline(!navigator.onLine);
  }

  // ====== Mobile Paste Bar + Chips ======
  function extractUrls(text){
    if(!text) return [];
    var lines = text.split(/\n+/);
    var out = [];
    for(var i=0;i<lines.length;i++){
      var s = (lines[i]||"").trim();
      if(!s) continue;
      // keep only plausible URL lines
      if(/^https?:\/\//i.test(s)) out.push(s);
      else if(/x\.com\/.+\/status\/\d+/i.test(s)) out.push("https://" + s.replace(/^\/+/, ""));
      else if(/twitter\.com\/.+\/status\/\d+/i.test(s)) out.push("https://" + s.replace(/^\/+/, ""));
    }
    // de-dupe
    var map = {};
    var uniq = [];
    for(var j=0;j<out.length;j++){
      var u = out[j];
      if(map[u]) continue;
      map[u]=1;
      uniq.push(u);
    }
    return uniq;
  }

  function renderChips(urls){
    var chipsEl = document.getElementById("ctUrlChips");
    if(!chipsEl) return;
    chipsEl.innerHTML = "";
    if(!urls || !urls.length) return;
    for(var i=0;i<urls.length;i++){
      (function(u){
        var chip = document.createElement("span");
        chip.className = "ct-chip";
        chip.innerHTML = '<span class="ct-chip-text"></span><button type="button" class="ct-chip-x" aria-label="Remove">×</button>';
        chip.querySelector(".ct-chip-text").textContent = u;
        chip.querySelector(".ct-chip-x").addEventListener("click", function(){
          var input = document.getElementById("urlInput");
          if(!input) return;
          var cur = extractUrls(input.value);
          var next = [];
          for(var k=0;k<cur.length;k++){ if(cur[k] !== u) next.push(cur[k]); }
          input.value = next.join("\n");
          try{ input.dispatchEvent(new Event("input", { bubbles:true })); } catch(e){}
          renderChips(next);
          try{ localStorage.setItem(LAST_URLS_KEY, input.value); }catch(e){}
        });
        chipsEl.appendChild(chip);
      })(urls[i]);
    }
  }

  function bindMobileTools(){
    var input = document.getElementById("urlInput");
    if(!input) return;

    var pasteBtn = document.getElementById("ctPasteBarPaste");
    var lastBtn  = document.getElementById("ctPasteBarLast");
    var clearBtn = document.getElementById("ctPasteBarClear");

    if(pasteBtn){
      pasteBtn.addEventListener("click", async function(){
        var text = "";
        // Clipboard API best effort
        try{
          if(navigator.clipboard && navigator.clipboard.readText){
            text = await navigator.clipboard.readText();
          }
        }catch(e){}
        if(!text){
          text = prompt("Paste your X/Twitter link(s) here (one per line):", "") || "";
        }
        if(text){
          input.value = (input.value ? (input.value.trim() + "\n") : "") + text.trim();
          try{ input.dispatchEvent(new Event("input", { bubbles:true })); } catch(e){}
        }
      });
    }

    if(lastBtn){
      lastBtn.addEventListener("click", function(){
        var last = localStorage.getItem(LAST_URLS_KEY) || localStorage.getItem(LAST_URLS_KEY) || "";
        if(!last){
          last = localStorage.getItem("ct_last_urls") || ""; // compatibility if any older key exists
        }
        if(last){
          input.value = last;
          try{ input.dispatchEvent(new Event("input", { bubbles:true })); } catch(e){}
        }
      });
    }

    if(clearBtn){
      clearBtn.addEventListener("click", function(){
        // reuse existing clear button if present
        var legacy = document.getElementById("clearBtn");
        if(legacy){ legacy.click(); return; }
        input.value = "";
        try{ input.dispatchEvent(new Event("input", { bubbles:true })); } catch(e){}
      });
    }

    // Keep last urls + chips updated
    var t = null;
    input.addEventListener("input", function(){
      if(t) clearTimeout(t);
      t = setTimeout(function(){
        var urls = extractUrls(input.value);
        renderChips(urls);
        try{ localStorage.setItem(LAST_URLS_KEY, input.value); }catch(e){}
      }, 120);
    });

    // initial render
    try{
      var initUrls = extractUrls(input.value);
      renderChips(initUrls);
    }catch(e){}
  }

  // ====== Custom Theme Studio ======
  function loadCustomTheme(){
    var saved = safeJsonParse(localStorage.getItem(CUSTOM_THEME_KEY));
    return saved || { h: 280, bg: 8, r: 18, blur: 14, border: 0.14 };
  }

  function applyCustomTheme(t){
    if(!t) return;
    var root = document.documentElement;
    root.style.setProperty("--ct-custom-accent-h", String(t.h));
    root.style.setProperty("--ct-custom-bg-l", String(t.bg) + "%");
    root.style.setProperty("--ct-custom-radius", String(t.r) + "px");
    root.style.setProperty("--ct-custom-blur", String(t.blur) + "px");
    root.style.setProperty("--ct-custom-border", String(t.border));
  }

  function openThemeStudio(){
    var existing = document.getElementById("ctThemeStudio");
    if(existing){ existing.remove(); }

    var t = loadCustomTheme();
    var modal = document.createElement("div");
    modal.id = "ctThemeStudio";
    modal.style.position = "fixed";
    modal.style.inset = "0";
    modal.style.zIndex = "99998";
    modal.style.display = "grid";
    modal.style.placeItems = "center";
    modal.style.padding = "16px";
    modal.style.background = "rgba(0,0,0,0.55)";
    modal.style.backdropFilter = "blur(6px)";

    var card = document.createElement("div");
    card.style.width = "min(520px, 92vw)";
    card.style.borderRadius = "18px";
    card.style.border = "1px solid rgba(255,255,255,0.14)";
    card.style.background = "rgba(14,16,28,0.92)";
    card.style.boxShadow = "0 28px 100px rgba(0,0,0,0.55)";
    card.style.backdropFilter = "blur(14px)";
    card.style.padding = "16px";

    card.innerHTML = [
      '<div style="display:flex;justify-content:space-between;align-items:center;gap:10px;">',
        '<div style="font-weight:700;color:rgba(240,245,255,0.94);">Custom Theme Studio</div>',
        '<button type="button" id="ctThemeClose" style="border:1px solid rgba(255,255,255,0.14);background:rgba(255,255,255,0.06);color:rgba(240,245,255,0.92);border-radius:999px;padding:8px 10px;cursor:pointer;">Close</button>',
      '</div>',
      '<div style="margin-top:14px;display:grid;gap:12px;">',
        sliderRow("Accent Hue", "ctHue", 0, 360, t.h),
        sliderRow("Background", "ctBg", 3, 18, t.bg),
        sliderRow("Radius", "ctRad", 12, 28, t.r),
        sliderRow("Blur", "ctBlur", 8, 22, t.blur),
        sliderRow("Border", "ctBorder", 0.08, 0.22, t.border, 0.01),
      '</div>',
      '<div style="margin-top:14px;display:flex;gap:10px;flex-wrap:wrap;">',
        '<button type="button" id="ctThemeApply" style="border:1px solid rgba(255,255,255,0.14);background:rgba(255,255,255,0.10);color:rgba(240,245,255,0.95);border-radius:999px;padding:10px 12px;cursor:pointer;">Save</button>',
        '<button type="button" id="ctThemeReset" style="border:1px solid rgba(255,255,255,0.14);background:rgba(255,255,255,0.06);color:rgba(240,245,255,0.92);border-radius:999px;padding:10px 12px;cursor:pointer;">Reset</button>',
        '<div style="opacity:0.72;color:rgba(240,245,255,0.85);font-size:12px;align-self:center;">Tip: switch to <b>Custom</b> theme to see changes</div>',
      '</div>'
    ].join("");

    modal.appendChild(card);
    document.body.appendChild(modal);

    function sliderRow(label, id, min, max, val, step){
      step = (step == null) ? 1 : step;
      return [
        '<label style="display:grid;gap:6px;">',
          '<div style="display:flex;justify-content:space-between;gap:10px;color:rgba(240,245,255,0.88);font-size:13px;">',
            '<span>', label, '</span>',
            '<span id="', id,'Val" style="opacity:0.78;">', String(val), '</span>',
          '</div>',
          '<input id="', id, '" type="range" min="', String(min), '" max="', String(max), '" step="', String(step), '" value="', String(val), '" />',
        '</label>'
      ].join("");
    }

    function onInput(id, cb){
      var el = document.getElementById(id);
      if(!el) return;
      el.addEventListener("input", function(){
        var v = el.value;
        var out = document.getElementById(id+"Val");
        if(out) out.textContent = v;
        cb(v);
        applyCustomTheme(t);
      });
    }

    onInput("ctHue", function(v){ t.h = parseInt(v,10)||280; });
    onInput("ctBg", function(v){ t.bg = parseInt(v,10)||8; });
    onInput("ctRad", function(v){ t.r = parseInt(v,10)||18; });
    onInput("ctBlur", function(v){ t.blur = parseInt(v,10)||14; });
    onInput("ctBorder", function(v){ t.border = parseFloat(v)||0.14; });

    document.getElementById("ctThemeClose").addEventListener("click", function(){ modal.remove(); });
    modal.addEventListener("click", function(e){ if(e.target === modal) modal.remove(); });

    document.getElementById("ctThemeApply").addEventListener("click", function(){
      try{ localStorage.setItem(CUSTOM_THEME_KEY, JSON.stringify(t)); }catch(e){}
      applyCustomTheme(t);
      flashCached(); // small feedback reuse
    });

    document.getElementById("ctThemeReset").addEventListener("click", function(){
      t = { h: 280, bg: 8, r: 18, blur: 14, border: 0.14 };
      try{ localStorage.setItem(CUSTOM_THEME_KEY, JSON.stringify(t)); }catch(e){}
      applyCustomTheme(t);
      // refresh UI values
      modal.remove();
      openThemeStudio();
    });
  }

  // ====== Desktop Command Palette ======
  var cmdOpen = false;

  function ensureCommandPalette(){
    var overlay = document.getElementById("ctCommandOverlay");
    if(!overlay) return null;

    if(overlay.getAttribute("data-ct-built") === "1") return overlay;
    overlay.setAttribute("data-ct-built", "1");

    overlay.innerHTML = [
      '<div class="ct-cmd-backdrop" style="position:fixed;inset:0;background:rgba(0,0,0,0.55);backdrop-filter:blur(10px);display:flex;align-items:flex-start;justify-content:center;padding:6vh 16px;">',
        '<div class="ct-cmd" style="width:min(720px,96vw);border-radius:18px;border:1px solid rgba(255,255,255,0.14);background:rgba(14,16,28,0.92);box-shadow:0 30px 120px rgba(0,0,0,0.65);overflow:hidden;">',
          '<div style="display:flex;gap:10px;align-items:center;padding:12px 12px;border-bottom:1px solid rgba(255,255,255,0.10);">',
            '<span style="opacity:0.85;color:rgba(240,245,255,0.92);font-weight:650;">Command</span>',
            '<input id="ctCmdInput" type="text" placeholder="Type a command..." autocomplete="off" style="flex:1;border:1px solid rgba(255,255,255,0.12);background:rgba(255,255,255,0.06);color:rgba(240,245,255,0.92);border-radius:12px;padding:10px 12px;outline:none;" />',
            '<kbd style="opacity:0.75;color:rgba(240,245,255,0.88);border:1px solid rgba(255,255,255,0.12);border-radius:10px;padding:6px 8px;background:rgba(255,255,255,0.05);">Esc</kbd>',
          '</div>',
          '<div id="ctCmdList" style="max-height:52vh;overflow:auto;padding:8px;"></div>',
        '</div>',
      '</div>'
    ].join("");

    overlay.style.display = "none";
    overlay.setAttribute("aria-hidden", "true");

    var backdrop = overlay.querySelector(".ct-cmd-backdrop");
    backdrop.addEventListener("click", function(e){
      if(e.target === backdrop) closeCmd();
    });

    return overlay;
  }

  function getCommands(){
    return [
      { id:"gen", label:"Generate now", hint:"Runs the generator", run:function(){ clickId("generateBtn"); } },
      { id:"safe", label:"Toggle safe mode", hint:"On/Off", run:function(){ clickId("safeModeToggle"); } },
      { id:"lite", label:"Toggle lite mode", hint:"On/Off", run:function(){ clickId("liteModeToggle"); } },
      { id:"export", label:"Export results", hint:"Copies/exports all", run:function(){ clickId("exportAllBtn"); } },
      { id:"resume", label:"Load last session", hint:"Resume last run", run:function(){ clickId("ctResumeBtn"); } },
      { id:"focus", label:"Focus URL input", hint:"Jump to input", run:function(){ var el=document.getElementById("urlInput"); if(el) el.focus(); } },
      { id:"theme-studio", label:"Open Theme Studio", hint:"Custom sliders", run:function(){ openThemeStudio(); } },
      { id:"theme-aurora", label:"Theme: Aurora", hint:"Switch theme", run:function(){ setTheme("aurora"); } },
      { id:"theme-sakura", label:"Theme: Sakura", hint:"Switch theme", run:function(){ setTheme("sakura"); } },
      { id:"theme-mono", label:"Theme: Mono", hint:"Switch theme", run:function(){ setTheme("mono"); } },
      { id:"theme-custom", label:"Theme: Custom", hint:"Switch theme", run:function(){ setTheme("custom"); } }
    ];
  }

  function clickId(id){
    var el = document.getElementById(id);
    if(el) el.click();
  }

  function setTheme(theme){
    try{
      document.documentElement.setAttribute("data-theme", theme);
      // update theme radios if present
      var inputs = document.querySelectorAll(".ct-theme-input[data-theme]");
      for(var i=0;i<inputs.length;i++){
        if(inputs[i].getAttribute("data-theme") === theme) inputs[i].checked = true;
      }
      localStorage.setItem("crowntalk_theme", theme);
    }catch(e){}
  }

  function openCmd(){
    var overlay = ensureCommandPalette();
    if(!overlay) return;
    overlay.style.display = "block";
    overlay.setAttribute("aria-hidden", "false");
    cmdOpen = true;

    var input = document.getElementById("ctCmdInput");
    var list = document.getElementById("ctCmdList");
    var cmds = getCommands();

    function render(filter){
      filter = (filter||"").toLowerCase().trim();
      list.innerHTML = "";
      var shown = 0;
      for(var i=0;i<cmds.length;i++){
        var c = cmds[i];
        var hay = (c.label + " " + (c.hint||"") + " " + c.id).toLowerCase();
        if(filter && hay.indexOf(filter) === -1) continue;
        shown++;
        var item = document.createElement("button");
        item.type = "button";
        item.style.width = "100%";
        item.style.textAlign = "left";
        item.style.border = "1px solid rgba(255,255,255,0.10)";
        item.style.background = "rgba(255,255,255,0.05)";
        item.style.color = "rgba(240,245,255,0.92)";
        item.style.borderRadius = "14px";
        item.style.padding = "12px 12px";
        item.style.margin = "6px 0";
        item.style.cursor = "pointer";
        item.innerHTML = '<div style="display:flex;justify-content:space-between;gap:12px;align-items:center;"><div style="font-weight:650;">'+escapeHtml(c.label)+'</div><div style="opacity:0.72;font-size:12px;">'+escapeHtml(c.hint||"")+'</div></div>';
        item.addEventListener("click", function(cc){ return function(){ closeCmd(); try{ cc.run(); }catch(e){} }; }(c));
        list.appendChild(item);
      }
      if(!shown){
        var empty = document.createElement("div");
        empty.style.opacity = "0.72";
        empty.style.padding = "12px";
        empty.textContent = "No matches.";
        list.appendChild(empty);
      }
    }

    function onKey(e){
      if(e.key === "Escape"){ e.preventDefault(); closeCmd(); return; }
      if(e.key === "Enter"){
        e.preventDefault();
        // click first visible item
        var btn = list.querySelector("button");
        if(btn) btn.click();
      }
    }

    input.value = "";
    render("");
    input.focus();
    input.oninput = function(){ render(input.value); };
    input.onkeydown = onKey;
  }

  function closeCmd(){
    var overlay = document.getElementById("ctCommandOverlay");
    if(!overlay) return;
    overlay.style.display = "none";
    overlay.setAttribute("aria-hidden", "true");
    cmdOpen = false;
  }

  function escapeHtml(s){
    return String(s).replace(/[&<>"']/g, function(m){
      return ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", "\"":"&quot;", "'":"&#039;" })[m];
    });
  }

  function bindCmdHotkey(){
    document.addEventListener("keydown", function(e){
      // Desktop only: overlay has hide-on-mobile class already
      var isMac = /Mac|iPhone|iPad|iPod/i.test(navigator.platform || "");
      var cmdk = (isMac && e.metaKey && e.key.toLowerCase() === "k") || (!isMac && e.ctrlKey && e.key.toLowerCase() === "k");
      if(cmdk){
        e.preventDefault();
        if(cmdOpen) closeCmd();
        else openCmd();
      }
      if(e.key === "Escape" && cmdOpen){
        e.preventDefault();
        closeCmd();
      }
    });
  }

  // ====== Fetch wrapper cache (no changes needed in script.v2.js) ======
  function wrapFetch(){
    if(window.__ctFetchWrapped) return;
    window.__ctFetchWrapped = true;

    var origFetch = window.fetch;
    if(!origFetch) return;

    window.fetch = function(input, init){
      try{
        var url = "";
        if(typeof input === "string") url = input;
        else if(input && input.url) url = input.url;

        var method = (init && init.method) ? String(init.method).toUpperCase() : "GET";
        var body = init && init.body;

        var isComment = /\/comment(\?|$)/.test(url);
        if(isComment && method === "POST" && typeof body === "string" && body.length){
          var payload = safeJsonParse(body);
          var key = makeCacheKeyFromPayload(payload);
          if(key){
            // Save last urls for mobile "Last"
            try{ localStorage.setItem(LAST_URLS_KEY, (payload.urls||[]).join("\n")); }catch(e){}

            var hit = cacheGet(key);
            if(hit){
              flashCached();
              // return cached response
              return Promise.resolve(new Response(JSON.stringify(hit), {
                status: 200,
                headers: { "Content-Type": "application/json", "X-CT-Cache": "HIT" }
              }));
            }

            // Otherwise fetch and store
            return origFetch(input, init).then(function(resp){
              try{
                if(resp && resp.ok){
                  var clone = resp.clone();
                  return clone.json().then(function(data){
                    cacheSet(key, data);
                    return resp;
                  }).catch(function(){
                    return resp;
                  });
                }
              }catch(e){}
              return resp;
            }).catch(function(err){
              // If network fails but we have a cache entry, fallback (best effort)
              var fallback = cacheGet(key);
              if(fallback){
                flashCached();
                return new Response(JSON.stringify(fallback), {
                  status: 200,
                  headers: { "Content-Type": "application/json", "X-CT-Cache": "HIT-OFFLINE" }
                });
              }
              throw err;
            });
          }
        }
      }catch(e){}
      return origFetch(input, init);
    };
  }

  // Expose minimal API for future tweaks
  window.CT_UPGRADES = {
    cacheGet: cacheGet,
    cacheSet: cacheSet,
    makeCacheKeyFromPayload: makeCacheKeyFromPayload,
    flashCached: flashCached,
    openThemeStudio: openThemeStudio
  };

  // Init on DOM ready
  function init(){
    try{ ensurePills(); }catch(e){}
    try{ syncOnline(); }catch(e){}
    window.addEventListener("online", syncOnline);
    window.addEventListener("offline", syncOnline);

    // Apply custom theme vars early
    try{ applyCustomTheme(loadCustomTheme()); }catch(e){}

    // Mobile tools + chips
    try{ bindMobileTools(); }catch(e){}

    // Desktop hotkey
    try{ bindCmdHotkey(); }catch(e){}
  }

  // Run ASAP for fetch wrapper (before main script)
  try{ wrapFetch(); }catch(e){}

  if(document.readyState === "loading"){
    document.addEventListener("DOMContentLoaded", init);
  }else{
    init();
  }
})();