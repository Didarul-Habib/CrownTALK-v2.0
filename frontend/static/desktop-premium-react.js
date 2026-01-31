/* Desktop Premium React Enhancements (no build step; React UMD loaded dynamically) */
(function () {
  const DESKTOP_MIN = 1024;

  function isDesktop() {
    try { return window.innerWidth >= DESKTOP_MIN; } catch { return false; }
  }
  function prefersReducedMotion() {
    try { return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches; } catch { return false; }
  }
  function safeModeOn() {
    try { return !!window.safeModeOn; } catch { return false; }
  }

  if (!window.CT_PREMIUM_ENABLED) return;
  if (!isDesktop()) return;

  window.CT_USE_REACT_TIMELINE = true;

  const React = window.React;
  const ReactDOM = window.ReactDOM;
  if (!React || !ReactDOM) return;

  // ---------- Small helpers ----------
  function q(id) { return document.getElementById(id); }
  function click(id) { const el = q(id); if (el) el.click(); }
  function toast(msg) {
    try {
      if (typeof window.ctToast === "function") window.ctToast(msg);
      else console.log("[CrownTALK]", msg);
    } catch {}
  }

  // Extract URLs robustly, including concatenated URLs.
  const URL_RE = /(https?:\/\/(?:www\.)?(?:x\.com|twitter\.com)\/[A-Za-z0-9_]+\/status\/\d+[^ \n\r\t]*)|(https?:\/\/(?:www\.)?x\.com\/i\/status\/\d+[^ \n\r\t]*)/gi;

  function extractUrls(raw) {
    const s = String(raw || "");
    const found = [];
    let m;
    while ((m = URL_RE.exec(s)) !== null) {
      const u = (m[1] || m[2] || "").trim();
      if (u) found.push(u);
    }
    // Deduplicate while keeping order
    const seen = new Set();
    const out = [];
    for (const u of found) {
      const key = u.replace(/[#?].*$/, "");
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(u);
    }
    return out;
  }

  // ---------- Smart Paste (desktop) ----------
  function SmartPaste() {
    const [chips, setChips] = React.useState([]);
    const [draft, setDraft] = React.useState("");

    const textarea = React.useMemo(() => q("urlInput"), []);
    const wrapMount = React.useMemo(() => textarea ? textarea.parentElement : null, [textarea]);

    React.useEffect(() => {
      if (!textarea || !wrapMount) return;

      // Hide original textarea (still used as the source of truth for core script)
      textarea.style.position = "absolute";
      textarea.style.left = "-99999px";
      textarea.style.opacity = "0";

      // Initialize from existing value
      const init = extractUrls(textarea.value || "");
      setChips(init.map(u => ({ url: u, ok: true })));
      textarea.value = init.join("\n");
    }, [textarea, wrapMount]);

    function syncToTextarea(next) {
      try {
        if (!textarea) return;
        textarea.value = next.map(c => c.url).join("\n");
        textarea.dispatchEvent(new Event("input", { bubbles: true }));
      } catch {}
    }

    function addFromText(rawText) {
      const urls = extractUrls(rawText);
      if (!urls.length) return;
      setChips(prev => {
        const existing = new Set(prev.map(p => p.url.replace(/[#?].*$/, "")));
        const merged = [...prev];
        for (const u of urls) {
          const key = u.replace(/[#?].*$/, "");
          if (existing.has(key)) continue;
          existing.add(key);
          merged.push({ url: u, ok: true });
        }
        syncToTextarea(merged);
        return merged;
      });
    }

    function onPaste(e) {
      const text = (e.clipboardData && e.clipboardData.getData("text")) || "";
      if (text && extractUrls(text).length) {
        e.preventDefault();
        addFromText(text);
        setDraft("");
      }
    }

    function removeChip(url) {
      setChips(prev => {
        const next = prev.filter(c => c.url !== url);
        syncToTextarea(next);
        return next;
      });
    }

    function onKeyDown(e) {
      if (e.key === "Enter") {
        e.preventDefault();
        addFromText(draft);
        setDraft("");
      }
      if (e.key === "Backspace" && !draft) {
        // delete last chip
        const last = chips[chips.length - 1];
        if (last) removeChip(last.url);
      }
    }

    return React.createElement("div", { className: "ct-smartpaste-wrap", onPaste },
      chips.map((c) =>
        React.createElement("div", { key: c.url, className: "ct-chip-url" + (c.ok ? "" : " bad") },
          React.createElement("span", { className: "txt", title: c.url }, c.url),
          React.createElement("span", { className: "x", role: "button", tabIndex: 0, onClick: () => removeChip(c.url) }, "×")
        )
      ),
      React.createElement("input", {
        className: "ct-smartpaste-input",
        placeholder: chips.length ? "Paste more links, or type and press Enter…" : "Paste X links here… (Ctrl+V)",
        value: draft,
        onChange: (e) => setDraft(e.target.value),
        onKeyDown,
      })
    );
  }

  // Mount Smart Paste into the input section, only on desktop and only if not reduced-motion safe mode
  function mountSmartPaste() {
    try {
      const textarea = q("urlInput");
      if (!textarea) return;
      const inputSection = textarea.parentElement;
      if (!inputSection) return;

      // Insert mount container just before gradient overlay
      const existing = document.getElementById("ctSmartPasteMount");
      if (existing) return;

      const mount = document.createElement("div");
      mount.id = "ctSmartPasteMount";
      mount.style.marginTop = "0px";
      inputSection.insertBefore(mount, inputSection.querySelector(".gradient-overlay"));

      ReactDOM.createRoot(mount).render(React.createElement(SmartPaste));
    } catch (e) {
      console.warn("SmartPaste mount failed", e);
    }
  }

  // ---------- Queue Timeline ----------
  function TimelineView() {
    const [lanes, setLanes] = React.useState({ incoming: [], processing: [], ready: [] });

    React.useEffect(() => {
      if (!window.CROWN_PREMIUM || !window.CROWN_PREMIUM.hooks) return;
      // Subscribe to queue render snapshots
      window.CROWN_PREMIUM.hooks.onQueueRender.push(({ urlQueueState }) => {
        const items = Array.isArray(urlQueueState) ? urlQueueState : [];
        const incoming = items.filter(x => x.state === "incoming");
        const processing = items.filter(x => x.state === "processing");
        const ready = items.filter(x => x.state === "ready");
        setLanes({ incoming, processing, ready });
      });
    }, []);

    function Lane(title, items) {
      return React.createElement("div", { className: "ct-lane" },
        React.createElement("div", { className: "ct-lane-head" },
          React.createElement("div", { className: "ct-lane-title" }, title.toUpperCase()),
          React.createElement("div", { className: "ct-lane-count" }, String(items.length))
        ),
        React.createElement("div", { className: "ct-lane-body" },
          items.slice(0, 30).map((it) =>
            React.createElement("div", { key: it.url, className: "ct-mini-card" },
              React.createElement("div", { className: "ct-mini-url", title: it.url }, it.url),
              React.createElement("div", { className: "ct-mini-stage" }, it.stage || it.state || "—")
            )
          )
        )
      );
    }

    return React.createElement("div", { className: "ct-timeline" },
      Lane("Incoming", lanes.incoming),
      Lane("Processing", lanes.processing),
      Lane("Ready", lanes.ready),
    );
  }

  function mountTimeline() {
    const host = q("pipelineView");
    if (!host) return;
    host.innerHTML = "";
    const mount = document.createElement("div");
    host.appendChild(mount);
    ReactDOM.createRoot(mount).render(React.createElement(TimelineView));
  }

  // ---------- Premium progress smoothing ----------
  function enableProgressSheen() {
    try {
      const fill = q("progressBarFill");
      if (!fill) return;
      fill.classList.add("ct-premium-sheen");
      fill.style.position = "relative";
      fill.style.overflow = "hidden";
    } catch {}
  }

  // Listen to backend progress events (via premium ws) if available, else core script updates still work.
  function hookWsProgress() {
    // premium.v1.js handles ws. We'll also hook into ctPremiumEmit if present.
    if (!window.CROWN_PREMIUM || !window.CROWN_PREMIUM.hooks) return;
    window.CROWN_PREMIUM.hooks.onAnalytics.push((payload) => {
      // run_finish already reflected elsewhere; nothing required
    });
  }

  // ---------- Export / Share ----------
  function buildExportText() {
    const results = window.__CT_RESULTS || []; // set by core script if available
    if (!Array.isArray(results) || !results.length) return "";
    const lines = [];
    for (const r of results) {
      lines.push(r.url || "");
      const comments = r.comments || [];
      for (const c of comments) lines.push("• " + (c.text || c.comment || ""));
      lines.push("");
    }
    return lines.join("\n").trim();
  }

  function downloadFile(filename, content, mime) {
    const blob = new Blob([content], { type: mime || "text/plain;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      URL.revokeObjectURL(a.href);
      a.remove();
    }, 300);
  }

  function wireExportButtons() {
    const pop = document.querySelector(".results-menu-pop");
    if (!pop) return;
    if (document.getElementById("exportTxtBtn")) return;

    const group = document.createElement("div");
    group.className = "results-menu-group";

    group.innerHTML = `
      <div class="results-menu-title">Export</div>
      <div class="results-menu-actions">
        <button type="button" class="btn-xs" id="exportTxtBtn">Download .txt</button>
        <button type="button" class="btn-xs" id="exportJsonBtn">Download .json</button>
      </div>
    `;
    pop.appendChild(group);

    document.getElementById("exportTxtBtn").addEventListener("click", () => {
      const txt = buildExportText();
      if (!txt) return toast("Nothing to export yet.");
      downloadFile("crowntalk-results.txt", txt, "text/plain");
      toast("Downloaded .txt");
    });

    document.getElementById("exportJsonBtn").addEventListener("click", () => {
      const data = window.__CT_RESULTS || [];
      if (!Array.isArray(data) || !data.length) return toast("Nothing to export yet.");
      downloadFile("crowntalk-results.json", JSON.stringify(data, null, 2), "application/json");
      toast("Downloaded .json");
    });
  }

  // ---------- Command Palette (Ctrl+K) ----------
  function CommandPalette() {
    const [open, setOpen] = React.useState(false);
    const [qv, setQv] = React.useState("");

    const actions = React.useMemo(() => ([
      { id: "gen", title: "Generate", desc: "Start generating comments", keys: "Enter", run: () => click("generateBtn") },
      { id: "cancel", title: "Cancel", desc: "Stop current run", keys: "Esc", run: () => click("cancelBtn") },
      { id: "clear", title: "Clear input", desc: "Clear URLs box", keys: "", run: () => click("clearBtn") },
      { id: "copyall", title: "Copy all results", desc: "Copy all comments to clipboard", keys: "", run: () => click("exportAllBtn") },
      { id: "txt", title: "Download .txt", desc: "Export results as text", keys: "", run: () => document.getElementById("exportTxtBtn")?.click() },
      { id: "json", title: "Download .json", desc: "Export results as JSON", keys: "", run: () => document.getElementById("exportJsonBtn")?.click() },
    ]), []);

    const filtered = actions.filter(a =>
      (a.title + " " + a.desc).toLowerCase().includes(qv.toLowerCase().trim())
    );

    React.useEffect(() => {
      function onKey(e) {
        // Toggle
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
          e.preventDefault();
          setOpen(v => !v);
          setQv("");
        }
        if (e.key === "Escape") setOpen(false);
      }
      window.addEventListener("keydown", onKey);
      return () => window.removeEventListener("keydown", onKey);
    }, []);

    React.useEffect(() => {
      // reflect in existing overlay container state
      const overlay = document.getElementById("ctCommandOverlay");
      if (overlay) overlay.setAttribute("aria-hidden", open ? "false" : "true");
    }, [open]);

    if (!open) return null;

    return React.createElement("div", { className: "ct-cmdk-backdrop", onMouseDown: () => setOpen(false) },
      React.createElement("div", { className: "ct-cmdk", onMouseDown: (e) => e.stopPropagation() },
        React.createElement("div", { className: "ct-cmdk-header" },
          React.createElement("input", {
            className: "ct-cmdk-input",
            autoFocus: true,
            value: qv,
            placeholder: "Type a command…",
            onChange: (e) => setQv(e.target.value),
            onKeyDown: (e) => {
              if (e.key === "Enter" && filtered[0]) {
                filtered[0].run();
                setOpen(false);
              }
            }
          }),
          React.createElement("div", { className: "ct-cmdk-kbd" }, "Ctrl/⌘ K")
        ),
        React.createElement("div", { className: "ct-cmdk-list" },
          filtered.map(a =>
            React.createElement("div", { key: a.id, className: "ct-cmdk-item", onClick: () => { a.run(); setOpen(false);} },
              React.createElement("div", null,
                React.createElement("div", { className: "title" }, a.title),
                React.createElement("div", { className: "desc" }, a.desc)
              ),
              React.createElement("div", { className: "ct-cmdk-kbd" }, a.keys || "")
            )
          )
        )
      )
    );
  }

  function mountCommandPalette() {
    const host = q("ctCommandOverlay");
    if (!host) return;
    host.innerHTML = "";
    ReactDOM.createRoot(host).render(React.createElement(CommandPalette));
  }

  // ---------- Boot ----------
  function boot() {
    if (prefersReducedMotion() || safeModeOn()) {
      // Still mount command palette & export (lightweight), skip heavier visuals.
      wireExportButtons();
      mountCommandPalette();
      return;
    }
    mountSmartPaste();
    mountTimeline();
    enableProgressSheen();
    wireExportButtons();
    mountCommandPalette();

    // Optional: GSAP micro intro (if available)
    try {
      if (window.gsap) {
        window.gsap.from(".card", { duration: 0.55, y: 10, opacity: 0, stagger: 0.02, ease: "power2.out" });
      }
    } catch {}
  }

  // delay until core UI exists
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
