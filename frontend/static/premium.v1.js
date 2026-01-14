(function () {
  if (!window.CT_PREMIUM_ENABLED) return;
  const url = window.CT_PREMIUM_WS_URL || null;
  if (!url) return;
  let ws;
  try {
    ws = new WebSocket(url);
  } catch {
    return;
  }

  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (!msg || !msg.type) return;

    if (msg.type === "run_finish" && msg.payload) {
      ctPremiumEmit("onAnalytics", msg.payload); // reuse same hooks if you like
    }
    // you can also fan out to radar, pipeline, etc.
  };
})();


(function () {
  if (!window.CT_PREMIUM_ENABLED) return;
  const flags = window.CT_PREMIUM_FLAGS || {};
  if (!flags.pipelineView) return;

  const pipelineEl = document.getElementById("pipelineView");
  if (!pipelineEl) return;

  const isSafe = window.safeModeOn; // read-only usage
  const msPerHeatStep = 1000 * 25; // 25s per step

  let lastQueueSnapshot = [];

  function getBrandProfileForUrl(item) {
    // simple version: use preset as brand; later you can wire real brand profiles
    try {
      const presetSel = document.getElementById("presetSelect");
      return presetSel?.value || "default";
    } catch {
      return "default";
    }
  }

  function computeHeatClass(item) {
    const startedAt = item.startedAt || item.queuedAt;
    if (!startedAt) return "heat-fresh";
    const ageMs = Date.now() - startedAt;
    const steps = ageMs / msPerHeatStep;
    if (steps < 1.5) return "heat-fresh";
    if (steps < 3.5) return "heat-warm";
    return "heat-hot";
  }

  function renderPipeline(queueState) {
    if (!pipelineEl) return;
    lastQueueSnapshot = queueState || [];

    if (!queueState.length) {
      pipelineEl.setAttribute("aria-hidden", "true");
      pipelineEl.innerHTML = "";
      return;
    }
    pipelineEl.setAttribute("aria-hidden", "false");

    // columns by status
    const columns = {
      queued: [],
      processing: [],
      done: [],
      failed: []
    };

    queueState.forEach((item) => {
      const status = item.status || "queued";
      const bucket = (status === "processing")
        ? "processing"
        : (status === "done" ? "done" : (status === "failed" ? "failed" : "queued"));
      const brand = item.brandProfile || getBrandProfileForUrl(item);
      if (!columns[bucket][brand]) columns[bucket][brand] = [];
      columns[bucket][brand].push(item);
    });

    const colDefs = [
      { key: "queued", label: "Incoming" },
      { key: "processing", label: "Processing" },
      { key: "done", label: "Ready" }
    ];

    const frag = document.createDocumentFragment();

    colDefs.forEach((col) => {
      const colEl = document.createElement("div");
      colEl.className = "pipeline-column";

      const head = document.createElement("div");
      head.className = "pipeline-column-head";
      head.innerHTML = `<span>${col.label}</span><span>${(columns[col.key] && Object.values(columns[col.key]).flat().length) || 0}</span>`;
      colEl.appendChild(head);

      const swimlanes = columns[col.key] || {};
      Object.keys(swimlanes).forEach((brand) => {
        const laneEl = document.createElement("div");
        laneEl.className = "pipeline-swimlane";

        const labelEl = document.createElement("div");
        labelEl.className = "pipeline-swimlane-label";
        labelEl.textContent = brand;
        laneEl.appendChild(labelEl);

        swimlanes[brand].forEach((item) => {
          const card = document.createElement("button");
          card.type = "button";
          card.className = "pipeline-item";
          const heatClass = computeHeatClass(item);
          card.classList.add(heatClass);
          card.dataset.url = item.url;

          const short = (item.shortLabel || item.url || "").replace(/^https?:\/\//, "");
          card.textContent = short.slice(0, 40);

          card.addEventListener("click", () => {
            const resultsEl = document.getElementById("results");
            if (!resultsEl) return;
            const sel = `.tweet[data-url="${item.url}"]`;
            const el = resultsEl.querySelector(sel);
            if (el) {
              el.scrollIntoView({ behavior: "smooth", block: "start" });
              el.classList.add("ct-highlight");
              setTimeout(() => el.classList.remove("ct-highlight"), 900);
            }
          });

          laneEl.appendChild(card);
        });

        colEl.appendChild(laneEl);
      });

      frag.appendChild(colEl);
    });

    pipelineEl.innerHTML = "";
    pipelineEl.appendChild(frag);
  }

  // Receive queue updates from core
  window.CROWN_PREMIUM.hooks.onQueueRender.push(({ urlQueueState }) => {
    renderPipeline(urlQueueState || []);
  });

  // Heat halo updater (no heavy animation; just class changes)
  if (!isSafe) {
    setInterval(() => {
      if (!lastQueueSnapshot.length) return;
      const cards = pipelineEl.querySelectorAll(".pipeline-item");
      cards.forEach((card) => {
        const url = card.dataset.url;
        const item = lastQueueSnapshot.find((x) => x.url === url);
        if (!item) return;
        card.classList.remove("heat-fresh", "heat-warm", "heat-hot");
        card.classList.add(computeHeatClass(item));
      });
    }, 8000); // every 8s
  }
})();

(function () {
  if (!window.CT_PREMIUM_ENABLED) return;
  const flags = window.CT_PREMIUM_FLAGS || {};
  if (!flags.runHealthHud) return;

  const latencyEl = document.getElementById("runHealthLatency");
  const sparkSvg  = document.getElementById("runHealthSparkSvg");
  const stateEl   = document.getElementById("runHealthState");
  const stateLabel= document.getElementById("runHealthStateLabel");
  if (!latencyEl || !sparkSvg || !stateEl || !stateLabel) return;

  const values = []; // ms

  function redrawSpark() {
    const w = sparkSvg.viewBox.baseVal.width || 80;
    const h = sparkSvg.viewBox.baseVal.height || 18;
    sparkSvg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    sparkSvg.innerHTML = "";
    if (!values.length) return;

    const max = Math.max(...values);
    const min = Math.min(...values);
    const range = Math.max(1, max - min);
    const stepX = values.length > 1 ? w / (values.length - 1) : w;
    let d = "";
    values.forEach((v, i) => {
      const x = stepX * i;
      const norm = (v - min) / range;
      const y = h - norm * h;
      d += (i === 0 ? "M" : "L") + x + " " + y + " ";
    });
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", d.trim());
    path.setAttribute("fill", "none");
    path.setAttribute("stroke-width", "1.5");
    path.setAttribute("stroke-linejoin", "round");
    path.setAttribute("stroke-linecap", "round");
    sparkSvg.appendChild(path);
  }

  function updateState(durationSec, failed) {
    const ms = durationSec * 1000;
    let state = "chill";
    if (ms > 25000 || failed >= 5) state = "strained";
    else if (ms > 12000 || failed >= 2) state = "busy";

    stateEl.dataset.state = state;
    stateEl.dataset.animated = state === "strained" ? "1" : "0";

    if (state === "chill") stateLabel.textContent = "Chill";
    else if (state === "busy") stateLabel.textContent = "Busy";
    else stateLabel.textContent = "Strained";
  }

  window.CROWN_PREMIUM.hooks.onAnalytics.push((meta) => {
    const durationSec = meta.durationSec || 0;
    const failed = meta.failed || 0;
    const tweets = meta.tweets || 0;

    const ms = Math.max(1, durationSec * 1000 / Math.max(1, tweets));
    latencyEl.textContent = `${Math.round(ms)} ms/tweet`;

    values.push(ms);
    if (values.length > 10) values.shift();
    redrawSpark();
    updateState(durationSec, failed);
  });
})();

(function () {
  if (!window.CT_PREMIUM_ENABLED) return;
  const flags = window.CT_PREMIUM_FLAGS || {};
  if (!flags.altOverlay) return;
  if (window.matchMedia && window.matchMedia("(pointer: coarse)").matches) return;

  const overlay = document.getElementById("ctCommandOverlay");
  if (!overlay) return;

  let visible = false;
  let streakCount = 0;
  let lastWasKeyboard = false;
  let streakTimer = null;

  // Map hotkeys to elements (IDs)
  const HOTKEY_MAP = {
    "g": "generateBtn",
    "c": "clearBtn",
    "h": "historyBtn",
    "r": "retryFailedBtn"
  };

  function showOverlay() {
    if (visible) return;
    visible = true;
    overlay.classList.add("is-visible");
    overlay.setAttribute("aria-hidden", "false");
    overlay.innerHTML = "";

    Object.entries(HOTKEY_MAP).forEach(([key, id]) => {
      const el = document.getElementById(id);
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const label = document.createElement("div");
      label.className = "ct-command-label";
      label.innerHTML = `<span class="key">${key.toUpperCase()}</span>`;
      label.style.left = rect.left + rect.width / 2 + "px";
      label.style.top = rect.top + rect.height / 2 + "px";
      overlay.appendChild(label);
    });
  }

  function hideOverlay() {
    if (!visible) return;
    visible = false;
    overlay.classList.remove("is-visible");
    overlay.setAttribute("aria-hidden", "true");
    overlay.innerHTML = "";
  }

  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Alt") {
      showOverlay();
      return;
    }

    // hotkeys active only while overlay visible
    if (visible && HOTKEY_MAP[ev.key]) {
      const id = HOTKEY_MAP[ev.key];
      const btn = document.getElementById(id);
      if (btn) {
        btn.click();
        // command streak tracking
        streakCount = lastWasKeyboard ? streakCount + 1 : 1;
        lastWasKeyboard = true;
        if (streakTimer) clearTimeout(streakTimer);
        streakTimer = setTimeout(() => { streakCount = 0; lastWasKeyboard = false; }, 4000);
        if (streakCount === 3 || streakCount === 5 || streakCount === 10) {
          if (typeof ctToast === "function" && (window.CT_PREMIUM_FLAGS || {}).achievements !== false) {
            ctToast(`Command streak x${streakCount}`, "ok", 2000);
          }
        }
      }
    }
  });

  document.addEventListener("keyup", (ev) => {
    if (ev.key === "Alt") hideOverlay();
  });

  document.addEventListener("click", () => {
    lastWasKeyboard = false;
  });
})();


(function () {
  if (!window.CT_PREMIUM_ENABLED) return;
  const runSheet = document.getElementById("runSheet");
  const toggle = document.getElementById("runSheetToggle");
  const summaryEl = document.getElementById("runSheetSummary");
  const statsEl = document.getElementById("runSheetStats");
  if (!runSheet || !toggle) return;

  toggle.addEventListener("click", () => {
    runSheet.classList.toggle("is-open");
    runSheet.setAttribute("aria-hidden", runSheet.classList.contains("is-open") ? "false" : "true");
  });

  runSheet.addEventListener("click", (ev) => {
    if (ev.target === runSheet) {
      runSheet.classList.remove("is-open");
      runSheet.setAttribute("aria-hidden", "true");
    }
  });

  // update content from analytics hook
  window.CROWN_PREMIUM.hooks.onAnalytics.push((meta) => {
    const tweets = meta.tweets || 0;
    const failed = meta.failed || 0;
    const durationSec = meta.durationSec || 0;
    const comments = meta.totalComments || meta.comments || (tweets * 2);

    summaryEl.textContent = `Last run: ${tweets} tweets, ${comments} comments, ${failed} failed.`;
    statsEl.textContent = `Took ${durationSec}s • Approx ${tweets ? Math.round((durationSec*1000)/tweets) : 0} ms/tweet.`;
  });
})();

(function () {
  if (!window.CT_PREMIUM_ENABLED) return;
  const flags = window.CT_PREMIUM_FLAGS || {};
  if (!flags.radarRing) return;

  const canvas = document.getElementById("radarCanvas");
  const radarRunningEl = document.getElementById("radarRunning");
  const radarQueuedEl = document.getElementById("radarQueued");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const maxR = 30;
  let angle = 0;
  let lastQueue = [];

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // rotating sweep
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.arc(0, 0, maxR, -0.5, 0.5);
    ctx.closePath();
    ctx.fillStyle = "rgba(56,189,248,0.18)";
    ctx.fill();
    ctx.restore();

    // dots for jobs
    const jobs = lastQueue || [];
    jobs.forEach((item, idx) => {
      const t = idx / Math.max(1, jobs.length - 1);
      const r = 6 + t * (maxR - 6);
      const a = t * Math.PI * 2;
      ctx.beginPath();
      ctx.arc(cx + Math.cos(a) * r, cy + Math.sin(a) * r, 2, 0, Math.PI * 2);
      let status = item.status || "queued";
      let color = "rgba(148,163,184,0.9)";
      if (status === "processing") color = "rgba(56,189,248,0.9)";
      else if (status === "done") color = "rgba(34,197,94,0.9)";
      else if (status === "failed") color = "rgba(239,68,68,0.9)";
      ctx.fillStyle = color;
      ctx.fill();
    });

    angle += 0.03;
    requestAnimationFrame(draw);
  }

  // update from queue hook
  window.CROWN_PREMIUM.hooks.onQueueRender.push(({ urlQueueState }) => {
    lastQueue = urlQueueState || [];
    const running = lastQueue.filter((x) => x.status === "processing").length;
    const queued = lastQueue.filter((x) => x.status === "queued").length;
    if (radarRunningEl) radarRunningEl.textContent = String(running);
    if (radarQueuedEl) radarQueuedEl.textContent = String(queued);

    const shell = document.getElementById("radarRing");
    const summary = document.getElementById("radarSummary");
    const visible = lastQueue.length > 0;
    if (shell) shell.setAttribute("aria-hidden", visible ? "false" : "true");
    if (summary) summary.setAttribute("aria-hidden", visible ? "false" : "true");
  });

  draw();
})();

(function () {
  if (!window.CT_PREMIUM_ENABLED) return;
  const flags = window.CT_PREMIUM_FLAGS || {};
  if (!flags.timeline) return;

  const timelineEl = document.getElementById("runTimeline");
  if (!timelineEl) return;

  const runs = [];

  window.CROWN_PREMIUM.hooks.onRunFinish.push((meta) => {
    const t = new Date();
    runs.push({
      timeLabel: t.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      tweets: meta.tweets || 0,
      failed: meta.failed || 0,
      sessionId: window.ctActiveSessionId || null
    });
    if (runs.length > 40) runs.shift();
    render();
  });

  function render() {
    if (!runs.length) {
      timelineEl.setAttribute("aria-hidden", "true");
      timelineEl.innerHTML = "";
      return;
    }
    timelineEl.setAttribute("aria-hidden", "false");
    const frag = document.createDocumentFragment();
    runs.forEach((run) => {
      const dot = document.createElement("button");
      dot.type = "button";
      dot.className = "run-dot";
      dot.innerHTML = `<span class="run-dot-time">${run.timeLabel}</span><span class="run-dot-size">${run.tweets} tweets</span>`;
      dot.addEventListener("click", () => {
        if (run.sessionId && typeof window.restoreSessionSnapshot === "function") {
          window.restoreSessionSnapshot(run.sessionId);
        }
      });
      frag.appendChild(dot);
    });
    timelineEl.innerHTML = "";
    timelineEl.appendChild(frag);
  }
})();

(function () {
  if (!window.CT_PREMIUM_ENABLED) return;
  const flags = window.CT_PREMIUM_FLAGS || {};
  if (!flags.achievements) return;
  if (typeof ctToast !== "function") return;

  let totalCommentsToday = 0;
  let zeroFailStreak = 0;

  window.CROWN_PREMIUM.hooks.onRunFinish.push((meta) => {
    const tweets = meta.tweets || 0;
    const comments = meta.totalComments || meta.comments || (tweets * 2);
    const failed = meta.failed || 0;

    totalCommentsToday += comments;
    if (failed === 0 && tweets > 0) {
      zeroFailStreak += 1;
    } else {
      zeroFailStreak = 0;
    }

    if (totalCommentsToday >= 100 && totalCommentsToday - comments < 100) {
      ctToast("100+ comments generated today. Grinding. 💪", "ok", 2600);
    }
    if (zeroFailStreak === 3) {
      ctToast("3 perfect runs in a row!", "ok", 2300);
    }
    if (zeroFailStreak === 10) {
      ctToast("10 flawless runs. You are a machine.", "ok", 3000);
    }
  });
})();

