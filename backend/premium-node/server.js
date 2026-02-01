import express from "express";
import { WebSocketServer } from "ws";

const app = express();
app.use(express.json({ limit: "256kb" }));

// ---- WebSocket setup (upgrade-based) ----
const wss = new WebSocketServer({ noServer: true });
const clients = new Set();

function broadcast(type, payload) {
  const msg = JSON.stringify({
    type,
    payload: payload ?? {},
    ts: Date.now(),
  });

  for (const ws of clients) {
    if (ws.readyState === ws.OPEN) {
      try {
        ws.send(msg);
      } catch {
        // ignore send errors
      }
    }
  }
}

// ---- Health / Debug ----
app.get("/", (_req, res) => res.send("premium-node ok"));
app.get("/health", (_req, res) =>
  res.json({ ok: true, clients: clients.size, ts: Date.now() })
);

// ---- Event ingest (backend -> premium-node) ----
app.post("/event", (req, res) => {
  const body = req.body || {};
  const { type, payload } = body;

  if (!type || typeof type !== "string") {
    return res.status(400).json({ ok: false, error: "missing type" });
  }

  broadcast(type, payload);
  return res.json({ ok: true });
});

// ---- Start server (Render-compatible) ----
// Render sets process.env.PORT. Keep PREMIUM_NODE_PORT for local overrides.
const PORT = process.env.PORT || process.env.PREMIUM_NODE_PORT || 8081;

const server = app.listen(PORT, "0.0.0.0", () => {
  console.log("Premium Node listening on", PORT);
});

// ---- Upgrade handler for WS path /ws ----
server.on("upgrade", (request, socket, head) => {
  try {
    // request.url may contain query params — strip them
    const url = request.url || "";
    const path = url.split("?")[0];

    if (path !== "/ws") {
      socket.destroy();
      return;
    }

    wss.handleUpgrade(request, socket, head, (ws) => {
      clients.add(ws);

      ws.on("close", () => clients.delete(ws));
      ws.on("error", () => clients.delete(ws));

      // Optional: send a hello packet so frontend knows it connected
      try {
        ws.send(
          JSON.stringify({
            type: "ws_connected",
            payload: { ok: true, ts: Date.now() },
            ts: Date.now(),
          })
        );
      } catch {}
    });
  } catch {
    socket.destroy();
  }
});
