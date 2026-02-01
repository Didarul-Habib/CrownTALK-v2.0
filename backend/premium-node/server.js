import express from "express";
import { WebSocketServer } from "ws";

const app = express();
app.use(express.json());

const wss = new WebSocketServer({ noServer: true });
const clients = new Set();

function broadcast(type, payload) {
  const msg = JSON.stringify({ type, payload });
  for (const ws of clients) {
    if (ws.readyState === ws.OPEN) ws.send(msg);
  }
}

app.get("/health", (_req, res) => res.json({ ok: true }));

app.post("/event", (req, res) => {
  const { type, payload } = req.body || {};
  if (!type) return res.status(400).json({ error: "missing type" });
  broadcast(type, payload || {});
  res.json({ ok: true });
});

// ✅ Render uses PORT. Keep PREMIUM_NODE_PORT as fallback for local.
const PORT = process.env.PORT || process.env.PREMIUM_NODE_PORT || 8081;

const server = app.listen(PORT, "0.0.0.0", () => {
  console.log("Premium Node listening on", PORT);
});

server.on("upgrade", (request, socket, head) => {
  if (request.url !== "/ws") {
    socket.destroy();
    return;
  }
  wss.handleUpgrade(request, socket, head, (ws) => {
    clients.add(ws);
    ws.on("close", () => clients.delete(ws));
  });
});
