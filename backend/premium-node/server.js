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

app.post("/event", (req, res) => {
  const { type, payload } = req.body || {};
  if (!type) return res.status(400).json({ error: "missing type" });
  broadcast(type, payload || {});
  res.json({ ok: true });
});

const server = app.listen(process.env.PREMIUM_NODE_PORT || 8081, () => {
  console.log("Premium Node listening");
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
