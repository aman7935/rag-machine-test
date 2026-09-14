const BASE = "http://localhost:8000";
const $ = (id) => document.getElementById(id);
const st = $("st");
const show = (id) => $(id)?.classList.remove("hidden");
const setSt = (t, err) => { if (st) { st.textContent = t; st.className = err ? "err" : ""; } };

function addMsg(text, who) {
  const d = document.createElement("div");
  d.className = `msg ${who}`;
  const whoEl = document.createElement("div");
  whoEl.className = "who";
  whoEl.textContent = who === "user" ? "You" : "Assistant";
  const body = document.createElement("div");
  body.className = "body";
  body.textContent = text;
  d.appendChild(whoEl);
  d.appendChild(body);
  $("msgs").appendChild(d);
  $("msgs").scrollTop = $("msgs").scrollHeight;
  return body;
}

function parseSSE(buf) {
  const events = [];
  const parts = buf.split("\n\n");
  for (const p of parts) {
    const line = p.trim();
    if (!line.startsWith("data:")) continue;
    events.push(line.slice(5).trim());
  }
  return { events, tail: parts.pop() || "" };
}

// Upload
$("up").onclick = async () => {
  const file = $("file").files[0];
  if (!file) return setSt("pick a PDF", true);
  $("up").disabled = true;
  setSt("processing...");
  try {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch(`${BASE}/upload`, { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "upload failed");
    setSt(`${d.filename} — ${d.table_rows} rows indexed`);
    if ($("file-info")) $("file-info").textContent = d.filename;
    show("chat-card");
    addMsg(`Ready. Ask me anything about ${d.filename}.`, "bot");
  } catch (e) { setSt(e.message, true); }
  finally { $("up").disabled = false; }
};

// Ask (streaming)
$("f").onsubmit = async (e) => {
  e.preventDefault();
  const q = $("q").value.trim();
  if (!q) return;
  addMsg(q, "user");
  $("q").value = "";
  const btn = $("f").querySelector("button");
  btn.disabled = true;
  const bubble = addMsg("", "bot");

  try {
    const r = await fetch(`${BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });
    if (!r.ok) {
      const d = await r.json();
      throw new Error(d.detail || "ask failed");
    }
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const { events, tail } = parseSSE(buf);
      buf = tail;

      for (const payload of events) {
        if (payload === "[DONE]") { btn.disabled = false; $("q").focus(); return; }
        try {
          const obj = JSON.parse(payload);
          if (obj.error) throw new Error(obj.error);
          bubble.textContent += obj.token;
          $("msgs").scrollTop = $("msgs").scrollHeight;
        } catch (e) { if (e.message !== "[DONE]") throw e; }
      }
    }
  } catch (err) { bubble.textContent = bubble.textContent || `Error: ${err.message}`; }
  finally { btn.disabled = false; $("q").focus(); }
};

// Boot: show status if already uploaded
(async () => {
  try {
    const r = await fetch(`${BASE}/status`);
    const d = await r.json();
    if (d.ready) { setSt(`${d.filename} ready`); if ($("file-info")) $("file-info").textContent = d.filename; show("chat-card"); }
  } catch {}
})();