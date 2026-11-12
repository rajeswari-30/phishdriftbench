const form = document.getElementById("form");
const urlInput = document.getElementById("url");
const checkBtn = document.getElementById("checkBtn");
const checkTabBtn = document.getElementById("checkTabBtn");
const resultDiv = document.getElementById("result");
const serverInput = document.getElementById("server");

const LABEL = { phishing: "Phishing", suspicious: "Suspicious", safe: "Looks safe" };

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

function render(data, { stale = false } = {}) {
  const cls = data.verdict in LABEL ? data.verdict : "suspicious";
  let html = `<div class="verdict-row">
      <span class="verdict ${cls}">${esc(LABEL[cls])}</span>
      <span class="conf">${esc(data.risk)}/100</span>
    </div>
    <div class="checked-url">${esc(data.url)}</div>
    <p class="summary">${esc(data.summary || "")}</p>`;
  const top = (data.findings || []).filter((f) => f.weight > 0).slice(0, 4);
  if (top.length) {
    html += `<div class="sect">Why</div><ul>`;
    top.forEach((f) => (html += `<li>${esc(f.title)}</li>`));
    html += `</ul>`;
  }
  if (stale) html += `<span class="stale">from your last scan</span>`;
  resultDiv.innerHTML = html;
}

async function runCheck(url) {
  if (!url) return;
  checkBtn.disabled = checkTabBtn.disabled = true;
  resultDiv.innerHTML = `<p class="muted">Scanning…</p>`;
  const res = await new Promise((resolve) => chrome.runtime.sendMessage({ type: "SCAN_URL", url }, resolve));
  if (res && res.ok) render(res.data);
  else resultDiv.innerHTML = `<p class="error">${esc((res && res.error) || "PhishLens is unreachable.")}<br>
      Check the server address below.</p>`;
  checkBtn.disabled = checkTabBtn.disabled = false;
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  runCheck(urlInput.value.trim());
});

checkTabBtn.addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab && tab.url) {
    urlInput.value = tab.url;
    runCheck(tab.url);
  }
});

chrome.storage.sync.get("apiBase", ({ apiBase }) => {
  serverInput.value = apiBase || "http://localhost:3000";
});
serverInput.addEventListener("change", () => {
  chrome.storage.sync.set({ apiBase: serverInput.value.trim() });
});

chrome.storage.local.get("lastResult", ({ lastResult }) => {
  if (lastResult && lastResult.verdict in LABEL) render(lastResult, { stale: true });
});
