const form = document.getElementById("form");
const urlInput = document.getElementById("url");
const checkBtn = document.getElementById("checkBtn");
const checkTabBtn = document.getElementById("checkTabBtn");
const resultDiv = document.getElementById("result");

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function askBackground(url) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ type: "CHECK_URL", url }, resolve);
  });
}

function render(data, { stale = false } = {}) {
  if (data.error) {
    resultDiv.innerHTML = `<p class="error">${esc(data.error)}</p>`;
    return;
  }
  const cls =
    data.verdict === "PHISHING" ? "phishing" : data.verdict === "UNCERTAIN" ? "uncertain" : "legitimate";

  let html = `<div class="verdict-row">
    <span class="verdict ${cls}">${esc(data.verdict)}</span>`;
  if (!data.abstained) {
    html += `<span class="conf">${(data.confidence * 100).toFixed(1)}%</span>`;
  }
  html += `</div><div class="checked-url">${esc(data.url)}</div>`;

  if (data.abstained) {
    html += `<div class="abstain"><strong>No verdict:</strong> ${esc(data.abstain_reason)}.</div>`;
  }

  if (data.squatting_reasons && data.squatting_reasons.length) {
    html += `<div class="sect">Brand-jacking screen</div><ul>`;
    data.squatting_reasons.forEach((r) => (html += `<li>${esc(r)}</li>`));
    html += `</ul>`;
  }

  if (data.top_reasons && data.top_reasons.length) {
    html += `<div class="sect">Top factors</div><ul>`;
    data.top_reasons.slice(0, 3).forEach((r) => (html += `<li>${esc(r)}</li>`));
    html += `</ul>`;
  }

  if (stale) html += `<span class="stale">from your last check</span>`;
  resultDiv.innerHTML = html;
}

function renderConnectionError() {
  resultDiv.innerHTML = `<p class="error">Can't reach PhishLens.<br>
    Make sure the local server is running:<br>
    <code>python webapp/app.py</code></p>`;
}

async function runCheck(url) {
  if (!url) return;
  checkBtn.disabled = true;
  checkTabBtn.disabled = true;
  resultDiv.innerHTML = `<p class="muted">Checking…</p>`;
  const res = await askBackground(url);
  if (res && res.ok) {
    render(res.data);
  } else {
    renderConnectionError();
  }
  checkBtn.disabled = false;
  checkTabBtn.disabled = false;
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

// On open, show the last result from a right-click check, if any, so the
// popup isn't empty the moment after you've just checked a link.
chrome.storage.local.get("lastResult", ({ lastResult }) => {
  if (lastResult) render(lastResult, { stale: true });
});
