/**
 * PhishLens extension: background service worker.
 *
 * A thin client for a PhishLens deployment's /api/scan endpoint. It holds no
 * model logic. The server address is a setting (popup → Server), defaulting
 * to the local dev server.
 */

const DEFAULT_API = "http://localhost:3000";
const MENU_ID = "phishlens-check-link";

const BADGE = {
  phishing: { text: "!", color: "#c2372b", title: "⚠ Likely phishing" },
  suspicious: { text: "?", color: "#a16a07", title: "Suspicious link" },
  safe: { text: "OK", color: "#1d7d3f", title: "✓ No warning signs" },
};

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({ id: MENU_ID, title: "Scan this link with PhishLens", contexts: ["link"] });
});

chrome.contextMenus.onClicked.addListener((info) => {
  if (info.menuItemId === MENU_ID && info.linkUrl) scanAndNotify(info.linkUrl);
});

async function apiBase() {
  const { apiBase } = await chrome.storage.sync.get("apiBase");
  return (apiBase || DEFAULT_API).replace(/\/+$/, "");
}

async function scanUrl(url) {
  const res = await fetch(`${await apiBase()}/api/scan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, deep: true }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${res.status}`);
  return body;
}

async function scanAndNotify(url) {
  chrome.action.setBadgeText({ text: "…" });
  try {
    const data = await scanUrl(url);
    await chrome.storage.local.set({ lastResult: { ...data, checkedAt: Date.now() } });
    const b = BADGE[data.verdict] || BADGE.suspicious;
    chrome.action.setBadgeText({ text: b.text });
    chrome.action.setBadgeBackgroundColor({ color: b.color });
    chrome.notifications.create({
      type: "basic",
      iconUrl: "icons/icon128.png",
      title: `${b.title} (risk ${data.risk}/100)`,
      message: data.summary || url,
      priority: data.verdict === "phishing" ? 2 : 0,
    });
  } catch (err) {
    chrome.action.setBadgeText({ text: "×" });
    chrome.action.setBadgeBackgroundColor({ color: "#6f6e69" });
    chrome.notifications.create({
      type: "basic",
      iconUrl: "icons/icon128.png",
      title: "PhishLens could not scan this link",
      message: String(err.message || err).slice(0, 160),
    });
  }
}

// MV3 service workers sleep, so the popup talks to this one by message.
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "SCAN_URL") {
    scanUrl(msg.url)
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err.message || err) }));
    return true;
  }
});
