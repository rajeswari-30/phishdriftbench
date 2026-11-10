/**
 * PhishLens extension — background service worker.
 *
 * This talks to your EXISTING PhishLens website (webapp/app.py), running
 * locally at http://127.0.0.1:5001 — the exact same /api/check endpoint the
 * browser page itself calls. No model logic lives in this extension; it is
 * a thin client. That is a deliberate scope choice, not a shortcut: because
 * the server stays on 127.0.0.1 (loopback), the checked URL never actually
 * leaves this machine, so the "no data is fetched or sent anywhere" claim
 * the project makes still holds — it just means the local Flask server
 * must be running for the extension to work. See extension/README.md.
 */

const API_BASE_CANDIDATES = ["http://127.0.0.1:5001", "http://localhost:5001"];
const MENU_ID = "phishlens-check-link";

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: MENU_ID,
    title: 'Check this link with PhishLens',
    contexts: ["link"],
  });
});

chrome.contextMenus.onClicked.addListener((info) => {
  if (info.menuItemId === MENU_ID && info.linkUrl) {
    checkAndNotify(info.linkUrl);
  }
});

/** Try 127.0.0.1 first, fall back to localhost — some setups only resolve one. */
async function callApi(path, body) {
  let lastErr;
  for (const base of API_BASE_CANDIDATES) {
    try {
      const res = await fetch(base + path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr;
}

async function checkUrl(url) {
  return callApi("/api/check", { url });
}

const BADGE_COLORS = {
  PHISHING: "#a02d1c",
  LEGITIMATE: "#0f6b45",
  UNCERTAIN: "#8a5a10",
};
const BADGE_TEXT = { PHISHING: "P", LEGITIMATE: "OK", UNCERTAIN: "?" };

async function checkAndNotify(url) {
  chrome.action.setBadgeText({ text: "…" });
  try {
    const data = await checkUrl(url);
    if (data.error) throw new Error(data.error);

    await chrome.storage.local.set({ lastResult: { ...data, checkedAt: Date.now() } });
    chrome.action.setBadgeText({ text: BADGE_TEXT[data.verdict] || "?" });
    chrome.action.setBadgeBackgroundColor({ color: BADGE_COLORS[data.verdict] || "#6b7482" });

    const titleByVerdict = {
      PHISHING: "⚠ Likely phishing",
      LEGITIMATE: "✓ Looks legitimate",
      UNCERTAIN: "? Could not verify",
    };
    chrome.notifications.create({
      type: "basic",
      iconUrl: "icons/icon128.png",
      title: titleByVerdict[data.verdict] || "PhishLens result",
      message: url.length > 90 ? url.slice(0, 87) + "…" : url,
      priority: data.verdict === "PHISHING" ? 2 : 0,
    });
  } catch (err) {
    chrome.action.setBadgeText({ text: "!" });
    chrome.action.setBadgeBackgroundColor({ color: "#a02d1c" });
    chrome.notifications.create({
      type: "basic",
      iconUrl: "icons/icon128.png",
      title: "PhishLens is unreachable",
      message: "Make sure the local server is running: python webapp/app.py",
    });
  }
}

// The popup can't always reach the background's local variables directly in
// MV3 (the service worker may be asleep), so it talks through messages.
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "CHECK_URL") {
    checkUrl(msg.url)
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: String(err) }));
    return true; // keep the message channel open for the async response
  }
});
