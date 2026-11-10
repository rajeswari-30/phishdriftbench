# PhishLens browser extension

A thin client for your existing PhishLens website (`webapp/app.py`). It does
not run any model logic itself — it calls the same `/api/check` endpoint the
browser page already uses, running locally at `http://127.0.0.1:5001`.

**Why this still keeps the "nothing is fetched or sent anywhere" promise:**
the server stays on `127.0.0.1` (loopback) — a checked URL never leaves this
machine. That is a deliberate design choice, not an afterthought: the
alternative (porting the model to JavaScript so the extension needs no
server at all) is the *more* private option long-term, but this is the
fast, honest path that reuses everything already built and verified.

## Requirements

The local PhishLens server must be running for the extension to work:

```bash
python webapp/app.py
```

Leave that running in a terminal. The extension has no model of its own —
if the server isn't up, it will tell you plainly rather than fail silently.

## Installing (Chrome / Edge / Brave — any Chromium browser)

1. Open `chrome://extensions` (or `edge://extensions`)
2. Turn on **Developer mode** (top-right toggle)
3. Click **Load unpacked**
4. Select this `extension/` folder
5. The PhishLens icon appears in your toolbar

This is a manual, one-time step you do yourself — Chrome's file picker for
"Load unpacked" can't be driven by an automated tool.

## Using it

- **Right-click any link on any page** → "Check this link with PhishLens".
  A notification pops up with the verdict, and the toolbar badge shows a
  colored result (red = phishing, green = legitimate, orange = uncertain).
- **Click the toolbar icon** to open the popup: paste any URL, or click
  "Check current page" to check the tab you're on right now.

## What it can and can't see

A browser extension can only see links **inside the browser** — any
website, WhatsApp Web, Gmail, etc. It cannot see a link inside the native
WhatsApp mobile app on a phone, because that isn't a webpage the extension
has access to. Phone-side protection would need a different delivery
mechanism (e.g. an Android "Share to PhishLens" target), not a browser
extension.

## Files

| File | Purpose |
|---|---|
| `manifest.json` | Manifest V3 config — permissions, icons, entry points |
| `background.js` | Service worker: right-click menu, API calls, badge/notification |
| `popup.html` / `popup.js` | The toolbar popup UI |
| `icons/` | Extension icons (same logo as the website), generated at 16/32/48/128px |
