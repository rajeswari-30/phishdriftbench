# PhishLens browser extension

A thin client for a PhishLens deployment. It calls the same `POST /api/scan` endpoint the website
uses and holds no model logic of its own.

## Install (Chrome, Edge, Brave or any Chromium browser)

1. Open `chrome://extensions` and turn on **Developer mode**.
2. Click **Load unpacked** and select this `extension/` folder.
3. Open the PhishLens popup and set **Server** to your deployment, e.g.
   `https://your-app.vercel.app`, or leave the default `http://localhost:3000` for local
   development (`npm run dev` plus `npm run dev:api`).

`*.vercel.app` and localhost are allowed out of the box. A custom domain needs the optional
`https://*/*` host permission, which Chrome asks for once.

## Use

- **Right-click any link → "Scan this link with PhishLens".** A notification shows the verdict
  and risk score, and the toolbar badge turns red (phishing), amber (suspicious) or green.
- **Click the toolbar icon** to paste a link or scan the current tab.

The extension only sees links inside the browser. It can't see links in native apps.

| File | Purpose |
|---|---|
| `manifest.json` | Manifest V3 config |
| `background.js` | Right-click menu, API calls, badge and notification |
| `popup.html` / `popup.js` | Toolbar popup and the server setting |
| `icons/` | Extension icons |
