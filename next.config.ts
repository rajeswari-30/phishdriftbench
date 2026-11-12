import type { NextConfig } from "next";

// On Vercel, vercel.json routes /api/* to the Python function in
// api/index.py, on the same domain as this site. Locally the API runs as
// its own process (`npm run dev:api`, port 8000), so the dev server proxies
// to it. PHISHLENS_API_URL points the proxy elsewhere, e.g. at a deployed API.
const api =
  process.env.PHISHLENS_API_URL ??
  (process.env.NODE_ENV === "development" ? "http://127.0.0.1:8000" : null);

const nextConfig: NextConfig = {
  async rewrites() {
    return api ? [{ source: "/api/:path*", destination: `${api}/api/:path*` }] : [];
  },
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
        ],
      },
    ];
  },
};

export default nextConfig;
