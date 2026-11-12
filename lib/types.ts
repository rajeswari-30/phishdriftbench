// Mirrors the dict returned by src/phishdriftbench/scan/engine.py:scan().

export type Verdict = "phishing" | "suspicious" | "safe";
export type Severity = "critical" | "high" | "medium" | "low" | "info" | "good";

export interface Finding {
  id: string;
  category: "url" | "domain" | "page" | "intel" | "ai";
  title: string;
  detail: string;
  weight: number;
  severity: Severity;
}

export interface IntelResult {
  source: string;
  status: "hit" | "clean" | "unavailable" | "not_configured";
  detail: string | null;
}

export interface Signal {
  name: string;
  label: string;
  value: number;
  contribution: number;
  direction: "phishing" | "legitimate" | "neutral";
}

export interface FormInfo {
  action: string;
  action_domain: string;
  method: string;
  has_password: boolean;
  sensitive_fields: string[];
  external: boolean;
  exfil: string | null;
}

export interface PageAnalysis {
  title: string;
  text_length: number;
  forms: FormInfo[];
  has_password: boolean;
  sensitive_fields: string[];
  brand_claims: string[];
  link_count: number;
  link_domains: Record<string, number>;
  dead_link_ratio: number;
  external_resource_ratio: number;
  favicon_domain: string | null;
  meta_refresh: string | null;
  js_redirects: string[];
  obfuscation: string[];
  anti_inspect: boolean;
  hidden_iframes: number;
  login_language: boolean;
  parked: boolean;
  provider_flagged: boolean;
  provider_takedown: boolean;
  is_html: boolean;
}

export interface DomainInfo {
  domain: string;
  registered: string | null;
  age_days: number | null;
  expires: string | null;
  registrar: string | null;
  rdap_error: string | null;
  a_records: string[];
  nameservers: string[];
  mx: boolean | null;
  dns_error: string | null;
  cert_issuer: string | null;
  cert_subject: string | null;
  cert_not_before: string | null;
  cert_not_after: string | null;
  cert_age_days: number | null;
  cert_sans: string[];
  cert_error: string | null;
  hosted_platform: boolean;
}

export interface PageInfo {
  fetched: boolean;
  status: number | null;
  error: string | null;
  final_url: string;
  hops: { url: string; status: number | null }[];
  tls_valid: boolean | null;
  tls_error: string | null;
  elapsed_ms: number;
  bytes: number;
  analysis: PageAnalysis | null;
}

export interface AiResult {
  enabled?: boolean;
  ran?: boolean;
  error?: string;
  verdict?: string;
  probability_phishing?: number;
  impersonated_brand?: string | null;
  lure?: string | null;
  reason?: string;
  model?: string;
}

export interface ScanResult {
  id: string;
  input: string;
  url: string;
  final_url: string;
  scanned_at: string;
  elapsed_ms: number;
  deep: boolean;
  risk: number;
  verdict: Verdict;
  overridden_by: string | null;
  evidence: "strong" | "moderate" | "limited";
  coverage: { url: boolean; domain_age: boolean; page: boolean; intel: string[]; ai: boolean };
  summary: string;
  findings: Finding[];
  url_model: {
    verdict: string;
    ml_score: number;
    decision_threshold: number;
    decided_by: string;
    decision_reason: string | null;
    brands_referenced: string[];
    suspicious_words: string[];
    squatting_reasons: string[];
    signals: Signal[];
  };
  domain: DomainInfo;
  page: PageInfo | null;
  intel: IntelResult[];
  ai: AiResult;
}

export interface Health {
  ok: boolean;
  version: string;
  sources: Record<string, boolean>;
}
