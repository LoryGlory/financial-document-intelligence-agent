export type DocumentStatus = "processing" | "ready" | "error";

export interface IngestResponse {
  document_id: string;
  filename: string;
  status: DocumentStatus;
  chunks_count: number;
  sections_found: string[];
}

export interface Citation {
  section_title: string;
  excerpt: string;
  score: number;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
}

export interface RevenueMetric {
  value: number | null;
  unit: string | null;
  period: string | null;
}

export interface EpsMetric {
  value: number | null;
  diluted: boolean | null;
}

export interface FinancialMetrics {
  revenue: RevenueMetric | null;
  eps: EpsMetric | null;
  net_income: { value: number | null; unit: string | null } | null;
  gross_margin: { value: number | null } | null;
  guidance: {
    revenue_low: number | null;
    revenue_high: number | null;
    period: string | null;
  } | null;
  yoy_deltas: {
    revenue: number | null;
    eps: number | null;
    net_income: number | null;
  } | null;
}

export interface ExtractionResponse {
  document_id: string;
  company_name: string | null;
  fiscal_year: string | null;
  filing_type: string | null;
  metrics: FinancialMetrics;
}

export interface Document extends IngestResponse {
  extraction?: ExtractionResponse;
}
