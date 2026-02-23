from pydantic import BaseModel


class Citation(BaseModel):
    """One retrieved chunk used to ground an answer."""

    section_title: str
    excerpt: str
    score: float


class QueryResponse(BaseModel):
    """Answer + supporting citations from the RAG pipeline."""

    answer: str
    citations: list[Citation]


class RevenueMetric(BaseModel):
    value: float | None = None
    unit: str | None = None
    period: str | None = None


class EpsMetric(BaseModel):
    value: float | None = None
    diluted: bool | None = None


class NetIncomeMetric(BaseModel):
    value: float | None = None
    unit: str | None = None


class GrossMarginMetric(BaseModel):
    value: float | None = None


class GuidanceMetric(BaseModel):
    revenue_low: float | None = None
    revenue_high: float | None = None
    period: str | None = None


class YoyDeltas(BaseModel):
    revenue: float | None = None
    eps: float | None = None
    net_income: float | None = None


class FinancialMetrics(BaseModel):
    revenue: RevenueMetric | None = None
    eps: EpsMetric | None = None
    net_income: NetIncomeMetric | None = None
    gross_margin: GrossMarginMetric | None = None
    guidance: GuidanceMetric | None = None
    yoy_deltas: YoyDeltas | None = None


class ExtractionResponse(BaseModel):
    document_id: str
    company_name: str | None = None
    fiscal_year: str | None = None
    filing_type: str | None = None
    metrics: FinancialMetrics = FinancialMetrics()
