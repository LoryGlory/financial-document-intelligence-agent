"use client";

import { TrendingDown, TrendingUp, Minus } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ExtractionResponse } from "@/lib/types";

interface MetricCardProps {
  title: string;
  value: string | null;
  delta?: number | null;
  subtitle?: string;
}

function MetricCard({ title, value, delta, subtitle }: MetricCardProps) {
  const DeltaIcon =
    delta == null ? null : delta > 0 ? TrendingUp : delta < 0 ? TrendingDown : Minus;
  const deltaColor =
    delta == null
      ? ""
      : delta > 0
        ? "text-emerald-600"
        : delta < 0
          ? "text-red-500"
          : "text-slate-500";

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs font-medium uppercase tracking-wide text-slate-500">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-2xl font-bold text-slate-900">
          {value ?? <span className="text-slate-300">—</span>}
        </p>
        {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
        {delta != null && DeltaIcon && (
          <div className={`mt-1.5 flex items-center gap-1 text-sm font-medium ${deltaColor}`}>
            <DeltaIcon className="h-4 w-4" />
            <span>
              {delta > 0 ? "+" : ""}
              {delta.toFixed(1)}% YoY
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function formatBillions(value: number | null | undefined, unit?: string | null): string | null {
  if (value == null) return null;
  const label = unit === "USD_billions" ? "B" : unit === "USD_millions" ? "M" : "";
  return `$${value.toFixed(1)}${label}`;
}

interface MetricsDashboardProps {
  extraction: ExtractionResponse;
}

export function MetricsDashboard({ extraction }: MetricsDashboardProps) {
  const { metrics, company_name, fiscal_year, filing_type } = extraction;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <h3 className="text-lg font-semibold text-slate-900">
          {company_name ?? "Company"} — {fiscal_year ?? "Unknown Year"}
        </h3>
        {filing_type && <Badge variant="outline">{filing_type}</Badge>}
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <MetricCard
          title="Revenue"
          value={formatBillions(metrics.revenue?.value, metrics.revenue?.unit)}
          subtitle={metrics.revenue?.period ?? undefined}
          delta={metrics.yoy_deltas?.revenue}
        />
        <MetricCard
          title="EPS"
          value={metrics.eps?.value != null ? `$${metrics.eps.value.toFixed(2)}` : null}
          subtitle={metrics.eps?.diluted ? "diluted" : undefined}
          delta={metrics.yoy_deltas?.eps}
        />
        <MetricCard
          title="Net Income"
          value={formatBillions(metrics.net_income?.value, metrics.net_income?.unit)}
          delta={metrics.yoy_deltas?.net_income}
        />
        <MetricCard
          title="Gross Margin"
          value={
            metrics.gross_margin?.value != null ? `${metrics.gross_margin.value.toFixed(1)}%` : null
          }
        />
        {metrics.guidance && (
          <MetricCard
            title="Guidance"
            value={
              metrics.guidance.revenue_low != null
                ? `$${metrics.guidance.revenue_low}–${metrics.guidance.revenue_high}B`
                : null
            }
            subtitle={metrics.guidance.period ?? undefined}
          />
        )}
      </div>
    </div>
  );
}
