import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import type { Citation } from "@/lib/types";

interface CitationCardProps {
  citation: Citation;
  index: number;
}

function scoreColor(score: number) {
  if (score >= 0.8) return "border-emerald-200 bg-emerald-50 text-emerald-700";
  if (score >= 0.6) return "border-amber-200 bg-amber-50 text-amber-700";
  return "border-slate-200 bg-slate-50 text-slate-600";
}

export function CitationCard({ citation, index }: CitationCardProps) {
  return (
    <Card className={`border ${scoreColor(citation.score)}`}>
      <CardContent className="p-3">
        <div className="mb-2 flex items-start justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full bg-slate-700 text-xs font-bold text-white">
              {index}
            </span>
            <span className="text-xs font-semibold">{citation.section_title}</span>
          </div>
          <Badge variant="outline" className="flex-shrink-0 text-xs">
            {(citation.score * 100).toFixed(0)}% match
          </Badge>
        </div>
        <p className="line-clamp-3 text-sm leading-relaxed">{citation.excerpt}</p>
      </CardContent>
    </Card>
  );
}
