"use client";

import { FileText } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import type { Document } from "@/lib/types";
import { UploadZone } from "./UploadZone";

interface DocumentSidebarProps {
  documents: Document[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onUpload: (file: File) => void;
  uploading: boolean;
}

export function DocumentSidebar({
  documents,
  selectedId,
  onSelect,
  onUpload,
  uploading,
}: DocumentSidebarProps) {
  return (
    <aside className="flex h-full w-72 flex-shrink-0 flex-col border-r bg-slate-50">
      {/* Logo / brand */}
      <div className="flex items-center gap-2.5 border-b px-4 py-3.5">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 32 32"
          fill="none"
          className="h-7 w-7 shrink-0"
          aria-hidden="true"
        >
          <rect x="4" y="2" width="18" height="24" rx="2" fill="url(#sb-grad)" />
          <path d="M18 2 L22 6 L18 6 Z" fill="#0f172a" />
          <rect x="7" y="17" width="3" height="6" rx="0.5" fill="#60a5fa" />
          <rect x="11" y="13" width="3" height="10" rx="0.5" fill="#3b82f6" />
          <rect x="15" y="10" width="3" height="13" rx="0.5" fill="#2563eb" />
          <polyline
            points="7,18 11,14 15,11 18,9"
            stroke="#93c5fd"
            strokeWidth="1.2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <defs>
            <linearGradient
              id="sb-grad"
              x1="4"
              y1="2"
              x2="22"
              y2="26"
              gradientUnits="userSpaceOnUse"
            >
              <stop offset="0%" stopColor="#1e40af" />
              <stop offset="100%" stopColor="#1e3a5f" />
            </linearGradient>
          </defs>
        </svg>
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-slate-800">FinDoc Intelligence</p>
          <p className="truncate text-xs text-slate-400">RAG-powered 10-K analysis</p>
        </div>
      </div>

      <div className="p-4">
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500">
          Documents
        </h2>
        <UploadZone onUpload={onUpload} uploading={uploading} />
      </div>

      <Separator />

      <ScrollArea className="flex-1 p-2">
        {documents.length === 0 ? (
          <p className="p-4 text-center text-xs text-slate-400">No documents yet</p>
        ) : (
          <ul className="space-y-1">
            {documents.map((doc) => (
              <li key={doc.document_id}>
                <button
                  onClick={() => onSelect(doc.document_id)}
                  className={cn(
                    "w-full rounded-lg px-3 py-2.5 text-left transition-colors",
                    selectedId === doc.document_id
                      ? "bg-blue-100 text-blue-900"
                      : "text-slate-700 hover:bg-slate-100"
                  )}
                >
                  <div className="flex items-start gap-2">
                    <FileText className="mt-0.5 h-4 w-4 flex-shrink-0" />
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium">
                        {doc.extraction?.company_name ?? doc.filename}
                      </p>
                      <div className="mt-1 flex items-center gap-1.5">
                        {doc.extraction?.fiscal_year && (
                          <Badge variant="secondary" className="text-xs">
                            {doc.extraction.fiscal_year}
                          </Badge>
                        )}
                        {doc.extraction?.filing_type && (
                          <Badge variant="outline" className="text-xs">
                            {doc.extraction.filing_type}
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>
                </button>
              </li>
            ))}
          </ul>
        )}
      </ScrollArea>
    </aside>
  );
}
