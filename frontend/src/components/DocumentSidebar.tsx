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
