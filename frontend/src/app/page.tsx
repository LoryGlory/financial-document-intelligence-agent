"use client";

import { FileSearch } from "lucide-react";

import { Separator } from "@/components/ui/separator";
import { DocumentSidebar } from "@/components/DocumentSidebar";
import { MetricsDashboard } from "@/components/MetricsDashboard";
import { QueryInterface } from "@/components/QueryInterface";
import { useDocuments } from "@/hooks/useDocuments";

export default function Home() {
  const { documents, selectedDocument, selectedId, setSelectedId, upload, uploading, error } =
    useDocuments();

  return (
    <div className="flex h-screen overflow-hidden">
      <DocumentSidebar
        documents={documents}
        selectedId={selectedId}
        onSelect={setSelectedId}
        onUpload={upload}
        uploading={uploading}
      />

      <main className="flex flex-1 flex-col overflow-hidden">
        {error && (
          <div className="border-b border-red-200 bg-red-50 px-6 py-2 text-sm text-red-700">
            {error}
          </div>
        )}

        {!selectedDocument ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-4 text-slate-400">
            <FileSearch className="h-16 w-16" />
            <div className="text-center">
              <p className="text-lg font-medium">No document selected</p>
              <p className="mt-1 text-sm">Upload a 10-K or earnings report to get started</p>
            </div>
          </div>
        ) : (
          <div className="flex flex-1 flex-col overflow-y-auto">
            {selectedDocument.extraction && (
              <>
                <div className="px-8 py-6">
                  <MetricsDashboard extraction={selectedDocument.extraction} />
                </div>
                <Separator />
              </>
            )}

            <div className="flex-1 px-8 py-6">
              <h3 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-500">
                Ask a Question
              </h3>
              <QueryInterface documentId={selectedDocument.document_id} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
