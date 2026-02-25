"use client";

import { useCallback, useState } from "react";

import { extractMetrics, ingestDocument } from "@/lib/api";
import type { Document } from "@/lib/types";

export function useDocuments() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const upload = useCallback(async (file: File) => {
    setUploading(true);
    setError(null);
    try {
      const ingestResult = await ingestDocument(file);
      const extraction = await extractMetrics(ingestResult.document_id);
      const doc: Document = { ...ingestResult, extraction };
      setDocuments((prev) => [doc, ...prev]);
      setSelectedId(doc.document_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }, []);

  const selectedDocument = documents.find((d) => d.document_id === selectedId);

  return {
    documents,
    selectedDocument,
    selectedId,
    setSelectedId,
    upload,
    uploading,
    error,
  };
}
