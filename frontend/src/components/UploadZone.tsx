"use client";

import { useCallback, useState } from "react";
import { Upload } from "lucide-react";
import { cn } from "@/lib/utils";

interface UploadZoneProps {
  onUpload: (file: File) => void;
  uploading: boolean;
}

export function UploadZone({ onUpload, uploading }: UploadZoneProps) {
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file?.type === "application/pdf") onUpload(file);
    },
    [onUpload]
  );

  return (
    <label
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-8 transition-colors",
        dragging
          ? "border-blue-500 bg-blue-50"
          : "border-slate-300 hover:border-slate-400 hover:bg-slate-50"
      )}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <input
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onUpload(file);
        }}
        disabled={uploading}
      />
      {uploading ? (
        <>
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent" />
          <p className="text-sm text-slate-600">Processing document…</p>
        </>
      ) : (
        <>
          <Upload className="h-8 w-8 text-slate-400" />
          <div className="text-center">
            <p className="text-sm font-medium text-slate-700">Drop a PDF here</p>
            <p className="mt-1 text-xs text-slate-500">or click to browse</p>
          </div>
        </>
      )}
    </label>
  );
}
