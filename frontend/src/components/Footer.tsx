import { ExternalLink } from "lucide-react";

export function Footer() {
  return (
    <footer className="flex items-center justify-center border-t bg-slate-50 px-6 py-2.5 text-xs text-slate-400">
      <span>
        Created with Claude as a study project by&nbsp;
        <a
          href="https://www.linkedin.com/in/laura-roganovic/"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-0.5 font-medium text-slate-600 underline-offset-2 hover:text-blue-600 hover:underline"
        >
          Laura Roganovic
          <ExternalLink className="h-3 w-3" />
        </a>
      </span>
    </footer>
  );
}
