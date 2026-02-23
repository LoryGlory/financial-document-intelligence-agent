import re
import uuid
from dataclasses import dataclass

import pdfplumber

from app.models.document import Chunk

SECTION_PATTERNS = [
    r"^item\s+\d+[a-z]?\.\s+.+",
    r"^management.s discussion",
    r"^notes to (consolidated )?financial",
    r"^selected financial data",
    r"^quantitative and qualitative",
    r"^controls and procedures",
]

SECTION_REGEX = re.compile("|".join(SECTION_PATTERNS), re.MULTILINE | re.IGNORECASE)


@dataclass
class Section:
    title: str
    content: str
    page_number: int


@dataclass
class PdfProcessor:
    max_chunk_tokens: int = 800
    overlap_tokens: int = 100

    def extract_text(self, pdf_path: str) -> tuple[str, dict[int, int]]:
        full_text = ""
        page_char_map: dict[int, int] = {}
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                page_char_map[len(full_text)] = page_num
                full_text += page_text + "\n"
        return full_text, page_char_map

    def detect_sections(self, text: str) -> list[Section]:
        matches = list(SECTION_REGEX.finditer(text))
        if not matches:
            return [Section(title="Document", content=text.strip(), page_number=1)]
        sections: list[Section] = []
        for i, match in enumerate(matches):
            title = match.group(0).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            if content:
                sections.append(Section(title=title, content=content, page_number=1))
        return sections

    def chunk_section(self, section: Section, document_id: str, chunk_offset: int) -> list[Chunk]:
        words = section.content.split()
        approx_max_words = int(self.max_chunk_tokens * 0.75)
        approx_overlap_words = int(self.overlap_tokens * 0.75)
        if len(words) <= approx_max_words:
            return [
                Chunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    section_title=section.title,
                    section_type=self._classify_section(section.title),
                    text=section.content,
                    page_number=section.page_number,
                    chunk_index=chunk_offset,
                )
            ]
        chunks: list[Chunk] = []
        step = max(1, approx_max_words - approx_overlap_words)
        starts = range(0, len(words), step)
        for chunk_idx, start in enumerate(starts):
            chunk_words = words[start : start + approx_max_words]
            if not chunk_words:
                break
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    section_title=section.title,
                    section_type=self._classify_section(section.title),
                    text=" ".join(chunk_words),
                    page_number=section.page_number,
                    chunk_index=chunk_offset + chunk_idx,
                )
            )
        return chunks

    def process(self, pdf_path: str, document_id: str) -> list[Chunk]:
        text, _ = self.extract_text(pdf_path)
        sections = self.detect_sections(text)
        all_chunks: list[Chunk] = []
        chunk_offset = 0
        for section in sections:
            chunks = self.chunk_section(section, document_id, chunk_offset)
            all_chunks.extend(chunks)
            chunk_offset += len(chunks)
        return all_chunks

    @staticmethod
    def _classify_section(title: str) -> str:
        title_lower = title.lower()
        if "risk factor" in title_lower:
            return "risk_factors"
        if "management" in title_lower or "md&a" in title_lower:
            return "mda"
        if "financial statement" in title_lower:
            return "financial_statements"
        if "business" in title_lower:
            return "business"
        if "note" in title_lower:
            return "notes"
        return "other"
