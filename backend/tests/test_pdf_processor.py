from app.services.pdf_processor import PdfProcessor, Section


def test_detect_10k_sections():
    processor = PdfProcessor(max_chunk_tokens=800, overlap_tokens=100)
    text = """
UNITED STATES SECURITIES AND EXCHANGE COMMISSION

Item 1. Business
Apple Inc. designs, manufactures and markets smartphones.

Item 1A. Risk Factors
The company operates in a highly competitive market.

Item 7. Management's Discussion and Analysis
Revenue for fiscal 2024 was $391 billion.
"""
    sections = processor.detect_sections(text)
    titles = [s.title for s in sections]
    assert "Item 1. Business" in titles
    assert "Item 1A. Risk Factors" in titles
    assert "Item 7. Management's Discussion and Analysis" in titles


def test_detect_sections_returns_content():
    processor = PdfProcessor(max_chunk_tokens=800, overlap_tokens=100)
    text = """
Item 1. Business
Apple Inc. designs smartphones.

Item 1A. Risk Factors
Competition is intense.
"""
    sections = processor.detect_sections(text)
    business = next(s for s in sections if "Business" in s.title)
    assert "Apple Inc." in business.content


def test_chunk_short_section_returns_one_chunk():
    processor = PdfProcessor(max_chunk_tokens=800, overlap_tokens=100)
    section = Section(title="Risk Factors", content="Short content.", page_number=5)
    chunks = processor.chunk_section(section, document_id="doc-1", chunk_offset=0)
    assert len(chunks) == 1
    assert chunks[0].section_title == "Risk Factors"
    assert chunks[0].document_id == "doc-1"


def test_chunk_long_section_produces_multiple_chunks():
    processor = PdfProcessor(max_chunk_tokens=50, overlap_tokens=10)
    long_content = " ".join(["word"] * 200)
    section = Section(title="MD&A", content=long_content, page_number=10)
    chunks = processor.chunk_section(section, document_id="doc-1", chunk_offset=0)
    assert len(chunks) > 1


def test_chunks_have_sequential_indices():
    processor = PdfProcessor(max_chunk_tokens=50, overlap_tokens=10)
    long_content = " ".join(["word"] * 200)
    section = Section(title="MD&A", content=long_content, page_number=10)
    chunks = processor.chunk_section(section, document_id="doc-1", chunk_offset=0)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
