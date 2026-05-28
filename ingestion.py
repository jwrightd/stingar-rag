import os
import re
import tempfile
import requests
import pdfplumber
import chromadb
from pathlib import Path
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

SECTION_NAMES = {
    "abstract", "introduction", "related work", "background",
    "method", "methods", "methodology", "approach", "model",
    "experiments", "experimental setup", "results", "evaluation",
    "discussion", "conclusion", "conclusions", "limitations",
    "future work",
}


def _is_section_header(line: str) -> str | None:
    """Return the section name if the line looks like a section header, else None."""
    stripped = line.strip()
    if not stripped:
        return None
    # ALL CAPS line (short)
    if stripped.isupper() and 3 < len(stripped) < 60:
        return stripped.title()
    # Matches known section names (with optional numbering like "1. Introduction")
    cleaned = re.sub(r"^\d+[\.\s]+", "", stripped).lower().rstrip(".")
    if cleaned in SECTION_NAMES:
        return stripped.title()
    return None


def _chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # Start new chunk with overlap from end of previous
            current = (current[-overlap:] + " " + sentence).strip() if current else sentence
    if current:
        chunks.append(current)
    return chunks


def ingest_paper(pdf_url: str, arxiv_id: str):
    """
    Download a PDF, extract text by section, chunk it, embed it,
    and store in a fresh ChromaDB in-memory collection.
    Returns (collection, list_of_chunk_dicts).
    """
    # Download PDF to a temp file
    response = requests.get(pdf_url, timeout=30)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(response.content)
        tmp_path = f.name

    # Extract text page by page, detect sections
    sections = []
    current_section = "Abstract"
    current_text = ""
    current_page = 1

    with pdfplumber.open(tmp_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for line in text.split("\n"):
                header = _is_section_header(line)
                if header:
                    if current_text.strip():
                        sections.append({
                            "section": current_section,
                            "text": current_text.strip(),
                            "page": current_page,
                        })
                    current_section = header
                    current_text = ""
                    current_page = page_num
                else:
                    current_text += " " + line

    # Flush final section
    if current_text.strip():
        sections.append({
            "section": current_section,
            "text": current_text.strip(),
            "page": current_page,
        })

    os.unlink(tmp_path)

    # Chunk all sections
    all_chunks = []
    chunk_index = 0
    for section in sections:
        for chunk_text in _chunk_text(section["text"]):
            if len(chunk_text.strip()) < 50:
                continue
            all_chunks.append({
                "id": f"{arxiv_id}_chunk_{chunk_index}",
                "text": chunk_text,
                "metadata": {
                    "section": section["section"],
                    "page": section["page"],
                    "arxiv_id": arxiv_id,
                    "chunk_index": chunk_index,
                },
            })
            chunk_index += 1

    # Store in ChromaDB
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small",
    )
    chroma_client = chromadb.Client()
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", arxiv_id)
    collection = chroma_client.get_or_create_collection(
        name=safe_id,
        embedding_function=embedding_fn,
    )

    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        metadatas=[c["metadata"] for c in all_chunks],
    )

    # Extract figures from arXiv HTML for use in architecture slide
    from figures import extract_paper_figures
    out_dir = Path("output") / arxiv_id
    figures = extract_paper_figures(arxiv_id, out_dir)

    return collection, all_chunks, figures
