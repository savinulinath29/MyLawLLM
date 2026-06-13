import os
import sys
import subprocess
import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# =============================================================
#  CONFIGURATION
# =============================================================
PDF_DIR = r"D:\KnowUrRights\PDFs"
CHROMA_DIR = r"D:\KnowUrRights\VectorDB"
POPPLER_PATH = r"D:\poppler\poppler-25.12.0\Library\bin"
TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
OCR_DPI = 200
EMBED_MODEL = "all-MiniLM-L6-v2"  # ~90 MB, downloads once, runs fully offline


# =============================================================
#  STARTUP CHECK — fail loudly before touching any PDF
# =============================================================
def validate_tesseract():
    if not os.path.isfile(TESSERACT):
        print(f"FATAL: Tesseract not found at:\n   {TESSERACT}")
        print("   Download: https://github.com/UB-Mannheim/tesseract/wiki")
        sys.exit(1)
    try:
        result = subprocess.run(
            [TESSERACT, "--version"],
            capture_output=True, text=True, timeout=10
        )
        version = (result.stdout or result.stderr).splitlines()[0]
        print(f"Tesseract ready : {version}")
    except Exception as e:
        print(f"FATAL: Tesseract found but won't run: {e}")
        sys.exit(1)
    pytesseract.pytesseract.tesseract_cmd = TESSERACT


# =============================================================
#  PHASE 1 — HYBRID TEXT EXTRACTION  (digital → OCR fallback)
# =============================================================
def extract_text_hybrid(filepath):
    filename = os.path.basename(filepath)
    text_content = []

    try:
        with open(filepath, "rb") as f:
            if f.read(4) != b"%PDF":
                print(f"Not a valid PDF — skipping")
                return None

        reader = PdfReader(filepath)
        total_pages = len(reader.pages)

        for i in range(total_pages):
            page_text = ""

            # Try digital extraction first
            try:
                page_text = reader.pages[i].extract_text() or ""
            except Exception as e:
                print(f"pypdf error page {i + 1}: {type(e).__name__} — falling back to OCR")

            # Fall back to OCR if page is blank / scanned
            if len(page_text.strip()) < 50:
                print(f"  [OCR] page {i + 1}/{total_pages}")
                try:
                    images = convert_from_path(
                        filepath,
                        first_page=i + 1,
                        last_page=i + 1,
                        poppler_path=POPPLER_PATH,
                        dpi=OCR_DPI,
                    )
                    page_text = pytesseract.image_to_string(images[0], lang="eng")
                except FileNotFoundError:
                    print("  FATAL: Tesseract lost — aborting.")
                    sys.exit(1)
                except Exception as e:
                    print(f"  OCR failed page {i + 1}: {e}")
                    page_text = ""

            text_content.append(page_text)

    except Exception as e:
        print(f"  Fatal error reading file: {e}")
        return None

    joined = "\n".join(text_content).strip()
    return joined if joined else None


# =============================================================
#  PHASE 2 — CHUNKING
# =============================================================
def chunk_documents(raw_docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(raw_docs)


# =============================================================
#  PHASE 3 — EMBED & SAVE TO CHROMA
# =============================================================
def save_to_vectordb(chunks):
    print(f"\n  Loading embedding model ({EMBED_MODEL}) ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # If the DB already exists, ADD to it instead of wiping it
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print(f"  Existing VectorDB found — appending {len(chunks):,} new chunks ...")
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        db.add_documents(chunks)
    else:
        print(f"  Creating new VectorDB at {CHROMA_DIR} ...")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
        )

    total = db._collection.count()
    print(f"  VectorDB saved — {total:,} total chunks on disk.")


# =============================================================
#  MAIN
# =============================================================
def main():
    validate_tesseract()

    print("\n" + "=" * 60)
    print("  Phase 1: Hybrid Text Extraction")
    print("=" * 60)

    if not os.path.exists(PDF_DIR):
        print(f"ERROR: PDF folder not found — {PDF_DIR}")
        sys.exit(1)

    pdf_files = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
    total = len(pdf_files)
    print(f"  Found {total} PDF files in {PDF_DIR}\n")

    raw_docs = []
    skipped = []

    for idx, filename in enumerate(pdf_files, 1):
        print(f"[{idx}/{total}]  {filename}")
        filepath = os.path.join(PDF_DIR, filename)
        content = extract_text_hybrid(filepath)

        if content:
            raw_docs.append(Document(
                page_content=content,
                metadata={"source": filename}
            ))
            print(f"{len(content):,} chars extracted")
        else:
            skipped.append(filename)
            print(f"Skipped — no content")

    print("\n" + "=" * 60)
    print("  Phase 1 Summary")
    print("=" * 60)
    print(f"  Extracted : {len(raw_docs)}/{total}")
    if skipped:
        print(f"  Skipped   : {len(skipped)}")
        for s in skipped:
            print(f"    → {s}")

    if not raw_docs:
        print("\nNothing to embed. Exiting.")
        sys.exit(0)

    # ── Phase 2 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Phase 2: Chunking")
    print("=" * 60)
    chunks = chunk_documents(raw_docs)
    print(f"{len(chunks):,} chunks created from {len(raw_docs)} documents")

    # ── Phase 3 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Phase 3: Embedding & Saving to VectorDB")
    print("=" * 60)
    save_to_vectordb(chunks)

    print("\n" + "=" * 60)
    print("  ALL DONE")
    print("=" * 60)
    print(f"  VectorDB location : {CHROMA_DIR}")
    print(f"  Total chunks      : {len(chunks):,}")
    print(f"  Documents         : {len(raw_docs)}")


if __name__ == "__main__":
    main()
