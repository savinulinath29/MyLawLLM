import os
import pickle
import logging
from typing import List
from dotenv import load_dotenv

# PDF and Text extraction
import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Store
try:
    from langchain_qdrant import QdrantVectorStore
except ImportError:
    try:
        from langchain_qdrant import Qdrant as QdrantVectorStore
    except ImportError:
        from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Hybrid Search (BM25)
from rank_bm25 import BM25Okapi

# Load environment
load_dotenv()

# Configuration
PDF_DIR = "PDFs"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "legal_docs")

# Paths for local hybrid search data
BM25_PATH = "bm25_index.pkl"
TEXTS_PATH = "all_texts.pkl"

# Model Config
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# OCR Config (Fallback for scanned PDFs)
POPPLER_PATH = r"D:\poppler\poppler-25.12.0\Library\bin" # Update if different
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SyncToQdrant")

def extract_text_from_pdf(filepath: str) -> str:
    """Extract text using digital retrieval with OCR fallback."""
    try:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if not page_text or len(page_text.strip()) < 50:
                # Scanned page likely, skipping digital for this page and using OCR if available
                # Selective OCR could be done here, but for simplicity we join digital text first
                pass
            else:
                text += page_text + "\n"
        
        # If total extraction is weak, try OCR on the whole thing (simplified for script)
        if len(text.strip()) < 200:
            logger.info(f"Low text density for {os.path.basename(filepath)}, attempting OCR...")
            images = convert_from_path(filepath, poppler_path=POPPLER_PATH)
            text = ""
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"
                
        return text.strip()
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        return ""

def main():
    if not QDRANT_URL or not QDRANT_API_KEY:
        logger.error("Missing QDRANT_URL or QDRANT_API_KEY in .env file")
        return

    # 1. Load Documents
    if not os.path.exists(PDF_DIR):
        logger.error(f"PDF directory {PDF_DIR} not found.")
        return

    pdfs = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    logger.info(f"Found {len(pdfs)} PDF files.")

    documents = []
    for pdf in pdfs:
        logger.info(f"Extracting text from {pdf}...")
        text = extract_text_from_pdf(os.path.join(PDF_DIR, pdf))
        if text:
            documents.append(Document(page_content=text, metadata={"source": pdf}))
    
    if not documents:
        logger.error("No text extracted from documents.")
        return

    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")

    # 3. Embeddings & Qdrant Upload
    logger.info("Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    logger.info(f"Connecting to Qdrant at {QDRANT_URL}...")
    is_cloud = "qdrant.io" in QDRANT_URL
    client = QdrantClient(
        url=QDRANT_URL, 
        port=443 if is_cloud else 6333, 
        api_key=QDRANT_API_KEY,
        timeout=120.0,
    )

    # Create collection if it doesn't exist
    logger.info(f"Checking collection: {QDRANT_COLLECTION}")
    collections = client.get_collections().collections
    exists = any(c.name == QDRANT_COLLECTION for c in collections)
    
    if not exists:
        logger.info(f"Creating collection: {QDRANT_COLLECTION}")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )
    else:
        logger.info(f"Collection {QDRANT_COLLECTION} already exists. Appending/Updating data.")

    logger.info(f"Uploading {len(chunks)} chunks to Qdrant...")
    
    # Initialize Vector Store using our explicit client (handling API variations)
    try:
        qdrant = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION,
            embedding=embeddings,
        )
    except TypeError:
        # Fallback for langchain_community versions
        qdrant = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION,
            embeddings=embeddings,
        )
    
    # Upload in smaller batches to avoid httpx ReadTimeout errors
    batch_size = 32
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        qdrant.add_documents(batch)
        logger.info(f"Uploaded batch {i // batch_size + 1} of {total_batches}")
        
    logger.info("Upload to Qdrant Cloud complete.")

    # 4. Update Local BM25 Index (for Hybrid Search)
    logger.info("Updating local BM25 index...")
    all_texts = [doc.page_content for doc in chunks]
    all_metadatas = [doc.metadata for doc in chunks]
    
    tokenized_corpus = [text.lower().split() for text in all_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump((all_texts, all_metadatas), f)

    logger.info("Local BM25 index saved successfully.")
    logger.info("SUCCESS: All systems synced.")

if __name__ == "__main__":
    main()
