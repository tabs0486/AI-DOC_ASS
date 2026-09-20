"""
Simple AI Document Assistant
----------------------------
Pipeline:  upload / Google Drive  ->  extract  ->  chunk  ->  embed  ->  hybrid search  ->  Groq answer

Everything heavy (embeddings, FAISS index) is built ONCE and kept in st.session_state,
so asking a question never re-processes the documents.
"""

import io
import os
import re
import json

import faiss
import numpy as np
import requests
import streamlit as st
from docx import Document as DocxDocument
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 900          # characters per chunk
CHUNK_OVERLAP = 150       # characters shared between neighbouring chunks
TOP_K = 4                 # chunks sent to the LLM
SUPPORTED = (".pdf", ".docx", ".txt", ".md")

st.set_page_config(page_title="AI Document Assistant", page_icon="📄", layout="wide")


# ----------------------------------------------------------------------------
# 1. TEXT EXTRACTION  (one small function per file type)
# ----------------------------------------------------------------------------
def extract_pdf(file_bytes, filename):
    """Return a list of {text, filename, page} — one entry per PDF page."""
    pages = []
    reader = PdfReader(io.BytesIO(file_bytes))
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"text": text, "filename": filename, "page": i})
    return pages


def extract_docx(file_bytes, filename):
    """DOCX has no real pages, so we return a single block with page = None."""
    doc = DocxDocument(io.BytesIO(file_bytes))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]

    # also pull text out of tables
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    text = "\n".join(parts).strip()
    return [{"text": text, "filename": filename, "page": None}] if text else []


def extract_txt(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="ignore").strip()
    return [{"text": text, "filename": filename, "page": None}] if text else []


def extract_md(file_bytes, filename):
    # Markdown is plain text; we keep the markup, it carries structure.
    return extract_txt(file_bytes, filename)


def extract_document(file_bytes, filename):
    """Router: pick the right extractor based on the file extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return extract_pdf(file_bytes, filename)
    if ext == ".docx":
        return extract_docx(file_bytes, filename)
    if ext == ".txt":
        return extract_txt(file_bytes, filename)
    if ext == ".md":
        return extract_md(file_bytes, filename)
    return []


# ----------------------------------------------------------------------------
# 2. CHUNKING  (overlapping windows, metadata carried along)
# ----------------------------------------------------------------------------
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split a string into overlapping character windows."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]

        # try to end on a sentence boundary so chunks read naturally
        if end < len(text):
            cut = max(chunk.rfind(". "), chunk.rfind("? "), chunk.rfind("! "))
            if cut > size * 0.5:
                chunk = chunk[: cut + 1]
                end = start + cut + 1

        chunks.append(chunk.strip())
        start = max(end - overlap, start + 1)

    return [c for c in chunks if c]


def build_chunks(pages):
    """Turn extracted pages into chunk records that keep filename + page."""
    records = []
    for page in pages:
        for piece in chunk_text(page["text"]):
            records.append(
                {
                    "text": piece,
                    "filename": page["filename"],
                    "page": page["page"],
                }
            )
    return records


# ----------------------------------------------------------------------------
# 3. EMBEDDINGS + FAISS INDEX  (built once, cached)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    """Loaded once per session, shared by every call."""
    return SentenceTransformer(EMBED_MODEL_NAME)


def embed_texts(texts):
    model = get_embedder()
    vectors = model.encode(texts, batch_size=32, show_progress_bar=False,
                           convert_to_numpy=True, normalize_embeddings=True)
    return vectors.astype("float32")


def build_faiss_index(vectors):
    """Inner product on normalized vectors == cosine similarity."""
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


# ----------------------------------------------------------------------------
# 4. SEARCH:  semantic (FAISS) + keyword, combined into hybrid
# ----------------------------------------------------------------------------
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
    "for", "on", "with", "as", "at", "by", "it", "this", "that", "be", "from",
    "what", "which", "who", "how", "when", "where", "why", "do", "does", "did",
    "can", "could", "should", "would", "about", "tell", "me", "please", "give",
}


def keywords_of(question):
    """Important words = lowercase tokens longer than 2 chars, minus stopwords."""
    tokens = re.findall(r"[a-z0-9]+", question.lower())
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


def semantic_search(question, index, n):
    """Return {chunk_index: cosine_score}."""
    qvec = embed_texts([question])
    scores, ids = index.search(qvec, min(n, index.ntotal))
    return {int(i): float(s) for i, s in zip(ids[0], scores[0]) if i != -1}


def keyword_search(question, chunks, n):
    """Score = share of the question's keywords that appear in the chunk."""
    words = keywords_of(question)
    if not words:
        return {}

    scored = {}
    for i, chunk in enumerate(chunks):
        lower = chunk["text"].lower()
        hits = sum(1 for w in words if w in lower)
        if hits:
            scored[i] = hits / len(words)

    top = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return dict(top)


def normalize(scores):
    """Scale a score dict into 0..1 so the two searches are comparable."""
    if not scores:
        return {}
    values = list(scores.values())
    low, high = min(values), max(values)
    if high == low:
        return {k: 1.0 for k in scores}
    return {k: (v - low) / (high - low) for k, v in scores.items()}


def hybrid_search(question, chunks, index, top_k=TOP_K,
                  semantic_weight=0.7, keyword_weight=0.3):
    """Merge both rankings, return the best chunks with their metadata."""
    pool = max(top_k * 3, 10)
    sem = normalize(semantic_search(question, index, pool))
    key = normalize(keyword_search(question, chunks, pool))

    combined = {}
    for i in set(sem) | set(key):
        combined[i] = semantic_weight * sem.get(i, 0.0) + keyword_weight * key.get(i, 0.0)

    best = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    results = []
    for i, score in best:
        record = dict(chunks[i])          # keeps text, filename, page
        record["score"] = round(score, 3)
        record["semantic"] = round(sem.get(i, 0.0), 3)
        record["keyword"] = round(key.get(i, 0.0), 3)
        results.append(record)
    return results


# ----------------------------------------------------------------------------
# 5. GROQ ANSWER  (context only, no outside knowledge)
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a document assistant. Answer ONLY using the context provided below. "
    "Do not use outside knowledge and do not guess. "
    "If the answer is not in the context, reply exactly: "
    "'The information is not available in the provided documents.' "
    "Cite the filename (and page, when given) inside your answer."
)


def get_groq_client():
    """API key comes from Streamlit secrets — never hardcoded."""
    key = st.secrets.get("GROQ_API_KEY")
    if not key:
        st.error("GROQ_API_KEY is missing. Add it to .streamlit/secrets.toml")
        st.stop()
    return Groq(api_key=key)


def format_context(results):
    blocks = []
    for n, r in enumerate(results, start=1):
        where = r["filename"] + (f", page {r['page']}" if r["page"] else "")
        blocks.append(f"[Source {n} — {where}]\n{r['text']}")
    return "\n\n".join(blocks)


def answer_question(question, results):
    client = get_groq_client()
    user_msg = f"Context:\n\n{format_context(results)}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return response.choices[0].message.content


# ----------------------------------------------------------------------------
# 6. GOOGLE DRIVE LOADING
# ----------------------------------------------------------------------------
def parse_drive_id(link):
    """Pull the file/folder id out of any common Drive URL shape."""
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"/folders/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
        r"/document/d/([a-zA-Z0-9_-]+)",
    ]
    for p in patterns:
        m = re.search(p, link)
        if m:
            return m.group(1)
    # user may have pasted a bare id
    return link.strip() if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", link.strip()) else None


def drive_api_key():
    return st.secrets.get("GDRIVE_API_KEY")


def drive_list_folder(folder_id):
    """List supported files inside a public folder. Needs GDRIVE_API_KEY."""
    key = drive_api_key()
    if not key:
        raise RuntimeError(
            "Folder links need GDRIVE_API_KEY in secrets. "
            "A single file link works without it."
        )
    url = "https://www.googleapis.com/drive/v3/files"
    params = {
        "q": f"'{folder_id}' in parents and trashed=false",
        "key": key,
        "fields": "files(id,name,mimeType)",
        "pageSize": 200,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    files = r.json().get("files", [])
    return [f for f in files if f["name"].lower().endswith(SUPPORTED)]


def drive_download(file_id, filename=None):
    """Download a public Drive file. Returns (filename, bytes)."""
    key = drive_api_key()
    if key:
        meta = requests.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            params={"key": key, "fields": "name"}, timeout=30,
        )
        meta.raise_for_status()
        filename = filename or meta.json().get("name", f"{file_id}.pdf")
        data = requests.get(
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            params={"key": key, "alt": "media"}, timeout=60,
        )
    else:
        # anyone-with-the-link direct download, no API key required
        data = requests.get(
            "https://drive.google.com/uc",
            params={"export": "download", "id": file_id}, timeout=60,
        )
        if filename is None:
            match = re.search(r'filename="([^"]+)"',
                              data.headers.get("content-disposition", ""))
            filename = match.group(1) if match else f"{file_id}.pdf"

    data.raise_for_status()
    return filename, data.content


def load_from_drive(link):
    """Return a list of (filename, bytes) for a file OR folder link."""
    drive_id = parse_drive_id(link)
    if not drive_id:
        raise ValueError("Could not read a file or folder id from that link.")

    if "/folders/" in link:
        return [drive_download(f["id"], f["name"]) for f in drive_list_folder(drive_id)]
    return [drive_download(drive_id)]


# ----------------------------------------------------------------------------
# 7. INDEXING — runs once per file, skips anything already processed
# ----------------------------------------------------------------------------
def init_state():
    st.session_state.setdefault("chunks", [])
    st.session_state.setdefault("vectors", None)
    st.session_state.setdefault("index", None)
    st.session_state.setdefault("processed", {})   # filename -> chunk count


def add_documents(files):
    """files: list of (filename, bytes). Embeds only the new ones."""
    new_chunks = []
    skipped = []

    for filename, data in files:
        if filename in st.session_state.processed:
            skipped.append(filename)
            continue
        pages = extract_document(data, filename)
        chunks = build_chunks(pages)
        if not chunks:
            st.warning(f"No text found in {filename}")
            continue
        st.session_state.processed[filename] = len(chunks)
        new_chunks.extend(chunks)

    if skipped:
        st.info("Already processed (embeddings reused): " + ", ".join(skipped))
    if not new_chunks:
        return 0

    with st.spinner(f"Embedding {len(new_chunks)} new chunks..."):
        vectors = embed_texts([c["text"] for c in new_chunks])

    if st.session_state.vectors is None:
        st.session_state.vectors = vectors
    else:
        st.session_state.vectors = np.vstack([st.session_state.vectors, vectors])

    st.session_state.chunks.extend(new_chunks)
    st.session_state.index = build_faiss_index(st.session_state.vectors)
    return len(new_chunks)


# ----------------------------------------------------------------------------
# 8. UI
# ----------------------------------------------------------------------------
init_state()

st.title("📄 AI Document Assistant")
st.caption("Upload documents or load them from Google Drive, then ask questions about them.")

with st.sidebar:
    st.header("1 · Add documents")

    uploaded = st.file_uploader(
        "Local files", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True
    )
    if uploaded and st.button("Process uploads", use_container_width=True):
        added = add_documents([(f.name, f.read()) for f in uploaded])
        if added:
            st.success(f"Added {added} new chunks.")

    st.divider()
    st.subheader("Google Drive")
    drive_link = st.text_input("Paste a file or folder link")
    if drive_link and st.button("Load from Drive", use_container_width=True):
        try:
            with st.spinner("Downloading from Drive..."):
                files = [f for f in load_from_drive(drive_link)
                         if f[0].lower().endswith(SUPPORTED)]
            if not files:
                st.warning("No supported files found at that link.")
            else:
                added = add_documents(files)
                if added:
                    st.success(f"Loaded {len(files)} file(s), {added} new chunks.")
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.header("2 · Index status")
    st.metric("Documents", len(st.session_state.processed))
    st.metric("Total chunks", len(st.session_state.chunks))

    if st.session_state.processed:
        with st.expander("Chunks per document"):
            for name, count in st.session_state.processed.items():
                st.write(f"• **{name}** — {count} chunks")

    if st.button("Clear everything", use_container_width=True):
        for k in ("chunks", "vectors", "index", "processed"):
            st.session_state.pop(k, None)
        init_state()
        st.rerun()

# ---- question area -----------------------------------------------------------
if not st.session_state.chunks:
    st.info("Add at least one document from the sidebar to get started.")
    st.stop()

st.success(
    f"Ready — {len(st.session_state.chunks)} chunks indexed "
    f"from {len(st.session_state.processed)} document(s)."
)

question = st.text_input("Ask a question about your documents")

if question:
    with st.spinner("Searching..."):
        results = hybrid_search(question, st.session_state.chunks, st.session_state.index)

    if not results:
        st.warning("Nothing relevant found.")
    else:
        with st.spinner("Thinking..."):
            answer = answer_question(question, results)

        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Sources")
        for n, r in enumerate(results, start=1):
            where = r["filename"] + (f" — page {r['page']}" if r["page"] else "")
            with st.expander(f"Source {n}: {where}  ·  score {r['score']}"):
                st.caption(
                    f"semantic {r['semantic']} · keyword {r['keyword']}"
                )
                st.write(r["text"])
