# AI Document Assistant

A single-file Streamlit app that lets you upload documents (or load them from Google Drive),
then ask questions answered **only** from those documents, with sources shown underneath.

```
upload / Drive → extract → chunk → embed → FAISS + keyword hybrid search → Groq answer → sources
```

## Setup

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` (never put keys in `app.py`):

```toml
GROQ_API_KEY = "gsk_your_key_here"

# Optional — only needed for Google Drive *folder* links
GDRIVE_API_KEY = "your_google_api_key"
```

Run it:

```bash
streamlit run app.py
```

## How it works

**1. Extraction** — one function per format, all returning the same shape
`{text, filename, page}`:

| Format | Library | Page numbers |
|---|---|---|
| PDF | `pypdf` | yes, one record per page |
| DOCX | `python-docx` | no (`page = None`), paragraphs + tables |
| TXT | built-in | no |
| MD | built-in | no, markup kept as-is |

**2. Chunking** — `chunk_text()` cuts the text into ~900-character windows with a
150-character overlap so a sentence split across two chunks still has context on both
sides. Chunks try to end on a sentence boundary. Every chunk keeps its `filename` and
`page`. The sidebar shows the chunk count per document and in total.

**3. Embeddings** — `all-MiniLM-L6-v2` from `sentence-transformers`. The model is loaded
once with `@st.cache_resource`. Vectors are normalized and stored in
`st.session_state.vectors` alongside the chunk metadata.

**4. FAISS search** — an `IndexFlatIP` index over normalized vectors, so inner product
equals cosine similarity. Your question is embedded and matched against the index.

**5. Keyword search** — the question is stripped of stopwords, and each chunk is scored by
the fraction of remaining keywords it contains. This catches exact terms (names, codes,
acronyms) that embeddings sometimes blur.

**6. Hybrid search** — both score sets are normalized to 0–1 and blended
(`0.7 × semantic + 0.3 × keyword`). The top 4 chunks are returned with their metadata and
per-method scores.

**7. Answering** — the retrieved chunks are sent to Groq (`llama-3.3-70b-versatile`) with a
system prompt that forbids outside knowledge. If the answer isn't in the context, the model
replies *"The information is not available in the provided documents."*

**8. Sources** — every answer is followed by expandable source cards showing the filename,
page number (when available), the hybrid score, and the exact retrieved text.

## Google Drive

Paste either:

- a **file** link — `https://drive.google.com/file/d/FILE_ID/view`
  Works with no API key as long as the file is shared with "anyone with the link".
- a **folder** link — `https://drive.google.com/drive/folders/FOLDER_ID`
  Requires `GDRIVE_API_KEY` in secrets (enable the Drive API in Google Cloud and create an
  API key). Only `.pdf`, `.docx`, `.txt`, `.md` files in the folder are loaded.

Drive files go through exactly the same extract → chunk → embed → search pipeline as local
uploads. Local upload keeps working independently.

## Processing happens once

`st.session_state.processed` maps each filename to its chunk count. `add_documents()` skips
any file already in that map, so re-running or adding a second batch only embeds the new
chunks — existing vectors are reused and the FAISS index is rebuilt from the stored array.
Asking a question never re-embeds documents; it only embeds the short question string.

Use **Clear everything** in the sidebar to reset the index.

## Tuning

Constants at the top of `app.py`:

| Name | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` | 900 | bigger = more context per chunk, less precise retrieval |
| `CHUNK_OVERLAP` | 150 | bigger = less chance of splitting an answer |
| `TOP_K` | 4 | how many chunks are sent to the model |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | swap for a smaller/faster Groq model |

Search weights live in `hybrid_search(semantic_weight=0.7, keyword_weight=0.3)`.

## Notes

- Everything is in memory, so the index resets when the Streamlit session ends. To persist,
  write `st.session_state.vectors` and `chunks` to disk with `faiss.write_index()` and a
  JSON file.
- The first run downloads the embedding model (~90 MB).
