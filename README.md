# Bank Statement Chatbot

A small RAG app that answers questions about PDF bank statements. Upload a PDF, ask something like *"who is the borrower?"*, and it streams back the answer pulled from the statement.

## How it works

```
        PDF
         │
      Docling
         │
┌────────┴─────────┐
│                  │
Text               Tables
 │                  │
└────────┬─────────┘
         │
Structured Markdown
         │
Semantic Chunking
         │
     Embedding
         │
      Chroma
         │
     Retriever
         │
     Reranker
         │  
        LLM
         │
       Answer
```

1. **Read the PDF** – `docling` opens the file and pulls out the text and tables from each page.
2. **Chunk it** – every table row and text block becomes a small chunk of text.
3. **Embed it** – each chunk is turned into a vector using the `BAAI/bge-small-en-v1.5` embedding model.
4. **Store it** – the vectors are saved in ChromaDB (local, in `chroma_db/`).
5. **Answer** – the question is embedded with the same model, ChromaDB finds the most relevant chunks, and those go to Groq's LLM (`openai/gpt-oss-20b`), which writes the answer and streams it back to the page.

## Run the server

```bash
uv sync
uv run --active uvicorn api:app --reload
```

Then open `frontend/index.html` in a browser. The API runs at `http://localhost:8000`.

> I use `uv` to manage packages — faster and cleaner than pip.

## Files

- `main.py` – extraction, chunking, embedding, retrieval, prompting
- `api.py` – FastAPI server (`/upload` and `/ask`)
- `frontend/` – the chat UI
- `chroma_db/` – saved vectors (created on first upload)

## Notes

- The same embedding model is used for both the document and the query, so they always match.
- Set `GROQ_API_KEY` in `.env` (the app reads it from there).
- `HF_HUB_OFFLINE=1` is in `.env`, so the embedding model loads straight from the local cache.