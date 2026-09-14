"""
Simple local RAG chatbot for structured/tabular PDFs (e.g. bank loan statements).

Stack (all local, no API keys):
  - pdfplumber   -> pulls tables out as real rows/columns, not flat text
  - Ollama       -> local embedding model + local LLM
  - Chroma       -> local vector store (just a folder on disk)

Setup (run once in your terminal, not in this script):
  ollama pull nomic-embed-text      # embedding model
  ollama pull llama3.1              # or qwen2.5, mistral, whatever you have pulled
  pip install pdfplumber chromadb ollama
"""

import chromadb
import ollama
import pdfplumber

PDF_PATH = "loan_statement.pdf"  # change to your file
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1"
COLLECTION_NAME = "loan_statement"


# ---------------------------------------------------------------------------
# STEP 1: Extract text + tables SEPARATELY
# ---------------------------------------------------------------------------
# The whole trick for tabular PDFs: don't let pdfplumber's plain .extract_text()
# flatten a table into a wall of numbers. Pull tables with .extract_tables()
# so you keep the row/column structure, and only use extract_text() for the
# surrounding narrative text (headers, notes, etc).
def extract_pdf(path):
    text_chunks = []
    table_rows = []  # list of dicts: {page, header, row}

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()

            # keep the header row of each table so every chunk can carry it
            for table in tables:
                if not table or len(table) < 2:
                    continue
                header = table[0]
                for row in table[1:]:
                    table_rows.append({"page": page_num, "header": header, "row": row})

            # grab non-table text too (loan summary fields etc. that aren't
            # inside a detected table)
            page_text = page.extract_text() or ""
            text_chunks.append({"page": page_num, "text": page_text})

    return text_chunks, table_rows


# ---------------------------------------------------------------------------
# STEP 2: Turn each table row into a markdown-style sentence with headers attached
# ---------------------------------------------------------------------------
# This is the part that actually matters. A raw row like
#   ['07/06/2025', 'Payment Received', '07/06/2025', 'S116456321/1-52', '13,434.00 CR', '0.00 DR']
# means nothing to an embedding model on its own. Pairing it with the header
# turns it into something the model (and the LLM later) can actually read.
def row_to_text(header, row):
    pairs = [f"{h.strip()}: {v.strip()}" for h, v in zip(header, row) if h and v]
    return "Transaction — " + ", ".join(pairs)


# ---------------------------------------------------------------------------
# STEP 3: Chunk by logical unit (here: one chunk per row, since each row is
# already a complete, self-contained fact). For plain narrative text, you'd
# chunk by paragraph or section instead.
# ---------------------------------------------------------------------------
def build_chunks(text_chunks, table_rows):
    chunks = []

    for row_info in table_rows:
        chunks.append(
            {
                "text": row_to_text(row_info["header"], row_info["row"]),
                "metadata": {"page": row_info["page"], "type": "transaction"},
            }
        )

    for tc in text_chunks:
        if tc["text"].strip():
            chunks.append(
                {
                    "text": tc["text"],
                    "metadata": {"page": tc["page"], "type": "summary_text"},
                }
            )

    return chunks


# ---------------------------------------------------------------------------
# STEP 4: Embed with Ollama's local embedding model and store in Chroma
# ---------------------------------------------------------------------------
def index_chunks(chunks):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(COLLECTION_NAME)

    for i, chunk in enumerate(chunks):
        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=chunk["text"])[
            "embedding"
        ]
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[chunk["metadata"]],
        )
    return collection


# ---------------------------------------------------------------------------
# STEP 5: Retrieve + generate
# ---------------------------------------------------------------------------
def answer_question(collection, question, top_k=5):
    q_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=question)["embedding"]
    results = collection.query(query_embeddings=[q_embedding], n_results=top_k)
    retrieved = results["documents"][0]

    context = "\n".join(retrieved)
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't have that information.
Quote the exact numbers/dates you used.

Context:
{context}

Question: {question}
Answer:"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"], retrieved


# ---------------------------------------------------------------------------
# Run it
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Extracting PDF...")
    text_chunks, table_rows = extract_pdf(PDF_PATH)
    print(f"Found {len(table_rows)} table rows and {len(text_chunks)} text blocks")

    print("Building chunks...")
    chunks = build_chunks(text_chunks, table_rows)

    print("Embedding + indexing (this calls Ollama once per chunk)...")
    collection = index_chunks(chunks)

    print("\nReady. Ask questions (type 'exit' to quit).\n")
    while True:
        q = input("You: ")
        if q.strip().lower() == "exit":
            break
        answer, sources = answer_question(collection, q)
        print(f"\nBot: {answer}\n")
