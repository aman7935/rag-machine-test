import os

import chromadb
import pdfplumber
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

CHAT_MODEL = "openai/gpt-oss-20b"
COLLECTION_NAME = "loan_statement"
MAX_OUTPUT_TOKENS = 512
CONTEXT_BUDGET_CHARS = 40000

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise SystemExit("GROQ_API_KEY not found in .env")


def get_llm():
    return Groq(api_key=GROQ_API_KEY)


def extract_pdf(path):
    text_chunks = []
    table_rows = []

    pdf = pdfplumber.open(path)

    for page_num, page in enumerate(pdf.pages, start=1):
        tables = page.find_tables()

        for table in tables:
            grid = table.extract()

            if len(grid) < 2:
                continue

            header = grid[0]

            for row in grid[1:]:
                table_rows.append({"page": page_num, "header": header, "row": row})

            # for row_1_detection in grid[1::]:
            #     table_rows.append({ page_num,  header,  row})

        x0, top, x1, bottom = page.bbox

        if tables:
            above = page.crop((x0, top, x1, tables[0].bbox[1])).extract_text() or ""

            below = page.crop((x0, tables[-1].bbox[3], x1, bottom)).extract_text() or ""

            text = " ".join(t for t in (above.strip(), below.strip()) if t)
        else:
            text = page.extract_text() or ""

        if text:
            text_chunks.append({"page": page_num, "text": text})

    return text_chunks, table_rows


def row_to_text(header, row):
    pairs = [f"{h.strip()}: {v.strip()}" for h, v in zip(header, row) if h and v]
    return "Transaction — " + ", ".join(pairs)


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


def index_chunks(chunks):
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[str(i)],
            documents=[chunk["text"]],
            metadatas=[chunk["metadata"]],
        )
    return collection


def retrieve_context(collection, question, top_k=10):
    results = collection.query(query_texts=[question], n_results=top_k)
    retrieved = list(results["documents"][0])

    summary_chunks = collection.get(where={"type": "summary_text"})
    existing = set(retrieved)
    for doc in summary_chunks["documents"]:
        if doc not in existing:
            retrieved.append(doc)

    context = ""
    for doc in retrieved:
        if len(context) + len(doc) > CONTEXT_BUDGET_CHARS:
            break
        context += "\n\n" + doc if context else doc
    return context


def build_prompt(question, context):
    return f"""You are a friendly, human bank customer-support agent. Respond ONLY using the facts in the context below.

Rules:
- Write like a real person talking to a customer: warm, natural, complete sentences, no labels or bullet points.
- Use the facts exactly as they appear; don't invent or merge separate fields.
- If the fact isn't in the context, say you don't have that information.
- Keep it short (1-3 sentences) unless the question needs more.

Context:
{context}

Question: {question}
Customer service agent:"""


def stream_answer_question(collection, question, top_k=10):
    """Yield the answer token-by-token as Groq generates it (SSE-ready)."""
    context = retrieve_context(collection, question, top_k)
    prompt = build_prompt(question, context)

    client = get_llm()
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=MAX_OUTPUT_TOKENS,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
