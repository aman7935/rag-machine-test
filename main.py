import os

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

converter = DocumentConverter()
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5"
)


def get_llm():
    return Groq(api_key=GROQ_API_KEY)


def extract_pdf(path):
    text_chunks = []
    table_rows = []

    document = converter.convert(path).document

    for item, _level in document.iterate_items():
        page = item.prov[0].page_no if item.prov else 0

        dataframe = getattr(item, "export_to_dataframe", None)
        if dataframe is None:
            text = getattr(item, "text", "") or ""
            if text.strip():
                text_chunks.append({"page": page, "text": text.strip()})
            continue

        table = dataframe()
        if table is None or table.empty:
            continue
        header = [str(c) for c in table.columns]
        for row in table.values:
            table_rows.append(
                {
                    "page": page,
                    "header": header,
                    "row": ["" if v is None else str(v) for v in row],
                }
            )

    return text_chunks, table_rows


# def row_to_text(header, row):
#     pairs = [f"{h.strip()}: if h and v]
#     return "trans — ".join(pairs)


def row_to_text(header, row):
    pairs = []

    for h, v in zip(header, row):
        if h and v:
            h = h.strip()
            v = v.strip()

            pair = f"{h}: {v}"
            pairs.append(pair)

    text = "transaction = " + ", ".join(pairs)

    return text


def build_chunks(text_chunks, table_rows):
    chunks = []

    # for row_info in table_rows:
    #     print(f"row_info=--=-=-=-=-=-=-------->>>", row_info)
    #     chunks.append(
    #         {
    #             "text": row_to_text(row_info["row"]),
    #             "metadata": {"page": row_info["page"], "type": "transaction"},
    #         }
    #     )

    for row_info in table_rows:
        header = row_info["header"]
        row = row_info["row"]
        page = row_info["page"]

        text = row_to_text(header, row)

        chunk = {"text": text, "metadata": {"page": page, "type": "transaction"}}

        print(f"chunk -----=-=-=---=-=-=-=->>>  {chunk}")

        chunks.append(chunk)

    for text_info in text_chunks:
        text = text_info["text"]
        page = text_info["page"]

        if not text.strip():
            continue

        chunk = {"text": text, "metadata": {"page": page, "type": "summary_text"}}

        chunks.append(chunk)

    return chunks


def index_chunks(chunks):
    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        client.delete_collection("loan_statement")
    except Exception:
        pass

    collection = client.create_collection(
        "loan_statement",
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    for i, chunk in enumerate(chunks):
        chunk_id = str(i)
        text = chunk["text"]
        metadata = chunk["metadata"]
        collection.add(
            ids=[chunk_id],
            documents=[text],
            metadatas=[metadata],
        )
    return collection


def retrieve_context(collection, question, top_k=10):
    query_embedding = embedding_function([question])
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    retrieved = list(results["documents"][0])

    print(f"resultsssssssss-==-=-=-=----->>>>{results['documents'][0]}")

    summary_chunks = collection.get(where={"type": "summary_text"})
    existing = set(retrieved)

    for doc in summary_chunks["documents"]:
        if doc not in existing:
            retrieved.append(doc)

    context = ""
    for doc in retrieved:
        if len(context) + len(doc) > 40000:
            break
        # context +=  doc if context else
        if context:
            context += "\n\n"
        context += doc
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
    context = retrieve_context(collection, question, top_k)
    prompt = build_prompt(question, context)

    client = get_llm()
    stream = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=512,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
