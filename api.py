import asyncio
import json
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from main import build_chunks, extract_pdf, index_chunks, stream_answer_question

app = FastAPI()
STREAM_DELAY = 0.03

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

collection = None
filename = None


class AskRequest(BaseModel):
    question: str


# @app.get("/status")
# def status():
#     return {"ready": collection is not None, "filename": filename}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global collection, filename
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="only pdf are acepted")

    data = await file.read()
    text_chunks, table_rows = extract_pdf(BytesIO(data))
    chunks = build_chunks(text_chunks, table_rows)
    collection = index_chunks(chunks)
    # filename = file.filename

    return {
        # "message": "PDF processed",
        # "filename": file.filename,
        "table_rows": len(table_rows),
        "text_blocks": len(text_chunks),
    }


@app.post("/ask")
def ask(req: AskRequest):
    if collection is None:
        raise HTTPException(status_code=400, detail="upload the pdf first")

    async def stream():
        try:
            for token in stream_answer_question(collection, req.question):
                yield f"data: {json.dumps({'token': token})}\n\n"
                await asyncio.sleep(STREAM_DELAY)
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
