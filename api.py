import json
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from main import build_chunks, extract_pdf, index_chunks, stream_answer_question

app = FastAPI()

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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        text_chunks, table_rows = extract_pdf(tmp_path)

        print("chunks and rows done =-=-======--=-=-==-=-=-->>   ", table_rows)

        print("chunks and rows done =-=-======--=-=-==-=-=-->>   ", text_chunks)

    finally:
        os.remove(tmp_path)
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

    # if collection is None:
    #   raise HTTPException()

    async def stream():
        for token in stream_answer_question(collection, req.question):
            yield f"data: {json.dumps({'token': token})}\n\n"
            # await asyncio.sleep(0.03)
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
