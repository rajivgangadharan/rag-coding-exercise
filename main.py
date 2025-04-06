# Coding exercise for RAG app
# Rajiv Gangadharan

import traceback

from dotenv import load_dotenv
from typing import Optional, Union
from exceptions import VectorStoreError
import vexstor

# Add logging facility
import logging
import os
import sys

# Environment variables
from dotenv import load_dotenv

# for making it into a service
import fastapi
import fastapi_cli
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4
import os
import shutil
from fastapi import FastAPI
from pydantic import BaseModel
from utils import extract_metadata
from os import path
import pathlib

# Generating unique ids
from uuid import uuid4

# Loading the environment
load_dotenv()


# Setting up logging infrastructure
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4
import os
import shutil

from vexstor import Embedder
from vexstor import VexStor
from vexstor import LanguageModel
from prompts import prompt


app = FastAPI()

# Initialize embedder and vector store
embedder = Embedder(api_key=os.getenv("COHERE_API_KEY"), model=os.getenv("MODEL_ID"))
vector_store = VexStor(
    embedder=embedder,
    collection="default_collection",
    host=os.getenv("QDRANT_CLOUD_HOST"),
    port=os.getenv("QDRANT_CLOUD_PORT"),
    api_key=os.getenv("QDRANT_CLOUD_API_KEY"),
)
model_id = os.getenv("LLM_MODEL_ID", "command-r-plus")
llm = LanguageModel(model=model_id, vexstor=vector_store)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    collection: Optional[str] = "default_collection"


class SearchResult(BaseModel):
    document_id: str
    content: str
    llm_response: str
    metadata: dict
    score: float


class DocumentIn(BaseModel):
    content: str
    metadata: Optional[dict] = {}


@app.post("/search", response_model=List[SearchResult])
def semantic_search(req: SearchRequest):
    search_results: list[SearchResult] = list()
    try:
        logger.debug(f"QUERY: {req.query}")
        results = vector_store.qdrant.search(
            query=req.query, search_type="similarity", k=req.top_k
        )
        for doc in results:
            response = llm(doc.page_content, query=req.query, prompt=prompt)
            sr = SearchResult(
                document_id=doc.metadata["_id"],  # Replace if IDs are stored
                content=doc.page_content,
                llm_response=response.generations[0].text,
                metadata=doc.metadata,
                score=doc.metadata.get("score", 1.0),
            )
            logger.debug(f"Result Item: {sr}")
            search_results.append(sr)
            logger.debug(f"Returning list of length {len(search_results)}")

        return search_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload")
def upload_pdf(file: UploadFile = File(...)):
    try:
        # Generate a unique session directory
        session_id = str(uuid4())
        session_dir = f"/tmp/uploads/{session_id}"
        os.makedirs(session_dir, exist_ok=True)

        # Save uploaded file to session folder
        file_path = os.path.join(session_dir, file.filename)
        if file_path is None:
            logger.error(f"File is none, cannot be processed")
            raise Exception(f"File {file_path} is none")

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logger.debug(f"✅ File saved to: {file_path}")
        file_saved_dir_name = os.path.dirname(file_path)

        # Process just that folder
        logger.debug(f"processing directory {file_saved_dir_name}")
        vector_store.process_pdf_dir(file_saved_dir_name)

        return {"status": "success", "file": file.filename, "session_id": session_id}

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during upload: {str(e)}")


@app.post("/documents")
def add_document(doc: DocumentIn):
    from langchain_core.documents import Document

    try:
        document = Document(page_content=doc.content, metadata=doc.metadata)
        vector_store.add_to_vectorstore([document])
        return {"status": "success", "id": str(uuid4())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def list_documents():
    try:
        collections = vector_store.get_collections()
        return {"collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections")
def list_collections():
    return list_documents()


@app.post("/collections/{collection_name}")
def create_collection(collection_name: str):
    try:
        vector_store.ensure_collection_exists(collection_name)
        return {"status": "created", "collection": collection_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collections/{collection_name}")
def delete_collection(collection_name: str):
    try:
        vector_store.delete_collection(collection_name)
        return {"status": "deleted", "collection": collection_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
