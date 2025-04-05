# Coding exercise for RAG app
# Rajiv Gangadharan

from dotenv import load_dotenv
from typing import Optional, Union
from exceptions import VectorStoreError

# Add logging facility
import logging
import os
import sys

import cohere
from langchain_qdrant import QdrantVectorStore, Qdrant
from langchain_qdrant.qdrant import QdrantVectorStoreError
from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse


from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_cohere import CohereEmbeddings
from langchain_qdrant import sparse_embeddings
from langchain_qdrant._utils import maximal_marginal_relevance
from qdrant_client.http.models import Distance, VectorParams

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import gradio as gr

from utils import extract_metadata

from uuid import uuid4

# Loading the environment
load_dotenv()


# Setting up logging infrastructure
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


class Embedder:
    _api_key: Optional[str]
    _model: Optional[str]
    _client: cohere.Client

    def __init__(
        self,
        api_key,
        model,
    ):
        if api_key is None:
            self._api_key = os.getenv("COHERE_API_KEY")
            if self._api_key is None:
                raise ValueError(f"api_key is not provied as a parameter on in env")
            logger.info(f"Got api_key {self._api_key}")
        else:
            self._api_key = api_key

        if model is None:
            self._model = os.getenv("MODEL_ID")
            if self._model is None:
                raise ValueError(f"model is not provided as parameter or in env")
        else:
            self._model = model

        self._client = cohere.Client(self._api_key)
        self.embeddings = CohereEmbeddings(
            client=self._client, model=self._model, async_client=False
        )

    def embed(self, chunks: list):
        try:
            return self._client.embed(
                texts=chunks, input_type="search_document", model=self._model
            )
        except Exception as e:
            print(f"Embedder.embed() - caught while embedding {e}")
            raise

    def embed_documents(self, docs: list[Document]) -> list[Document]:
        texts = [doc.page_content for doc in docs]
        embeddings = self.embeddings.embed_documents(texts)
        d: list[Document] = list()

        for doc, embedding in zip(docs, embeddings):
            doc.metadata["embedding"] = embedding
            logger.debug(f"Assigning {embedding[:10]} to {doc}")
            d.append(doc)

        return d


class VexStor:

    _host: Optional[str]
    _api_key: Optional[str]
    _port: Optional[int]
    _embedder: Embedder
    vectorstore: QdrantVectorStore

    def __init__(
        self,
        embedder: Embedder,
        collection="default_embeddings",
        *,
        host,
        port,
        api_key,
    ):
        """
        Accepts collection name and the embedding model
        initialize the vector, keep the vector for access in the variable vs
        need three things, client, collection and embedding model
        """

        assert collection != None
        self._collection = collection
        self._api_key = api_key
        self._host = host
        self._port = port
        self._embedder = embedder

        # Setting the port to none will get it from the env
        if port is None:
            port = os.getenv("QDRANT_CLOUD_PORT")
            self._port = int(port) if port is not None else None
            if self._port is None:
                self._port = 6333

        # Ensuring that we have a host to connect to
        if host is None:
            logger.info("None host passed, trying from env")
            self._host = os.getenv("QDRANT_CLOUD_HOST")
            if self._host is None:
                raise ValueError(
                    "Host is empty.\n either provide in instantiation or env"
                )

        self._embedder = Embedder(
            api_key=os.getenv("COHERE_API_KEY"), model=os.getenv("MODEL_ID")
        )

        logger.info(f"Instantiating the client with {self._host} and port {self._port}")

        try:
            self.client = QdrantClient(
                url=self._host, port=self._port, api_key=self._api_key
            )

            logger.info(f"Client instantiated")
            logger.info(f"Check if colllection {self._collection} exists.")
            if self._collection:
                self.ensure_collection_exists(self._collection)

            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self._collection,
                embedding=self._embedder.embeddings,
            )
            self.qdrant = Qdrant(
                self.client, "default_collection", self._embedder.embeddings
            )
        except Exception as e:
            VectorStoreError.handle_qdrant_exception(e)
            raise

    def get_collections(
        self,
    ) -> list[str]:
        collections = [c.name for c in self.client.get_collections().collections]
        return collections

    def exists_collection(
        self,
        name: str,
    ) -> bool:
        return True if name in self.get_collections() else False

    def delete_collection(self, col_name: str) -> None:
        if col_name in [col.name for col in self.client.get_collections().collections]:
            self.client.delete_collection(col_name)

    def ingest_pdfs(self, pdf_dir: str) -> list[Document]:

        # file_extractor = {".pdf": LlamaParse()}
        loader = DirectoryLoader(
            path=pdf_dir,
            recursive=True,
        )

        raw_docs: list = loader.load()
        logger.info(f"The following documents are going to be processed {raw_docs}\n")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents=raw_docs)
        processed_docs = list()

        for chunk in chunks:
            logger.debug(f"Processing chunk from file {chunk.metadata.get('source')}")
            metadata = extract_metadata(chunk.page_content)
            metadata["file_name"] = chunk.metadata.get(
                "source"
            )  # capturing the file name
            new_doc = Document(page_content=chunk.page_content, metadata=metadata)
            logger.debug(f"Appending doc {new_doc}")
            processed_docs.append(new_doc)

        return processed_docs

    def add_to_vectorstore(self, documents: list[Document]) -> list[str]:
        _length = len(documents)
        try:
            logger.debug(
                f"add_to_vectorstore() - Adding # {_length} to the vector store"
            )
            doc_ids: list[str] = self.vectorstore.add_documents(documents=documents)
        except Exception as e:
            logger.error(f"While adding documents to vector store {e}")
            raise

        logger.debug(f"added {len(doc_ids)}")
        return doc_ids

    def process_pdf_dir(self, pdf_dir: str) -> list[str]:
        doc_ids: list[str]
        try:
            assert pdf_dir != None
            documents = self.ingest_pdfs(pdf_dir)
            logger.debug(f"#{len(documents)} documents to be processed")
            doc_ids = self.add_to_vectorstore(documents)
            logger.debug(f"#{len(documents)} processed.")
        except Exception as e:
            logger.error(f"process_pdf_path() - While processing pdfs {e}")
            raise
        return doc_ids

    # Ensure Qdrant collection exists
    def ensure_collection_exists(self, collection_name):
        try:
            logger.info(f"Checking existing collection {collection_name}")
            response = self.client.get_collection(collection_name)
            logger.info(f"Got Collection {response}")

        except QdrantVectorStoreError as qvse:
            logger.error(f"Exception getting collection for {collection_name}: {qvse}")
            raise
        except Exception as e:
            logger.info(f"Getting an existing collection failed! {e}")
            try:
                embedding_dim = len(self._embedder.embed(["test"]).embeddings[0])
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dim, distance=Distance.COSINE
                    ),  # Adjust size based on model
                )

            except Exception as e:
                logger.info(f"Creating collection failed! {e}")
                raise

            logger.info(f"Created Qdrant collection: {collection_name}")

    def similarity_search(
        self,
        query: str,
        collection: str = "default_collection",
        top_k: int = 5,
        **kwargs,
    ):
        search_results = self.vectorstore.similarity_search(
            query=query, k=top_k, **kwargs
        )
        #
        return search_results


def main():
    embedder = Embedder(
        api_key=os.getenv("COHERE_API_KEY"), model=os.getenv("MODEL_ID")
    )

    vs = VexStor(
        embedder,
        "default_collection",
        host=os.getenv("QDRANT_CLOUD_HOST"),
        port=os.getenv("QDRANT_CLOUD_PORT"),
        api_key=os.getenv("QDRANT_CLOUD_API_KEY"),
    )
    doc_ids = vs.process_pdf_dir("/home/rajivg/Code/rag-coding-exercise/data")
    logger.debug(f"Added {len(doc_ids)} to the vectorstore")

    logger.info(f"{'=' * 60}")

    # Example retrieval
    query = "Was Byron the father of Ada?"

    results = vs.qdrant.search(query=query, search_type="similarity", k=3)
    logger.info(">>>>>>")
    for r in results:
        print(f"{r}\n\n")
    logger.info("<<<<<<")
    logger.info(f"{'-' * 60}")


if __name__ == "__main__":
    main()
