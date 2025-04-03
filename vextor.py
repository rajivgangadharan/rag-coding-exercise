# Coding exercise for RAG app
# Rajiv Gangadharan

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from typing import Optional, Union


# Add logging facility
import logging
import os
import sys


import cohere
from langchain_qdrant import QdrantVectorStore, Qdrant

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
            self._model = os.getenv("MODEl_ID")
            if self._model is None:
                raise ValueError(f"model is not provided as parameter or in env")
        else:
            self._model = model

        self._client = cohere.Client(self._api_key)

    def embed(self, chunks: list):
        try:
            return self._client.embed(
                texts=chunks, input_type="search_document", model=self._model
            )
        except Exception as e:
            print(f"Embedder.embed() - caught while embedding {e}")
            raise


class VexStor:

    _host: Optional[str]
    _api_key: Optional[str]
    _port: Optional[int]
    _embedder: Embedder

    def __init__(
        self,
        collection="default_embeddings",
        embedder: Embedder,
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



def main():
    embedder = Embedder(
        api_key=os.getenv("COHERE_API_KEY"), model=os.getenv("MODEL_ID")
    )

    vs = VexStor("default_collection", embedder, 
                 host=os.getenv("QDRANT_CLOUD_HOST"), 
                 port=os.getenv("QDRANT_CLOUD_PORT"), 
                 api_key=os.getenv("QDRANT_CLOUD_API_KEY"))  
    texts = ["this is some text1", "this is another piece of text2"]
    embeddings = embedder.embed(chunks=texts)
    print(embeddings)


if __name__ == "__main__":
    main()
