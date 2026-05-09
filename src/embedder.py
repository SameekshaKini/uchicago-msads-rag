import os
import json
import logging
from langchain.schema import Document

log = logging.getLogger(__name__)

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
VECTOR_STORE_TYPE  = os.getenv("VECTOR_STORE_TYPE", "chroma")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_EMB_DEPLOY = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")


def chunks_to_documents(chunks_path: str = "data/all_chunks.json") -> list:
    """Convert scraped chunks JSON -> LangChain Documents."""
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    docs = []
    for c in chunks:
        if not c.get("text", "").strip():
            continue
        docs.append(Document(
            page_content=c["text"],
            metadata={
                "source": c["url"],
                "page_title": c.get("page_title", ""),
                "section": c.get("section", ""),
                "chunk_index": c.get("chunk_index", 0),
            }
        ))
    log.info("Loaded %d documents from %s", len(docs), chunks_path)
    return docs


def get_embeddings(provider: str = EMBEDDING_PROVIDER):
    provider = provider.lower()
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        log.info("Loading HuggingFace embeddings: %s", HF_EMBEDDING_MODEL)
        return HuggingFaceEmbeddings(
            model_name=HF_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    elif provider == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings
        log.info("Loading Azure OpenAI embeddings: %s", AZURE_EMB_DEPLOY)
        return AzureOpenAIEmbeddings(
            azure_deployment=AZURE_EMB_DEPLOY,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION,
        )
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        raise ValueError(f"Unknown embedding provider: {provider!r}")


class VectorStoreManager:
    def __init__(self, store_type=VECTOR_STORE_TYPE, embeddings=None,
                 persist_dir=CHROMA_PERSIST_DIR):
        self.store_type  = store_type.lower()
        self.persist_dir = persist_dir
        self.embeddings  = embeddings or get_embeddings()
        self._store      = None

    def build(self, documents: list) -> None:
        log.info("Building %s vector store with %d documents...",
                 self.store_type, len(documents))
        if self.store_type == "chroma":
            from langchain_community.vectorstores import Chroma
            os.makedirs(self.persist_dir, exist_ok=True)
            self._store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
                collection_name="msads_rag",
            )
        elif self.store_type == "faiss":
            from langchain_community.vectorstores import FAISS
            self._store = FAISS.from_documents(documents, self.embeddings)
        else:
            raise ValueError(f"Unknown store type: {self.store_type!r}")
        log.info("Vector store built successfully.")

    def load(self) -> None:
        log.info("Loading %s vector store from %s", self.store_type, self.persist_dir)
        if self.store_type == "chroma":
            from langchain_community.vectorstores import Chroma
            self._store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                collection_name="msads_rag",
            )
        elif self.store_type == "faiss":
            from langchain_community.vectorstores import FAISS
            self._store = FAISS.load_local(
                self.persist_dir, self.embeddings,
                allow_dangerous_deserialization=True,
            )

    def retriever(self, k: int = 5):
        if self._store is None:
            raise RuntimeError("Call build() or load() first.")
        return self._store.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )

    def similarity_search(self, query: str, k: int = 5):
        if self._store is None:
            raise RuntimeError("Call build() or load() first.")
        return self._store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 5):
        if self._store is None:
            raise RuntimeError("Call build() or load() first.")
        return self._store.similarity_search_with_score(query, k=k)


def build_vector_store(documents, provider=EMBEDDING_PROVIDER,
                       store_type=VECTOR_STORE_TYPE) -> VectorStoreManager:
    vsm = VectorStoreManager(store_type=store_type,
                              embeddings=get_embeddings(provider))
    vsm.build(documents)
    return vsm


def load_vector_store(provider=EMBEDDING_PROVIDER,
                      store_type=VECTOR_STORE_TYPE,
                      persist_dir=CHROMA_PERSIST_DIR) -> VectorStoreManager:
    vsm = VectorStoreManager(store_type=store_type,
                              embeddings=get_embeddings(provider),
                              persist_dir=persist_dir)
    vsm.load()
    return vsm
