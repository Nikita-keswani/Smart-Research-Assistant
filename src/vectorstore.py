from typing import List , Optional

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from src.embeddings import get_embedding_model
from config import config
from pinecone import Pinecone, ServerlessSpec


class VectorStoreManager:
    """Builds, loads, and updates a Pinecone-backed vector store."""
    
    def __init__(self , config = config , embeddings = None):
        if not config.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is not set. Add it to your .env file.")
        self.config = config
        self.embeddings = embeddings or get_embedding_model()
        self.client = Pinecone(api_key = self.config.pinecone_api_key)
        self.store = None

    def ensure_index_exists(self) -> None:
        """Create Pinecone index if it doesn't exist."""
        print(f"[VectorStore] Checking if Pinecone index '{self.config.pinecone_index_name}' exists...")
        if self.config.pinecone_index_name not in self.client.list_indexes().names():
            print(f"[VectorStore] Index '{self.config.pinecone_index_name}' not found. Creating a new one...")
            self.client.create_index(
                name=self.config.pinecone_index_name,
                dimension=self.config.pinecone_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.config.pinecone_cloud,
                    region=self.config.pinecone_region,
                ),
            )
            print(f"[VectorStore] Index '{self.config.pinecone_index_name}' created successfully.")
        else:
            print(f"[VectorStore] Index '{self.config.pinecone_index_name}' already exists.")
    
    def build(self , chunks : List[Document]) -> PineconeVectorStore:
        """Create a new Pinecone vector store from chunks."""
        self.ensure_index_exists()

        print(f"[VectorStore] Building new vector store with {len(chunks)} chunks...")
        self.store = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            index_name=self.config.pinecone_index_name,
        )
        print("[VectorStore] Vector store built successfully.")
        return self.store
    
    def load(self) -> PineconeVectorStore:
        """Load existing Pinecone vector store."""
        if self.store is None:
            print(f"[VectorStore] Connecting to existing index '{self.config.pinecone_index_name}'...")
            self.store = PineconeVectorStore(
                index_name=self.config.pinecone_index_name,
                embedding=self.embeddings,
            )
            print("[VectorStore] Vector store connection established.")
        return self.store
    
    def update(self , chunks : List[Document]) -> PineconeVectorStore:
        """Add new chunks to existing Pinecone vector store."""
        self.ensure_index_exists()

        print(f"[VectorStore] Adding {len(chunks)} new chunks to vector store...")
        self.store = self.load()
        self.store.add_documents(chunks)
        print("[VectorStore] Vector store update complete.")
        return self.store

    def delete_all(self) -> None:
        """Delete all vectors inside the vector store."""
        print("[VectorStore] Resetting vector store index...")
        self.store = self.load()
        try:
            self.store.delete(delete_all=True)
            print("[VectorStore] All vectors deleted successfully.")
        except Exception as e:
            print(f"[VectorStore] delete(delete_all=True) failed: {e}. Recreating index...")
            if self.config.pinecone_index_name in self.client.list_indexes().names():
                self.client.delete_index(self.config.pinecone_index_name)
                print(f"[VectorStore] Deleted index '{self.config.pinecone_index_name}' successfully.")
            self.ensure_index_exists()
            print("[VectorStore] Re-created index successfully.")
            self.store = None
            self.load()