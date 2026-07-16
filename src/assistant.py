from typing import List, Optional
from langchain_core.documents import Document
from src.ingestion import Ingestor
from src.vectorstore import VectorStoreManager
from src.rag_chain import RAGChain
from src.memory import ConversationBufferMemory
from src.evaluation import AnswerEvaluator


class RAGAssistant:
    """End-to-end RAG pipeline: ingest -> index -> ask -> (optionally) evaluate."""

    def __init__(self):
        print("[Assistant] Setting up helper components (Ingestor, VectorStore, Memory, Evaluator)...")
        self.ingestor = Ingestor()
        self.vectorstore = VectorStoreManager()
        self.memory_manager = ConversationBufferMemory()
        self.evaluator = AnswerEvaluator()
        self._chain: Optional[RAGChain] = None

        # Try to pick up an existing persisted index on startup.
        print("[Assistant] Loading existing vector index if it exists...")
        self.vectorstore.load()
        print("[Assistant] Initialization complete.")

    # -- Indexing -----------------------------------------------------
    def ingest_file(self, file_path: str) -> List[Document]:
        print(f"[Assistant] Starting ingestion of file: {file_path}")
        chunks = self.ingestor.ingest_file(file_path)
        print(f"[Assistant] File chunked into {len(chunks)} parts. Updating vector store...")
        self.vectorstore.update(chunks)
        self._invalidate_chain()
        print(f"[Assistant] Finished indexing file: {file_path}")
        return chunks

    def ingest_directory(self, directory: Optional[str] = None) -> List[Document]:
        print(f"[Assistant] Starting ingestion of directory: {directory or 'default upload folder'}")
        chunks = self.ingestor.ingest_directory(directory)
        print(f"[Assistant] Directory chunked into {len(chunks)} parts. Updating vector store...")
        self.vectorstore.update(chunks)
        self._invalidate_chain()
        print(f"[Assistant] Finished indexing directory.")
        return chunks

    def has_index(self) -> bool:
        return self.vectorstore.store is not None

    # -- Querying -------------------------------------------------------
    def ask(self, question: str) -> dict:
        if not self.has_index():
            raise RuntimeError("No documents indexed yet. Call ingest_file() or ingest_directory() first.")
        if self._chain is None:
            print("[Assistant] Creating a new RAG chain with the loaded vector store...")
            self._chain = RAGChain(self.vectorstore.store, self.memory_manager)
        print(f"[Assistant] Sending question to RAG chain: {question}")
        result = self._chain.ask(question)
        print("[Assistant] Got answer from RAG chain.")
        return result

    def reset_conversation(self):
        print("[Assistant] Resetting conversation memory...")
        self.memory_manager.reset()
        self._invalidate_chain()
        print("[Assistant] Conversation history reset.")

    # -- Evaluation -------------------------------------------------------
    def evaluate(self, question: str, answer: str, source_documents) -> dict:
        print("[Assistant] Extracting contexts from source documents for evaluation...")
        contexts = self.evaluator.contexts_from_source_documents(source_documents)
        print("[Assistant] Evaluating answer using RAGAS evaluator...")
        return self.evaluator.evaluate(question, answer, contexts)

    # -- Internal ---------------------------------------------------------
    def _invalidate_chain(self):
        """Force the chain to be rebuilt against the freshest vector store."""
        self._chain = None

    def clear_index(self):
        """Reset and wipe the vector store."""
        print("[Assistant] Resetting vector database index...")
        self.vectorstore.delete_all()
        self._invalidate_chain()
        print("[Assistant] Vector database index reset completed.")
