from pathlib import Path
from typing import List , Optional

from src.document_loader import DocumentLoaderFactory
from config import config , config as default_config

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class TextChunker:
    
    def __init__(self , config: config = default_config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def split(self , documents:List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(documents)


class Ingestor:
    """Orchestrates: load a file -> chunk it -> tag metadata."""
    def __init__(self, config: config = default_config, chunker: Optional[TextChunker] = None):
        self.config = config
        self.chunker = chunker or TextChunker(config)

    
    def ingest_file(self , file_path:str) -> List[Document]:
        print(f"[Ingestor] Getting appropriate document loader for: {file_path}")
        loader = DocumentLoaderFactory.get_loader(file_path)
        print("[Ingestor] Loading file contents into memory...")
        documents = loader.load()
        print(f"[Ingestor] Loaded {len(documents)} page/document objects. Splitting them into smaller chunks...")
        chunks = self.chunker.split(documents)

        print(f"[Ingestor] Attaching metadata tags to all {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["source_file"] = Path(file_path).name
        print("[Ingestor] Ingestion of file completed.")
        return chunks
    
    def ingest_directory(self , directory : Optional[str] = None) -> List[Document]:
        if directory is None:
            directory = self.config.upload_dir
        
        print(f"[Ingestor] Scanning folder: {directory}")
        all_chunks: List[Document] = []
        for file_path in directory.glob("**/*"):
            if file_path.suffix.lower() in DocumentLoaderFactory.supported_extensions():
                print(f"[Ingestor] Found supported file: {file_path.name}. Starting ingestion...")
                all_chunks.extend(self.ingest_file(str(file_path)))
        print(f"[Ingestor] Directory scan done. Total chunks in directory: {len(all_chunks)}")
        return all_chunks
                