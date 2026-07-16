from abc import ABC , abstractmethod
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

class DocumentLoader(ABC):

    def __init__(self , file_path: str):
        self.file_path = Path(file_path)
        print(f"Ready to read file: {self.file_path.name}")

    @abstractmethod
    def load(self) -> List[Document]:
        pass

    # @abstractmethod
    # def split_document(self , )

class PDFDocumentLoader(DocumentLoader):
    
    def load(self) -> List[Document]:
        docs = PyPDFLoader(str(self.file_path)).load()
        print("Successfully loaded PDF.")
        return docs
        
         
class TextDocumentLoader(DocumentLoader):

    def load(self) -> List[Document]:
        docs = TextLoader(str(self.file_path) , encoding = "utf-8").load()
        print("Successfully loaded text.")
        return docs


class DocumentLoaderFactory:

    @staticmethod
    def get_loader(file_path: str) -> DocumentLoader:
        file_extension = Path(file_path).suffix.lower()
        print(f"Checking file type: {file_extension}")
        if file_extension == ".pdf":
            return PDFDocumentLoader(file_path)
        elif file_extension == ".txt":
            return TextDocumentLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    @staticmethod
    def supported_extensions() -> tuple:
        return (".pdf", ".txt")