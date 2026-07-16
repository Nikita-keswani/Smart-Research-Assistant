import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class config:

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.upload_dir = self.base_dir/"data"/"upload"
        self.upload_dir.mkdir(parents = True , exist_ok = True)

        # API Keys
        self.azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")

        # azure openai settings
        self.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_language_model = os.getenv("AZURE_OPENAI_LANGUAGE_MODEL")
        self.azure_openai_embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")

        # pinecone settings
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
        self.pinecone_cloud = os.getenv("PINECONE_CLOUD")
        self.pinecone_region = os.getenv("PINECONE_REGION")
        self.pinecone_dimension = int(os.getenv("PINECONE_DIMENSION"))

        # rag settings
        self.chunk_size = int(os.getenv("CHUNK_SIZE"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))
        self.top_k = int(os.getenv("TOP_K"))

        #Backend Toggles
        self.vector_db = os.getenv("VECTOR_DB")
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER")
        
        
        self.llm_model = self.azure_openai_language_model
        self.embedding_model = self.azure_openai_embedding_model
        
        


config = config()
