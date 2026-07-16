from langchain_openai import AzureOpenAIEmbeddings
from config import config

class AzureOpenAIEmbeddingProvider:

    def __init__(self , model_name : str = None):
        self.model_name = model_name or config.embedding_model

    def get_model(self) -> AzureOpenAIEmbeddings:
        print(f"[Embeddings] Initializing AzureOpenAIEmbeddings model: {self.model_name}")
        return AzureOpenAIEmbeddings(
            api_key=config.azure_openai_key,
            azure_endpoint=config.azure_openai_endpoint,
            api_version=config.azure_openai_api_version,
            azure_deployment=config.azure_openai_embedding_model,
        )

def get_embedding_model() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddingProvider().get_model()