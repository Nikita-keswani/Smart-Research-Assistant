from typing import Optional

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

from src.vectorstore import VectorStoreManager
from config import config
from src.memory import ConversationBufferMemory

SYSTEM_PROMPT = """You are an expert research assistant. Your task is to provide comprehensive, detailed, and clear answers based strictly on the provided documents. Your response should be easy for a user to follow and understand, explaining complex terms or concepts where appropriate.

Follow these guidelines:
1. DETAILED EXPLANATIONS: Provide thorough, well-structured explanations instead of overly brief answers. Break down the information logically and explain it step-by-step so the user can easily understand.
2. GROUNDING: Use ONLY the information in the context below. Do not use external or general knowledge, and do not make assumptions.
3. NO ANSWER FOUND: If the context does not contain the answer, respond exactly with: "I couldn't find this in the provided documents."
4. STRUCTURE: Use clear formatting like paragraphs, bullet points, and headings where appropriate to make your response highly readable.
5. CITATIONS: After your answer, list the sources used in this format:
   Sources: [Source 1], [Source 2]

Context:
{context}

Question: {question}

Answer:"""

class RAGChain:

   def __init__(self , vectorstore , memory_manager: Optional[ConversationBufferMemory] = None):
      print("[RAGChain] Initializing RAGChain...")
      self.vectorstore = vectorstore
      self.memory_manager = memory_manager or ConversationBufferMemory()

      print("[RAGChain] Setting up LLM model connections...")
      llm = AzureChatOpenAI(
         azure_deployment = config.llm_model,
         api_version = config.azure_openai_api_version,
         azure_endpoint = config.azure_openai_endpoint,
         api_key = config.azure_openai_key,
         temperature= 0
      )

      retriever = vectorstore.as_retriever(search_kwargs={"k": config.top_k})
      prompt = PromptTemplate.from_template(SYSTEM_PROMPT)

      print("[RAGChain] Creating LangChain conversational retrieval chain...")
      self._chain: ConversationalRetrievalChain = ConversationalRetrievalChain.from_llm(
         llm = llm,
         retriever = retriever,
         memory = self.memory_manager.memory,
         return_source_documents = True,
         combine_docs_chain_kwargs = {"prompt" : prompt}
      )
      print("[RAGChain] RAGChain initialization complete.")

   def ask(self , ques : str) ->dict:
      print(f"[RAGChain] Invoking chain for question: '{ques}'")
      result =  self._chain.invoke({'question' : ques})
      print(f"[RAGChain] Answer generated successfully. Retrieved {len(result.get('source_documents', []))} source document(s).")
      return {
         "answer": result["answer"],
         "source_documents": result.get("source_documents", []),
      }
