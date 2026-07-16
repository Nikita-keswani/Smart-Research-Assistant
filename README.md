# 📚 Smart Research Assistant

A lightweight, local Retrieval-Augmented Generation (RAG) tool built with **Streamlit**, **LangChain**, **Azure OpenAI**, and **Pinecone**. This assistant allows you to upload documents (PDF and TXT), index them into a high-performance vector store, ask questions using an LLM, view references to exact source document chunks, and evaluate answers with **Ragas** reliability scores.

---

## 🌟 Key Features

- **Document Ingestion:** Easily upload and chunk multiple PDF or TXT documents.
- **Vector Database Integration:** Automatically creates/updates embeddings and indexes documents using a Pinecone serverless vector store.
- **Conversational Memory:** Remembers chat history to support natural follow-up questions.
- **Sources & Citations:** Highlights the exact page/chunks of document sources used for generating the answer.
- **Answer Evaluation:** Calculates real-time reliability metrics using Ragas (Faithfulness, Answer Relevance, and Context Precision).

---

## 📂 Project Structure

```text
├── app.py                # Main Streamlit UI application
├── config.py             # Configuration & Environment variable loader
├── requirements.txt      # Python dependencies
├── src/
│   ├── assistant.py      # Main pipeline coordinator (RAGAssistant)
│   ├── document_loader.py# PDF & Text document readers
│   ├── embeddings.py     # Embedding provider helper (Azure OpenAI)
│   ├── evaluation.py     # Ragas reliability evaluator
│   ├── ingestion.py      # Text chunker and metadata tagger
│   ├── memory.py         # Conversation memory buffer
│   ├── rag_chain.py      # RAG prompt template and chain setup
│   └── vectorstore.py    # Pinecone vector store manager
└── data/
    └── upload/           # Storing uploaded files (auto-created)
```

---

## 🛠️ Setup & Installation

### 1. Clone or Open the Project
Ensure you are in the project's root directory:
```bash
cd Research_Assistant_practice
```

### 2. Install Dependencies
Install all required libraries using pip:
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a file named `.env` in the root directory of the project and populate it with your API keys and configuration settings:

```ini
# --- API Keys ---
AZURE_OPENAI_KEY="your-azure-openai-api-key"
PINECONE_API_KEY="your-pinecone-api-key"

# --- Azure OpenAI Settings ---
AZURE_OPENAI_API_VERSION="2023-05-15"
AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
AZURE_OPENAI_LANGUAGE_MODEL="your-gpt-deployment-name"
AZURE_OPENAI_EMBEDDING_MODEL="your-embedding-deployment-name"

# --- Pinecone Vector Database Settings ---
PINECONE_INDEX_NAME="your-index-name"
PINECONE_CLOUD="aws"               # e.g., aws, gcp
PINECONE_REGION="us-east-1"        # e.g., us-east-1, us-west-2
PINECONE_DIMENSION=1536            # e.g., 1536 for text-embedding-ada-002

# --- RAG Settings ---
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=3

# --- Backend Toggles ---
VECTOR_DB="pinecone"
EMBEDDING_PROVIDER="azure_openai"
```

---

## 🚀 How to Run the App

Launch the Streamlit dashboard by running:
```bash
streamlit run app.py
```

This will automatically start a local server (typically at `http://localhost:8501`) and open it in your default web browser.

---

## 💡 How to Use the Assistant

1. **Upload Documents:** Use the sidebar on the left to drag-and-drop or select your PDF/TXT files.
2. **Process Files:** Check or uncheck *“Clear existing documents before uploading”*, then click **Process Documents**. The assistant will segment, embed, and upload the chunks to Pinecone.
3. **Chat:** Enter your queries in the bottom chat bar. The assistant answers using context retrieved from the database.
4. **Inspect Sources:** Expand the **Source chunks used** dropdown underneath the answer to view specific text snippets and their metadata.
5. **Check Reliability (Optional):** Check the *“Show reliability score (RAGAS)”* box to run evaluations on Faithfulness, Answer Relevance, and Context Precision.
