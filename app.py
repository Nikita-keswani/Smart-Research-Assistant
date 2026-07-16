"""
Streamlit UI -- document upload + chat interaction.
Run with: streamlit run app.py
"""
import streamlit as st

# Shim to fix Ragas/Langchain compatibility issue where it tries to import removed ChatVertexAI
import sys
import types
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    class ChatVertexAI: pass
    _stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

from config import config
from src.assistant import RAGAssistant

st.set_page_config(page_title="Smart Research Assistant", page_icon="📚", layout="wide")
st.title(" Smart Research Assistant")
st.caption("Upload documents, ask questions, get grounded answers with reliability scores.")

# --- Session state ---
# The whole pipeline's state (vector store, chain, memory) lives inside
# one RAGAssistant object, so we only need to stash *that* in session_state.
if "assistant" not in st.session_state:
    st.session_state.assistant = RAGAssistant()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, content, sources)

assistant: RAGAssistant = st.session_state.assistant

# --- Sidebar: upload + settings ---
with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )
    clear_existing = st.checkbox("Clear existing documents before uploading", value=True)

    if st.button("Process Documents", disabled=not uploaded_files):
        print(f"[UI] Starting process for {len(uploaded_files)} uploaded file(s)...")
        with st.spinner("Ingesting and indexing documents..."):
            if clear_existing:
                print("[UI] Clearing existing database index first...")
                try:
                    assistant.clear_index()
                except Exception as e:
                    print(f"[UI] Warning: Failed to clear index: {e}")
                    st.warning(f"Could not clear existing index: {e}")
            total_chunks = 0
            for uf in uploaded_files:
                save_path = config.upload_dir / uf.name
                save_path.write_bytes(uf.getbuffer())
                try:
                    chunks = assistant.ingest_file(str(save_path))
                    total_chunks += len(chunks)
                except Exception as e:
                    print(f"[UI] Error processing file '{uf.name}': {e}")
                    st.error(f"Failed to process '{uf.name}': {e}")
                    raise
        print(f"[UI] Document processing completed successfully. Total chunks: {total_chunks}")
        st.success(f"Indexed {total_chunks} chunks from {len(uploaded_files)} file(s).")

    st.divider()
    st.header("⚙️ Settings")
    st.write(f"**Vector DB:** {config.vector_db}")
    st.write(f"**Embedding provider:** {"text-embedding-3-small"}")
    st.write(f"**LLM model:** {"gpt-4o-mini"}")
    run_eval = st.checkbox("Show reliability score (RAGAS)", value=True)
    
    if st.button("Clear Vector Database Index", type="secondary"):
        with st.spinner("Clearing vector database..."):
            try:
                assistant.clear_index()
                st.success("Vector database index cleared successfully.")
            except Exception as e:
                st.error(f"Failed to clear vector database: {e}")

# --- Main: chat interface ---
if not assistant.has_index():
    st.info("👈 Upload and process at least one document to get started.")
else:
    # Render chat history
    for role, content, sources in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(content)
            if role == "assistant" and sources:
                with st.expander("📄 Source chunks used"):
                    for s in sources:
                        st.markdown(f"**{s['source']}** (chunk {s['chunk_id']})")
                        st.text(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))

    question = st.chat_input("Ask a question about your documents...")
    if question:
        st.session_state.chat_history.append(("user", question, None))
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    print(f"[UI] Asking assistant: '{question}'")
                    result = assistant.ask(question)
                    answer = result["answer"]
                    source_docs = result["source_documents"]
                except Exception as e:
                    print(f"[UI] Error during assistant.ask(): {e}")
                    st.error(f"Failed to ask assistant: {e}")
                    raise

                st.write(answer)

                sources_display = [
                    {
                        "source": d.metadata.get("source_file", "unknown"),
                        "chunk_id": d.metadata.get("chunk_id", "?"),
                        "text": d.page_content,
                    }
                    for d in source_docs
                ]
                if sources_display:
                    with st.expander("📄 Source chunks used"):
                        for s in sources_display:
                            st.markdown(f"**{s['source']}** (chunk {s['chunk_id']})")
                            st.text(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))

                if run_eval:
                    print("[UI] Requesting reliability score scoring...")
                    with st.spinner("Scoring answer reliability..."):
                        try:
                            scores = assistant.evaluate(question, answer, source_docs)
                            cols = st.columns(3)
                            cols[0].metric("Faithfulness", f"{scores.get('faithfulness', 0):.2f}")
                            cols[1].metric("Answer Relevance", f"{scores.get('answer_relevancy', 0):.2f}")
                            cols[2].metric("Context Precision", f"{scores.get('context_precision', 0):.2f}")
                            print(f"[UI] Reliability scores displayed: {scores}")
                        except Exception as e:
                            print(f"[UI] Scoring skipped/failed: {e}")
                            st.warning(f"Evaluation skipped: {e}")

        st.session_state.chat_history.append(("assistant", answer, sources_display))
