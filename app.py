"""
Streamlit UI -- document upload + chat interaction.
Run with: streamlit run app.py
"""
import streamlit as st
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

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

# ------------------------------------------------------------------
# Chat persistence -- each conversation is saved as its own JSON file
# on disk, so it survives a browser refresh / server restart. The
# active chat's id is kept in the URL (?chat_id=...) so a refresh
# reopens the same conversation instead of starting a blank one.
# ------------------------------------------------------------------
CHATS_DIR = Path(__file__).parent / "chat_sessions"
CHATS_DIR.mkdir(exist_ok=True)


def _chat_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"


def save_chat(chat_id: str, history: list) -> None:
    title = "New chat"
    for role, content, _ in history:
        if role == "user":
            title = content[:40] + ("..." if len(content) > 40 else "")
            break
    data = {
        "id": chat_id,
        "title": title,
        "updated_at": datetime.now().isoformat(),
        "messages": [
            {"role": role, "content": content, "sources": sources}
            for role, content, sources in history
        ],
    }
    try:
        _chat_path(chat_id).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[UI] Saved chat to {_chat_path(chat_id)}")
    except Exception as e:
        print(f"[UI] ERROR: failed to save chat '{chat_id}' to {_chat_path(chat_id)}: {e}")
        st.warning(f"Could not save chat history: {e}")


def load_chat(chat_id: str) -> list:
    path = _chat_path(chat_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [(m["role"], m["content"], m.get("sources")) for m in data.get("messages", [])]
    except Exception as e:
        print(f"[UI] Warning: failed to load chat '{chat_id}': {e}")
        return []


def list_chats() -> list:
    chats = []
    for f in CHATS_DIR.glob("*.json"):
        try:
            chats.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    chats.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
    return chats


_LAST_ACTIVE_FILE = CHATS_DIR / "_last_active.txt"


def set_active_chat(chat_id: str, history: Optional[list] = None) -> None:
    """Make chat_id the active chat: store it in session_state, in the URL,
    and in a small fallback file on disk. The fallback file matters because
    on some setups a hard browser refresh doesn't reliably keep custom URL
    query params, so without it a refresh would always start a new chat."""
    st.session_state.chat_id = chat_id
    st.session_state.chat_history = history if history is not None else load_chat(chat_id)
    st.query_params["chat_id"] = chat_id
    try:
        _LAST_ACTIVE_FILE.write_text(chat_id, encoding="utf-8")
    except Exception as e:
        print(f"[UI] Warning: could not write last-active chat file: {e}")

st.set_page_config(page_title="Smart Research Assistant", page_icon="📚", layout="wide")

# ------------------------------------------------------------------
# Custom CSS -- makes the chat page look closer to ChatGPT:
# centered column, rounded message bubbles, clean sidebar buttons.
# ------------------------------------------------------------------
st.markdown("""
<style>
    /* Center the main content like ChatGPT's chat column */
    .block-container {
        max-width: 800px;
        padding-top: 2rem;
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }

    /* User bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #1a1a1a;
    }

    /* Assistant bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #0d0d0d;
        border: 1px solid #333333;
    }

    /* Sidebar nav buttons */
    .sidebar-nav-btn button {
        width: 100%;
        text-align: left;
        border: none;
        background: transparent;
        font-size: 1rem;
    }

    /* Hide the default streamlit top padding a bit */
    header[data-testid="stHeader"] { background: transparent; }

    /* ---- Sidebar: black background to match the app theme ---- */
    section[data-testid="stSidebar"] {
        background-color: #000000;
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    /* Sidebar buttons */
    section[data-testid="stSidebar"] button {
        background-color: #1a1a1a !important;
        border: 1px solid #333333 !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] button:hover {
        background-color: #2a2a2a !important;
        border-color: #555555 !important;
    }
    /* Primary (active) sidebar button gets a highlight */
    section[data-testid="stSidebar"] button[kind="primary"] {
        background-color: #333333 !important;
        border: 1px solid #555555 !important;
    }
    /* Divider line visible on black */
    section[data-testid="stSidebar"] hr {
        border-color: #333333 !important;
    }

    /* ---- Main app: black background to match the sidebar ---- */
    .stApp {
        background-color: #000000;
    }
    .main .block-container {
        background-color: #000000;
        color: #ffffff;
    }
    body, p, span, label, div, h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
    }

    /* Chat input box */
    [data-testid="stChatInput"] {
        background-color: #111111;
    }
    [data-testid="stChatInput"] textarea {
        background-color: #111111 !important;
        color: #ffffff !important;
    }

    /* File uploader box */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #111111 !important;
        border: 1px dashed #444444 !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #ffffff !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background-color: #0d0d0d !important;
        border: 1px solid #333333 !important;
    }

    /* Buttons in main area */
    .main button {
        background-color: #1a1a1a !important;
        border: 1px solid #333333 !important;
        color: #ffffff !important;
    }
    .main button:hover {
        background-color: #2a2a2a !important;
        border-color: #555555 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Session state ---
# The whole pipeline's state (vector store, chain, memory) lives inside
# one RAGAssistant object, so we only need to stash *that* in session_state.
# NOTE: we deliberately do NOT use st.cache_resource here -- that would
# share a single RAGAssistant (and its "has an index" status) across every
# browser session/user, which is wrong. Each session gets its own instance.
if "assistant" not in st.session_state:
    st.session_state.assistant = RAGAssistant()
if "chat_id" not in st.session_state:
    qp_chat_id = st.query_params.get("chat_id")
    if qp_chat_id and _chat_path(qp_chat_id).exists():
        # URL points at a real, existing chat -- reopen it.
        set_active_chat(qp_chat_id)
    elif _LAST_ACTIVE_FILE.exists() and _chat_path(_LAST_ACTIVE_FILE.read_text(encoding="utf-8").strip()).exists():
        # URL didn't have a usable chat_id (e.g. it got lost on refresh) --
        # fall back to whichever chat was last active.
        set_active_chat(_LAST_ACTIVE_FILE.read_text(encoding="utf-8").strip())
    else:
        set_active_chat(str(uuid.uuid4()), history=[])
if "page" not in st.session_state:
    st.session_state.page = "Upload Documents"
if "docs_ready" not in st.session_state:
    # Only becomes True after THIS session successfully processes an upload.
    # We deliberately don't derive this from assistant.has_index() alone,
    # since that could be True just because of leftover data from a
    # previous run -- we only want to show "ready" after a fresh, real
    # upload+index in the current session.
    st.session_state.docs_ready = False

assistant: RAGAssistant = st.session_state.assistant

# ------------------------------------------------------------------
# Sidebar -- just two choices: Upload Documents / Chat
# ------------------------------------------------------------------
with st.sidebar:
    st.title("📚 Research Assistant")
    st.write("")

    if st.button("📁  Upload Documents", use_container_width=True,
                 type="primary" if st.session_state.page == "Upload Documents" else "secondary"):
        st.session_state.page = "Upload Documents"
        st.rerun()

    if st.button("💬  Chat", use_container_width=True,
                 type="primary" if st.session_state.page == "Chat" else "secondary"):
        st.session_state.page = "Chat"
        st.rerun()

    st.divider()

    if st.button("➕  New Chat", use_container_width=True):
        set_active_chat(str(uuid.uuid4()), history=[])
        st.session_state.page = "Chat"
        st.rerun()

    all_chats = list_chats()
    if all_chats:
        st.caption("Recent chats")
        for c in all_chats[:15]:
            is_active = c["id"] == st.session_state.chat_id
            label = ("🟢 " if is_active else "💭 ") + (c.get("title") or "New chat")
            if st.button(label, key=f"chat_btn_{c['id']}", use_container_width=True):
                set_active_chat(c["id"])
                st.session_state.page = "Chat"
                st.rerun()

    st.divider()
    st.caption(f"Chat ID: `{st.session_state.chat_id[:8]}...`")

    if st.session_state.docs_ready:
        st.caption("✅ Documents indexed and ready.")
    else:
        st.caption("⚠️ No documents indexed yet.")

    st.divider()
    run_eval = st.checkbox("Show reliability score (RAGAS)", value=True)

    with st.expander("⚙️ Settings"):
        st.write(f"**Vector DB:** {config.vector_db}")
        st.write("**Embedding provider:** text-embedding-3-small")
        st.write("**LLM model:** gpt-4o-mini")
        if st.button("Clear Vector Database Index", type="secondary"):
            with st.spinner("Clearing vector database..."):
                try:
                    assistant.clear_index()
                    st.success("Vector database index cleared successfully.")
                except Exception as e:
                    st.error(f"Failed to clear vector database: {e}")

# ==================================================================
# PAGE 1: Upload Documents
# ==================================================================
if st.session_state.page == "Upload Documents":
    st.header("📁 Upload Documents")
    st.caption("Drag and drop files below, or click to browse.")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Drag and drop files here, or click to browse your computer.",
    )
    clear_existing = st.checkbox("Clear existing documents before uploading", value=True)

    if st.button("Process Documents", disabled=not uploaded_files, type="primary"):
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
        st.session_state.docs_ready = True
        st.success(f"Indexed {total_chunks} chunks from {len(uploaded_files)} file(s).")

    if st.session_state.docs_ready:
        st.info("Documents are indexed. Head to the **Chat** tab in the sidebar to start asking questions.")

# ==================================================================
# PAGE 2: Chat  (ChatGPT-style)
# ==================================================================
elif st.session_state.page == "Chat":
    if not assistant.has_index():
        st.info("📁 Upload and process at least one document first (see the sidebar).")
    else:
        st.header("💬 Chat with your documents")

        # Render chat history
        for role, content, sources in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(content)
                if role == "assistant" and sources:
                    with st.expander("📄 Source chunks used"):
                        for s in sources:
                            st.markdown(f"**{s['source']}** (chunk {s['chunk_id']})")
                            st.text(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))

        question = st.chat_input("Message Research Assistant...")
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
            save_chat(st.session_state.chat_id, st.session_state.chat_history)