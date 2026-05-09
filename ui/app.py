import os
import sys
sys.path.insert(0, "/content/src")

os.environ.setdefault("CHROMA_PERSIST_DIR", "./data/chroma_db")
os.environ.setdefault("VECTOR_STORE_TYPE",  "chroma")
os.environ.setdefault("EMBEDDING_PROVIDER", "huggingface")
os.environ.setdefault("HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
os.environ.setdefault("RETRIEVAL_K",        "5")
os.environ.setdefault("LLM_TEMPERATURE",    "0.1")
os.environ.setdefault("LLM_MAX_TOKENS",     "1024")

import streamlit as st
from embedder    import load_vector_store
from llm_factory import get_llm
from rag_chain   import build_rag_chain

st.set_page_config(
    page_title="UChicago MS-ADS Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .uc-header {
    background: #800000;
    color: white;
    padding: 1.1rem 1.5rem;
    border-radius: 8px;
    margin-bottom: 1rem;
  }
  .uc-header h1 { margin: 0; font-size: 1.5rem; }
  .uc-header p  { margin: 0.2rem 0 0; opacity: 0.85; font-size: 0.9rem; }
  .source-tag {
    display: inline-block;
    background: #800000;
    color: white;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 12px;
    margin: 2px 2px 0 0;
    text-decoration: none;
  }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="uc-header">
  <h1>🎓 MS in Applied Data Science AI Assistant</h1>
  <p>University of Chicago | Ask me anything about the MS-ADS program</p>
</div>
""", unsafe_allow_html=True)

if "messages"     not in st.session_state: st.session_state.messages = []
if "rag_chain"    not in st.session_state: st.session_state.rag_chain = None
if "system_ready" not in st.session_state: st.session_state.system_ready = False
if "query_count"  not in st.session_state: st.session_state.query_count = 0

@st.cache_resource(show_spinner="Loading AI system — please wait...")
def load_system():
    vsm   = load_vector_store()
    llm   = get_llm()
    chain = build_rag_chain(vsm, llm)
    return chain

if not st.session_state.system_ready:
    st.session_state.rag_chain    = load_system()
    st.session_state.system_ready = True

with st.sidebar:
    st.markdown("### 🎓 MS-ADS Assistant")
    st.caption("University of Chicago")
    st.divider()

    st.markdown("**💡 Quick Questions**")
    quick_qs = [
        "What are the core and elective courses?",
        "Admission requirements?",
        "Online vs in-person?",
        "Tell me about the capstone",
        "What are the tuition fees?",
        "Career outcomes for graduates?",
        "Application deadlines?",
        "Who are the instructors?",
    ]
    for q in quick_qs:
        if st.button(q, use_container_width=True, key=f"quick_{q}"):
            st.session_state["pending"] = q

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages    = []
            st.session_state.query_count = 0
            if st.session_state.rag_chain:
                st.session_state.rag_chain.reset()
            st.rerun()
    with col2:
        st.metric("Queries", st.session_state.query_count)

    st.divider()
    k_val = st.slider("Retrieved chunks (k)", 1, 10, 5)
    st.caption("More chunks = more context, slightly slower")

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🎓"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            src_html = " ".join(
                f'<a class="source-tag" href="{s}" target="_blank">🔗 source</a>'
                for s in msg["sources"]
            )
            st.markdown(src_html, unsafe_allow_html=True)

if not st.session_state.messages:
    st.info(
        "**Welcome!** I can answer questions about the UChicago MS-ADS program - "
        "courses, admissions, tuition, instructors, capstone projects, and more. "
        "Use the Quick Questions in the sidebar or type your own below."
    )

pending    = st.session_state.pop("pending", None)
user_input = st.chat_input("Ask anything about the MS-ADS program...") or pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Searching knowledge base..."):
            try:
                result  = st.session_state.rag_chain.ask(user_input, k=k_val)
                answer  = result["answer"]
                sources = result["sources"]
                docs    = result.get("docs", [])

                st.markdown(answer)

                if sources:
                    src_html = " ".join(
                        f'<a class="source-tag" href="{s}" target="_blank">🔗 source</a>'
                        for s in sources
                    )
                    st.markdown(src_html, unsafe_allow_html=True)

                if docs:
                    with st.expander(f"🔍 View {len(docs)} retrieved chunks"):
                        for i, doc in enumerate(docs, 1):
                            m = doc.metadata
                            st.markdown(
                                f"**Chunk {i}** | Section: `{m.get('section','')}` | "
                                f"[{m.get('page_title','')}]({m.get('source','')})"
                            )
                            st.text(doc.page_content[:400] + (" ..." if len(doc.page_content) > 400 else ""))
                            if i < len(docs):
                                st.divider()

            except Exception as exc:
                answer  = f"Sorry, an error occurred: {exc}"
                sources = []
                st.error(answer)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })
    st.session_state.query_count += 1
    st.rerun()