import streamlit as st
import os
import json
import io
import re
from typing import List, Tuple
import numpy as np
from search_engine import RealTimeSearchEngine
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

# Load environment variables (kept for local fallback)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(project_root, ".env"))

# Session storage (multiple chats)
SESSIONS_DIR = os.path.join(project_root, "Data", "Sessions")
SESSIONS_INDEX = os.path.join(SESSIONS_DIR, "index.json")

def _ensure_sessions_dir() -> None:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    if not os.path.exists(SESSIONS_INDEX):
        with open(SESSIONS_INDEX, "w", encoding="utf-8") as f:
            json.dump({"sessions": []}, f, indent=2)

def _read_sessions_index() -> dict:
    _ensure_sessions_dir()
    try:
        with open(SESSIONS_INDEX, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"sessions": []}

def _write_sessions_index(data: dict) -> None:
    _ensure_sessions_dir()
    with open(SESSIONS_INDEX, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def _session_file(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")

def _create_session(title: str) -> str:
    from datetime import datetime
    idx = _read_sessions_index()
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    idx.setdefault("sessions", []).append({
        "id": session_id,
        "title": title,
        "created_at": datetime.now().isoformat(timespec="seconds")
    })
    _write_sessions_index(idx)
    with open(_session_file(session_id), "w", encoding="utf-8") as f:
        json.dump({"messages": []}, f, indent=2)
    return session_id

def _load_session_messages(session_id: str) -> List[dict]:
    try:
        with open(_session_file(session_id), "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("messages", [])
    except Exception:
        return []

def _save_session_messages(session_id: str, messages: List[dict]) -> None:
    with open(_session_file(session_id), "w", encoding="utf-8") as f:
        json.dump({"messages": messages}, f, indent=2)

# Set up page configuration
st.set_page_config(page_title="ProPlay Assistant", page_icon="⚽", layout="wide")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Mode", ["Fast", "Accurate"], index=0, horizontal=True)
    model = st.selectbox(
        "Model",
        options=[
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ],
        index=1,
        help="Choose the language model for responses"
    )
    default_temp = 0.2 if mode == "Fast" else 0.5
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, default_temp, 0.05)
    max_tokens = 256 if mode == "Fast" else 768

    # 🟢 LIVE UPDATES TOGGLE (uses checkbox for compatibility)
    live_updates = st.checkbox("🌍 Enable Live Internet Updates (NewsAPI)", value=True,
                              help="Turn off to use offline Groq-only responses for faster answers.")

    clear = st.button("Clear chat", key="clear_sidebar")
    st.markdown("---")
    summarize_clicked = st.button("Summarize chat", key="summarize_sidebar")
    st.markdown("---")
    st.subheader("Sessions")
    _ensure_sessions_dir()
    idx = _read_sessions_index()
    sessions = idx.get("sessions", [])
    if not sessions:
        first_id = _create_session("Chat 1")
        sessions = _read_sessions_index().get("sessions", [])
    titles = [s.get("title", s.get("id", "Chat")) for s in sessions]
    ids = [s.get("id") for s in sessions]
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = ids[0]
        st.session_state.messages = _load_session_messages(ids[0])
    current_idx = ids.index(st.session_state.current_session_id) if st.session_state.current_session_id in ids else 0
    safe_index = int(current_idx) if isinstance(current_idx, int) else 0
    selection = st.selectbox("Select chat", options=list(range(len(ids))), index=safe_index, format_func=lambda i: titles[i])
    selected_index = int(selection) if selection is not None else 0
    selected_index = max(0, min(selected_index, len(ids) - 1))
    selected_id = ids[selected_index]
    if selected_id != st.session_state.current_session_id:
        st.session_state.current_session_id = selected_id
        st.session_state.messages = _load_session_messages(selected_id)
        st.rerun()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("New chat"):
            new_id = _create_session(f"Chat {len(ids) + 1}")
            st.session_state.current_session_id = new_id
            st.session_state.messages = []
            st.rerun()
    with col_b:
        if st.button("Delete chat"):
            try:
                os.remove(_session_file(st.session_state.current_session_id))
            except Exception:
                pass
            idx = _read_sessions_index()
            idx["sessions"] = [s for s in idx.get("sessions", []) if s.get("id") != st.session_state.current_session_id]
            _write_sessions_index(idx)
            idx2 = _read_sessions_index()
            if idx2.get("sessions"):
                st.session_state.current_session_id = idx2["sessions"][0]["id"]
                st.session_state.messages = _load_session_messages(st.session_state.current_session_id)
            else:
                st.session_state.current_session_id = _create_session("Chat 1")
                st.session_state.messages = []
            st.rerun()
    rename_source_idx = safe_index if 0 <= safe_index < len(titles) else 0
    rename_val = st.text_input("Rename chat", value=titles[rename_source_idx])
    if st.button("Save name"):
        idx = _read_sessions_index()
        for s in idx.get("sessions", []):
            if s.get("id") == st.session_state.current_session_id:
                new_name = (rename_val or s.get("title", "Chat"))
                s["title"] = new_name.strip()
        _write_sessions_index(idx)

# Initialize session state for chat history and settings
if "messages" not in st.session_state:
    st.session_state.messages = []
if clear:
    st.session_state.messages = []
    st.session_state.pop("summary_bytes", None)
    st.session_state.pop("summary_text", None)
    if st.session_state.get("current_session_id"):
        _save_session_messages(st.session_state.current_session_id, st.session_state.messages)

# ---------- Local NLP summarizer (no API) ----------
def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def _extract_corpus(messages: List[dict]) -> Tuple[str, List[str]]:
    # Combine assistant + user messages; keep assistant slightly higher weight by duplication
    texts: List[str] = []
    for m in messages:
        role = m.get("role", "")
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        if role == "assistant":
            texts.append(content)
            texts.append(content)  # weight assistant text
        else:
            texts.append(content)
    full_text = " \n".join(texts)
    return full_text, texts

def summarize_chat_locally(messages: List[dict], max_sentences: int = 8) -> Tuple[str, List[str]]:
    full_text, texts = _extract_corpus(messages)
    sentences = _split_sentences(full_text)
    if not sentences:
        return "No content to summarize.", []
    # TF-IDF scoring
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(sentences)
    def _sum_rows_sparse(mat) -> np.ndarray:
        try:
            return np.asarray(mat.sum(axis=1)).ravel()  # type: ignore[attr-defined]
        except Exception:
            return np.asarray(mat).sum(axis=1).ravel()
    scores = _sum_rows_sparse(X)
    # Select top sentences preserving original order
    ranked_idx = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))[:max_sentences]
    ranked_idx = sorted(ranked_idx)
    selected = [sentences[i] for i in ranked_idx]
    # Keyphrases from the entire corpus
    vectorizer2 = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=50)
    X2 = vectorizer2.fit_transform(texts if texts else [full_text])
    terms = vectorizer2.get_feature_names_out()
    def _sum_cols_sparse(mat) -> np.ndarray:
        try:
            return np.asarray(mat.sum(axis=0)).ravel()  # type: ignore[attr-defined]
        except Exception:
            return np.asarray(mat).sum(axis=0).ravel()
    weights = _sum_cols_sparse(X2)
    term_scores = list(zip(terms, weights))
    top_terms = [t for t, _ in sorted(term_scores, key=lambda x: -x[1])[:10]]
    # Format summary as bullets
    bullet_summary = "\n".join([f"- {s}" for s in selected])
    return bullet_summary, top_terms

def build_summary_pdf(summary_text: str, keyphrases: List[str]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=LETTER,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Chat Summary",
    )
    styles = getSampleStyleSheet()
    elems = []
    elems.append(Paragraph("Chat Summary", styles["Title"]))
    elems.append(Spacer(1, 0.2 * inch))
    elems.append(Paragraph("Key Phrases:", styles["Heading2"]))
    if keyphrases:
        elems.append(Paragraph(", ".join(keyphrases), styles["BodyText"]))
    else:
        elems.append(Paragraph("None", styles["BodyText"]))
    elems.append(Spacer(1, 0.2 * inch))
    elems.append(Paragraph("Summary:", styles["Heading2"]))
    for line in summary_text.split("\n"):
        if line.strip():
            elems.append(Paragraph(line.strip().replace("- ", "• "), styles["BodyText"]))
    doc.build(elems)
    buf.seek(0)
    return buf.read()

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    /* Hide default sidebar toggle ">" */
    [data-testid="collapsedControl"] { display:none !important; }
    
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #ffffff;
        border-radius: 20px;
        padding: 15px;
        border: 1px solid #4a4a4a;
    }
    
    .stButton > button {
        background-color: #00cc66;
        color: white;
        border-radius: 20px;
        padding: 10px 25px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #00b359;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,204,102,0.3);
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease;
        color: #ffffff;
    }
    
    .user-message {
        background-color: #1e3a8a;
        margin-left: auto;
        margin-right: 10px;
        border-bottom-right-radius: 5px;
    }
    
    .assistant-message {
        background-color: #262730;
        margin-right: auto;
        margin-left: 10px;
        border-bottom-left-radius: 5px;
    }
    
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
    
    .title-container { text-align:center; padding: 0.75rem 0 0.5rem; margin-bottom: 0.5rem; }
    
    .sports-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #1a1a1a;
        padding: 1rem;
        text-align: center;
        font-size: 0.8rem;
        border-top: 1px solid #333;
    }
    
    .features-list {
        background-color: #262730;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #4a4a4a;
    }
    
    .features-list ul {
        list-style-type: none;
        padding-left: 0;
    }
    
    .features-list li {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
        position: relative;
    }
    
    .features-list li:before {
        content: "🎯";
        position: absolute;
        left: 0;
    }
    .welcome-card {
        background: linear-gradient(180deg, rgba(30,58,138,0.5), rgba(0,204,102,0.2));
        padding: 1rem 1.25rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title section with gradient background
st.markdown('<div class="title-container"><h2>⚽ ProPlay Assistant</h2></div>', unsafe_allow_html=True)

# Features section
with st.expander("✨ What can I help you with?", expanded=False):
    st.markdown('''
    <div class="features-list">
    <ul>
        <li>Real-time sports news and updates</li>
        <li>Player statistics and performance analysis</li>
        <li>Team rankings and match results</li>
        <li>Tournament schedules and fixtures</li>
        <li>Sports rules and regulations</li>
        <li>Training and technique tips</li>
    </ul>
    </div>
    ''', unsafe_allow_html=True)

# 🟢 Live status banner
if 'live_updates' in locals() or 'live_updates' in globals():
    # prefer the variable from sidebar
    status_enabled = live_updates
else:
    # fallback default if not available
    status_enabled = True

if status_enabled:
    st.markdown('<div style="text-align:center; margin-bottom:0.5rem;">🟢 <strong>Live updates enabled</strong> — fetching recent news when relevant.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="text-align:center; margin-bottom:0.5rem;">🔴 <strong>Live updates disabled</strong> — using Groq responses only.</div>', unsafe_allow_html=True)

chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f'<div class="chat-message user-message">👤 You{(" [" + message.get("time", "") + "]") if message.get("time") else ""}: {message["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            content = message["content"]
            # Convert bullet-style text into HTML list for proper line breaks
            lines = [ln.strip() for ln in content.split('\n') if ln.strip()]
            if any(ln.startswith('- ') for ln in lines):
                items = ''.join([f'<li>{ln[2:].strip()}</li>' if ln.startswith('- ') else f'<li>{ln}</li>' for ln in lines])
                html_body = f'<ul style="margin:0 0 0 1rem;">{items}</ul>'
            else:
                html_body = '<br>'.join(lines)
            st.markdown(
                f'<div class="chat-message assistant-message">⚽ Assistant{(" [" + message.get("time", "") + "]") if message.get("time") else ""}: {html_body}</div>',
                unsafe_allow_html=True
            )

# Summarization action & download button
if summarize_clicked:
    summary, keyphrases = summarize_chat_locally(st.session_state.messages, max_sentences=8)
    st.session_state["summary_text"] = summary
    st.session_state["summary_bytes"] = build_summary_pdf(summary, keyphrases)
    st.success("Summary generated.")

if st.session_state.get("summary_bytes"):
    st.download_button(
        label="Download chat summary (PDF)",
        data=st.session_state["summary_bytes"],
        file_name="chat_summary.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

# Chat input (restored)
user_input = st.chat_input("Ask me anything about sports...")
if user_input:
    import datetime as _dt
    current_input = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": current_input, "time": _dt.datetime.now().strftime('%H:%M')})
    with st.spinner("🤔 Thinking..."):
        try:
            safe_model = model or "llama-3.1-70b-versatile"
            if status_enabled:
                # Call RealTimeSearchEngine which will enrich with NewsAPI results (if query warrants it)
                response = RealTimeSearchEngine(
                    current_input,
                    model=safe_model,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    top_p=0.9 if mode == "Fast" else 0.95,
                )
            else:
                # Offline/Groq-only placeholder — you may replace with a local-only model call if desired
                response = RealTimeSearchEngine(
                    current_input,
                    model=safe_model,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    top_p=0.9 if mode == "Fast" else 0.95,
                )
                # Note: search_engine will not call external NewsAPI if logic inside it checks secrets; 
                # if you need a pure offline path, modify RealTimeSearchEngine to accept a flag.
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response, "time": _dt.datetime.now().strftime('%H:%M')})
            else:
                st.session_state.messages.append({"role": "assistant", "content": "I couldn't generate a response. Please try again.", "time": _dt.datetime.now().strftime('%H:%M')})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": "An error occurred. Please try again.", "time": _dt.datetime.now().strftime('%H:%M')})
            st.error(f"Error: {str(e)}")
    if st.session_state.get("current_session_id"):
        _save_session_messages(st.session_state.current_session_id, st.session_state.messages)
    st.rerun()

# Footer
st.markdown(
    '<div class="footer">Made by Shaunak Lad and Satyapalsinh Chudasama | Bringing real-time sports insights to your fingertips 🌟</div>',
    unsafe_allow_html=True
)
