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

    # 🟢 NEW: Live Updates toggle
    live_updates = st.toggle("🌍 Enable Live Internet Updates (NewsAPI)", value=True,
                             help="Turn off to use offline Groq-only responses for faster answers.")

    clear = st.button("Clear chat", key="clear_sidebar")
    st.markdown("---")
    summarize_clicked = st.button("Summarize chat", key="summarize_sidebar")
    st.markdown("---")

    # Session management
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


# ---------- Local summarizer (same as before) ----------
def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]

def _extract_corpus(messages: List[dict]) -> Tuple[str, List[str]]:
    texts: List[str] = []
    for m in messages:
        content = str(m.get("content", "")).strip()
        if content:
            texts.append(content)
    return " \n".join(texts), texts

def summarize_chat_locally(messages: List[dict], max_sentences: int = 8) -> Tuple[str, List[str]]:
    full_text, texts = _extract_corpus(messages)
    sentences = _split_sentences(full_text)
    if not sentences:
        return "No content to summarize.", []
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(sentences)
    scores = np.asarray(X.sum(axis=1)).ravel()
    ranked_idx = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))[:max_sentences]
    ranked_idx = sorted(ranked_idx)
    selected = [sentences[i] for i in ranked_idx]
    vectorizer2 = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=50)
    X2 = vectorizer2.fit_transform(texts)
    terms = vectorizer2.get_feature_names_out()
    weights = np.asarray(X2.sum(axis=0)).ravel()
    term_scores = list(zip(terms, weights))
    top_terms = [t for t, _ in sorted(term_scores, key=lambda x: -x[1])[:10]]
    bullet_summary = "\n".join([f"- {s}" for s in selected])
    return bullet_summary, top_terms


def build_summary_pdf(summary_text: str, keyphrases: List[str]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER, title="Chat Summary")
    styles = getSampleStyleSheet()
    elems = [
        Paragraph("Chat Summary", styles["Title"]),
        Spacer(1, 0.2 * inch),
        Paragraph("Key Phrases:", styles["Heading2"]),
        Paragraph(", ".join(keyphrases) if keyphrases else "None", styles["BodyText"]),
        Spacer(1, 0.2 * inch),
        Paragraph("Summary:", styles["Heading2"]),
    ]
    for line in summary_text.split("\n"):
        if line.strip():
            elems.append(Paragraph(line.strip().replace("- ", "• "), styles["BodyText"]))
    doc.build(elems)
    buf.seek(0)
    return buf.read()


# ---------- Chat Area ----------
st.markdown('<div class="title-container"><h2>⚽ ProPlay Assistant</h2></div>', unsafe_allow_html=True)

chat_container = st.container()
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    css_class = "user-message" if role == "user" else "assistant-message"
    st.markdown(f'<div class="chat-message {css_class}">{content}</div>', unsafe_allow_html=True)

# ---------- User Input ----------
user_input = st.chat_input("Ask me anything about sports...")

if user_input:
    import datetime as _dt
    current_input = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": current_input, "time": _dt.datetime.now().strftime('%H:%M')})
    with st.spinner("🤔 Thinking..."):
        try:
            safe_model = model or "llama-3.1-70b-versatile"
            # 🟢 Pass toggle state to RealTimeSearchEngine
            if live_updates:
                response = RealTimeSearchEngine(
                    current_input,
                    model=safe_model,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    top_p=0.9 if mode == "Fast" else 0.95,
                )
            else:
                response = "Live updates are disabled. Enable them in the sidebar to get current news."
            st.session_state.messages.append({"role": "assistant", "content": response, "time": _dt.datetime.now().strftime('%H:%M')})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": "An error occurred. Please try again.", "time": _dt.datetime.now().strftime('%H:%M')})
            st.error(f"Error: {str(e)}")
    if st.session_state.get("current_session_id"):
        _save_session_messages(st.session_state.current_session_id, st.session_state.messages)
    st.rerun()

# Footer
st.markdown('<div class="footer">Made by Shaunak Lad and Satyapalsinh Chudasama | Bringing real-time sports insights to your fingertips 🌟</div>', unsafe_allow_html=True)
