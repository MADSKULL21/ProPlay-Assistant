from groq import Groq
from json import load, dump
import requests
import datetime
import streamlit as st
import os
import re

# ✅ Load configuration from Streamlit Secrets only
Username = st.secrets.get("USERNAME", "User")
Assistantname = st.secrets.get("ASSISTANTNAME", "Sports Assistant")
GroqAPIKey = st.secrets.get("GROQ_API_KEY")
NewsAPIKey = st.secrets.get("NEWS_API_KEY")

# ✅ Initialize Groq client
client = Groq(api_key=GroqAPIKey) if GroqAPIKey else None


# ---------- Utility Functions ----------
def _ensure_chatlog_path() -> str:
    """Ensure Data directory exists and return ChatLog path."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, "Data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "ChatLog.json")


# ---------- System Prompt ----------
System = f"""Hello, I am {Username}. You are an advanced AI Sports Assistant named {Assistantname}, 
which provides real-time, up-to-date information about sports from the internet.

Your expertise includes:
- Latest sports news and updates
- Player statistics and performance analysis
- Team rankings and match results
- Tournament schedules and fixtures
- Sports rules and regulations
- Training and technique advice

*** Provide answers in a professional manner with correct grammar, punctuation, and clarity. ***
*** Format all answers as concise bullet points. Use short sentences, each starting with "- ". ***
*** Focus on providing accurate sports-related information based on the provided data. ***
*** If a query is not sports-related, politely redirect the user to ask sports-related questions. ***
"""


# ---------- Live News via NewsAPI ----------
def GoogleSearch(query: str) -> str:
    """Fetch live sports updates using NewsAPI."""
    if not NewsAPIKey:
        return "[start]\nNo NewsAPI key found. Please add NEWS_API_KEY to Streamlit secrets.\n[end]"

    try:
        url = (
            "https://newsapi.org/v2/everything?"
            f"q={query}+sports&"
            "language=en&"
            "sortBy=publishedAt&"
            "pageSize=5&"
            f"apiKey={NewsAPIKey}"
        )
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200 or "articles" not in data:
            return f"[start]\nFailed to fetch live updates (HTTP {response.status_code}).\n[end]"

        articles = data.get("articles", [])
        if not articles:
            return f"[start]\nNo recent updates found for '{query}'.\n[end]"

        # Format NewsAPI response
        answer = f"Here are the latest sports updates for '{query}':\n[start]\n"
        for art in articles:
            title = art.get("title", "No title available")
            desc = art.get("description", "")
            url = art.get("url", "")
            published = art.get("publishedAt", "").replace("T", " ").replace("Z", "")
            answer += f"Title: {title}\nDescription: {desc}\nPublished: {published}\nURL: {url}\n\n"
        answer += "[end]"
        return answer

    except Exception as e:
        return f"[start]\nError fetching live data: {e}\n[end]"


# ---------- Text Formatting ----------
def AnswerModifier(answer: str) -> str:
    """Normalize or bulletize the model’s responses."""
    lines = [ln.strip() for ln in answer.split('\n') if ln.strip()]
    if not lines:
        return ""

    has_bullets = any(ln.lstrip().startswith(('- ', '* ', '• ')) for ln in lines)
    if has_bullets:
        normalized = [('- ' + ln.lstrip()[2:].strip()) if ln.lstrip().startswith(('* ', '• ')) else ln for ln in lines]
        return '\n'.join(normalized)

    text = ' '.join(lines)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    bullets = [f"- {s.rstrip('.')}" for s in sentences]
    return '\n'.join(bullets)


# ---------- Optional Score Extractor ----------
def _extract_cricket_score(summary_block: str) -> str | None:
    """Extract quick cricket scores from summaries."""
    block = summary_block.lower()
    if not any(k in block for k in ["india", "ind", "west indies", "wi", "runs", "wickets", "/"]):
        return None

    m = re.search(r"(india|ind)[^\n]*\b(\d+(/\d+)?)([^\n]*)", block)
    n = re.search(r"(west\s*indies|\bwi\b)[^\n]*\b(\d+(/\d+)?)([^\n]*)", block)
    r = re.search(r"(won by|beat|defeated|draw|tie|tied)[^\n]*", block)

    parts = []
    if m:
        parts.append(f"- India: {m.group(2)}")
    if n:
        parts.append(f"- West Indies: {n.group(2)}")
    if r:
        parts.append(f"- Result: {r.group(0).capitalize()}")

    return '\n'.join(parts) if parts else None


# ---------- Date / Time Info ----------
def Information() -> str:
    """Generate dynamic date and time for context."""
    current_date_time = datetime.datetime.now()
    return (
        f"Current information for sports updates:\n"
        f"Day: {current_date_time.strftime('%A')}\n"
        f"Date: {current_date_time.strftime('%d')}\n"
        f"Month: {current_date_time.strftime('%B')}\n"
        f"Year: {current_date_time.strftime('%Y')}\n"
        f"Time: {current_date_time.strftime('%H:%M:%S')}.\n"
    )


# ---------- Main Chat Engine ----------
def RealTimeSearchEngine(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    top_p: float = 0.9
) -> str:
    """Main chat logic combining Groq + live search data."""
    try:
        lowered = prompt.lower().strip()
        if lowered in ['hi', 'hello', 'hey'] or any(lowered.startswith(g) for g in ['hi ', 'hello ', 'hey ']):
            return "Hello! I'm your Sports Assistant. How can I help you with sports-related information today?"

        if client is None:
            return "⚠️ Missing Groq API key. Please add GROQ_API_KEY in Streamlit secrets."

        chat_log_path = _ensure_chatlog_path()
        if not os.path.exists(chat_log_path):
            with open(chat_log_path, "w") as f:
                dump([], f)

        try:
            with open(chat_log_path, "r") as f:
                messages = load(f)
        except:
            messages = []

        # Add live search enrichment
        event_keywords = re.compile(r"\b(score|result|won|lost|beat|defeated|draw|tie|fixture|match|news|update)\b", re.I)
        conversation = [{"role": "system", "content": System}, {"role": "user", "content": prompt}]

        if event_keywords.search(prompt) or len(prompt.split()) > 3:
            try:
                search_result = GoogleSearch(prompt)
                conversation.append({"role": "system", "content": search_result})
            except Exception:
                pass

        conversation.append({"role": "system", "content": Information()})

        # Try multiple models in fallback order
        candidate_models = [m for m in [model, "llama-3.1-8b-instant", "mixtral-8x7b-32768"] if m]
        response, last_error = None, None

        for candidate in candidate_models:
            try:
                completion = client.chat.completions.create(
                    model=candidate,
                    messages=conversation,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
                response = completion.choices[0].message.content
                break
            except Exception as e:
                last_error = str(e)
                continue

        if response is None:
            raise Exception(last_error or "No response generated from Groq API.")

        response = response.strip().replace("</s", "")
        messages.append({"role": "user", "content": prompt})
        formatted = AnswerModifier(response)
        messages.append({"role": "assistant", "content": formatted})

        # Limit chat history
        messages = messages[-20:] if len(messages) > 20 else messages

        # Save chat log
        try:
            with open(chat_log_path, "w") as f:
                dump(messages, f, indent=4)
        except Exception:
            pass

        return formatted

    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try again."
