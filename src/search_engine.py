import os
import datetime
import re
from dotenv import dotenv_values
from groq import Groq

# --- Load Config (Secrets first, .env fallback) ---
Username = os.getenv("USERNAME")
Assistantname = os.getenv("ASSISTANTNAME")
GroqAPIKey = os.getenv("GROQ_API_KEY")

if not (Username and Assistantname and GroqAPIKey):
    env_vars = dotenv_values(os.path.join(os.path.dirname(__file__), "..", ".env"))
    Username = Username or env_vars.get("Username", "User")
    Assistantname = Assistantname or env_vars.get("Assistantname", "Sports Assistant")
    GroqAPIKey = GroqAPIKey or env_vars.get("GroqAPIKey", "")

# Initialize Groq client
client = Groq(api_key=GroqAPIKey) if GroqAPIKey else None

# --- System Prompt ---
System = f"""
Hello, I am {Username}. You are an advanced AI Sports Assistant named {Assistantname}, 
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

def GoogleSearch(query: str) -> str:
    """Perform a Google search with sports context and return a concise summary block.

    Includes title, description, and URL when available.
    """
    try:
        # Prefer authoritative sports sources when user asks about events
        lowered = query.lower()
        site_bias = (
            " site:uefa.com OR site:laliga.com OR site:fifa.com OR site:premierleague.com"
            " OR site:espn.com OR site:bbc.com OR site:skysports.com"
            " OR site:flashscore.com OR site:sofascore.com OR site:realmadrid.com"
            " OR site:espncricinfo.com OR site:cricbuzz.com OR site:icc-cricket.com"
        )
        if "real madrid" in lowered:
            site_bias += " OR site:realmadrid.com"
        if any(k in lowered for k in ["fixture", "match", "schedule", "next", "upcoming", "game"]):
            sports_query = f"{query} fixtures date time{site_bias}"
        elif any(k in lowered for k in ["last", "previous", "result", "score", "final score", "won", "lost"]):
            sports_query = f"{query} result score date{site_bias}"
        else:
            sports_query = f"{query} sports news stats results"
        results = list(search(sports_query, advanced=True, num_results=8))
        Answer = f"The sports-related search results for '{query}' are:\n[start]\n"
        for result in results:
            url = getattr(result, "url", "")
            Answer += (
                f"Title: {getattr(result, 'title', '')}\n"
                f"Description: {getattr(result, 'description', '')}\n"
                f"URL: {url}\n\n"
            )
        Answer += "[end]"
        return Answer
    except Exception:
        return "[start]\nNo live search results available at the moment.\n[end]"

def AnswerModifier(answer: str) -> str:
    lines = [ln.strip() for ln in answer.split('\n') if ln.strip()]
    if not lines:
        return ""
    # If content already looks like a list, keep as-is with normalization
    has_bullets = any(ln.lstrip().startswith(('- ', '* ', '• ')) for ln in lines)
    if has_bullets:
        normalized = [('- ' + ln.lstrip()[2:].strip()) if ln.lstrip().startswith(('* ', '• ')) else ln for ln in lines]
        return '\n'.join(normalized)
    # Otherwise, split into sentences and bulletize
    import re
    text = ' '.join(lines)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    bullets = [f"- {s.rstrip('.')}" for s in sentences]
    return '\n'.join(bullets)

def _extract_cricket_score(summary_block: str) -> str | None:
    # Look for patterns like IND 245/7, India 245 & 123, won by 5 wickets, etc.
    block = summary_block.lower()
    # Quick signals
    if not any(k in block for k in ["india", "ind", "west indies", "wi", "runs", "wickets", "/"]):
        return None
    # Try to find concise outcome lines
    m = re.search(r"(india|ind)[^\n]*\b(\d+(/\d+)?)([^\n]*)", block)
    n = re.search(r"(west\s*indies|\bwi\b)[^\n]*\b(\d+(/\d+)?)([^\n]*)", block)
    # Result phrase
    r = re.search(r"(won by|beat|defeated|draw|tie|tied)[^\n]*", block)
    parts = []
    if m:
        parts.append(f"- India: {m.group(2)}")
    if n:
        parts.append(f"- West Indies: {n.group(2)}")
    if r:
        parts.append(f"- Result: {r.group(0).capitalize()}")
    return '\n'.join(parts) if parts else None

def Information() -> str:
    current_date_time = datetime.datetime.now()
    return (
        f"Current information for sports updates:\n"
        f"Day: {current_date_time.strftime('%A')}\n"
        f"Date: {current_date_time.strftime('%d')}\n"
        f"Month: {current_date_time.strftime('%B')}\n"
        f"Year: {current_date_time.strftime('%Y')}\n"
        f"Time: {current_date_time.strftime('%H')} hours, "
        f"{current_date_time.strftime('%M')} minutes, "
        f"{current_date_time.strftime('%S')} seconds.\n"
    )

def RealTimeSearchEngine(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    top_p: float = 0.9
) -> str:
    try:
        # Handle basic greetings
        lowered = prompt.lower().strip()
        if lowered in ['hi', 'hello', 'hey'] or any(lowered.startswith(g) for g in ['hi ', 'hello ', 'hey ']):
            return "Hello! I'm your Sports Assistant. How can I help you with sports-related information today?"

        # Check if client is initialized (after greeting so we can always respond to greetings)
        if client is None:
            return (
                "I'm ready to chat about sports, but I need an API key to fetch live answers. "
                "Please set GroqAPIKey in your .env and restart the app."
            )

        # Load chat history
        chat_log_path = _ensure_chatlog_path()
        if not os.path.exists(chat_log_path):
            with open(chat_log_path, "w") as f:
                dump([], f)

        try:
            with open(chat_log_path, "r") as f:
                messages = load(f)
        except:
            messages = []

        # Shortcut: for event score queries, try extracting directly from search
        event_keywords = re.compile(r"\b(score|result|won|lost|beat|defeated|draw|tie|fixture|match)\b", re.I)
        if event_keywords.search(prompt):
            try:
                search_block = GoogleSearch(prompt)
                cricket = _extract_cricket_score(search_block)
                if cricket:
                    return AnswerModifier(cricket)
            except Exception:
                pass

        # Initialize conversation
        conversation = [
            {"role": "system", "content": System},
            {"role": "user", "content": prompt}
        ]

        # Add search results for longer queries only (reduces latency)
        # Always enrich if the user asks about match timing/fixtures; otherwise, enrich longer queries only
        import re as _re
        asks_events = bool(_re.search(r"\b(next|upcoming|match|fixture|schedule|game|last|previous|result|score)\b", prompt, _re.I))
        if asks_events or len(prompt.split()) > 3:
            try:
                search_result = GoogleSearch(prompt)
                conversation.append({"role": "system", "content": search_result})
            except:
                pass  # Continue without search results if search fails

        # Add current time information
        conversation.append({"role": "system", "content": Information()})

        # Choose models (backend will try fallbacks to avoid decommission errors)
        candidate_models = [
            m for m in [
                model,
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
            ] if m
        ]
        last_error: str | None = None
        response: str | None = None
        for candidate in candidate_models:
            try:
                completion = client.chat.completions.create(
                    model=candidate,
                    messages=conversation,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
                response = completion.choices[0].message.content
                break
            except Exception as e:  # Fallback on decommission or invalid model
                err_text = str(e)
                last_error = err_text
                if "model_decommissioned" in err_text or "decommissioned" in err_text or "invalid_request_error" in err_text:
                    continue
                # If it's another transient error, continue to next; otherwise keep last_error
                continue
        if response is None:
            raise Exception(last_error or "Model error")

        # Clean up the response
        response = response.strip().replace("</s", "")
        
        # Update chat history
        messages.append({"role": "user", "content": prompt})
        formatted = AnswerModifier(response)
        messages.append({"role": "assistant", "content": formatted})
        
        # Keep only recent messages (last 10 exchanges)
        messages = messages[-20:] if len(messages) > 20 else messages
        
        # Save chat history
        try:
            with open(chat_log_path, "w") as f:
                dump(messages, f, indent=4)
        except Exception:
            pass  # Continue if saving fails
        
        return formatted

    except Exception as e:
        if "hi" in prompt.lower() or "hello" in prompt.lower():
            return "Hello! I'm your Sports Assistant. How can I help you with sports-related information today?"
        return f"I apologize, but I encountered an error: {str(e)}. Please try again."
