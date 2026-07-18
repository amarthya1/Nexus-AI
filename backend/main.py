import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from openai import OpenAI
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("ai_router")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Environment — load from backend/.env
# ---------------------------------------------------------------------------
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# ---------------------------------------------------------------------------
# Service clients
# ---------------------------------------------------------------------------
groq_client: Optional[Groq] = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    logger.warning("GROQ_API_KEY is not configured; Groq client unavailable.")

openai_client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    openai_client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://models.inference.ai.azure.com",
    )
else:
    logger.warning("OPENAI_API_KEY is not configured; OpenAI client unavailable.")

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully.")
    except Exception as exc:
        logger.exception("Failed to initialize Supabase client: %s", exc)
else:
    logger.warning("SUPABASE_URL or SUPABASE_KEY missing; Supabase client unavailable.")

# ---------------------------------------------------------------------------
# FastAPI application  — CORS MUST be added immediately after init
# ---------------------------------------------------------------------------
app = FastAPI(title="AI Router API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    model_used: str
    intent: str
    session_id: Optional[str]


class SessionSummary(BaseModel):
    id: str
    title: str
    created_at: Optional[str]


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    model_used: Optional[str] = None
    intent: Optional[str] = None


# ---------------------------------------------------------------------------
# Database helpers  — every DB call is wrapped in try/except
# ---------------------------------------------------------------------------

def _ensure_supabase() -> Client:
    """Return the Supabase client or raise a clear HTTP 500."""
    if not supabase:
        logger.error("Supabase client not initialized.")
        raise HTTPException(
            status_code=500,
            detail="Database client is not configured. Please check SUPABASE_URL and SUPABASE_KEY.",
        )
    return supabase


def _execute_db(action: str, query: Any) -> Any:
    """Execute a Supabase query with full error handling."""
    try:
        response = query.execute()
    except Exception as exc:
        logger.exception("Database error while %s: %s", action, exc)
        raise HTTPException(status_code=500, detail=f"Database error while {action}.")

    data = getattr(response, "data", None)
    if data is None:
        logger.error("Database returned no data while %s", action)
        raise HTTPException(status_code=500, detail=f"Database returned no data while {action}.")

    return response


def _generate_session_title(first_prompt: str) -> str:
    """Create a short title from the first user prompt."""
    prompt = (first_prompt or "").strip()
    if not prompt:
        return "New chat"
    if len(prompt) <= 80:
        return prompt
    truncated = prompt[:80].rsplit(" ", 1)[0]
    return f"{truncated}..."


def _find_first_user_message(messages: List[Message]) -> Optional[str]:
    for message in messages:
        if message.role == "user" and message.content.strip():
            return message.content.strip()
    return None


def _find_last_user_message(messages: List[Message]) -> Optional[str]:
    for message in reversed(messages):
        if message.role == "user" and message.content.strip():
            return message.content.strip()
    return None


def _create_chat_session(first_prompt: str) -> str:
    """Insert a new row into chat_sessions and return its UUID."""
    client = _ensure_supabase()
    session_title = _generate_session_title(first_prompt)
    try:
        response = _execute_db(
            "create chat session",
            client.table("chat_sessions").insert({"title": session_title}).select("id"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error creating chat session: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to create chat session.")

    data = getattr(response, "data", None)
    if not data or not isinstance(data, list) or not data[0].get("id"):
        logger.error("Unexpected response while creating chat session: %s", data)
        raise HTTPException(status_code=500, detail="Unable to create chat session.")
    return str(data[0]["id"])


def _save_chat_message(
    session_id: str,
    sender: str,
    content: str,
    model_used: Optional[str] = None,
    intent: Optional[str] = None,
) -> None:
    """Insert a new row into chat_messages."""
    client = _ensure_supabase()
    payload: Dict[str, Any] = {
        "session_id": session_id,
        "sender": sender,
        "content": content,
    }
    if model_used is not None:
        payload["model_used"] = model_used
    if intent is not None:
        payload["intent"] = intent
    try:
        _execute_db(
            f"save {sender} message",
            client.table("chat_messages").insert(payload),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error saving %s message: %s", sender, exc)
        raise HTTPException(status_code=500, detail=f"Failed to save {sender} message.")


def to_openai_messages(messages: List[Message]) -> List[Dict[str, str]]:
    """Convert internal Message list to the OpenAI-compatible dict format."""
    return [{"role": m.role, "content": m.content} for m in messages]


# ---------------------------------------------------------------------------
# AI routing helpers
# ---------------------------------------------------------------------------
CLASSIFIER_PROMPT = (
    "You are an intent classifier. Given the user's message, respond with EXACTLY one word:\n"
    "CODE — for programming, debugging, algorithms, technical tasks\n"
    "CREATIVE — for creative writing, poetry, storytelling, brainstorming\n"
    "GENERAL — for everything else\n\n"
    "Respond with only: CODE, CREATIVE, or GENERAL"
)

# Model configuration
GROQ_MODEL = "llama-3.1-70b-versatile"
OPENAI_MODEL = "gpt-4o-mini"


def classify_intent(last_user_message: str) -> str:
    """Use OpenAI (gpt-4o-mini) to classify intent; fallback to GENERAL on error."""
    if not openai_client:
        logger.warning("OpenAI client unavailable; defaulting intent to GENERAL.")
        return "GENERAL"
    try:
        completion = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user", "content": last_user_message},
            ],
            max_tokens=5,
            temperature=0,
        )
        content = (completion.choices[0].message.content or "").strip().upper()
        return content if content in {"CODE", "CREATIVE", "GENERAL"} else "GENERAL"
    except Exception as exc:
        logger.exception("Intent classification failed: %s", exc)
        return "GENERAL"


def _extract_response_text(choice: Any) -> str:
    """Pull the text content out of an API completion choice."""
    if not choice:
        return ""
    if hasattr(choice, "message"):
        return getattr(choice.message, "content", None) or ""
    return getattr(choice, "text", None) or ""


def _call_openai(messages: List[Message]) -> Dict[str, str]:
    """Call OpenAI gpt-4o-mini and return response + model label."""
    if not openai_client:
        raise RuntimeError("OpenAI client is not configured.")
    completion = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=to_openai_messages(messages),
    )
    return {
        "response": _extract_response_text(completion.choices[0]),
        "model_used": f"OpenAI / {OPENAI_MODEL}",
    }


def _call_groq(messages: List[Message]) -> Dict[str, str]:
    """Call Groq llama-3.1-70b-versatile and return response + model label."""
    if not groq_client:
        raise RuntimeError("Groq client is not configured.")
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=to_openai_messages(messages),
    )
    return {
        "response": _extract_response_text(completion.choices[0]),
        "model_used": f"Groq / {GROQ_MODEL}",
    }


def _get_ai_response(messages: List[Message], intent: str) -> Dict[str, str]:
    """
    Route to the correct provider based on intent, with automatic fallback.

    Routing rules:
      • CODE  → Groq (llama-3.1-70b-versatile) primary, OpenAI fallback
      • CREATIVE / GENERAL → OpenAI (gpt-4o-mini) primary, Groq fallback
    """
    if intent == "CODE":
        primary_fn = _call_groq
        fallback_fn = _call_openai
        primary_label = "Groq"
        fallback_label = "OpenAI"
    else:
        primary_fn = _call_openai
        fallback_fn = _call_groq
        primary_label = "OpenAI"
        fallback_label = "Groq"

    # --- Attempt primary provider ---
    try:
        logger.info("Attempting %s for intent=%s", primary_label, intent)
        result = primary_fn(messages)
        logger.info("Success via %s", primary_label)
        return result
    except Exception as primary_exc:
        logger.warning(
            "%s failed for intent=%s: %s — falling back to %s",
            primary_label, intent, primary_exc, fallback_label,
        )

    # --- Attempt fallback provider ---
    try:
        logger.info("Attempting fallback via %s", fallback_label)
        result = fallback_fn(messages)
        result["model_used"] += " (fallback)"
        logger.info("Success via %s (fallback)", fallback_label)
        return result
    except Exception as fallback_exc:
        logger.exception(
            "Both %s and %s failed: %s",
            primary_label, fallback_label, fallback_exc,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                f"All AI providers failed. "
                f"{primary_label}: {primary_exc}  |  "
                f"{fallback_label}: {fallback_exc}"
            ),
        )


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Main chat endpoint — classifies intent, routes to AI, persists to Supabase."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="'messages' list cannot be empty.")

    first_user_prompt = _find_first_user_message(request.messages)
    if not first_user_prompt:
        raise HTTPException(status_code=400, detail="At least one user message is required.")

    last_user_msg = _find_last_user_message(request.messages)
    if not last_user_msg:
        raise HTTPException(status_code=400, detail="Unable to locate the latest user message.")

    # Create or reuse session
    session_id = request.session_id or _create_chat_session(first_user_prompt)

    # Classify and route
    intent = classify_intent(last_user_msg)
    ai_result = _get_ai_response(request.messages, intent)

    # Persist messages AFTER the AI call succeeds — no orphaned records
    try:
        _save_chat_message(session_id, "user", last_user_msg)
        _save_chat_message(
            session_id,
            "assistant",
            ai_result["response"],
            model_used=ai_result["model_used"],
            intent=intent,
        )
    except Exception as exc:
        # Log but don't fail the response — the AI answer was already generated
        logger.exception("Failed to persist messages to database: %s", exc)

    return ChatResponse(
        response=ai_result["response"],
        model_used=ai_result["model_used"],
        intent=intent,
        session_id=session_id,
    )


@app.get("/api/sessions")
def get_all_sessions() -> Dict[str, List[SessionSummary]]:
    """Return all chat sessions ordered by most recent first."""
    client = _ensure_supabase()
    try:
        response = _execute_db(
            "fetch chat sessions",
            client.table("chat_sessions")
            .select("id,title,created_at")
            .order("created_at", desc=True),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error fetching sessions: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch chat sessions.")

    sessions = getattr(response, "data", []) or []
    return {"sessions": sessions}


@app.get("/api/chat/history/{session_id}")
def get_chat_history(session_id: str) -> Dict[str, List[ChatHistoryItem]]:
    """Return all messages for a given session in chronological order."""
    client = _ensure_supabase()
    try:
        response = _execute_db(
            "fetch chat history",
            client.table("chat_messages")
            .select("sender,content,model_used,intent,created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=False),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error fetching chat history: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch chat history.")

    rows = getattr(response, "data", []) or []
    history: List[ChatHistoryItem] = []
    for row in rows:
        sender = (row.get("sender") or "").lower()
        role = "assistant" if sender in {"assistant", "bot"} else "user"
        history.append(
            ChatHistoryItem(
                role=role,
                content=row.get("content", ""),
                model_used=row.get("model_used"),
                intent=row.get("intent"),
            )
        )

    return {"history": history}


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "message": "AI Router API is running 🚀"}


# ---------------------------------------------------------------------------
# Entrypoint — run with: python main.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
