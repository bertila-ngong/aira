import asyncio
import json
import logging
import base64
import re
import time
import os
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from api.deps import get_db
from core.security import decode_access_token
from agents.aira_agent import AIRAAgent
from agents.goal_planner import GoalPlanner
from models.user import User
from models.session import Session
from services.gemini_vision import GeminiVisionService
from api.routes.browser import get_browser_agent
from agents.desktop_agent import DesktopAgent

logger = logging.getLogger("aira.voice_route")

router = APIRouter(prefix="/voice", tags=["Voice"])

_last_executed_queries: dict[str, float] = {}
LAST_ACTION_COOLDOWN_SEC = 10  # Reduced from 90s — was silently dropping repeated requests

_desktop_agent = DesktopAgent()


async def get_user_from_token(token: str, db: AsyncSession):
    payload = decode_access_token(token)
    if not payload:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


def extract_search_query(user_text: str, aira_text: str) -> str | None:
    """
    Try to extract the search query from user_text first (most reliable),
    then fall back to parsing aira_text.
    """
    # 1. Try user text directly — strip filler phrases
    if user_text:
        t = user_text.strip()
        filler = re.compile(
            r'^(?:hey aira[,\s]*|aira[,\s]*|can you[,\s]*|could you[,\s]*|please[,\s]*|'
            r'i want you to[,\s]*|i\'d like you to[,\s]*)',
            re.IGNORECASE
        )
        t = filler.sub('', t).strip()
        action_prefix = re.compile(
            r'^(?:search(?:\s+(?:google|the\s+web|online))?\s+for\s+|'
            r'look\s+up\s+|find\s+(?:info(?:rmation)?\s+(?:on|about)\s+)?|'
            r'google\s+|search\s+|can\s+you\s+search\s+for\s+|'
            r'open\s+(?:up\s+)?|go\s+to\s+|navigate\s+to\s+|'
            r'browse\s+(?:to\s+)?|pull\s+up\s+)',
            re.IGNORECASE
        )
        cleaned = action_prefix.sub('', t).strip().strip('"\'.,?!')
        if len(cleaned) > 2 and len(cleaned) < 200:
            return cleaned

    # 2. Fall back to quoted strings in aira_text
    if aira_text:
        quoted = re.findall(r'"([^"]{3,})"', aira_text)
        if quoted:
            return quoted[0]

        patterns = [
            r'search(?:ing)?\s+(?:google\s+)?for\s+"?(.+?)"?(?:\.|,|$)',
            r'look(?:ing)?\s+up\s+"?(.+?)"?(?:\.|,|$)',
            r'find(?:ing)?\s+(?:information\s+(?:on|about)\s+)?"?(.+?)"?(?:\.|,|$)',
            r'(?:open|opening|navigate|navigating)\s+(?:to\s+)?"?(.+?)"?(?:\.|,|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, aira_text.lower())
            if match:
                query = match.group(1).strip().strip('"\'')
                if len(query) > 3:
                    return query

    return None


def extract_music_query(text: str) -> str | None:
    t = text.lower().strip()
    # Strip platform suffix before extracting
    t = re.sub(r'\s+on\s+(youtube|spotify|music|apple music|tidal)\s*$', '', t)
    t = re.sub(r'\s+in\s+(youtube|spotify|music|apple music|tidal)\s*$', '', t)
    patterns = [
        r'play\s+(?:me\s+)?(?:the\s+song\s+)?["\'"](.+?)["\'"]',
        r'play\s+(?:me\s+)?(.+)',
        r'(?:search|find)\s+(.+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, t)
        if match:
            q = match.group(1).strip().strip('"\'.,')
            if len(q) > 1:
                return q
    return None


def extract_url(aira_text: str) -> str | None:
    urls = re.findall(r'https?://[^\s,]+', aira_text)
    if urls:
        return urls[0]
    sites = re.findall(r'(?:open|go to|navigate to|visit)\s+([\w.]+\.(?:com|org|net|io|co))', aira_text.lower())
    if sites:
        return f"https://{sites[0]}"
    return None


def classify_command(user_text: str, aira_text: str) -> str:
    """
    Returns: 'music' | 'app' | 'browser' | 'none'

    Priority: check user_text first (intent is clearest there),
    then use aira_text as confirmation.
    """
    combined = (user_text + " " + aira_text).lower()
    user_lower = user_text.lower()

    music_platform = _desktop_agent.detect_music_platform(combined)
    app_key = _desktop_agent.detect_app(combined)

    has_play = "play" in user_lower
    has_music_word = any(w in user_lower for w in ["song", "music", "track", "album", "artist"])

    has_launch = any(w in user_lower for w in ["open", "launch", "start", "run"])

    # Expanded browser keywords — check user_text primarily
    browser_keywords = [
        "search", "google", "look up", "look it up", "find", "navigate",
        "browse", "website", "url", "go to", "open youtube", "youtube",
        "what is", "what are", "who is", "how to", "how do", "tell me about",
        "show me", "research", "check", "i want to know", "can you find",
        "pull up", "bring up",
    ]
    has_browser_word = any(w in user_lower for w in browser_keywords)

    # Music: play intent + platform or music keyword
    if has_play and (music_platform or has_music_word):
        return "music"
    if music_platform and has_play:
        return "music"

    # App launch: launch keyword + known app
    if app_key and has_launch and not has_browser_word:
        return "app"

    # Browser: search/navigation intent in user text
    if has_browser_word:
        return "browser"

    # Final fallback: check aira_text for browser intent signals
    aira_browser_signals = [
        "i'll search", "let me search", "searching", "pulling up",
        "loading", "i'm opening", "navigating to", "looking up",
        "i found", "here are the results", "according to",
    ]
    if any(s in aira_text.lower() for s in aira_browser_signals):
        return "browser"

    return "none"


@router.get("/status")
async def voice_status() -> dict:
    return {"status": "Voice WebSocket service is running"}


@router.websocket("/stream")
async def voice_stream(
    websocket: WebSocket,
    token: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    await websocket.accept()

    user = await get_user_from_token(token, db)
    if not user:
        await websocket.send_json({"type": "error", "message": "Invalid authentication token"})
        await websocket.close(code=4001)
        return

    db_session = Session(user_id=user.id, session_type="voice", status="active")
    db.add(db_session)
    await db.flush()
    session_id = str(db_session.id)

    agent = AIRAAgent(user=user, db=db)
    connected = await agent.initialize(session_id=session_id)

    if not connected:
        await websocket.send_json({"type": "error", "message": "AIRA failed to connect to Gemini."})
        await websocket.close(code=4002)
        return

    await websocket.send_json({
        "type": "session_started",
        "session_id": session_id,
        "message": f"Hello {user.full_name.split()[0]}, I am AIRA. How can I help you?",
        "interruptions_enabled": True,
    })

    goal_planner = GoalPlanner()
    vision_service = GeminiVisionService()
    browser = get_browser_agent()

    aira_text_buffer: list[str] = []
    last_user_message: list[str] = [""]
    browser_action_fired: list[bool] = [False]
    allow_interruptions: list[bool] = [True]
    aira_is_processing: list[bool] = [False]

    async def handle_turn_complete():
        full_text = " ".join(aira_text_buffer).strip()
        aira_text_buffer.clear()
        aira_is_processing[0] = False

        if not full_text:
            return

        if browser_action_fired[0]:
            logger.info("turn_complete ignored — action already fired")
            return

        user_text = last_user_message[0]
        action_type = classify_command(user_text, full_text)
        logger.info(
            f"Command classified as: {action_type} | "
            f"user='{user_text[:80]}' | aira='{full_text[:80]}'"
        )

        if action_type == "none":
            return

        action_key = user_text.strip().lower()[:80] or full_text[:60]
        now = time.time()
        if now - _last_executed_queries.get(action_key, 0) < LAST_ACTION_COOLDOWN_SEC:
            logger.info(f"Cooldown active — skipping '{action_key}'")
            return

        browser_action_fired[0] = True
        _last_executed_queries[action_key] = now
        stale = [k for k, v in _last_executed_queries.items() if now - v > 300]
        for k in stale:
            del _last_executed_queries[k]

        combined = (user_text + " " + full_text).lower()

        try:
            if action_type == "music":
                platform = _desktop_agent.detect_music_platform(combined) or "youtube"
                music_query = extract_music_query(user_text) or user_text.strip()
                if not music_query or len(music_query) > 100:
                    quoted = re.findall(r'"([^"]{2,60})"', full_text)
                    music_query = quoted[0] if quoted else None
                if not music_query or len(music_query) > 100:
                    logger.warning(f"Bad music_query: {music_query!r} — aborting")
                    browser_action_fired[0] = False
                    _last_executed_queries.pop(action_key, None)
                    return
                logger.info(f"Music: platform={platform} query={music_query}")

                browser_pref = "chrome"
                for b, keywords in [
                    ("chrome", ["google chrome", "chrome"]),
                    ("firefox", ["firefox"]),
                    ("edge", ["edge", "microsoft edge"]),
                ]:
                    if any(k in combined for k in keywords):
                        browser_pref = b
                        break

                if not browser.is_running:
                    await browser.start(browser=browser_pref)
                    await asyncio.sleep(2)

                if platform == "spotify":
                    result = await browser.spotify_search(music_query)
                elif platform in ("apple music", "apple_music"):
                    result = await browser.apple_music_search(music_query)
                else:
                    result = await browser.youtube_search(music_query)

                if result.get("success"):
                    await browser.keep_alive()
                    await websocket.send_json({
                        "type": "goal_plan",
                        "plan": {
                            "goal_summary": f"Here are the results for '{music_query}' on {platform}",
                            "requires_confirmation": False,
                            "steps": [{
                                "step": 1,
                                "action": f"Searched {platform} for '{music_query}' — click a result to play",
                                "type": "music",
                                "details": music_query,
                                "status": "completed",
                            }],
                        },
                    })
                else:
                    logger.error(f"Music action failed: {result.get('error')}")
                    browser_action_fired[0] = False
                    _last_executed_queries.pop(action_key, None)

            elif action_type == "app":
                app_key_name = _desktop_agent.detect_app(combined)
                logger.info(f"App launch: {app_key_name}")
                result = await _desktop_agent.launch_app(app_key_name)
                if result.get("success"):
                    await websocket.send_json({
                        "type": "goal_plan",
                        "plan": {
                            "goal_summary": f"Launching {app_key_name}",
                            "requires_confirmation": False,
                            "steps": [{
                                "step": 1,
                                "action": f"Launched {app_key_name}",
                                "type": "app_launch",
                                "details": app_key_name,
                                "status": "completed",
                            }],
                        },
                    })
                else:
                    browser_action_fired[0] = False
                    _last_executed_queries.pop(action_key, None)

            elif action_type == "browser":
                # Pass user_text into extract_search_query for better accuracy
                query = extract_search_query(user_text, full_text)
                url = extract_url(full_text) or extract_url(user_text)
                logger.info(f"Browser: query={query!r} url={url!r}")

                browser_pref = "chrome"
                for b, keywords in [
                    ("chrome", ["google chrome", "chrome"]),
                    ("firefox", ["firefox"]),
                    ("edge", ["edge", "microsoft edge"]),
                ]:
                    if any(k in combined for k in keywords):
                        browser_pref = b
                        break

                if not browser.is_running:
                    await browser.start(browser=browser_pref)
                    await asyncio.sleep(2)

                if "youtube" in combined and query:
                    result = await browser.youtube_search(query)
                elif url:
                    result = await browser.navigate(url)
                elif query:
                    result = await browser.search_google(query)
                else:
                    # Last resort: search whatever the user said
                    result = await browser.search_google(user_text)

                if result.get("success"):
                    await websocket.send_json({
                        "type": "goal_plan",
                        "plan": {
                            "goal_summary": query or user_text or "Browser search",
                            "requires_confirmation": False,
                            "steps": [{
                                "step": 1,
                                "action": f"Search: {query or user_text}",
                                "type": "search",
                                "details": query or user_text,
                                "status": "completed",
                            }],
                        },
                    })
                else:
                    logger.error(f"Browser action failed: {result.get('error')}")
                    browser_action_fired[0] = False
                    _last_executed_queries.pop(action_key, None)

        except Exception as e:
            logger.error(f"Action failed [{action_type}]: {e}", exc_info=True)
            browser_action_fired[0] = False
            _last_executed_queries.pop(action_key, None)

    async def stream_gemini_responses():
        try:
            async for response in agent.gemini_live.receive_responses():
                rtype = response.get("type")

                if rtype == "audio":
                    aira_is_processing[0] = True
                    await websocket.send_json({
                        "type": "audio",
                        "data": base64.b64encode(response["data"]).decode("utf-8"),
                        "mime_type": "audio/pcm;rate=24000",
                    })

                elif rtype == "text":
                    aira_is_processing[0] = True
                    text = response["data"]
                    agent.add_to_transcript("aira", text)
                    if not browser_action_fired[0]:
                        aira_text_buffer.append(text)
                    await websocket.send_json({
                        "type": "transcript",
                        "role": "aira",
                        "text": text,
                    })

                elif rtype == "user_transcript":
                    user_text = response["data"]
                    last_user_message[0] = user_text
                    browser_action_fired[0] = False
                    agent.add_to_transcript("user", user_text)
                    await websocket.send_json({
                        "type": "transcript",
                        "role": "user",
                        "text": user_text,
                    })

                elif rtype == "turn_complete":
                    await handle_turn_complete()
                    await websocket.send_json({"type": "turn_complete"})

                elif rtype == "interrupted":
                    aira_text_buffer.clear()
                    aira_is_processing[0] = False
                    browser_action_fired[0] = False
                    await websocket.send_json({"type": "interrupted"})

                elif rtype in ("connection_closed", "error"):
                    await websocket.send_json({
                        "type": "error",
                        "message": response.get("message", "Connection lost"),
                    })
                    break

        except Exception as e:
            logger.error(f"Gemini stream error: {e}")

    response_task = asyncio.create_task(stream_gemini_responses())

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = message.get("type")

            if msg_type == "audio":
                if aira_is_processing[0] and not allow_interruptions[0]:
                    continue
                audio_bytes = base64.b64decode(message["data"])
                await agent.process_audio(audio_bytes)

            elif msg_type == "set_interruptions":
                allow_interruptions[0] = bool(message.get("enabled", True))
                await websocket.send_json({
                    "type": "interruptions_updated",
                    "enabled": allow_interruptions[0],
                })

            elif msg_type == "text":
                user_text = message.get("data", "").strip()
                if user_text:
                    last_user_message[0] = user_text
                    browser_action_fired[0] = False
                    agent.add_to_transcript("user", user_text)
                    await websocket.send_json({
                        "type": "transcript",
                        "role": "user",
                        "text": user_text,
                    })
                    await agent.process_text(user_text)

            elif msg_type == "screen_context":
                data = message.get("data", "")
                if message.get("is_image"):
                    image_bytes = base64.b64decode(data)
                    description = await vision_service.describe_screenshot(image_bytes)
                    await agent.inject_screen_context(description)
                    await websocket.send_json({"type": "screen_analyzed", "description": description})
                else:
                    await agent.inject_screen_context(data)

            elif msg_type == "interrupt":
                if allow_interruptions[0]:
                    await agent.interrupt()

            elif msg_type == "end_session":
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Voice stream error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        response_task.cancel()
        try:
            await response_task
        except asyncio.CancelledError:
            pass

        memories_saved = await agent.end_session()
        agent.last_screen_context = ""

        from datetime import datetime, timezone
        db_session.status = "ended"
        db_session.ended_at = datetime.now(timezone.utc)
        db_session.transcript = json.dumps(agent.session_transcript)
        db_session.total_turns = len(agent.session_transcript)
        await db.flush()

        try:
            await websocket.send_json({"type": "session_ended", "memories_saved": memories_saved})
        except Exception:
            pass

        logger.info(f"Session {session_id} ended.")