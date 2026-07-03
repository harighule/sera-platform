import httpx
from config import AI_API_KEY, AI_MODEL, AI_BASE_URL
from core.entity_resolution import entity_registry

SYSTEM_PROMPT = """You are SERA-AI, the intelligent query interface for the SERA Intelligence Platform.
You help analysts understand the real-time behavioral intelligence data flowing through the platform.
The platform tracks entities (people, assets, processes) across financial, healthcare, IoT, and social domains.
You answer questions about entities, entropy readings, predictions, and data streams.
Be concise, analytical, and use technical precision. Keep answers under 150 words unless asked for detail."""

async def chat(user_message: str) -> str:
    """Send a message to the AI and return the response."""
   
    if not AI_API_KEY:
        return _mock_response(user_message)

    entities = entity_registry.get_all()
    pre_transition = [e for e in entities if e["status"] == "pre-transition"]
    context = f"\nCurrent platform state: {len(entities)} entities tracked, {len(pre_transition)} in pre-transition state."

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{AI_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {AI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": AI_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT + context},
                        {"role": "user", "content": user_message}
                    ],
                    "max_tokens": 300,
                    "temperature": 0.7
                }
            )
            try:
                data = response.json()
                if "error" in data:
                    return f"AI service error: {data['error'].get('message', 'Unknown error')}. Please check the API key in your .env file."
                if "choices" in data:
                    return data["choices"][0]["message"]["content"]
                if "content" in data:
                    return data["content"][0]["text"]
                return "AI response format not recognised. Please check API configuration."
            except Exception as e:
                return f"Failed to parse AI response: {str(e)}"
    except Exception as e:
        return f"AI service error: {str(e)}. Check your API key and connection."

def _mock_response(message: str) -> str:
    msg = message.lower()
    entities = entity_registry.get_all()
    pre_transition = [e for e in entities if e["status"] == "pre-transition"]
    if "entropy" in msg:
        return f"Current platform entropy analysis: {len(pre_transition)} of {len(entities)} entities show elevated entropy. AXIOM-Φ is monitoring all behavioral channels for phase transitions."
    elif "entity" in msg or "entities" in msg:
        return f"Platform is tracking {len(entities)} resolved entities across financial, healthcare, IoT, and social domains. {len(pre_transition)} are currently flagged as pre-transition."
    elif "predict" in msg or "zola" in msg:
        return "ZOLA has generated intervention briefs for all pre-transition entities. Success probability ranges from 65% to 92% depending on the behavioral transition type."
    else:
        return "SERA Intelligence Platform is operating normally. All data streams are active. Ask me about entities, entropy levels, predictions, or data stream status."