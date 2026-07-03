from fastapi import APIRouter
from pydantic import BaseModel
from ai.chat_service import chat

router = APIRouter(prefix="/api/chat", tags=["chat"])

class ChatRequest(BaseModel):
    message: str

@router.post("/")
async def send_message(req: ChatRequest):
    response = await chat(req.message)
    return {"response": response}