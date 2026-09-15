from typing import List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: List[ChatMessage] = Field(default_factory=list, max_length=40)
    session_id: Optional[str] = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    intent: str
    sources: List[str] = Field(default_factory=list)


class ClearRequest(BaseModel):
    session_id: str = Field(..., min_length=8, max_length=64)


class ClearResponse(BaseModel):
    ok: bool
    session_id: str
