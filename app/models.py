import datetime
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key_value = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), default="Default Key")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)

class ChatSession(Base):
    __tablename__ = "chats"

    id = Column(String(255), primary_key=True, index=True)
    title = Column(String(255), default="New Conversation")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String(255), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(50), nullable=False)  # system, user, assistant, tool
    content = Column(Text, nullable=False)
    search_results = Column(Text, nullable=True)  # Context scraped from web search
    embedding = Column(Text, nullable=True)  # JSON-encoded array of floats
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    chat = relationship("ChatSession", back_populates="messages")

class SearchCache(Base):
    __tablename__ = "search_cache"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(String(255), unique=True, index=True, nullable=False)
    results_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    tags = Column(String(255), nullable=True)  # comma-separated tags e.g. "lua,roblox,vector3"
    embedding = Column(Text, nullable=True)  # JSON-encoded array of floats
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class LanguageGuideline(Base):
    __tablename__ = "language_guidelines"

    id = Column(Integer, primary_key=True, index=True)
    language_name = Column(String(255), unique=True, index=True, nullable=False)
    keywords = Column(Text, nullable=True)  # comma-separated keywords e.g. "lua,luau,roblox"
    instructions = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
