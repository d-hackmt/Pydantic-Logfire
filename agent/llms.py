import os
from langchain_openai import ChatOpenAI

# GROQ

def groq_llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=temperature,
    )


# GEMINI

def gemini_llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-2.5-flash-lite",
        temperature=temperature,
    )
