"""
agents/llm.py

Fábrica única del LLM (Google Gemini) para el Hub y los Spokes.
No existe fallback local: si falta GOOGLE_API_KEY se lanza un ConfigError,
que es un error terminal (main.py lo muestra y termina el proceso).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(Path(__file__).parent.parent / ".env")

DEFAULT_MODEL = "gemini-3.5-flash"


class ConfigError(RuntimeError):
    """Error terminal de configuración: el sistema no puede continuar hasta corregir el .env."""


def require_google_api_key() -> str:
    """Retorna GOOGLE_API_KEY o lanza ConfigError si no está definida."""
    api_key = (os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise ConfigError(
            "GOOGLE_API_KEY no está configurada. Todos los agentes usan Google Gemini y no hay "
            "modelo local de respaldo. Define GOOGLE_API_KEY en el archivo .env y vuelve a ejecutar."
        )
    return api_key


def get_gemini_llm(model: str | None = None) -> ChatGoogleGenerativeAI:
    """Construye el cliente de Gemini; lanza ConfigError si falta la API key."""
    return ChatGoogleGenerativeAI(
        model=model or DEFAULT_MODEL,
        google_api_key=require_google_api_key(),
        temperature=0,
    )
