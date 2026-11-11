"""
agents/web_search_agent.py

Spoke especializado en inteligencia externa y búsqueda web mediante Tavily Search.
Utilizado para información pública en tiempo real de Banamex (sucursales, noticias corporativas,
indicadores macroeconómicos o comunicados externos).
Sintetiza con Google Gemini (gemini-3.5-flash).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv(Path(__file__).parent.parent / ".env")

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
COORDINATOR_MODEL = os.environ.get("COORDINATOR_MODEL", "gemini-3.5-flash")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


def get_web_llm():
    """Retorna Gemini si GOOGLE_API_KEY está configurada; fallback a Ollama local."""
    if GOOGLE_API_KEY:
        return ChatGoogleGenerativeAI(
            model=COORDINATOR_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )
    return ChatOllama(model="qwen2.5:3b", temperature=0)


def run_web_search_agent(question: str) -> dict:
    """
    Ejecuta una búsqueda externa con Tavily y sintetiza los hallazgos con Gemini 3.5 Flash.
    """
    if not TAVILY_API_KEY:
        return {
            "query": question,
            "raw_results": None,
            "summary": "Error: TAVILY_API_KEY no configurada en el archivo .env",
            "error": "TAVILY_API_KEY_MISSING"
        }

    try:
        tavily = TavilySearch(max_results=3, tavily_api_key=TAVILY_API_KEY)
        search_results = tavily.invoke(question)

        llm = get_web_llm()
        prompt = PromptTemplate.from_template(
            """Eres el Analista de Inteligencia Externa de Banamex.
Sintetiza de forma clara y objetiva la información pública obtenida de Internet para responder a la duda del usuario.

### RESULTADOS DE BÚSQUEDA WEB:
{results}

### PREGUNTA DEL USUARIO:
{question}

### INSTRUCCIONES:
- Responde directamente al punto con datos factuales y fechas.
- Si hay fuentes o enlaces relevantes, inclúyelos.
- Aclara que es información pública externa a los sistemas transaccionales del banco.
"""
        )

        chain = prompt | llm
        response = chain.invoke({"results": str(search_results), "question": question})
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            summary_text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        else:
            summary_text = str(content)

        return {
            "query": question,
            "raw_results": search_results,
            "summary": summary_text,
            "error": None
        }

    except Exception as e:
        return {
            "query": question,
            "raw_results": None,
            "summary": f"Error al ejecutar búsqueda web: {str(e)}",
            "error": str(e)
        }


def web_search_agent_node(state: dict) -> dict:
    """Nodo de LangGraph para el agente de búsqueda web."""
    instruction = state.get("spoke_instruction") or state.get("question", "")
    result = run_web_search_agent(instruction)

    msg_content = f"### [EVIDENCIA WEB - TAVILY SEARCH]\n{result['summary']}"

    return {
        "evidences": {"web_evidence": result},
        "messages": [AIMessage(content=msg_content, name="web_search_agent")]
    }
