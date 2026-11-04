"""
agents/web_search_agent.py

Spoke especializado en inteligencia externa y búsqueda web mediante Tavily Search.
Utilizado para información pública en tiempo real de Banamex (sucursales, noticias corporativas,
indicadores macroeconómicos o comunicados externos).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv(Path(__file__).parent.parent / ".env")

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
COORDINATOR_MODEL = os.environ.get("COORDINATOR_MODEL", "qwen2.5:3b")


def run_web_search_agent(question: str) -> dict:
    """
    Ejecuta una búsqueda externa con Tavily y sintetiza los hallazgos con qwen2.5:3b.
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

        llm = ChatOllama(model=COORDINATOR_MODEL, temperature=0)
        prompt = PromptTemplate.from_template(
            """Eres el Analista de Inteligencia Externa de Banamex.
Sintetiza de forma clara y objetiva la información pública obtenida de Internet para responder a la duda del usuario.

### RESULTADOS DE BÚSQUEDA WEB:
{results}

### PREGUNTA DEL USUARIO:
{question}

### INSTRUCCIONES:
- Responde directamente al punto.
- Si hay enlaces o fechas relevantes, inclúyelos.
- Aclara que es información pública externa a los sistemas transaccionales del banco.
"""
        )

        chain = prompt | llm
        response = chain.invoke({"results": str(search_results), "question": question})
        summary_text = response.content if hasattr(response, "content") else str(response)

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
    question = state.get("question", "")
    result = run_web_search_agent(question)

    msg_content = f"### [EVIDENCIA WEB - TAVILY SEARCH]\n{result['summary']}"

    return {
        "evidences": {"web_evidence": result},
        "messages": [AIMessage(content=msg_content, name="web_search_agent")]
    }
