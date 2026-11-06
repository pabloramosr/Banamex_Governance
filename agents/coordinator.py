"""
agents/coordinator.py

HUB / ORQUESTADOR CENTRAL (Data Steward Supervisor).
Responsable de:
1. Analizar la intención del usuario y trazar un plan.
2. Rutear la consulta al Spoke correcto:
   - 'sql_agent': Datos cuantitativos en PostgreSQL (clientes, saldos, cuentas, transacciones).
   - 'rag_agent': Normativas y Términos y Condiciones en Pinecone (privacidad, cookies, geolocalización).
   - 'web_search_agent': Información pública y actual en Internet (noticias, eventos, mercado).
   - 'FINISH': Saludos o consultas que no requieren búsqueda.
3. Sintetizar la respuesta final con enfoque de Data Stewardship y gobernanza.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, SystemMessage
from langchain_ollama import ChatOllama

load_dotenv(Path(__file__).parent.parent / ".env")

COORDINATOR_MODEL = os.environ.get("COORDINATOR_MODEL", "qwen2.5:3b")


ROUTING_PROMPT = """Eres el Data Steward y Coordinador Principal de la Plataforma de Datos de Banamex.
Tu función es analizar la pregunta del usuario y seleccionar el AGENTE ESPECIALIZADO adecuado.

AGENTES DISPONIBLES:
1. 'sql_agent':
   - Usar para preguntas numéricas, agregaciones, conteos y métricas de negocio.
   - Tablas disponibles: clientes, cuentas, productos (cheques, ahorro, tarjeta de crédito, crédito personal), saldos mensuales y movimientos/transacciones.
   - Ejemplos: "¿Cuántos clientes tenemos en segmento PREMIUM?", "¿Cuál fue el saldo total en cuentas de ahorro?", "Listar transacciones de retiro".

2. 'rag_agent':
   - Usar para preguntas sobre los Términos y Condiciones del Portal Banamex, políticas de uso, privacidad, geolocalización, cookies, propiedad intelectual y responsabilidades de los usuarios.
   - Ejemplos: "¿Qué dice Banamex sobre la geolocalización?", "¿Cuáles son las reglas sobre las cookies en el portal?", "¿Quién es responsable de las claves de acceso?".

3. 'web_search_agent':
   - Usar para información pública externa, noticias recientes de Banamex, presencia institucional, eventos públicos o indicadores de mercado no contenidos en la base interna.
   - Ejemplos: "¿Cuáles son las noticias recientes sobre la venta de Banamex?", "¿Qué servicios ofrece Citibanamex en sucursales hoy?".

4. 'FINISH':
   - Usar únicamente para saludos, despedidas o preguntas triviales que no requieren datos.

INSTRUCCIÓN:
Responde ÚNICAMENTE un objeto JSON válido con este formato:
{
  "plan": "Breve explicación de 1 línea del por qué eliges este agente",
  "next_agent": "sql_agent" | "rag_agent" | "web_search_agent" | "FINISH"
}
No incluyas explicaciones fuera del JSON.
"""

SYNTHESIS_PROMPT = """Eres el Data Steward Oficial de Banamex.
Tu misión es presentar una respuesta clara, confiable, profesional y bien estructurada al usuario, basándote en la evidencia recopilada por tu equipo de agentes.

PREGUNTA DEL USUARIO:
{question}

EVIDENCIA RECOPILADA:
{evidence}

REGLAS DE GOBERNANZA Y COMUNICACIÓN:
1. Responde con precisión matemática y sustento documental.
2. Menciona claramente de qué fuente provino el dato (ej. Base de Datos Transaccional PostgreSQL de Banamex, Términos y Condiciones Oficiales en Pinecone, o Búsqueda Pública Web con Tavily).
3. Mantén un tono formal, bancario y transparente ("Data-Driven").
"""

# def route_question(messages: Sequence[BaseMessage]) -> dict:
def route_question(question: str) -> dict:
    """Invoca al LLM para determinar el ruteo hacia el spoke adecuado."""
    llm = ChatOllama(model=COORDINATOR_MODEL, temperature=0)
    messages = [
        SystemMessage(content=ROUTING_PROMPT),
        {"role": "user", "content": f"Pregunta: {question}"}
    ]
    response = llm.invoke(messages)
    content = response.content if hasattr(response, "content") else str(response)

    # Limpieza para extraer JSON
    try:
        content = content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        parsed = json.loads(content)
        next_agent = parsed.get("next_agent", "sql_agent")
        if next_agent not in ["sql_agent", "rag_agent", "web_search_agent", "FINISH"]:
            next_agent = "sql_agent"
        return {
            "plan": parsed.get("plan", "Ruteo automático hacia agente especializado."),
            "next_agent": next_agent
        }
    except Exception:
        # Fallback heurístico si el LLM no formatea JSON
        q_lower = question.lower()
        if any(w in q_lower for w in ["cuantos", "cuántos", "saldo", "cliente", "transaccion", "cuenta", "premium", "deposito"]):
            return {"plan": "Detección de consulta cuantitativa sobre clientes o saldos.", "next_agent": "sql_agent"}
        elif any(w in q_lower for w in ["terminos", "términos", "condicion", "condición", "privacidad", "geolocalizacion", "cookie"]):
            return {"plan": "Detección de consulta sobre términos y normativas de Banamex.", "next_agent": "rag_agent"}
        elif any(w in q_lower for w in ["noticia", "internet", "web", "sucursal"]):
            return {"plan": "Detección de consulta de información pública exterior.", "next_agent": "web_search_agent"}
        else:
            return {"plan": "Consulta general canalizada a análisis relacional.", "next_agent": "sql_agent"}


def synthesize_response(question: str, evidences: dict) -> str:
    """Sintetiza la respuesta final del Data Steward con toda la evidencia reunida."""
    llm = ChatOllama(model=COORDINATOR_MODEL, temperature=0)
    
    evidence_text = ""
    for k, v in (evidences or {}).items():
        summary = v.get("summary", "") if isinstance(v, dict) else str(v)
        evidence_text += f"\n--- [{k.upper()}] ---\n{summary}\n"

    if not evidence_text.strip():
        evidence_text = "No se requirieron fuentes externas para esta consulta."

    prompt = SYNTHESIS_PROMPT.format(question=question, evidence=evidence_text)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content if hasattr(response, "content") else str(response)


def coordinator_node(state: dict) -> dict:
    """
    Nodo del Coordinador (Hub).
    - Si aún no hay evidencias de ningún Spoke, rutea la pregunta.
    - Si ya hay evidencia recopilada, sintetiza la respuesta final y finaliza.
    """
    evidences = state.get("evidences", {})
    question = state.get("question", "")

    # Si ya tenemos evidencia de algún agente spoke, sintetizamos
    if evidences:
        final_answer = synthesize_response(question, evidences)
        return {
            "final_answer": final_answer,
            "next_agent": "FINISH",
            "messages": [AIMessage(content=final_answer, name="coordinator")]
        }

    # Primera pasada: determinar hacia qué Spoke enviar la pregunta
    decision = route_question(question)
    plan = decision["plan"]
    next_agent = decision["next_agent"]

    if next_agent == "FINISH":
        direct_reply = "¡Hola! Soy el Agente Data Steward de Banamex. ¿En qué consulta analítica, normativa o transaccional puedo apoyarte hoy?"
        return {
            "plan": plan,
            "next_agent": "FINISH",
            "final_answer": direct_reply,
            "messages": [AIMessage(content=direct_reply, name="coordinator")]
        }

    return {
        "plan": plan,
        "next_agent": next_agent
    }

