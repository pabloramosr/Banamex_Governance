"""
agents/coordinator.py

HUB / ORQUESTADOR CENTRAL (Data Steward Supervisor).
Responsable de:
1. Analizar la intención del usuario considerando todo el historial de la conversación.
2. Discernir si la nueva pregunta se puede responder DIRECTAMENTE con la memoria existente
   o si requiere delegar a un nuevo Spoke especializado.
3. Enriquecer la consulta técnica mediante el canal 'spoke_instruction' para evitar ambigüedades.
4. Rutear al Spoke correcto:
   - 'sql_agent': Datos relacionales y métricas financieras en PostgreSQL.
   - 'rag_agent': Normativas y Términos y Condiciones en Pinecone.
   - 'web_search_agent': Información pública y actual en Internet con Tavily.
   - 'FINISH': Saludos, aclaraciones o preguntas que se responden con la memoria conversacional.
5. Sintetizar la respuesta final con enfoque de Data Stewardship y gobernanza.
"""

import os
import json
from pathlib import Path
from typing import Sequence, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(Path(__file__).parent.parent / ".env")

COORDINATOR_MODEL = os.environ.get("COORDINATOR_MODEL", "gemini-3.5-flash")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


def get_coordinator_llm():
    """Retorna Gemini si GOOGLE_API_KEY está configurada, o Ollama como fallback local."""
    if GOOGLE_API_KEY:
        return ChatGoogleGenerativeAI(
            model=COORDINATOR_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )
    return ChatOllama(model=COORDINATOR_MODEL, temperature=0)


def extract_content_text(response) -> str:
    """Extrae texto plano del contenido de respuesta, soportando bloques multimodales y strings."""
    content = response.content if hasattr(response, "content") else response
    if isinstance(content, list):
        text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    else:
        text = str(content)
    return clean_degenerative_repetitions(text)


def clean_degenerative_repetitions(text: str) -> str:
    """Detecta y poda bucles de repetición degenerativa generados accidentalmente por modelos de lenguaje."""
    lines = text.split("\n")
    cleaned_lines = []
    repeat_tracker = []

    for line in lines:
        stripped = line.strip()
        # Si la línea tiene contenido y se repite exactamente en las últimas 3 líneas
        if stripped and len(stripped) > 5 and stripped in repeat_tracker[-3:]:
            continue
        cleaned_lines.append(line)
        if stripped:
            repeat_tracker.append(stripped)

    return "\n".join(cleaned_lines)


def format_conversation_history(messages: Sequence[BaseMessage]) -> str:
    """Formatea el historial de mensajes para que el LLM tenga contexto de turnos anteriores."""
    if not messages:
        return "No hay conversación previa."
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append(f"Usuario: {msg.content}")
        elif isinstance(msg, AIMessage):
            speaker = getattr(msg, "name", None) or "Coordinador"
            # Omitir volcados técnicos crudos para mantener el contexto limpio
            content = str(msg.content)
            if content.startswith("### [EVIDENCIA"):
                continue
            history.append(f"{speaker}: {content.strip()}")
    return "\n".join(history) if history else "No hay conversación previa."


# ==============================================================================
# PROMPT DE DELEGACIÓN, ENRIQUECIMIENTO Y MEMORIA (ROUTING & ORCHESTRATION)
# ==============================================================================

ROUTING_PROMPT = """Eres el Data Steward y Coordinador Principal de la Plataforma de Datos de Banamex.
Tu rol es supervisar la conversación, razonar sobre el contexto y tomar una de dos decisiones:
A) Responder DIRECTAMENTE si la pregunta puede contestarse con el historial de la conversación o no requiere fuentes externas.
B) Delegar a un AGENTE ESPECIALIZADO enriqueciendo la consulta con los antecedentes conversacionales.

HISTORIAL DE LA CONVERSACIÓN PREVIA:
{history}

PREGUNTA ACTUAL DEL USUARIO:
{question}

AGENTES ESPECIALIZADOS DISPONIBLES:
1. 'sql_agent':
   - Usar para métricas numéricas, conteos y datos transaccionales en PostgreSQL.
   - Entidades: clientes (segmentos PREMIUM, PREFERENTE, MASIVO; género, tipo persona), productos (captación: cheques/ahorro; colocación: tarjeta de crédito/crédito personal), saldos mensuales y movimientos diarios.
   - Ejemplos: "¿Cuántos clientes tenemos?", "¿Cuántos de ellos son mujeres?", "¿Cuál es el saldo total en cuentas de ahorro?".

2. 'rag_agent':
   - Usar para normativas, Términos y Condiciones oficiales del Portal Banamex, privacidad, geolocalización, cookies y cláusulas de responsabilidad.
   - Ejemplos: "¿Qué dice Banamex sobre la geolocalización obligatoria?", "¿Cómo protegen mis datos?".

3. 'web_search_agent':
   - Usar para noticias recientes, información pública externa de Banamex, presencia en medios o eventos de mercado exterior.
   - Ejemplos: "¿Qué noticias recientes hay sobre la venta de Banamex?", "¿Qué horario tienen hoy las sucursales?".

4. 'FINISH' (Respuesta directa con memoria):
   - Usar cuando la pregunta se puede responder con lo que YA se dijo en el historial (ej. "explícame qué significa 250", "¿cuál fue el total que dijiste antes?", "¿por qué consideras ese dato importante?"), o para saludos/despedidas.

INSTRUCCIONES DE ENRIQUECIMIENTO:
- Si decides delegar a un spoke (ej. 'sql_agent'), genera en 'spoke_instruction' una instrucción técnica COMPLETA y SIN AMBIGÜEDADES, resolviendo pronombres y referencias anteriores.
  * Ejemplo: Si antes se habló de clientes PREMIUM y ahora el usuario dice "¿Cuántos son mujeres?", el 'spoke_instruction' debe ser:
    "De los clientes con segmento_cte = 'PREMIUM', contar cuántos tienen genero_cte = 'MUJER' en dim.dim_cliente".

FORMATO DE RESPUESTA OBLIGATORIO:
Responde ÚNICAMENTE un objeto JSON válido con la siguiente estructura (sin texto antes ni después):
{{
  "can_answer_directly": true | false,
  "direct_answer": "Respuesta completa si can_answer_directly es true; de lo contrario null",
  "next_agent": "sql_agent" | "rag_agent" | "web_search_agent" | "FINISH",
  "plan": "Explicación breve de 1 línea de tu decisión",
  "spoke_instruction": "Instrucción técnica contextualizada y enriquecida para el spoke (o null si direct_answer)"
}}
"""

SYNTHESIS_PROMPT = """Eres el Data Steward Oficial de Banamex.
Tu misión es presentar una respuesta clara, confiable, profesional y bien estructurada al usuario,
basándote en la evidencia técnica recién recopilada por tu equipo de agentes especializados.

PREGUNTA DEL USUARIO:
{question}

HISTORIAL CONVERSACIONAL:
{history}

EVIDENCIA RECIÉN RECOPILADA:
{evidence}

REGLAS DE GOBERNANZA, ESTRUCTURA Y COMUNICACIÓN BANCARIA:
1. Precisión matemática: Cita las cifras exactas del resultado sin alterarlas.
2. Trazabilidad de origen: Menciona claramente de qué fuente certificada provino el dato (ej. Base de Datos Transaccional PostgreSQL de Banamex [dim.dim_cliente], Términos Oficiales en Pinecone, o Inteligencia Web con Tavily).
3. Estructura concisa y limpia (MÁXIMO 3 SECCIONES):
   - Sección 1: Fuente y certificación del dato.
   - Sección 2: Resultados cuantitativos detallados (usa una tabla Markdown o viñetas limpias).
   - Sección 3: Conclusión ejecutiva / Data-Driven (máximo 1 párrafo breve).
4. REGLA ESTRICTA ANTI-REPETICIÓN:
   - PROHIBIDO repetir secciones, encabezados, subtítulos o frases ya escritas.
   - Presenta cada categoría y cifra UNA SOLA VEZ.
   - Concluye de inmediato con la firma del Data Steward y finaliza la respuesta. NO agregues notas redundantes después de la firma.
"""


def route_or_respond(question: str, messages: Sequence[BaseMessage]) -> dict:
    """
    Invoca a Gemini para decidir si la pregunta se resuelve con memoria o si se delega
    a un spoke con una instrucción enriquecida.
    """
    llm = get_coordinator_llm()
    history_str = format_conversation_history(messages)

    prompt = ROUTING_PROMPT.format(history=history_str, question=question)
    response = llm.invoke([
        SystemMessage(content="Eres el Supervisor de Gobernanza y Orquestador de Banamex. Responde estrictamente en formato JSON."),
        HumanMessage(content=prompt)
    ])
    content = extract_content_text(response).strip()

    # Limpieza de bloque JSON
    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(content)
        can_answer_directly = parsed.get("can_answer_directly", False)
        next_agent = parsed.get("next_agent", "sql_agent")
        
        if can_answer_directly or next_agent == "FINISH":
            return {
                "can_answer_directly": True,
                "direct_answer": parsed.get("direct_answer") or "Consulta atendida por el Oficial Data Steward.",
                "next_agent": "FINISH",
                "plan": parsed.get("plan", "Respuesta generada con la memoria y contexto previo."),
                "spoke_instruction": None
            }

        if next_agent not in ["sql_agent", "rag_agent", "web_search_agent"]:
            next_agent = "sql_agent"

        return {
            "can_answer_directly": False,
            "direct_answer": None,
            "next_agent": next_agent,
            "plan": parsed.get("plan", "Ruteo hacia agente especializado."),
            "spoke_instruction": parsed.get("spoke_instruction") or question
        }

    except Exception:
        # Fallback de seguridad heurístico si el parseo JSON fallara
        q_lower = question.lower()
        if any(w in q_lower for w in ["cuantos", "cuántos", "saldo", "cliente", "transaccion", "cuenta", "premium"]):
            return {
                "can_answer_directly": False,
                "direct_answer": None,
                "next_agent": "sql_agent",
                "plan": "Ruteo a análisis relacional por presencia de términos cuantitativos.",
                "spoke_instruction": question
            }
        return {
            "can_answer_directly": True,
            "direct_answer": "¡Hola! Soy el Agente Data Steward de Banamex. ¿En qué consulta analítica, normativa o transaccional puedo apoyarte hoy?",
            "next_agent": "FINISH",
            "plan": "Respuesta general directa.",
            "spoke_instruction": None
        }


def synthesize_response(question: str, evidences: dict, messages: Sequence[BaseMessage]) -> str:
    """Sintetiza la respuesta final del Data Steward con la evidencia recopilada e historial."""
    llm = get_coordinator_llm()
    history_str = format_conversation_history(messages)

    evidence_text = ""
    for k, v in (evidences or {}).items():
        summary = v.get("summary", "") if isinstance(v, dict) else str(v)
        evidence_text += f"\n--- [{k.upper()}] ---\n{summary}\n"

    if not evidence_text.strip():
        evidence_text = "No se requirieron fuentes externas para esta consulta."

    prompt = SYNTHESIS_PROMPT.format(question=question, history=history_str, evidence=evidence_text)
    response = llm.invoke([
        SystemMessage(content="Eres el Oficial Data Steward de Banamex. Redacta la respuesta final con máxima precisión y claridad."),
        HumanMessage(content=prompt)
    ])
    return extract_content_text(response)


def coordinator_node(state: dict) -> dict:
    """
    Nodo del Coordinador (Hub).
    Discrimina el momento del ciclo de vida del agente:
    - Si el último mensaje fue emitido por un Spoke especializado (retorno con evidencia fresca),
      sintetiza la respuesta final y finaliza el ciclo en FINISH.
    - Si el último mensaje es de un Humano (nueva consulta), decide si responder desde memoria
      o enriquecer y delegar a un nuevo Spoke limpiando evidencias obsoletas.
    """
    messages = state.get("messages", [])
    question = state.get("question", "")
    evidences = state.get("evidences", {})

    # Detectar si el último mensaje proviene de un spoke que acaba de finalizar su trabajo
    last_message = messages[-1] if messages else None
    is_spoke_return = bool(last_message and getattr(last_message, "name", None) in ["sql_agent", "rag_agent", "web_search_agent"])

    # --------------------------------------------------------------------------
    # FASE 2: SÍNTESIS DE EVIDENCIA (Regreso de un Spoke)
    # --------------------------------------------------------------------------
    if is_spoke_return and evidences:
        final_answer = synthesize_response(question, evidences, messages)
        return {
            "final_answer": final_answer,
            "next_agent": "FINISH",
            "spoke_instruction": None,
            "messages": [AIMessage(content=final_answer, name="coordinator")]
        }

    # --------------------------------------------------------------------------
    # FASE 1: NUEVA PREGUNTA (Evaluación de memoria vs delegación enriquecida)
    # --------------------------------------------------------------------------
    decision = route_or_respond(question, messages)

    # Caso A: Respuesta directa desde memoria o saludo
    if decision["can_answer_directly"] or decision["next_agent"] == "FINISH":
        direct_reply = decision["direct_answer"]
        return {
            "plan": decision["plan"],
            "next_agent": "FINISH",
            "spoke_instruction": None,
            "final_answer": direct_reply,
            "messages": [AIMessage(content=direct_reply, name="coordinator")]
        }

    # Caso B: Requiere delegar a un Spoke con pregunta enriquecida y evidencia limpia
    return {
        "plan": decision["plan"],
        "next_agent": decision["next_agent"],
        "spoke_instruction": decision["spoke_instruction"],
        "evidences": {}  # Se limpia la evidencia previa para que el Spoke deposite la fresca
    }
