"""
agents/coordinator.py

HUB / ORQUESTADOR CENTRAL (Data Steward Supervisor).
Responsable de:
1. Analizar la intención del usuario considerando todo el historial de la conversación.
2. BLINDAJE DE DOMINIO: Rechazar directamente consultas ajenas al ámbito financiero y bancario de Banamex.
3. Discernir si la nueva pregunta se puede responder DIRECTAMENTE con la memoria existente
   o si requiere delegar a un spoke especializado.
4. Enriquecer la consulta técnica mediante el canal 'spoke_instruction' para evitar ambigüedades.
5. Rutear al Spoke correcto:
   - 'sql_agent': Datos relacionales y métricas financieras en PostgreSQL.
   - 'rag_agent': Documentación oficial, contratos, comisiones y normativas en Pinecone.
   - 'FINISH': Saludos, aclaraciones de memoria o consultas fuera de dominio.
6. Sintetizar la respuesta final con enfoque de Data Stewardship y gobernanza.
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
            content = str(msg.content)
            if content.startswith("### [EVIDENCIA"):
                continue
            history.append(f"{speaker}: {content.strip()}")
    return "\n".join(history) if history else "No hay conversación previa."


# ==============================================================================
# PROMPT DE DELEGACIÓN, BLINDAJE DE DOMINIO Y ENRIQUECIMIENTO
# ==============================================================================

ROUTING_PROMPT = """Eres el Data Steward y Coordinador Principal de la Plataforma de Datos de Grupo Financiero Banamex.
Tu rol es supervisar la conversación, razonar sobre el contexto y tomar una decisión entre:
A) Responder DIRECTAMENTE si la pregunta puede contestarse con el historial previo, si es un saludo, o si está FUERA DE DOMINIO.
B) Delegar a un SPOKE ESPECIALIZADO enriqueciendo la consulta técnica.

HISTORIAL DE LA CONVERSACIÓN PREVIA:
{history}

PREGUNTA ACTUAL DEL USUARIO:
{question}

POLÍTICA ESTRICTA DE GOBERNANZA Y DOMINIO BANCARIO:
- Eres el Data Steward exclusivo de Grupo Financiero Banamex.
- Si la pregunta del usuario es AJENA al ámbito de Banamex (por ejemplo: deportes, partidos, entretenimiento, recetas, clima, política externa o cultura general no financiera), DEBES:
  1. Definir "can_answer_directly": true
  2. Definir "next_agent": "FINISH"
  3. Redactar en "direct_answer" una negativa cortés y profesional indicando que como Data Steward Oficial de Grupo Financiero Banamex únicamente tienes facultades para atender consultas sobre datos de clientes, cuentas, transacciones, contratos, comisiones y normativas oficiales de Banamex.

SPOKES ESPECIALIZADOS DISPONIBLES:
1. 'sql_agent':
   - Usar para métricas cuantitativas, agrupaciones, conteos y saldos en la base de datos relacional PostgreSQL de Banamex.
   - Entidades: clientes (segmentos PREMIUM, PREFERENTE, MASIVO; género, tipo persona), productos de captación y colocación, saldos mensuales y movimientos diarios.
   - Ejemplos: "¿Cuántos clientes tenemos?", "¿Cuántos son mujeres?", "¿Qué cuentas puede tener un cliente?".

2. 'rag_agent':
   - Usar para buscar en la documentación oficial de Banamex indexada en Pinecone (51 documentos oficiales).
   - Incluye: contratos maestros de tarjetas de crédito y cuentas, carátulas oficiales de comisiones, folletos comerciales, créditos (personal, nómina, hipotecario), inversiones y términos y condiciones del portal web.
   - Ejemplos: "¿Cuáles son las comisiones de la tarjeta Costco?", "¿Qué comisiones cobra MiCuenta Banamex?", "¿Cuál es el procedimiento para aclaraciones de cargos?".

3. 'FINISH' (Respuesta directa con memoria o de dominio):
   - Usar cuando la pregunta se puede responder con lo que YA se dijo en el historial previo, para saludos/despedidas, o para consultas fuera de dominio bancario.

INSTRUCCIONES DE ENRIQUECIMIENTO:
- Si decides delegar a un spoke (ej. 'sql_agent' o 'rag_agent'), genera en 'spoke_instruction' una instrucción técnica COMPLETA y SIN AMBIGÜEDADES, resolviendo pronombres y referencias anteriores.
  * Ejemplo: Si antes se habló de clientes PREMIUM y ahora el usuario dice "¿Cuántos son mujeres?", el 'spoke_instruction' debe ser:
    "De los clientes con segmento_cte = 'PREMIUM', contar cuántos tienen genero_cte = 'MUJER' en dim.dim_cliente".
  * Ejemplo RAG: Si el usuario dice "¿Cuánto cobran de anualidad?", el 'spoke_instruction' debe ser:
    "Comisión por anualidad de tarjetas de crédito Banamex en anexos de comisiones oficiales".

FORMATO DE RESPUESTA OBLIGATORIO:
Responde ÚNICAMENTE un objeto JSON válido con la siguiente estructura (sin texto antes ni después):
{{
  "can_answer_directly": true | false,
  "direct_answer": "Respuesta completa si can_answer_directly es true; de lo contrario null",
  "next_agent": "sql_agent" | "rag_agent" | "FINISH",
  "plan": "Explicación breve de 1 línea de tu decisión",
  "spoke_instruction": "Instrucción técnica contextualizada para el spoke (o null si direct_answer)"
}}
"""

SYNTHESIS_PROMPT = """Eres el Data Steward Oficial de Grupo Financiero Banamex.
Tu misión es presentar una respuesta clara, confiable, profesional y bien estructurada al usuario,
basándote en la evidencia técnica recopilada por tu equipo de spokes especializados.

PREGUNTA DEL USUARIO:
{question}

HISTORIAL CONVERSACIONAL:
{history}

EVIDENCIA RECIÉN RECOPILADA:
{evidence}

REGLAS DE GOBERNANZA, ESTRUCTURA Y COMUNICACIÓN BANCARIA:
1. Precisión matemática y citas documentales:
   - Si la evidencia proviene de SQL, cita las cifras y tablas exactas (ej. dim.dim_cliente).
   - Si la evidencia proviene de RAG (Pinecone), cita explícitamente el documento oficial (ej. anexo_costco.pdf, contrato_cuenta_switch.pdf), las páginas de referencia y los montos exactos en pesos o porcentajes de comisiones/CAT.
2. Trazabilidad de origen: Menciona claramente de qué fuente oficial provino el dato.
3. Estructura concisa y limpia (MÁXIMO 3 SECCIONES):
   - Sección 1: Fuente y certificación del dato.
   - Sección 2: Resultados detallados (usa viñetas limpias o tablas Markdown).
   - Sección 3: Conclusión ejecutiva / Dictamen de gobernanza (máximo 1 párrafo breve).
4. REGLA ESTRICTA ANTI-REPETICIÓN:
   - PROHIBIDO repetir secciones, encabezados o frases ya escritas.
   - Concluye de inmediato con la firma del Data Steward y finaliza la respuesta.
"""


def route_or_respond(question: str, messages: Sequence[BaseMessage]) -> dict:
    """
    Invoca a Gemini para decidir si la pregunta se resuelve con memoria o fuera de dominio,
    o si se delega a un spoke con una instrucción enriquecida.
    """
    llm = get_coordinator_llm()
    history_str = format_conversation_history(messages)

    prompt = ROUTING_PROMPT.format(history=history_str, question=question)
    response = llm.invoke([
        SystemMessage(content="Eres el Supervisor de Gobernanza y Orquestador de Banamex. Responde estrictamente en formato JSON."),
        HumanMessage(content=prompt)
    ])

    raw_text = extract_content_text(response).strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[-1]
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("```", 1)[0]
    raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
        can_answer = parsed.get("can_answer_directly", False)
        next_agent = parsed.get("next_agent", "FINISH")

        if can_answer or next_agent == "FINISH":
            return {
                "can_answer_directly": True,
                "direct_answer": parsed.get("direct_answer") or "Consulta resuelta directamente.",
                "next_agent": "FINISH",
                "plan": parsed.get("plan", "Respuesta directa de gobernanza o memoria."),
                "spoke_instruction": None
            }

        if next_agent not in ["sql_agent", "rag_agent"]:
            next_agent = "sql_agent"

        return {
            "can_answer_directly": False,
            "direct_answer": None,
            "next_agent": next_agent,
            "plan": parsed.get("plan", "Ruteo hacia spoke especializado."),
            "spoke_instruction": parsed.get("spoke_instruction") or question
        }

    except Exception:
        # Fallback de seguridad heurístico
        q_lower = question.lower()
        if any(w in q_lower for w in ["cuantos", "cuántos", "saldo", "cliente", "transaccion", "cuenta", "premium"]):
            return {
                "can_answer_directly": False,
                "direct_answer": None,
                "next_agent": "sql_agent",
                "plan": "Ruteo a análisis relacional por presencia de términos cuantitativos.",
                "spoke_instruction": question
            }
        if any(w in q_lower for w in ["comision", "comisión", "contrato", "cat", "anualidad", "terminos", "términos", "requisito", "costco"]):
            return {
                "can_answer_directly": False,
                "direct_answer": None,
                "next_agent": "rag_agent",
                "plan": "Ruteo a documentación oficial de Banamex.",
                "spoke_instruction": question
            }
        return {
            "can_answer_directly": True,
            "direct_answer": "Como Data Steward Oficial de Grupo Financiero Banamex, estoy a tu disposición para consultas analíticas sobre clientes, saldos, transacciones, normativas y contratos oficiales del banco.",
            "next_agent": "FINISH",
            "plan": "Respuesta institucional de Data Steward.",
            "spoke_instruction": None
        }


def synthesize_response(question: str, evidences: dict, messages: Sequence[BaseMessage]) -> str:
    """Sintetiza la respuesta final del Data Steward con la evidencia documental o transaccional recopilada."""
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
    - Si el último mensaje es de un Spoke especializado (retorno con evidencia), sintetiza la respuesta final y finaliza en FINISH.
    - Si el último mensaje es del usuario, decide si responder directamente (memoria/fuera de dominio) o delegar al Spoke.
    """
    messages = state.get("messages", [])
    question = state.get("question", "")
    evidences = state.get("evidences", {})

    last_message = messages[-1] if messages else None
    is_spoke_return = bool(last_message and getattr(last_message, "name", None) in ["sql_agent", "rag_agent"])

    # FASE 2: Síntesis de retorno de un Spoke
    if is_spoke_return and evidences:
        final_answer = synthesize_response(question, evidences, messages)
        return {
            "final_answer": final_answer,
            "next_agent": "FINISH",
            "plan": "Síntesis oficial finalizada por el Data Steward.",
            "messages": [AIMessage(content=final_answer, name="coordinator")]
        }

    # FASE 1: Análisis y decisión de ruteo
    decision = route_or_respond(question, messages)

    if decision["can_answer_directly"]:
        answer = decision["direct_answer"]
        return {
            "plan": decision["plan"],
            "next_agent": "FINISH",
            "final_answer": answer,
            "spoke_instruction": None,
            "evidences": {},
            "messages": [AIMessage(content=answer, name="coordinator")]
        }

    return {
        "plan": decision["plan"],
        "next_agent": decision["next_agent"],
        "spoke_instruction": decision["spoke_instruction"],
        "evidences": {},
        "final_answer": None
    }
