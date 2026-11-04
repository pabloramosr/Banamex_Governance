"""
state.py

Definición del estado global (AgentState) para la arquitectura Hub-and-Spoke.
Mantiene el historial de mensajes, la traza de ruteo del supervisor y las evidencias.
"""

from typing import Annotated, Sequence, TypedDict, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def merge_evidences(existing: Dict[str, Any], new_evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Combina evidencias recopiladas por los distintos agentes spoke."""
    updated = dict(existing or {})
    if new_evidence:
        updated.update(new_evidence)
    return updated


class AgentState(TypedDict):
    # Historial de conversación manejado automáticamente por LangGraph
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Pregunta actual del usuario
    question: str
    
    # Razonamiento y plan del Coordinador (Hub)
    plan: Optional[str]
    
    # Siguiente agente a invocar ('sql_agent', 'rag_agent', 'web_search_agent', o 'FINISH')
    next_agent: Optional[str]
    
    # Almacén de evidencias recopiladas por los agentes especializados
    evidences: Annotated[Dict[str, Any], merge_evidences]
    
    # Respuesta final formulada y validada por el Data Steward
    final_answer: Optional[str]
