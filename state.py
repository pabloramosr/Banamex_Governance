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
    # ----- Campos de Estado -----
    # ----- Canales de Estado -----
    # Libreta compartida que viaja de agente en agente a traves del grafo.

    # Historial de conversación manejado automáticamente por LangGraph
    # Sequence: Lista o Tupla
    # BaseMessage: No se guarda como str ya que tiene metadatos(HumanMessage, AIMessage, SystemMessage)
    # add_messages: Función que permite agregar mensajes al historial.
    # Annotated: Etiquetar o anotar para adjuntar funcion especial (add_messages).
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Pregunta actual del usuario
    question: str
    
    # Razonamiento y plan del Coordinador (Hub)
    # El Optioanl[str] indica que este campo puede ser de tipo String o None, depende de la etapa del flujo del sistema.
    # No puede haber una final_answer si no hay plan ni next_agent, por eso de deja como Optional.
    plan: Optional[str]
    
    # Siguiente agente a invocar ('sql_agent', 'rag_agent', 'web_search_agent', o 'FINISH')
    next_agent: Optional[str]
    
    # Almacén de evidencias recopiladas por los agentes especializados
    # Annotated: Etiquetar o anotar para adjuntar funcion especial (merge_evidences).
    evidences: Annotated[Dict[str, Any], merge_evidences]
    
    # Respuesta final formulada y validada por el Data Steward
    final_answer: Optional[str]

# Ajuste importante: Actualmente el spoke tiene como entrada el canal 'question' directamente, el coordinador
# solo se encarga de enviar la pregunta al spoke mas adecuado, no de enriquecer la pregunta con su propio razonamiento.
# Que el coordinador modifique el input de los spoke podria añadir facilidad para el spoke y hacer su tarea. Incluso
# se puede hacer que el coordinador use el campo semantico para orientar: 
# pregunta -> enriquecer con semantica(utilizando otro agente, como si los agentes refinaran la pregunta) -> spoke
# y asi intentar reducir posibles fallas de comprension del spoke.