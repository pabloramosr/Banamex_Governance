"""
workflow.py

Orquestación del Grafo Multi-Agente (Hub-and-Spoke) con LangGraph.
Define el flujo cíclico:
   [START] -> [Coordinador Hub]
                  │
        ┌─────────┼──────────┐
        ▼         ▼          ▼
   [SQL Spoke] [RAG Spoke] [Web Spoke]
        │         │          │
        └─────────┼──────────┘
                  ▼
          [Coordinador Hub] -> [Sintetizar y FINISH] -> [END]

Incluye memoria de checkpointing (MemorySaver) para recordar la sesión de consultas.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from agents.coordinator import coordinator_node
from agents.sql_agent import sql_agent_node
from agents.rag_agent import rag_agent_node
from agents.web_search_agent import web_search_agent_node


def route_next_spoke(state: AgentState) -> str:
    """Función de enrutamiento condicional basada en la decisión del Coordinador."""
    next_agent = state.get("next_agent", "FINISH")
    if next_agent in ["sql_agent", "rag_agent", "web_search_agent"]:
        return next_agent
    return END


def create_hub_spoke_graph():
    """Construye y compila el flujo StateGraph de LangGraph."""
    graph = StateGraph(AgentState)

    # 1. Registrar nodos
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("web_search_agent", web_search_agent_node)

    # 2. Conectar inicio con el Hub Coordinador
    graph.add_edge(START, "coordinator")

    # 3. Ruteo condicional desde el Coordinador hacia los Spokes o hacia END
    graph.add_conditional_edges(
        "coordinator",
        route_next_spoke, # paralelo
        {
            "sql_agent": "sql_agent",
            "rag_agent": "rag_agent",
            "web_search_agent": "web_search_agent",
            END: END,
        }
    )

    # 4. Todos los Spokes devuelven el control al Coordinador para sintetizar
    graph.add_edge("sql_agent", "coordinator")
    graph.add_edge("rag_agent", "coordinator")
    graph.add_edge("web_search_agent", "coordinator")

    # 5. Compilar con memoria local en memoria (checkpointing)
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    return app


# Instancia compilada lista para usar
multi_agent_app = create_hub_spoke_graph()

