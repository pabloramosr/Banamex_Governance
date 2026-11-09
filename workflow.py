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

# La funcion recibe como argumento 'state' el estado del agente(AgentState) y recupera el next_agent
# razonado por el coordinador, de no haber agente seleccionado regresa un FINISH,
# posteriormente verifica cual de los agentes fue llamado y ese sera el next_agent. De no llamar a ninguno
# y ser FINISH, el garfo termina con END.

def route_next_spoke(state: AgentState) -> str:
    """Función de enrutamiento condicional basada en la decisión del Coordinador."""
    next_agent = state.get("next_agent", "FINISH")
    if next_agent in ["sql_agent", "rag_agent", "web_search_agent"]:
        return next_agent
    return END


def create_hub_spoke_graph():
    """Construye y compila el flujo StateGraph de LangGraph."""
    # El grafo se construye utilizando el AgentState como estado, 
    # es el universo completo, y sigue las reglas de los 6 canales de estado.
    # El estado se mueve y actualiza de nodo en nodo.
    graph = StateGraph(AgentState)

    # 1. Registrar nodos
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("web_search_agent", web_search_agent_node)

    # 2. Conectar inicio con el Hub Coordinador
    graph.add_edge(START, "coordinator")

    # 3. Ruteo condicional desde el Coordinador hacia los Spokes o hacia END
    graph.add_conditional_edges( # Hacerlo paralelo!!!
        "coordinator", # Nodo de origen
        route_next_spoke, # Funcion de eleccion
        {               # Mapa de destinos disponibles.
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
    checkpointer = MemorySaver() # thread_id, se toma una foto cada que un nodo termina de ejecutarse.
    # {
    #"sesion_pablo_1": [ Foto_Paso_1, Foto_Paso_2, Foto_Paso_3 ],
    #"sesion_usuario_2": [ ... ]
    #}
    # En una sesion que solo involucra pregunta respuesta, se toman 3 fotos solamente
    # 1- Coordinator -> 2- SQL Spoke -> 3- Syntesis
    app = graph.compile(checkpointer=checkpointer) # Se construye el grafo con su memoria, y queda definida la arquitectura.
    return app


# Instancia compilada lista para usar
multi_agent_app = create_hub_spoke_graph()

