"""
main.py

Punto de entrada de prueba para la Plataforma Multi-Agente Hub-and-Spoke de Banamex.
Ejecuta preguntas interactivas o de demostración mostrando la trazabilidad completa.
"""

import sys
if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8")
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from workflow import multi_agent_app
from logging_config import (
    log_header,
    log_coordinator_decision,
    log_sql_execution,
    log_rag_retrieval,
    log_final_response,
)

load_dotenv(Path(__file__).parent / ".env")


def ask_data_steward(question: str, thread_id: str = "demo_thread"):
    """Ejecuta el grafo Hub-and-Spoke trazando cada paso en consola."""
    log_header(f"CONSULTA RECIBIDA: {question}")

    config = {"configurable": {"thread_id": thread_id}}
    initial_input = {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "plan": None,
        "next_agent": None,
        "evidences": {},
        "final_answer": None
    }

    final_state = None
    for event in multi_agent_app.stream(initial_input, config=config, stream_mode="updates"):
        for node_name, update in event.items():
            if node_name == "coordinator":
                if update.get("plan") and update.get("next_agent"):
                    log_coordinator_decision(update["plan"], update["next_agent"])
                if update.get("final_answer"):
                    final_state = update
            elif node_name == "sql_agent":
                ev = update.get("evidences", {}).get("sql_evidence", {})
                log_sql_execution(ev.get("sql_query", "N/A"), ev.get("summary", "N/A"))
            elif node_name == "rag_agent":
                ev = update.get("evidences", {}).get("rag_evidence", {})
                log_rag_retrieval(question, ev.get("summary", "N/A"))

    if final_state and final_state.get("final_answer"):
        log_final_response(final_state["final_answer"])


def run_demo():
    """Ejecuta una batería de pruebas demostrativas cubriendo SQL y RAG oficial."""
    demo_questions = [
        # Prueba 1: SQL Agent (PostgreSQL + Gemini)
        "¿Cuántos clientes tenemos registrados en el segmento PREMIUM?",
        
        # Prueba 2: RAG Tool (Pinecone + bge-large + Documentación Oficial)
        "¿Cuáles son las comisiones y costos de anualidad de la tarjeta Costco Banamex?",
        
        # Prueba 3: Blindaje de Dominio (Pregunta fuera de contexto Banamex)
        "¿Quién ganó el partido de fútbol de ayer en la noche?"
    ]

    print("\n" + "=" * 80)
    print(" INICIANDO PRUEBAS DEMOSTRATIVAS - PLATAFORMA MULTI-AGENTE BANAMEX")
    print("=" * 80)

    for i, q in enumerate(demo_questions, 1):
        ask_data_steward(q, thread_id=f"demo_session_{i}")


def run_interactive():
    """Modo interactivo en terminal para hacer cualquier pregunta."""
    print("\n" + "=" * 80)
    print(" BANAMEX DATA STEWARD - CONSOLA INTERACTIVA (HUB-AND-SPOKE)")
    print(" Escribe tu consulta o 'salir' para terminar.")
    print("=" * 80)

    thread_id = "interactive_user_session"
    while True:
        try:
            q = input("\n[Usuario] > ").strip()
            if not q:
                continue
            if q.lower() in ["salir", "exit", "quit"]:
                print("\nCerrando sesión del Data Steward. ¡Hasta pronto!")
                break
            ask_data_steward(q, thread_id=thread_id)
        except KeyboardInterrupt:
            print("\nOperación cancelada por el usuario.")
            break
        except Exception as e:
            print(f"\nError en la ejecución: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo()
    else:
        run_interactive()
