"""
agents/rag_agent.py

SPOKE DETERMINISTA DE GOBERNANZA DOCUMENTAL Y NORMATIVA BANAMEX (RAG TOOL).
Ejecuta la recuperación semántica sobre Pinecone ('banamex-governance') usando bge-large.

CAMBIO ARQUITECTÓNICO:
Se eliminó la llamada intermedia a un LLM en este spoke.
Ahora opera como un nodo-herramienta puro y determinista que extrae fragmentos oficiales,
metadatos de páginas y contratos, entregándolos crudos al Coordinador para que él sea
el único que sintetice la respuesta final. Esto ahorra tokens y elimina alucinaciones intermedias.
"""

from langchain_core.messages import AIMessage
from tools.rag_tool import search_banamex_governance_knowledge, get_retriever


def run_rag_tool(query: str) -> dict:
    """Ejecuta la herramienta de búsqueda semántica en Pinecone y devuelve la evidencia formateada."""
    try:
        retriever = get_retriever(k=6)
        docs = retriever.invoke(query)

        if not docs:
            return {
                "retrieved_docs": [],
                "summary": "No se encontró evidencia relevante en la documentación oficial de Banamex.",
                "error": None
            }

        context_blocks = []
        for i, d in enumerate(docs, 1):
            source = d.metadata.get("source", "documento_banamex")
            page = d.metadata.get("page", 1)
            tipo = d.metadata.get("tipo_documento", "documento")
            prod = d.metadata.get("producto_o_familia", "general")
            url = d.metadata.get("url_fuente", "https://www.banamex.com")

            block = (
                f"[Fragmento {i} | Fuente: {source} (Pág. {page}, Tipo: {tipo}, Familia: {prod})]\n"
                f"URL: {url}\n"
                f"{d.page_content.strip()}"
            )
            context_blocks.append(block)

        raw_evidence = "\n\n---\n\n".join(context_blocks)

        return {
            "retrieved_docs": context_blocks,
            "summary": raw_evidence,
            "error": None
        }

    except Exception as e:
        return {
            "retrieved_docs": [],
            "summary": f"Error técnico al recuperar documentación de Pinecone: {str(e)}",
            "error": str(e)
        }


def rag_agent_node(state: dict) -> dict:
    """Nodo de LangGraph para el spoke de documentación y gobernanza (Deterministic Tool Spoke)."""
    instruction = state.get("spoke_instruction") or state.get("question", "")
    result = run_rag_tool(instruction)

    msg_content = f"### [EVIDENCIA DOCUMENTAL RECUPERADA DE PINECONE]\n{result['summary']}"

    return {
        "evidences": {"rag_evidence": result},
        "messages": [AIMessage(content=msg_content, name="rag_agent")]
    }
