"""
tools/rag_tool.py

Herramienta determinista (Tool) para búsqueda y recuperación semántica sobre
el corpus oficial de gobernanza, contratos y comisiones de Banamex en Pinecone.
Sigue el patrón estándar @tool de LangChain (similar a 10_agentic_rag_pinecone.py).

NO contiene LLM intermedio: solo calcula el embedding con bge-large, consulta
Pinecone (k=6) y devuelve la evidencia documental cruda con metadatos de trazabilidad.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv(Path(__file__).parent.parent / ".env")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME", "banamex-governance")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-large")

# Inicialización diferida del Vector Store para optimizar tiempos de arranque
_vector_store = None
_retriever = None


def get_retriever(k: int = 6):
    global _vector_store, _retriever
    if _retriever is None:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        _vector_store = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
        )
        _retriever = _vector_store.as_retriever(
            search_kwargs={"k": k}
        )
    return _retriever


@tool
def search_banamex_governance_knowledge(query: str) -> str:
    """Busca en el corpus oficial de Banamex indexado en Pinecone.
    Incluye: contratos maestros de tarjetas de crédito (Clásica, Oro, Platinum, Costco, etc.),
    anexos oficiales de comisiones, folletos comerciales de cuentas (Priority, MiCuenta, Básica),
    créditos (personal, nómina, hipotecario), ahorro, inversiones y términos del portal web."""
    try:
        retriever = get_retriever(k=6)
        docs = retriever.invoke(query)

        if not docs:
            return "No se encontró evidencia documental relevante en el repositorio oficial de Banamex."

        evidence = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Documento no identificado")
            page = doc.metadata.get("page")
            tipo = doc.metadata.get("tipo_documento", "general")
            producto = doc.metadata.get("producto_o_familia", "general")
            url = doc.metadata.get("url_fuente", "https://www.banamex.com")

            header = f"[FUENTE #{i}]: {source}"
            if page is not None:
                header += f" | Página: {page}"
            header += f" | Tipo: {tipo} | Familia: {producto}"

            evidence.append(
                f"{header}\n"
                f"URL de referencia: {url}\n"
                f"Contenido oficial:\n{doc.page_content.strip()}"
            )

        return "\n\n---\n\n".join(evidence)
    except Exception as e:
        return f"Error técnico al consultar el repositorio documental de Banamex: {str(e)}"
