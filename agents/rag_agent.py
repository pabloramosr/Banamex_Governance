"""
agents/rag_agent.py

Spoke especializado en gobernanza documental y normativa bancaria (RAG).
Consulta el índice semántico 'banamex-governance' en Pinecone generado con bge-large
para responder sobre Términos y Condiciones, privacidad, cookies, geolocalización y obligaciones.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv(Path(__file__).parent.parent / ".env")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME", "banamex-governance")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-large")
COORDINATOR_MODEL = os.environ.get("COORDINATOR_MODEL", "qwen2.5:3b")


def get_vector_store() -> PineconeVectorStore:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )


def run_rag_agent(question: str) -> dict:
    """
    Recupera contexto de Pinecone y genera una respuesta fundamentada con qwen2.5:3b.
    """
    try:
        vector_store = get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)

        if not docs:
            return {
                "retrieved_docs": [],
                "summary": "No se encontró evidencia relevante en los Términos y Condiciones de Banamex.",
                "error": None
            }

        context_blocks = []
        for i, d in enumerate(docs, 1):
            source = d.metadata.get("source", "terminos_banamex.txt")
            context_blocks.append(f"[Fragmento {i} | Fuente: {source}]\n{d.page_content.strip()}")

        context_text = "\n\n---\n\n".join(context_blocks)

        llm = ChatOllama(model=COORDINATOR_MODEL, temperature=0)
        prompt = PromptTemplate.from_template(
            """Eres el Oficial de Cumplimiento y Gobernanza de Datos de Banamex.
Responde a la siguiente consulta del usuario basándote ESTRICTAMENTE en los Términos y Condiciones de Banamex proporcionados.
Si la información no se encuentra en el texto, indícalo claramente sin inventar.

### TEXTO OFICIAL DE BANAMEX:
{context}

### PREGUNTA DEL USUARIO:
{question}

### INSTRUCCIONES:
- Cita los numerales o conceptos clave (ej. Uso y Restricciones, Propiedad Intelectual, Cookies, Geolocalización).
- Sé conciso, profesional y fundamentado.
"""
        )

        chain = prompt | llm
        response = chain.invoke({"context": context_text, "question": question})
        rag_answer = response.content if hasattr(response, "content") else str(response)

        return {
            "retrieved_docs": context_blocks,
            "summary": rag_answer,
            "error": None
        }

    except Exception as e:
        return {
            "retrieved_docs": [],
            "summary": f"Error en RAG Agent: {str(e)}",
            "error": str(e)
        }


def rag_agent_node(state: dict) -> dict:
    """Nodo de LangGraph para el agente RAG normativo."""
    question = state.get("question", "")
    result = run_rag_agent(question)

    msg_content = f"### [EVIDENCIA RAG - TÉRMINOS Y CONDICIONES BANAMEX]\n{result['summary']}"

    return {
        "evidences": {"rag_evidence": result},
        "messages": [AIMessage(content=msg_content, name="rag_agent")]
    }

