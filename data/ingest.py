"""
data/ingest.py

Script para cargar, dividir y vectorizar los Términos y Condiciones de Banamex en Pinecone.
Índice: banamex-governance (dimension=1024, metric='cosine', serverless aws/us-east-1)
Embeddings: bge-large (local vía Ollama)
"""

import sys
if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8")
import os
import time
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

# Cargar variables de entorno
load_dotenv(Path(__file__).parent.parent / ".env")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME", "banamex-governance")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-large")
DOC_PATH = Path(__file__).parent / "terminos_banamex.txt"


def ensure_pinecone_index(pc: Pinecone, index_name: str, dimension: int = 1024):
    """Verifica si el índice existe; si no, lo crea de forma Serverless."""
    existing_indexes = [i.name for i in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Creando índice '{index_name}' en Pinecone (dim={dimension}, metric='cosine')...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            print("Esperando inicialización del índice...")
            time.sleep(2)
        print("Índice listo en Pinecone.")
    else:
        print(f"El índice '{index_name}' ya existe en Pinecone.")


def load_text_document(doc_path: Path) -> list[Document]:
    """Carga el documento de términos y condiciones de Banamex."""
    if not doc_path.exists():
        # Fallback a Data.txt en la raíz
        alt_path = doc_path.parent.parent / "Data.txt"
        if alt_path.exists():
            doc_path = alt_path
        else:
            raise FileNotFoundError(f"No se encontró el archivo: {doc_path}")

    print(f"Leyendo documento: {doc_path}")
    text = doc_path.read_text(encoding="utf-8")
    return [
        Document(
            page_content=text,
            metadata={
                "source": "terminos_condiciones_banamex.txt",
                "document_type": "terminos_y_condiciones_bancarios",
                "institucion": "Grupo Financiero Banamex",
            },
        )
    ]


def generate_stable_id(doc: Document) -> str:
    """Genera un hash determinista para evitar chunks duplicados al reejecutar."""
    identity = f"{doc.metadata.get('source')}|{doc.page_content[:120]}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def main():
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY no encontrada en .env")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_pinecone_index(pc, INDEX_NAME, dimension=1024)

    raw_docs = load_text_document(DOC_PATH)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Documento dividido en {len(chunks)} chunks semánticos.")

    print(f"Inicializando modelo de embeddings local '{EMBEDDING_MODEL}'...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    print(f"Conectando al vector store de Pinecone ('{INDEX_NAME}')...")
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )

    ids = [generate_stable_id(c) for c in chunks]
    batch_size = 20
    total_chunks = len(chunks)

    print(f"Indexando {total_chunks} chunks en Pinecone...")
    for i in range(0, total_chunks, batch_size):
        end = min(i + batch_size, total_chunks)
        vector_store.add_documents(documents=chunks[i:end], ids=ids[i:end])
        print(f"  -> Subidos chunks {i+1} a {end} de {total_chunks}...")

    print(f"\n¡Éxito total! Términos de Banamex indexados en Pinecone ('{INDEX_NAME}').")


if __name__ == "__main__":
    main()

