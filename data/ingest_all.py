"""
data/ingest_all.py

Script optimizado para cargar, dividir y vectorizar el Corpus Bancario Crítico de Banamex en Pinecone.
- 51 Documentos de Negocio: contratos maestros de tarjetas y cuentas, carátulas y anexos de comisiones oficiales,
  folletos comerciales de productos, créditos (personal, nómina, hipotecario), ahorro e inversiones.
- 1 Documento de Términos del Portal Web (terminos_banamex.txt).
- Excluye normativa y circulares macro de Banxico de cientos de páginas para evitar ruido semántico.
- Embedding: bge-large (1024 dims) acelerado con concurrencia (ThreadPoolExecutor).
- Vector DB: Pinecone (banamex-governance, cosine, 1024 dims).
"""

import sys
if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8")
import os
import csv
import time
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# Cargar variables de entorno
load_dotenv(Path(__file__).parent.parent / ".env")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME", "banamex-governance")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-large")

DATA_DIR = Path(__file__).parent
CORPUS_DIR = DATA_DIR / "reto_banamex_completo"
INVENTORY_CSV = CORPUS_DIR / "00_INDICE" / "inventario_documentos.csv"
TERMINOS_TXT = DATA_DIR / "terminos_banamex.txt"


def ensure_clean_pinecone_index(pc: Pinecone, index_name: str, dimension: int = 1024):
    """Verifica si el índice existe en Pinecone y borra vectores previos para una ingesta limpia."""
    existing_indexes = [i.name for i in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Creando índice '{index_name}' en Pinecone (dim={dimension}, metric='cosine')...", flush=True)
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(2)
        print("Índice listo en Pinecone.", flush=True)
    else:
        print(f"Índice '{index_name}' activo en Pinecone. Limpiando vectores previos...", flush=True)
        idx = pc.Index(index_name)
        try:
            idx.delete(delete_all=True)
            print("Vectores anteriores eliminados exitosamente.", flush=True)
        except Exception as e:
            print(f"Aviso al limpiar: {e}", flush=True)


def generate_stable_id(doc: Document) -> str:
    """Genera un hash determinista para identificar cada chunk."""
    identity = f"{doc.metadata.get('source')}|p{doc.metadata.get('page', 1)}|{doc.page_content[:120]}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def load_portal_terms() -> list[Document]:
    """Carga terminos_banamex.txt."""
    if not TERMINOS_TXT.exists():
        return []
    text = TERMINOS_TXT.read_text(encoding="utf-8")
    return [
        Document(
            page_content=text,
            metadata={
                "source": "terminos_banamex.txt",
                "tipo_documento": "terminos_portal_web",
                "producto_o_familia": "portal_banamex_general",
                "prioridad_rag": 4,
                "url_fuente": "https://www.banamex.com/terminos-y-condiciones.html",
                "institucion": "Grupo Financiero Banamex",
                "page": 1,
                "total_pages": 1
            }
        )
    ]


def load_corpus_documents() -> list[Document]:
    """Carga los 51 PDFs esenciales de negocio, descartando normativa macro."""
    if not INVENTORY_CSV.exists():
        print(f"Aviso: No se encontró {INVENTORY_CSV}", flush=True)
        return []

    docs = []
    skipped_count = 0
    with open(INVENTORY_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Excluir leyes federales y circulares pesadas de Banxico
            if row.get("tipo_documento") == "normativa":
                skipped_count += 1
                continue

            pdf_rel = row.get("ruta", "")
            pdf_path = CORPUS_DIR / pdf_rel
            if not pdf_path.exists():
                continue

            try:
                reader_pdf = PdfReader(pdf_path)
                num_pages = len(reader_pdf.pages)
                for page_idx, page in enumerate(reader_pdf.pages, 1):
                    text = (page.extract_text() or "").strip()
                    if not text:
                        continue

                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "tipo_documento": row.get("tipo_documento", "documento_bancario"),
                            "producto_o_familia": row.get("producto_o_familia", "general"),
                            "prioridad_rag": int(row.get("prioridad_rag", 3)),
                            "url_fuente": row.get("url_fuente", "https://www.banamex.com"),
                            "institucion": "Grupo Financiero Banamex",
                            "page": page_idx,
                            "total_pages": num_pages
                        }
                    )
                    docs.append(doc)
            except Exception as e:
                print(f"Error al leer {pdf_path.name}: {e}", flush=True)

    print(f"  * Documentos de normativa descartados para evitar ruido: {skipped_count}", flush=True)
    return docs


def main():
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY no encontrada en .env")

    print("\n" + "=" * 80, flush=True)
    print(" INGESTA DE CORPUS BANCARIO CRÍTICO BANAMEX EN PINECONE (ALTA PRECISIÓN)", flush=True)
    print("=" * 80, flush=True)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_clean_pinecone_index(pc, INDEX_NAME, dimension=1024)
    index = pc.Index(INDEX_NAME)

    # 1. Cargar documentos
    print("\n[1/4] Extrayendo páginas de PDFs de productos, comisiones y términos...", flush=True)
    raw_docs = load_portal_terms()
    corpus_docs = load_corpus_documents()
    all_raw_docs = raw_docs + corpus_docs
    print(f"  * Total de páginas con texto útil: {len(all_raw_docs)}", flush=True)

    # 2. Particionar en Chunks semánticos
    print("\n[2/4] Generando chunks semánticos (chunk_size=1000, overlap=150)...", flush=True)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(all_raw_docs)
    total_chunks = len(chunks)
    print(f"  * Total de chunks semánticos listos para indexar: {total_chunks}", flush=True)

    # 3. Inicializar embeddings acelerados
    print(f"\n[3/4] Conectando a Ollama ('{EMBEDDING_MODEL}') con concurrencia...", flush=True)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # 4. Upsert en lotes concurrentes a Pinecone
    print(f"\n[4/4] Indexando {total_chunks} chunks en Pinecone...", flush=True)
    batch_size = 50
    t0 = time.time()

    for i in range(0, total_chunks, batch_size):
        end = min(i + batch_size, total_chunks)
        batch_chunks = chunks[i:end]

        # Extraer textos y metadatos
        texts = [c.page_content for c in batch_chunks]
        ids = [generate_stable_id(c) for c in batch_chunks]

        # Concurrencia de 4 hilos para acelerar embeddings locales
        with ThreadPoolExecutor(max_workers=4) as executor:
            vectors = list(executor.map(embeddings.embed_query, texts))

        # Formato de vectores para Pinecone
        pinecone_records = []
        for vec_id, vec, c in zip(ids, vectors, batch_chunks):
            meta = dict(c.metadata)
            meta["text"] = c.page_content
            pinecone_records.append({
                "id": vec_id,
                "values": vec,
                "metadata": meta
            })

        # Upsert directo a Pinecone
        index.upsert(vectors=pinecone_records)

        elapsed = time.time() - t0
        pct = (end / total_chunks) * 100
        speed = end / elapsed if elapsed > 0 else 0
        eta = (total_chunks - end) / speed if speed > 0 else 0
        print(f"  -> [{pct:5.1f}%] {end}/{total_chunks} chunks ({speed:.1f} ch/s, transcurrido: {elapsed:.0f}s, ETA: {eta:.0f}s)...", flush=True)

    print(f"\n ¡Éxito total! Se indexaron {total_chunks} chunks en Pinecone ('{INDEX_NAME}').", flush=True)
    print(f" Tiempo total: {time.time()-t0:.1f} segundos.\n", flush=True)


if __name__ == "__main__":
    main()
