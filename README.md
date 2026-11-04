# 🏛️ Plataforma Multi-Agente Hub-and-Spoke - Data Steward Banamex

Sistema inteligente multi-agente diseñado bajo la arquitectura **Hub-and-Spoke (Supervisor Pattern)** con **LangGraph** para la democratización y gobernanza de datos en **Grupo Financiero Banamex**.

---

## 📐 Arquitectura del Sistema

```
                    ┌─────────────────────────┐
                    │  Usuario / Interfaz UI  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │   COORDINADOR (HUB)   │
                     │  (Data Steward Lead)  │
                     └───────────┬───────────┘
                                 │
       ┌─────────────────────────┼─────────────────────────┐
       ▼                         ▼                         ▼
┌───────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   SQL AGENT   │       │    RAG AGENT    │       │   WEB SEARCH    │
│    (Spoke)    │       │     (Spoke)     │       │     (Spoke)     │
├───────────────┤       ├─────────────────┤       ├─────────────────┤
│ Model:        │       │ Vector DB:      │       │ Engine:         │
│ sqlcoder:7b   │       │ Pinecone        │       │ Tavily Search   │
│               │       │                 │       │                 │
│ Engine:       │       │ Embeddings:     │       │ Model:          │
│ PostgreSQL    │       │ bge-large       │       │ qwen2.5:3b      │
│ (dim & cfg)   │       │                 │       │                 │
│               │       │ Docs: Términos  │       │ Info: Noticias  │
│ Datos:        │       │ y Condiciones   │       │ públicas y      │
│ Clientes y TXN│       │ Banamex         │       │ mercado externo │
└───────┬───────┘       └────────┬────────┘       └────────┬────────┘
        │                        │                         │
        └────────────────────────┼─────────────────────────┘
                                 │ (Evidencias)
                                 ▼
                     ┌───────────────────────┐
                     │   COORDINADOR (HUB)   │
                     │ (Síntesis y Auditoría)│
                     └───────────┬───────────┘
                                 │
                                 ▼
                       Respuesta Fundamentada
```

---

## 📁 Estructura del Repositorio

```
MultiAgentHS/
├── agents/
│   ├── __init__.py
│   ├── coordinator.py         # Hub: Enrutador de intenciones y sintetizador final
│   ├── sql_agent.py           # Spoke: Text-to-SQL sobre PostgreSQL con sqlcoder:7b
│   ├── rag_agent.py           # Spoke: RAG normativo con Pinecone y bge-large
│   └── web_search_agent.py    # Spoke: Búsqueda web externa mediante Tavily API
├── data/
│   ├── terminos_banamex.txt   # Corpus normativo oficial de Banamex
│   └── ingest.py              # Pipeline de chunking e indexación en Pinecone
├── .env.example               # Plantilla de variables de entorno para colaboradores
├── .gitignore                 # Exclusión de credenciales y entornos virtuales
├── logging_config.py          # Visualización enriquecida con Rich (logs y trazabilidad)
├── main.py                    # CLI de pruebas automatizadas y modo interactivo
├── pyproject.toml             # Gestión moderna y reproducible de dependencias con uv
├── state.py                   # Esquema tipado del AgentState para LangGraph
├── workflow.py                # Definición del StateGraph compilado con checkpointing
└── ui_streamlit.py            # Interfaz gráfica (en desarrollo)
```

---

## 🚀 Requisitos Previos

1. **Python >= 3.12** y [uv](https://docs.astral.sh/uv/) instalado.
2. **Ollama** ejecutándose localmente con los modelos descargados:
   ```bash
   ollama pull qwen2.5:3b
   ollama pull sqlcoder:7b
   ollama pull bge-large
   ```
3. **Servicio PostgreSQL** activo con la base de datos `reto_banamex` inicializada.
4. Cuentas y API keys de **Pinecone** y **Tavily**.

---

## 🛠️ Instalación y Puesta en Marcha

### 1. Clonar el repositorio y entrar a la carpeta
```bash
git clone <URL_DEL_REPOSITORIO>
cd MultiAgentHS
```

### 2. Sincronizar el entorno virtual con `uv`
```bash
uv sync
```

### 3. Configurar variables de entorno
Copia la plantilla y edita tus claves:
```bash
cp .env.example .env
```
Asegúrate de configurar:
- `TAVILY_API_KEY`
- `PINECONE_API_KEY`
- `INDEX_NAME=banamex-governance`
- `PGPASSWORD=tu_contraseña_postgres`

### 4. Indexar los Términos y Condiciones en Pinecone (una sola vez)
```bash
uv run python data/ingest.py
```

### 5. Ejecutar la batería de pruebas
```bash
# Modo demostrativo (ejecuta preguntas de SQL, RAG y Web):
uv run python main.py --demo

# Modo consola interactiva:
uv run python main.py --interactive
```

---

## 🧠 Características de Gobernanza y Data Stewardship

- **Cero Datos Huérfanos:** Las consultas del `sql_agent` se ejecutan sobre un modelo dimensional relacional validado con llaves foráneas (`dim_cliente`, `dim_producto`).
- **Verificabilidad y Trazabilidad:** Cada respuesta final del Coordinador cita explícitamente la fuente del dato (PostgreSQL interno, Pinecone normativo o Tavily externo).
- **Memoria de Sesión (LangGraph Checkpointing):** Utiliza `MemorySaver` para recordar el hilo de la conversación y mantener el contexto de preguntas anteriores.

