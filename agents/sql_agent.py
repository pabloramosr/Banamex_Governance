"""
agents/sql_agent.py

Spoke especializado en análisis relacional Text-to-SQL.
Conecta a la base de datos PostgreSQL de Banamex (reto_banamex) y utiliza Defog SQLCoder:7b
para generar consultas precisas con gobernanza sobre los esquemas 'dim' y 'cfg'.
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import AIMessage
from langchain_ollama import OllamaLLM
from langchain_community.utilities import SQLDatabase

load_dotenv(Path(__file__).parent.parent / ".env")

PGHOST = os.environ.get("PGHOST", "localhost")
PGPORT = os.environ.get("PGPORT", "5432")
PGUSER = os.environ.get("PGUSER", "postgres")
PGPASSWORD = os.environ.get("PGPASSWORD", "banamex_database")
PGDATABASE = os.environ.get("PGDATABASE", "reto_banamex")
SQL_MODEL = os.environ.get("SQL_MODEL", "sqlcoder:7b")

DB_URI = f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}?options=-csearch_path=dim,cfg,public"


def get_db_instance() -> SQLDatabase:
    return SQLDatabase.from_uri(DB_URI)


def extract_clean_sql(text: str) -> str:
    """Extrae la sentencia SQL limpia eliminando etiquetas markdown y tokens de completado."""
    text = text.replace("[SQL]", "").strip()

    match = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    match = re.search(r"\b(SELECT|WITH)\b.*?(?:;|\Z)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).strip()

    lineas = [l for l in text.splitlines() if not l.strip().startswith("###") and not l.strip().startswith("Given the")]
    return "\n".join(lineas).strip()


def run_sql_agent(question: str) -> dict:
    """
    Ejecuta el pipeline de Text-to-SQL:
    1. Obtiene el esquema de las tablas maestras y de hechos.
    2. Construye el prompt optimizado para sqlcoder:7b.
    3. Genera y limpia la consulta SQL.
    4. Ejecuta la consulta en PostgreSQL y retorna la evidencia.
    """
    db = get_db_instance()
    llm = OllamaLLM(model=SQL_MODEL, temperature=0)

    try:
        table_names = ["dim_cliente", "dim_producto", "dim_txn", "fact_sdos_captacoloca_m", "fact_txn_captacoloca_d"]
        schema_info = db.get_table_info(table_names)

        prompt = f"""### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Database Schema
The query will run on a database with the following schema:
{schema_info}

### Answer
Given the database schema, here is the SQL query that [QUESTION]{question}[/QUESTION]
[SQL]
"""
        raw_completion = llm.invoke(prompt)
        raw_text = raw_completion if isinstance(raw_completion, str) else getattr(raw_completion, "content", "")
        clean_sql = extract_clean_sql(raw_text)

        if not clean_sql:
            raise ValueError(f"No se pudo generar SQL válido a partir de la respuesta: {raw_text}")

        # Ejecución en PostgreSQL
        db_result = db.run(clean_sql)
        
        evidence_summary = (
            f"Consulta SQL ejecutada:\n```sql\n{clean_sql}\n```\n\n"
            f"Resultado crudo de la base de datos:\n{db_result}"
        )

        return {
            "sql_query": clean_sql,
            "raw_result": db_result,
            "summary": evidence_summary,
            "error": None
        }

    except Exception as e:
        error_msg = f"Error en SQL Agent: {str(e)}"
        return {
            "sql_query": None,
            "raw_result": None,
            "summary": error_msg,
            "error": str(e)
        }
    finally:
        db._engine.dispose()


def sql_agent_node(state: dict) -> dict:
    """Nodo de LangGraph para el agente de SQL."""
    question = state.get("question", "")
    result = run_sql_agent(question)
    
    msg_content = f"### [EVIDENCIA SQL - BASE DE DATOS BANAMEX]\n{result['summary']}"
    
    return {
        "evidences": {"sql_evidence": result},
        "messages": [AIMessage(content=msg_content, name="sql_agent")]
    }
