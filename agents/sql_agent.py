"""
agents/sql_agent.py

Spoke especializado en análisis relacional Text-to-SQL para Banamex.
Conecta a la base de datos PostgreSQL (reto_banamex) y utiliza Google Gemini (gemini-3.5-flash-lite)
incorporando la capa semántica y reglas de negocio bancarias (documentacion_arquitectura_sql.txt)
para garantizar consultas precisas, respetando los esquemas 'dim' y 'cfg'.
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_community.utilities import SQLDatabase

from agents.llm import get_gemini_llm

load_dotenv(Path(__file__).parent.parent / ".env")

PGHOST = os.environ.get("PGHOST", "localhost")
PGPORT = os.environ.get("PGPORT", "5432")
PGUSER = os.environ.get("PGUSER", "postgres")
PGPASSWORD = os.environ.get("PGPASSWORD", "banamex_database")
PGDATABASE = os.environ.get("PGDATABASE", "reto_banamex")
SQL_MODEL = os.environ.get("SQL_MODEL", "gemini-3.5-flash")

DB_URI = f"postgresql+psycopg2://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}?options=-csearch_path=dim,cfg,public"


def get_db_instance() -> SQLDatabase:
    return SQLDatabase.from_uri(DB_URI)


def get_sql_llm():
    """Retorna Gemini. Lanza ConfigError (error terminal) si falta GOOGLE_API_KEY."""
    return get_gemini_llm(SQL_MODEL)


def extract_clean_sql(text: str) -> str:
    """Extrae la sentencia SQL limpia eliminando etiquetas markdown y explicaciones."""
    text = text.replace("[SQL]", "").strip()

    # Bloque de código ```sql ... ```
    match = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    # Coincidencia con sentencia SQL (SELECT o WITH)
    match = re.search(r"\b(SELECT|WITH)\b.*?(?:;|\Z)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).strip()

    lineas = [l for l in text.splitlines() if not l.strip().startswith("###") and not l.strip().startswith("Given the")]
    return "\n".join(lineas).strip()


SQL_SYSTEM_PROMPT = """Eres el Especialista Técnico Text-to-SQL de la Plataforma de Datos de Banamex.
Tu única responsabilidad es generar consultas SQL en sintaxis PostgreSQL válidas, óptimas y precisas para responder la solicitud analítica.

================================================================================
ARQUITECTURA DE DATOS Y CAPA SEMÁNTICA BANAMEX (Modelo Estrella Kimball)
================================================================================

1. ESQUEMA DE DIMENSIONES (Contexto cualitativo, catálogo maestro):
   - `dim.dim_cliente` (5,000 registros):
     * `num_cte` (INTEGER, PK): Identificador único del cliente.
     * `fec_alta` (DATE): Fecha de ingreso del cliente.
     * `fec_nac` (DATE): Fecha de nacimiento.
     * `tipo_persona` (VARCHAR(10)): 'PF' (Persona Física, 96%) o 'PM' (Persona Moral, 4%).
     * `genero_cte` (VARCHAR(10)): 'HOMBRE' (52%), 'MUJER' (48%) o 'NO APLICA' (para PM).
     * `segmento_cte` (VARCHAR(20)): 'PREMIUM' (250), 'PREFERENTE' (1000), 'MASIVO' (3750).

   - `dim.dim_producto` (4 registros):
     * `cve_producto` (VARCHAR(40), PK): 'CHEQUES', 'AHORRO', 'C_PERSONAL', 'TDC'.
     * `cve_dominio` (VARCHAR(20)): 'DEPOSITS', 'LENDING', 'TDC'.
     * `grupo_producto` (VARCHAR(20)): 'CAPTACION' o 'COLOCACION'.
     * `descripcion` (VARCHAR(100)): Nombre comercial.

   - `dim.dim_txn` (9 registros):
     * `cve_producto`, `cve_txn` (100: DEPOSITO, 200: RETIRO, 300: PAGOS, 400: COMPRAS, 450: DISPOSICIONES).
     * `desc_txn` (VARCHAR(40)).

2. ESQUEMA DE HECHOS (Métricas cuantitativas y balances):
   - `cfg.fact_sdos_captacoloca_m` (Saldos mensuales, 120,000 filas):
     * Contiene 6 fotografías mensuales (cierres contables de 20260131 a 20260630).
     * Cada uno de los 5,000 clientes tiene 1 cuenta para cada uno de los 4 productos (20,000 cuentas activas por mes).
     * `num_cta` (INTEGER): Número de cuenta.
     * `num_cte` (INTEGER, FK -> dim_cliente.num_cte).
     * `cve_producto` (VARCHAR(40), FK -> dim_producto.cve_producto).
     * `saldo_apertura` (NUMERIC(18,2)): Saldo inicial del mes.
     * `saldo_mes` (NUMERIC(18,2)): Saldo de cierre del mes.
     * `fec_informacion` (INTEGER): Fecha de corte en formato YYYYMMDD (20260131, 20260228, 20260331, 20260430, 20260531, 20260630).

   - `cfg.fact_txn_captacoloca_d` (Transacciones diarias, 225,000 filas, Feb a Jun 2026):
     * `num_cta`, `num_cte`, `cve_producto`, `fec_txn`, `cve_mes_txn`, `saldo_txn` (importe positivo), `cve_txn`, `desc_txn`.

================================================================================
REGLAS DE NEGOCIO Y GOBERNANZA BANCARIA OBLIGATORIAS:
================================================================================
1. CONTEO DE CLIENTES:
   - Para saber cuántos clientes hay (por segmento, género, tipo de persona, etc.), SIEMPRE debes contar sobre la tabla maestra: `dim.dim_cliente` (ej: `SELECT COUNT(*) FROM dim.dim_cliente WHERE segmento_cte = 'PREMIUM'`).
   - NUNCA cuentes clientes sobre `fact_sdos_captacoloca_m` a menos que explícitamente se pida filtrar por clientes con cuentas con saldo > X.

2. FILTROS DE SEGMENTO:
   - Si la consulta pide clientes "PREMIUM", usa exactamente: `segmento_cte = 'PREMIUM'`. No uses `IN ('PREMIUM', 'PREFERENTE', 'MASIVO')` ni agrupes por todos si se pidió uno en específico.

3. FILTROS DE GÉNERO:
   - Los valores válidos son 'MUJER' y 'HOMBRE' en mayúsculas: `genero_cte = 'MUJER'`.

4. SALDOS TOTALES Y FECHAS DE INFORMACIÓN:
   - `fact_sdos_captacoloca_m` contiene 6 meses acumulados de fotos de saldo.
   - Si la consulta pide "saldo total actual" o no especifica mes, DEBES filtrar por el cierre más reciente:
     `WHERE fec_informacion = 20260630` (o `WHERE fec_informacion = (SELECT MAX(fec_informacion) FROM cfg.fact_sdos_captacoloca_m)`), de lo contrario sumarías 6 veces el dinero de los clientes.

5. DOMINIOS FINANCIEROS (Captación vs Colocación):
   - Captación (Depósitos de clientes): 'CHEQUES' y 'AHORRO' (o `p.grupo_producto = 'CAPTACION'`).
   - Colocación (Créditos y financiamiento): 'C_PERSONAL' y 'TDC' (o `p.grupo_producto = 'COLOCACION'`).

6. LLAVES FORÁNEAS Y JOINS:
   - Vincular saldos con clientes: `cfg.fact_sdos_captacoloca_m.num_cte = dim.dim_cliente.num_cte`
   - Vincular saldos con productos: `cfg.fact_sdos_captacoloca_m.cve_producto = dim.dim_producto.cve_producto`

7. FORMATO DE RESPUESTA:
   - Responde ÚNICAMENTE la consulta SQL limpia en PostgreSQL dentro de un bloque ```sql ... ```.
   - NO agregues saludos, explicaciones ni notas adicionales fuera de la consulta.
"""


def run_sql_agent(instruction: str) -> dict:
    """
    Ejecuta el pipeline Text-to-SQL con Gemini:
    1. Envía la instrucción técnica y la capa semántica.
    2. Extrae la consulta SQL limpia.
    3. Ejecuta en PostgreSQL reto_banamex y retorna el resultado.
    """
    # Fuera del try: un error de configuración es terminal y no debe convertirse en evidencia.
    llm = get_sql_llm()
    db = get_db_instance()

    try:
        user_message = f"Instrucción para generar SQL: {instruction}"
        
        # Invocar LLM
        messages = [
            SystemMessage(content=SQL_SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ]
        raw_response = llm.invoke(messages)
        content = raw_response.content
        if isinstance(content, list):
            raw_text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        else:
            raw_text = str(content)

        clean_sql = extract_clean_sql(raw_text)

        if not clean_sql:
            raise ValueError(f"No se pudo extraer SQL ejecutable de la respuesta: {raw_text}")

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
    """
    Nodo de LangGraph para el agente de SQL.
    Prioriza 'spoke_instruction' (la instrucción enriquecida del coordinador)
    sobre 'question' (la pregunta cruda del usuario).
    """
    instruction = state.get("spoke_instruction") or state.get("question", "")
    result = run_sql_agent(instruction)

    msg_content = f"### [EVIDENCIA SQL - BASE DE DATOS BANAMEX]\n{result['summary']}"

    return {
        "evidences": {"sql_evidence": result},
        "messages": [AIMessage(content=msg_content, name="sql_agent")]
    }
