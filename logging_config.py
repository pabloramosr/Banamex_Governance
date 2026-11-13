"""
logging_config.py

Formateo visual enriquecido para la trazabilidad del sistema multi-agente de Banamex.
Permite visualizar claramente qué decisiones toma el Hub y qué evidencia genera cada Spoke.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def log_header(title: str):
    console.print()
    console.print(Panel(Text(title, justify="center", style="bold cyan"), border_style="cyan"))


def log_coordinator_decision(plan: str, next_agent: str):
    table = Table(title="Decisión del Coordinador (Hub)", border_style="yellow", show_header=True)
    table.add_column("Métrica", style="bold yellow")
    table.add_column("Detalle", style="white")
    table.add_row("Plan / Razonamiento", plan)
    table.add_row("Próximo Agente Destino", f"[bold green]{next_agent}[/bold green]")
    console.print(table)


def log_sql_execution(query: str, result_summary: str):
    content = f"[bold green]SQL Generado:[/bold green]\n{query}\n\n[bold green]Resultado Obtenido:[/bold green]\n{result_summary}"
    console.print(Panel(content, title="Spoke: SQL Agent (Gemini + PostgreSQL)", border_style="green"))


def log_rag_retrieval(query: str, sources_summary: str):
    content = f"[bold blue]Búsqueda Semántica en Pinecone:[/bold blue] {query}\n\n[bold blue]Evidencia Documental Recuperada (bge-large):[/bold blue]\n{sources_summary}"
    console.print(Panel(content, title="Spoke: RAG Tool (Pinecone + bge-large)", border_style="blue"))


def log_final_response(answer: str):
    console.print()
    console.print(Panel(answer, title="Respuesta Final - Data Steward Banamex", border_style="bold bright_green"))


def log_fatal_error(message: str):
    console.print()
    console.print(Panel(message, title="ERROR TERMINAL - Ejecución detenida", border_style="bold red"))
