"""
Interactive Command-Line Interface (CLI) for SupportPearlz.
Provides a multi-turn support chat loop with visible source citations,
confidence indicators, query condensation logging, and session reset.
"""

import sys
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.config import settings
from src.retrieval.vector_store import load_or_build
from src.chains.rag_chain import execute_rag_pipeline
from src.chains.memory import ConversationMemory


def format_confidence_badge(confidence: str) -> str:
    """Format confidence signal with terminal colors."""
    if confidence == "high":
        return "[bold green]HIGH[/bold green]"
    elif confidence == "partial":
        return "[bold yellow]PARTIAL[/bold yellow]"
    return "[bold red]NONE (REFUSAL)[/bold red]"


def run_cli() -> None:
    """Main interactive REPL loop."""
    console = Console()
    console.print("\n[bold cyan]" + "=" * 70 + "[/bold cyan]")
    console.print("[bold white] SupportPearlz -- Enterprise RAG Customer Support Agent[/bold white]")
    console.print(f"[dim] Company: Pearlz Home Systems (Pvt.) Ltd. | Collection: {settings.COLLECTION_NAME}[/dim]")
    console.print(f"[dim] LLM: {settings.LLM_PROVIDER} ({settings.LLM_MODEL}) | Temperature: {settings.TEMPERATURE}[/dim]")
    console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")
    console.print("[yellow]Type your questions below. Commands: [bold]/reset[/bold] (clear memory), [bold]/help[/bold], [bold]/exit[/bold].[/yellow]\n")

    # Load vector store from disk
    with console.status("[bold green]Loading knowledge base index from disk...[/bold green]"):
        try:
            store = load_or_build()
            count = store._collection.count() if hasattr(store, "_collection") else "Persisted"
            console.print(f"[green]✔ Knowledge base loaded successfully ({count} chunks active).[/green]\n")
        except Exception as e:
            console.print(f"[bold red]Failed to load vector store: {e}[/bold red]")
            console.print("[yellow]Please run 'python -m src.ingestion.build_index' first.[/yellow]")
            sys.exit(1)

    memory = ConversationMemory(window_turns=settings.MEMORY_WINDOW_TURNS)
    session_id = "cli_session_01"

    while True:
        try:
            user_input = console.input("[bold cyan]You > [/bold cyan]").strip()
            if not user_input:
                continue

            # Command handling
            if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
                console.print("\n[cyan]Ending SupportPearlz session. Goodbye![/cyan]")
                break
            elif user_input.lower() == "/reset":
                memory.clear()
                console.print("[green]✔ Conversation history cleared. Started fresh session.[/green]\n")
                continue
            elif user_input.lower() == "/help":
                console.print("\n[bold]Available Commands:[/bold]")
                console.print("  [cyan]/reset[/cyan] - Clear conversation memory history")
                console.print("  [cyan]/help[/cyan]  - Display this help message")
                console.print("  [cyan]/exit[/cyan]  - Terminate the support session\n")
                continue

            with console.status("[bold blue]Retrieving grounded context and synthesizing answer...[/bold blue]"):
                response = execute_rag_pipeline(
                    question=user_input,
                    memory=memory,
                    session_id=session_id,
                    vector_store=store
                )

            # Display response
            conf_badge = format_confidence_badge(response.confidence)
            console.print(f"\n[bold white]Bot > [/bold white]{response.answer}")

            # Display sources
            if response.sources:
                console.print("\n[bold]Sources:[/bold]")
                for idx, src in enumerate(response.sources, start=1):
                    loc = f" -- {src.location}" if src.location else ""
                    console.print(f"  [{idx}] [cyan]{src.source}[/cyan]{loc}")
            else:
                if not response.answered:
                    console.print("\n[dim]Sources: None (No supporting document found or question out-of-scope)[/dim]")

            # Telemetry footer
            console.print(f"[dim]Confidence: {conf_badge} | Latency: {response.latency_ms:.1f}ms[/dim]")
            if response.rewritten_query and response.rewritten_query != user_input:
                console.print(f"[dim]Query Rewritten to: '{response.rewritten_query}'[/dim]")
            console.print()

        except KeyboardInterrupt:
            console.print("\n[cyan]Session terminated by user.[/cyan]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error in conversation pipeline: {e}[/bold red]\n")


if __name__ == "__main__":
    run_cli()
