from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich import box
except ImportError:
    print("Install CLI deps: pip install touchstone-llm[cli]")
    sys.exit(1)

from .judge import Judge
from .panel import Panel
from .rubric import Rubric

app = typer.Typer(
    name="touchstone",
    help="Bias-aware LLM judge. Evaluate, compare, and calibrate AI outputs.",
    no_args_is_help=True,
)
console = Console()


def _build_rubric(criteria: list[str]) -> Rubric | None:
    return Rubric.from_list(criteria) if criteria else None


@app.command()
def score(
    output: str = typer.Argument(..., help="Text to evaluate"),
    criteria: Optional[list[str]] = typer.Option(None, "--criteria", "-c", help="Evaluation criteria"),
    model: str = typer.Option("claude-sonnet-4-6", "--model", "-m"),
    threshold: float = typer.Option(0.7, "--threshold", "-t"),
    evaluated_model: Optional[str] = typer.Option(None, "--evaluated-model", help="Model that generated the output (for self-preference detection)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Score a single LLM output."""
    judge = Judge(model=model, threshold=threshold, evaluated_model=evaluated_model)
    rubric = _build_rubric(criteria or [])

    result = asyncio.run(judge.score(output, rubric))

    if json_output:
        console.print_json(result.model_dump_json())
        raise typer.Exit(0 if result.passed else 1)

    table = Table(box=box.ROUNDED, show_header=False, title=f"[bold]touchstone[/] — {model}")
    table.add_column("Field", style="dim")
    table.add_column("Value")

    color = "green" if result.passed else "red"
    table.add_row("Score (raw)", f"{result.score:.3f}")
    table.add_row("Score (adjusted)", f"[{color}]{result.adjusted_score:.3f}[/{color}]")
    table.add_row("Confidence", f"{result.confidence:.2f}")
    table.add_row("Passed", "[green]✓[/green]" if result.passed else "[red]✗[/red]")
    if result.bias_flags:
        table.add_row("Bias flags", ", ".join(result.bias_flags))
    table.add_row("Reasoning", result.reasoning)

    console.print(table)

    if result.criteria_scores:
        ct = Table(title="Criteria Breakdown", box=box.SIMPLE)
        ct.add_column("Criterion")
        ct.add_column("Score")
        ct.add_column("Notes")
        for cs in result.criteria_scores:
            c = "green" if cs.score >= threshold else "red"
            ct.add_row(cs.name, f"[{c}]{cs.score:.2f}[/{c}]", cs.reasoning)
        console.print(ct)

    raise typer.Exit(0 if result.passed else 1)


@app.command()
def compare(
    output_a: str = typer.Argument(..., help="Output A"),
    output_b: str = typer.Argument(..., help="Output B"),
    criteria: Optional[list[str]] = typer.Option(None, "--criteria", "-c"),
    model: str = typer.Option("claude-sonnet-4-6", "--model", "-m"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Pairwise comparison with position-bias correction."""
    judge = Judge(model=model)
    rubric = _build_rubric(criteria or [])
    result = asyncio.run(judge.compare(output_a, output_b, rubric))

    if json_output:
        console.print_json(result.model_dump_json())
        return

    winner_display = {"a": "[green]A wins[/green]", "b": "[green]B wins[/green]", "tie": "[yellow]Tie[/yellow]"}
    console.print(f"\nResult: {winner_display[result.winner]}")
    console.print(f"Confidence: {result.confidence:.2f}")
    console.print(f"Score A: {result.score_a:.3f}  |  Score B: {result.score_b:.3f}")
    if result.position_bias_detected:
        console.print("[yellow]⚠ Position bias detected — result may be unreliable[/yellow]")
    console.print(f"Reasoning: {result.reasoning}")


@app.command()
def panel_eval(
    output: str = typer.Argument(..., help="Text to evaluate"),
    models: list[str] = typer.Option(["claude-sonnet-4-6", "gpt-4o"], "--model", "-m"),
    criteria: Optional[list[str]] = typer.Option(None, "--criteria", "-c"),
    threshold: float = typer.Option(0.7, "--threshold", "-t"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Multi-model panel evaluation with consensus scoring."""
    p = Panel(models=models, threshold=threshold)
    rubric = _build_rubric(criteria or [])
    result = asyncio.run(p.evaluate(output, rubric))

    if json_output:
        console.print_json(result.model_dump_json())
        raise typer.Exit(0 if result.passed else 1)

    color = "green" if not result.disputed else "yellow"
    console.print(f"\nConsensus score: [{color}]{result.consensus_score:.3f}[/{color}]")
    console.print(f"Agreement: {result.agreement:.2f}  |  Disputed: {'⚠ yes' if result.disputed else 'no'}")
    console.print(f"Passed ({threshold}): {'[green]✓[/green]' if result.passed else '[red]✗[/red]'}")
    console.print("\nIndividual scores:")
    for m, r in result.individual_results.items():
        console.print(f"  {m}: {r.adjusted_score:.3f}")

    raise typer.Exit(0 if result.passed else 1)


def main() -> None:
    app()
