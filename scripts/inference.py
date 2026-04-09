#!/usr/bin/env python3
"""
KAIROSYN Interactive Inference
================================
Usage:
    python scripts/inference.py \
        --checkpoint ./checkpoints/kairosyn-maml-v1 \
        --introspection true

    # Or using defaults (Gemma 4 E4B without checkpoint):
    python scripts/inference.py
"""

import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from kairosyn.model.kairosyn_model import KairosynModel
from kairosyn.model.backbone import KairosynConfig

console = Console()


def print_banner():
    console.print(Panel.fit(
        "[bold purple]KAIROSYN-1[/bold purple]\n"
        "[dim]Recursive Multimodal Architecture for Epinoetic AI[/dim]\n"
        "[dim]Or4cl3 AI Solutions — research@or4cl3.ai[/dim]",
        border_style="purple",
    ))


def print_metrics(output):
    table = Table(title="Epinoetic Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Description", style="dim")

    metrics = [
        ("NCS", f"{output.ncs:.4f}", "Narrative Coherence Score ↑"),
        ("TCE", f"{output.tce:.4f}", "Temporal Continuity Error ↓"),
        ("AAC", f"{output.aac:.4f}", "Abstraction Alignment ↑"),
        ("MSA", f"{output.msa:.4f}", "Multimodal Synchrony ↑"),
        ("RCS", f"{output.rcs:.4f}", "Recursive Convergence ↑"),
        ("φ",   f"{output.phi:.4f}", "IIT Integration Measure ↑"),
    ]
    for metric, value, desc in metrics:
        table.add_row(metric, value, desc)

    console.print(table)


def run_interactive(model: KairosynModel):
    """Interactive chat loop with metrics display."""
    console.print("[dim]Type 'quit' to exit, 'reset' to start a new session.[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Session ended.[/dim]")
            break

        if user_input.lower() == "reset":
            model.reset_session()
            console.print("[yellow]Session state reset.[/yellow]\n")
            continue

        if not user_input:
            continue

        with console.status("[bold purple]KAIROSYN thinking...[/bold purple]"):
            output = model.generate(
                text=user_input,
                enable_introspection=True,
                enable_continuity=True,
                max_new_tokens=512,
            )

        console.print(f"\n[bold purple]KAIROSYN:[/bold purple] {output.generated_text}\n")
        print_metrics(output)
        console.print()


def parse_args():
    parser = argparse.ArgumentParser(description="KAIROSYN Interactive Inference")
    parser.add_argument("--config", type=str, default="configs/model/kairosyn_e4b.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt mode")
    parser.add_argument("--introspection", type=str, default="true")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    print_banner()

    from pathlib import Path
    config = KairosynConfig.from_yaml(args.config) \
        if Path(args.config).exists() \
        else KairosynConfig()

    with console.status("[bold green]Loading KAIROSYN model...[/bold green]"):
        model = KairosynModel(config)
        if args.checkpoint:
            model.load_pretrained(args.checkpoint)

    console.print(f"[green]✓ Model loaded[/green]\n")

    if args.prompt:
        # Single prompt mode
        output = model.generate(
            text=args.prompt,
            max_new_tokens=args.max_new_tokens,
            enable_introspection=(args.introspection.lower() == "true"),
        )
        console.print(f"[bold purple]KAIROSYN:[/bold purple] {output.generated_text}\n")
        print_metrics(output)
    else:
        # Interactive mode
        run_interactive(model)


if __name__ == "__main__":
    main()
