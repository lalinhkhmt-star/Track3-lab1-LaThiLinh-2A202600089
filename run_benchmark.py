from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print

try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except Exception:
    pass

from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.schemas import QAExample, RunRecord
from src.reflexion_lab.utils import load_dataset, save_jsonl

app = typer.Typer(add_completion=False)


def run_with_progress(agent: ReActAgent | ReflexionAgent, examples: list[QAExample]) -> list[RunRecord]:
    records: list[RunRecord] = []
    total = len(examples)
    label = agent.agent_type
    for index, example in enumerate(examples, start=1):
        print(f"[cyan]{label}[/cyan] sample {index}/{total} qid={example.qid}")
        records.append(agent.run(example))
    return records


@app.command()
def main(
    dataset: str = "data/hotpot_100.json",
    out_dir: str = "outputs/hotpot100_real",
    reflexion_attempts: int = 3,
    limit: Optional[int] = None,
) -> None:
    examples = load_dataset(dataset)
    if limit is not None:
        examples = examples[:limit]
    if len(examples) < 100:
        print(f"[yellow]Warning:[/yellow] dataset contains {len(examples)} examples; README asks for at least 100.")

    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)
    try:
        react_records = run_with_progress(react, examples)
        reflexion_records = run_with_progress(reflexion, examples)
    except RuntimeError as exc:
        print(f"[red]Benchmark stopped:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    all_records = react_records + reflexion_records

    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode="real_llm")
    json_path, md_path = save_report(report, out_path)

    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))


if __name__ == "__main__":
    app()
