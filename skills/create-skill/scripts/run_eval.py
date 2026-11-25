#!/usr/bin/env python3
"""Run trigger evaluation for a skill description.

Tests whether a skill's description causes the agent to load the skill for a
set of queries. Drives `opencode run --format json` and watches the event
stream for a `skill` tool call. Outputs results as JSON.
"""

import argparse
import json
import os
import re
import select
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from scripts.utils import parse_skill_md


OPENCODE_BIN = os.environ.get("OPENCODE_BIN", "opencode")


def find_project_root() -> Path:
    """Find the project root by walking up from cwd looking for .opencode/ or .claude/."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / ".opencode").is_dir() or (parent / ".claude").is_dir():
            return parent
    return current


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-") or "skill"


def _child_env(cwd: Path) -> dict[str, str]:
    """Environment for a child `opencode run`.

    The CLI resolves its project from the environment, so an inherited session
    ID or PWD would point the child at the caller's project instead of the
    sandbox. Strip those and pin PWD to the sandbox.
    """
    drop = {"OPENCODE_SESSION_ID", "OPENCODE", "OPENCODE_TERMINAL", "AI_AGENT"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env["PWD"] = str(cwd)
    return env


def run_single_query(
    query: str,
    skill_name: str,
    skill_description: str,
    timeout: int,
    project_root: str | None = None,
    model: str | None = None,
) -> bool:
    """Run a single query and return whether the skill was loaded.

    The skill is written into a throwaway project so the run is isolated and
    the candidate description is what the model actually sees. Then
    `opencode run --format json` is streamed and every tool call is checked for
    a load of this test skill.
    """
    clean_name = f"{_slug(skill_name)}-skill-{uuid.uuid4().hex[:8]}"
    sandbox = Path(tempfile.mkdtemp(prefix="skill-eval-"))
    skill_dir = sandbox / ".opencode" / "skills" / clean_name
    skill_file = skill_dir / "SKILL.md"

    try:
        skill_dir.mkdir(parents=True)
        # YAML block scalar so quotes and colons in the description are safe
        indented_desc = "\n  ".join(skill_description.split("\n"))
        skill_file.write_text(
            "---\n"
            f"name: {clean_name}\n"
            "description: |\n"
            f"  {indented_desc}\n"
            "---\n\n"
            f"# {skill_name}\n\n{skill_description}\n"
        )

        cmd = [OPENCODE_BIN, "run", "--standalone", "--archived", "--format", "json"]
        if model:
            cmd.extend(["--model", model])
        cmd.append(query)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            cwd=sandbox,
            env=_child_env(sandbox),
        )

        start_time = time.time()
        buffer = ""

        try:
            while time.time() - start_time < timeout:
                if process.poll() is not None:
                    remaining = process.stdout.read()
                    if remaining:
                        buffer += remaining.decode("utf-8", errors="replace")
                    break

                ready, _, _ = select.select([process.stdout], [], [], 1.0)
                if not ready:
                    continue

                chunk = os.read(process.stdout.fileno(), 8192)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if event.get("type") != "tool_use":
                        continue
                    part = event.get("part", {})
                    tool = part.get("tool", "")
                    state = part.get("state") or {}
                    tool_input = state.get("input") or {}
                    raw_text = state.get("rawInput") or ""

                    if tool == "skill":
                        skill_id = tool_input.get("id", "") if isinstance(tool_input, dict) else ""
                        if skill_id == clean_name:
                            return True
                    elif tool in ("read", "grep", "glob") and clean_name in raw_text:
                        return True
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            try:
                process.stdout.read()
            except Exception:
                pass

        return False
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)



def run_eval(
    eval_set: list[dict],
    skill_name: str,
    description: str,
    num_workers: int,
    timeout: int,
    project_root: Path,
    runs_per_query: int = 1,
    trigger_threshold: float = 0.5,
    model: str | None = None,
) -> dict:
    """Run the full eval set and return results."""
    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_info = {}
        for item in eval_set:
            for run_idx in range(runs_per_query):
                future = executor.submit(
                    run_single_query,
                    item["query"],
                    skill_name,
                    description,
                    timeout,
                    str(project_root),
                    model,
                )
                future_to_info[future] = (item, run_idx)

        query_triggers: dict[str, list[bool]] = {}
        query_items: dict[str, dict] = {}
        for future in as_completed(future_to_info):
            item, _ = future_to_info[future]
            query = item["query"]
            query_items[query] = item
            if query not in query_triggers:
                query_triggers[query] = []
            try:
                query_triggers[query].append(future.result())
            except Exception as e:
                print(f"Warning: query failed: {e}", file=sys.stderr)
                query_triggers[query].append(False)

    for query, triggers in query_triggers.items():
        item = query_items[query]
        trigger_rate = sum(triggers) / len(triggers)
        should_trigger = item["should_trigger"]
        if should_trigger:
            did_pass = trigger_rate >= trigger_threshold
        else:
            did_pass = trigger_rate < trigger_threshold
        results.append({
            "query": query,
            "should_trigger": should_trigger,
            "trigger_rate": trigger_rate,
            "triggers": sum(triggers),
            "runs": len(triggers),
            "pass": did_pass,
        })

    passed = sum(1 for r in results if r["pass"])
    total = len(results)

    return {
        "skill_name": skill_name,
        "description": description,
        "results": results,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run trigger evaluation for a skill description")
    parser.add_argument("--eval-set", required=True, help="Path to eval set JSON file")
    parser.add_argument("--skill-path", required=True, help="Path to skill directory")
    parser.add_argument("--description", default=None, help="Override description to test")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per query in seconds")
    parser.add_argument("--runs-per-query", type=int, default=3, help="Number of runs per query")
    parser.add_argument("--trigger-threshold", type=float, default=0.5, help="Trigger rate threshold")
    parser.add_argument("--model", default=None, help="Model for `opencode run -m provider/model` (default: OpenCode's configured model)")
    parser.add_argument("--verbose", action="store_true", help="Print progress to stderr")
    args = parser.parse_args()

    eval_set = json.loads(Path(args.eval_set).read_text())
    skill_path = Path(args.skill_path)

    if not (skill_path / "SKILL.md").exists():
        print(f"Error: No SKILL.md found at {skill_path}", file=sys.stderr)
        sys.exit(1)

    name, original_description, content = parse_skill_md(skill_path)
    description = args.description or original_description
    project_root = find_project_root()

    if args.verbose:
        print(f"Evaluating: {description}", file=sys.stderr)

    output = run_eval(
        eval_set=eval_set,
        skill_name=name,
        description=description,
        num_workers=args.num_workers,
        timeout=args.timeout,
        project_root=project_root,
        runs_per_query=args.runs_per_query,
        trigger_threshold=args.trigger_threshold,
        model=args.model,
    )

    if args.verbose:
        summary = output["summary"]
        print(f"Results: {summary['passed']}/{summary['total']} passed", file=sys.stderr)
        for r in output["results"]:
            status = "PASS" if r["pass"] else "FAIL"
            rate_str = f"{r['triggers']}/{r['runs']}"
            print(f"  [{status}] rate={rate_str} expected={r['should_trigger']}: {r['query'][:70]}", file=sys.stderr)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
