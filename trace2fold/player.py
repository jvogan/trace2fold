"""Run a single sandbox CLI turn without contestant tools or project context."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from .cases import digest

MODEL = "sandbox-cli-model"
FLAGS = [
    "--preset",
    "empty",
    "--no-session-log",
    "--no-foreign-personal-context",
    "--disable-reminders",
    "--disable-request-user-input",
    "--disable-web-tools",
    "--disable-shell",
    "--disable-write",
    "--sandbox-network",
    "restricted",
    "--approval-mode",
    "never",
    "--max-model-steps",
    "1",
]
EVENTS = set(
    """
runtime.command.accepted session.run.linked run.model.configured turn.input.user
run.lifecycle.started task.stream.linked task.lifecycle.proposed task.lifecycle.accepted
task.lifecycle.scheduled task.lifecycle.side_effect_intent task.lifecycle.started task.lifecycle.status
task.lifecycle.completed run.output.delta run.terminal.completed run.usage
""".split()
)


def audit_muse(raw, expected_prompt):
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    types = [e["payload_type"] for e in events]
    if set(types) - EVENTS:
        raise ValueError(f"unexpected Muse event(s): {sorted(set(types) - EVENTS)}")

    def payloads(t):
        return [event["payload"] for event in events if event["payload_type"] == t]

    def only(t):
        rows = payloads(t)
        if len(rows) != 1:
            raise ValueError("expected exactly one " + t)
        return rows[0]

    if only("turn.input.user")["prompt"] != expected_prompt:
        raise ValueError("prompt mismatch")
    configured = only("run.model.configured")
    if configured["model_id"] != MODEL or configured["provider_id"] != "meta":
        raise ValueError("unexpected model or provider")
    proposed = only("task.lifecycle.proposed")["event"]
    intended = only("task.lifecycle.side_effect_intent")["event"]
    if (
        proposed["task_kind"] != "model.meta.response"
        or intended["operation"] != proposed["task_kind"]
    ):
        raise ValueError("non-model operation in trace")
    if (
        intended["task_id"] != proposed["task_id"]
        or only("task.lifecycle.completed")["event"]["task_id"] != proposed["task_id"]
    ):
        raise ValueError("inconsistent task trace")
    opening = [
        p
        for p in payloads("task.lifecycle.status")
        if "opening meta model stream" in p["event"].get("message", "")
    ]
    if len(opening) != 1:
        raise ValueError("cannot verify a single provider attempt")
    usage = only("run.usage")
    if (
        usage["calls"] != 1
        or usage["completed_calls"] != 1
        or len(usage["models"]) != 1
    ):
        raise ValueError("unexpected provider call count")
    response = only("run.terminal.completed")["text"]
    return response, {
        "model": MODEL,
        "route": "cli-zero-tool",
        "effort": "xhigh",
        "model_calls": 1,
        "usage": usage["models"][0],
        "tool_calls_observed": 0,
        "trace_sha256": digest(raw),
        "flags": FLAGS,
    }


def muse(task, out):
    if shutil.which("muse") is None:
        raise RuntimeError(
            "No --endpoint configured in design.json; pass --endpoint http://127.0.0.1:8000/v1/chat/completions "
            "to `trace2fold plan`, or pass pre-recorded replies directly to `trace2fold grade-v4`."
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    # Empty unrelated workspace. Never give the provider a repository or a file path.
    with tempfile.TemporaryDirectory() as workspace:
        cmd = [
            "muse",
            "exec",
            "--workspace",
            workspace,
            *FLAGS,
            "--model",
            MODEL,
            "--reasoning-effort",
            "xhigh",
            "--json",
            task,
        ]
        env = os.environ.copy()
        with (
            (out / "events.jsonl").open("x") as trace,
            (out / "stderr.txt").open("x") as log,
        ):
            subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                cwd=workspace,
                env=env,
                stdout=trace,
                stderr=log,
                check=True,
                timeout=900,
            )
    return audit_muse((out / "events.jsonl").read_text(), task)
