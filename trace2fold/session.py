"""Execute a frozen roster. Each call has its own trace and locked answer."""

from pathlib import Path
import json
import random
from .audit import panel
from .cases import condition_case, input_condition, prompt
from .player import muse, audit_muse
from .trials import LEGACY_PARSER, revision_prompt, lock


def play(cases, run, case_id, condition):
    run = Path(run)
    design = json.loads((run / "design.json").read_text())
    model, endpoint = design["contestant"], design.get("endpoint")
    version = design.get("prompt_version", "v1")
    if design != panel(
        cases, design["case_ids"], design["condition_order"], model, endpoint, version,
        answer_parser=design.get("answer_parser"),
    ):
        raise ValueError("roster differs from the frozen targets")
    if case_id not in design["case_ids"] or condition not in design["condition_order"]:
        raise ValueError("trial outside frozen roster")
    case = cases[case_id]
    previous = None
    if condition == "review":
        first = json.loads((run / "records" / (case.id + "-geometry.json")).read_text())
        if (
            first["case"] != case_id
            or first["protocol"]["model"] != model
            or first["condition"] != "geometry"
        ):
            raise ValueError(
                "review requires the same contestant's first answer for this case"
            )
        task = revision_prompt(case, first["raw"], version)
        previous = first["protocol"]["trace_sha256"]
    else:
        task = prompt(case, input_condition(condition), version)
    key = case_id + "-" + condition
    folder = run / "traces" / key
    # Re-audit an existing trace directory before writing the locked record.
    if endpoint:
        from backends.plain_player import plain, audit_plain

        response, protocol = (
            audit_plain(folder, task, endpoint=endpoint, model=model)
            if folder.exists()
            else plain(task, folder, endpoint=endpoint, model=model)
        )
    elif folder.exists():
        response, protocol = audit_muse((folder / "events.jsonl").read_text(), task)
    else:
        response, protocol = muse(task, folder)
    return lock(
        condition_case(case, condition),
        run / "records" / (key + ".json"),
        condition,
        response,
        protocol,
        task,
        previous,
        version=version,
        parser=design.get("answer_parser", LEGACY_PARSER),
    )


def run_all(cases, run):
    design = json.loads((run / "design.json").read_text())
    trials = [(i, c) for i in design["case_ids"] for c in design["condition_order"]]
    # Spread condition order across cases while preserving each revision's prerequisite.
    rng = random.Random(261118)
    pending = set(trials)
    while pending:
        eligible = sorted(
            (i, c)
            for i, c in pending
            if c != "review" or (i, "geometry") not in pending
        )
        trial = rng.choice(eligible)
        play(cases, run, *trial)
        pending.remove(trial)
