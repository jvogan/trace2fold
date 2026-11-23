"""Recompute prompts and verify a predeclared panel against contestant traces."""

from pathlib import Path
import json
from .cases import (
    PROMPT_VERSION,
    STRUCTURE_CONDITIONS,
    V3_CONDITIONS,
    V4_CONDITIONS,
    V4_VERSIONS,
    condition_case,
    digest,
    input_condition,
    prompt,
    v4_conditions,
)
from .player import audit_muse, MODEL
from .trials import ANSWER_PARSER, LEGACY_PARSER, answer_sequence, revision_prompt

DEFAULT_CONDITIONS = ("geometry", "review", "restart", "length")
CONDITIONS = (
    *DEFAULT_CONDITIONS,
    *STRUCTURE_CONDITIONS[1:],
    *(condition for condition in V4_CONDITIONS if condition not in DEFAULT_CONDITIONS),
)


def panel(
    cases, ids, conditions, model=MODEL, endpoint=None, version=PROMPT_VERSION,
    *, answer_parser=None,
):
    if answer_parser not in (None, ANSWER_PARSER, LEGACY_PARSER):
        raise ValueError(f"Unknown answer parser: {answer_parser}")
    if endpoint is None and model != MODEL:
        raise ValueError("choose a loopback endpoint for this contestant")
    if endpoint is not None:
        from backends.plain_player import route_address

        route_address(endpoint)
        if model == "control":
            raise ValueError("reserved control model name")
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("select distinct cases")
    if version == "v3" and set(conditions) - set(V3_CONDITIONS):
        raise ValueError("v3 supports only geometry and length conditions")
    if version in V4_VERSIONS and set(conditions) - set(v4_conditions(version)):
        raise ValueError(f"{version} supports only {', '.join(v4_conditions(version))} conditions")
    if (
        not conditions
        or len(set(conditions)) != len(conditions)
        or set(conditions) - set(CONDITIONS)
        or ("review" in conditions and "geometry" not in conditions)
    ):
        raise ValueError("select distinct conditions and include geometry for review")
    additional = [c for c in conditions if c not in DEFAULT_CONDITIONS]
    if additional and version == "v1":
        raise ValueError("the archived v1 protocol supports only the original conditions")
    if version in ("v3", *V4_VERSIONS) and "length" in conditions:
        additional.append("length")
    # Keep legacy plans byte-compatible. Freeze every additional representation before inference.
    input_hashes = {
        c: [digest(prompt(cases[i], c, version)) for i in ids] for c in additional
    }
    return {
        **({"answer_parser": answer_parser} if answer_parser else {}),
        **({"representation_prompts": input_hashes} if additional else {}),
        "case_ids": ids,
        "condition_order": conditions,
        "contestant": model,
        **({"endpoint": endpoint} if endpoint else {}),
        # Plans made before prompt v2 carry no version and replay as v1.
        **({"prompt_version": version} if version != "v1" else {}),
        "calls_planned": len(ids) * len(conditions),
        "target_prompts": [digest(prompt(cases[i], version=version)) for i in ids],
    }


def audit(cases, run):
    run = Path(run)
    plan = json.loads((run / "design.json").read_text())
    ids, conditions = plan["case_ids"], plan["condition_order"]
    model, endpoint = plan["contestant"], plan.get("endpoint")
    version = plan.get("prompt_version", "v1")
    if plan != panel(
        cases, ids, conditions, model, endpoint, version,
        answer_parser=plan.get("answer_parser"),
    ):
        raise ValueError("roster differs from the frozen target or protocol")
    expected = {f"{i}-{c}" for i in ids for c in conditions}
    records = {
        p.stem: json.loads(p.read_text()) for p in (run / "records").glob("*.json")
    }
    if set(records) != expected:
        raise ValueError(
            f"trial coverage mismatch: missing {sorted(expected - set(records))}, unexpected {sorted(set(records) - expected)}"
        )
    traces = [p.name for p in (run / "traces").iterdir()]
    if set(traces) != expected:
        raise ValueError("trace directory coverage differs from the roster")
    valid = 0
    distinct = set()
    sessions = set()
    for i in ids:
        base_case = cases[i]
        for c in conditions:
            case = condition_case(base_case, c)
            name = f"{i}-{c}"
            record = records[name]
            task = prompt(base_case, input_condition(c), version)
            parent = None
            if c == "review":
                first = records[f"{i}-geometry"]
                task = revision_prompt(case, first["raw"], version)
                parent = first["protocol"]["trace_sha256"]
            trace_dir = run / "traces" / name
            if endpoint:
                from backends.plain_player import audit_plain

                raw, protocol = audit_plain(
                    trace_dir, task, endpoint=endpoint, model=model
                )
            else:
                trace = (trace_dir / "events.jsonl").read_text()
                raw, protocol = audit_muse(trace, task)
                session_ids = {
                    json.loads(line)["stream"]["id"]
                    for line in trace.splitlines()
                    if line.strip()
                }
                if len(session_ids) != 1 or sessions & session_ids:
                    raise ValueError("each trial needs a distinct fresh session")
                sessions.update(session_ids)
            parser = record.get("answer_parser", LEGACY_PARSER)
            if version != "v1" and plan.get("answer_parser", parser) != parser:
                raise ValueError(f"Answer parser differs from the plan for {name}")
            sequence = answer_sequence(case, raw.strip(), protocol, bare_only=version == "v1", parser=parser)
            if (
                raw.strip() != record["raw"]
                or record["protocol"] != protocol
                or record["previous"] != parent
                or record["prompt_sha256"] != digest(task)
                or record["case"] != i
                or record["condition"] != c
                or record["valid"] != (sequence is not None)
                or record["sequence"] != sequence
                or record["target_sha256"]
                != digest(case.xyz.astype("<f8").tobytes().hex() + case.template)
            ):
                raise ValueError(f"submission differs from trace or case: {name}")
            if record["valid"]:
                valid += 1
                distinct.add(record["sequence"])
    return {
        "model": model,
        "planned_calls": len(expected),
        "verified_calls": len(records),
        "valid": valid,
        "invalid": len(records) - valid,
        "unique_valid_sequences": len(distinct),
        "roster_sha256": digest((run / "design.json").read_text()),
        "traces": {
            name: r["protocol"]["trace_sha256"] for name, r in sorted(records.items())
        },
    }
