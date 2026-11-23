"""Create one-call trials and lock responses before any folding."""

import json
import random
import re
from collections import defaultdict
import numpy as np
from .cases import AA, PROMPT_VERSION, V4_VERSIONS, digest, dump, prompt


def controls(case):
    source = (case.info.get("sequence") or "").replace(":", "")
    if not source:
        return {}
    editable = np.where(case.mutable)[0]
    shuffled = list(source)
    moved = [source[i] for i in editable]
    random.Random(digest(case.id + ":shuffle-v1")).shuffle(moved)
    for i, letter in zip(editable, moved):
        shuffled[i] = letter
    return {"positive": case.split(source), "shuffled": case.split("".join(shuffled))}


LEGACY_PARSER = "last-token-v1"
ANSWER_PARSER = "length-aware-v2"
TOKEN_TRIM = "`*\"'.,;()[]"
FENCED_BLOCK = re.compile(
    r"^[ \t]*```[^`\r\n]*\r?\n(?P<body>.*?)^[ \t]*```[ \t]*\r?$",
    re.MULTILINE | re.DOTALL,
)


def answer_sequence(case, raw, protocol, *, bare_only=False, parser=ANSWER_PARSER):
    """Read the last whole candidate that satisfies the case, in reply order."""
    if parser not in (ANSWER_PARSER, LEGACY_PARSER):
        raise ValueError(f"Unknown answer parser: {parser}")
    if protocol.get("answer_status", "complete") != "complete":
        return None
    text = raw.strip()
    if case.validate(text):
        return text
    if bare_only:
        return None
    if parser == LEGACY_PARSER:
        tokens = (token.strip(TOKEN_TRIM) for token in text.split())
        found = [t for t in tokens if re.fullmatch(f"[{AA}:]{{8,}}", t)]
        return found[-1] if found and case.validate(found[-1]) else None

    compact = "".join(text.split())
    wrapper = re.fullmatch(r"(?P<ticks>`{1,3})(?P<body>[^`]+)(?P=ticks)", compact)
    if wrapper:
        compact = wrapper["body"]
    if case.validate(compact):
        return compact
    candidates = []
    for token in re.finditer(r"\S+", text):
        candidate = token.group().strip(TOKEN_TRIM)
        if case.validate(candidate):
            candidates.append((token.end(), candidate))
    for block in FENCED_BLOCK.finditer(text):
        candidate = "".join(block["body"].split())
        if case.validate(candidate):
            candidates.append((block.end("body"), candidate))
    return max(candidates, default=(0, None))[1]


def valid_answer(case, raw, protocol):
    return answer_sequence(case, raw, protocol) is not None


def lock(
    case,
    output,
    kind,
    raw,
    protocol,
    task=None,
    previous=None,
    *,
    version=PROMPT_VERSION,
    parser=ANSWER_PARSER,
):
    raw = raw.strip()
    sequence = answer_sequence(case, raw, protocol, bare_only=version == "v1", parser=parser)
    format_field = {}
    if version != "v1":
        answer_format = None
        if sequence is not None:
            answer_format = "bare" if sequence == raw else "extracted"
        format_field["format"] = answer_format
        if parser != LEGACY_PARSER:
            format_field["answer_parser"] = parser
    row = {
        "case": case.id,
        "condition": kind,
        "raw": raw,
        "valid": sequence is not None,
        **format_field,
        "sequence": sequence,
        "protocol": protocol,
        "prompt_sha256": digest(task) if task is not None else None,
        "target_sha256": digest(case.xyz.astype("<f8").tobytes().hex() + case.template),
        "previous": previous,
    }
    dump(output, row)
    return row


def revision_prompt(case, first, version=PROMPT_VERSION):
    if version == "v3":
        raise ValueError("v3 supports only geometry and length conditions")
    if version in V4_VERSIONS:
        raise ValueError(f"{version} has no review condition")
    if version == "v1":
        return (
            prompt(case, version="v1")
            + "\nYour first submitted answer was:\n"
            + json.dumps(first)
            + "\nReview it once. Return one final sequence using the same format."
        )
    return (
        prompt(case, version=version)
        + "\n\nYour earlier answer was:\n"
        + first
        + "\n\nCheck it against the backbone. Reply with your final sequence only."
    )


def fold_inputs(records):
    seqs = {r["sequence"] for r in records if r["valid"]}
    return {"items": [{"id": digest(s)[:32], "sequence": s} for s in sorted(seqs)]}


def decoys(cases):
    pairs = defaultdict(list)
    for case in cases.values():
        if case.info.get("pair"):
            pairs[case.info["pair"]].append(case)
    mates = {}
    for key, (a, b) in ((k, v) for k, v in pairs.items() if len(v) == 2):
        if a.lengths != b.lengths or a.template != b.template:
            raise ValueError(f"lengths or fixed positions differ in pair {key}")
        mates[a.id], mates[b.id] = b, a
    if any(len(v) != 2 for v in pairs.values()):
        raise ValueError("each declared pair must have two members")
    return mates
