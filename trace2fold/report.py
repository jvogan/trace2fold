"""Sequence-only fold scoring and pair-level uncertainty."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import math
import numpy as np
from .cases import digest, read_ca
from .metrics import SPACEFILL_PROFILE, measure
from .trials import LEGACY_PARSER, answer_sequence, decoys


def read_fold(directory, sequence):
    key = digest(sequence)[:32]
    pdb = Path(directory) / (key + ".pdb")
    if not pdb.exists():
        return None
    chains = list(
        dict.fromkeys(
            line[21]
            for line in pdb.read_text().splitlines()
            if line.startswith("ATOM") and line[12:16].strip() == "CA"
        )
    )
    xyz, seqs = read_ca(pdb, sorted(chains))
    if ":".join(seqs) != sequence:
        raise ValueError(f"fold sequence or chain mismatch in {pdb}")
    return xyz


def score(cases, records, directory, *, profile=None):
    mates = decoys(cases)
    protocol = json.loads((Path(directory) / "manifest.json").read_text())["protocol"]
    results = []
    seen = set()
    for row in records:
        case = cases[row["case"]]
        identity = (
            row["protocol"]["model"],
            row["condition"],
            case.id,
            row.get("trial_id"),
        )
        if identity in seen:
            raise ValueError(f"duplicate trial {identity}")
        seen.add(identity)
        locked_hash = digest(case.xyz.astype("<f8").tobytes().hex() + case.template)
        if row["target_sha256"] != locked_hash:
            raise ValueError(f"target changed after submission: {case.id}")
        result = {
            "case": case.id,
            "sequence_sha256": digest(row["sequence"]) if row["valid"] else None,
            "target_sha256": locked_hash,
            "condition": row["condition"],
            "model": row["protocol"]["model"],
            "pair": case.info.get("pair"),
            "tier": case.info.get("tier"),
            "family": case.info.get("family"),
            "generator": case.info.get("generator", "known"),
            "length": sum(case.lengths),
            "status": "invalid",
            "score": 0.0,
        }
        if "trial_id" in row:
            result["trial_id"] = row["trial_id"]
        sequence = answer_sequence(
            case, row["raw"], row["protocol"], bare_only="format" not in row,
            parser=row.get("answer_parser", LEGACY_PARSER),
        )
        if row["valid"] != (sequence is not None) or row["sequence"] != sequence:
            raise ValueError(f"submission integrity error: {case.id}")
        if row["valid"]:
            xyz = read_fold(directory, row["sequence"])
            if xyz is None:
                result.update(status="ungraded", score=None)
            else:
                terms, _ = measure(xyz, case, profile=profile)
                result.update(status="graded", **terms)
                sidecar = Path(directory) / (digest(row["sequence"])[:32] + ".json")
                evidence = json.loads(sidecar.read_text())
                raw_confidence = np.asarray(evidence["plddt"])
                if (
                    evidence["sequence"] != row["sequence"]
                    or raw_confidence.shape != case.mutable.shape
                    or evidence["provenance"]["protocol"] != protocol
                    or evidence["pdb_sha256"]
                    != hashlib.sha256(
                        sidecar.with_suffix(".pdb").read_bytes()
                    ).hexdigest()
                    or not np.isfinite(raw_confidence).all()
                    or not np.all((0 <= raw_confidence) & (raw_confidence <= 100))
                ):
                    raise ValueError("fold evidence and submission differ")
                result["mean_plddt"] = float(np.mean(raw_confidence[case.mutable]))
                if profile == SPACEFILL_PROFILE:
                    if (
                        evidence["id"] != sidecar.stem
                        or evidence["sequence_sha256"] != result["sequence_sha256"]
                        or not np.array_equal(evidence["ca"], xyz)
                    ):
                        raise ValueError(
                            "prediction sidecar and PDB coordinates disagree"
                        )
                    ptm = evidence.get("ptm")
                    if ptm is not None and not 0 <= ptm <= 1:
                        raise ValueError("pTM must be between zero and one")
                    result["ptm"] = ptm
                if case.id in mates:
                    other, _ = measure(xyz, mates[case.id], profile=profile)
                    result.update(
                        decoy_score=other["score"],
                        margin=terms["score"] - other["score"],
                    )
        results.append(result)
    return results


def paired_statistics(rows):
    pairs = defaultdict(list)
    for row in rows:
        if row.get("pair"):
            pairs[row["pair"]].append(row)
    margins = [
        sum(r["margin"] for r in pair) / 2
        for pair in pairs.values()
        if len(pair) == 2 and all("margin" in r for r in pair)
    ]
    wins = sum(m > 1e-8 for m in margins)
    losses = sum(m < -1e-8 for m in margins)
    # Under label exchange, each distinct pair has either sign with equal probability.
    n = wins + losses
    p = sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n if n else None
    k_min = min(wins, losses)
    p_two = (
        min(1.0, 2.0 * sum(math.comb(n, k) for k in range(0, k_min + 1)) / 2**n)
        if n
        else None
    )
    ci = None
    if len(margins) >= 2:
        rng = np.random.default_rng(261118)
        means = rng.choice(margins, (5000, len(margins))).mean(1)
        ci = np.percentile(means, [2.5, 97.5]).tolist()
    return {
        "complete_pairs": len(margins),
        "incomplete_pairs": len(pairs) - len(margins),
        "wins": wins,
        "losses": losses,
        "ties": len(margins) - n,
        "mean_pair_margin": float(np.mean(margins)) if margins else None,
        "bootstrap_95pct": ci,
        "one_sided_sign_p": p,
        "two_sided_sign_p": p_two,
    }


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                row["model"],
                row["condition"],
                "calibration" if row["generator"] == "known" else "generated",
            )
        ].append(row)
    summary = []
    for (model, condition, suite), group in groups.items():
        missing = sum(r["status"] == "ungraded" for r in group)
        summary.append(
            {
                "model": model,
                "condition": condition,
                "suite": suite,
                "attempts": len(group),
                "graded": sum(r["status"] == "graded" for r in group),
                "invalid": sum(r["status"] == "invalid" for r in group),
                "ungraded": missing,
                "mean_score": None
                if missing
                else float(np.mean([r["score"] for r in group])),
                **paired_statistics(group),
            }
        )
    return summary
