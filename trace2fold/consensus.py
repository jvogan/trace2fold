"""Aggregate two sequence-only predictors after both have finished."""

from collections import defaultdict


def consensus(first, second):
    def index(report):
        rows = {
            (r["case"], r["model"], r["condition"], r.get("trial_id")): r
            for r in report["rows"]
        }
        if len(rows) != len(report["rows"]):
            raise ValueError("duplicate grader records")
        return rows

    if (
        first["judge"]["model"] != "biohub/ESMFold2-Fast"
        or second["judge"]["model"] != "boltz-community/boltz-2"
    ):
        raise ValueError("expected ESMFold2-Fast and Boltz-2 reports in that order")
    if first.get("scoring_profile") != second.get("scoring_profile"):
        raise ValueError("judges used different scoring profiles")
    a, b = index(first), index(second)
    if set(a) != set(b):
        raise ValueError("graders scored different trial sets")
    out = []
    for key, left in a.items():
        right = b[key]
        if any(
            left[k] != right[k]
            for k in ("sequence_sha256", "target_sha256", "pair", "length")
        ):
            raise ValueError("judges received different inputs")
        if (
            left["status"] == "invalid"
            and right["status"] != "invalid"
            or right["status"] == "invalid"
            and left["status"] != "invalid"
        ):
            raise ValueError("inconsistent validity between judges")
        row = {
            k: left[k]
            for k in [
                "case",
                "condition",
                "model",
                "pair",
                "tier",
                "family",
                "generator",
                "length",
            ]
        }
        if "trial_id" in left:
            row["trial_id"] = left["trial_id"]
        row.update(
            esmfold2=left["score"], boltz2=right["score"], status="ungraded", score=None
        )
        if left["status"] == right["status"] == "graded":
            row.update(
                status="graded",
                score=(left["score"] + right["score"]) / 2,
                judge_gap=abs(left["score"] - right["score"]),
                both_within_2a=(left["rmsd"] <= 2 and right["rmsd"] <= 2),
            )
            for k in [
                "margin",
                "decoy_score",
                "contact_f1",
                "ca_lddt",
                "aligned_coverage",
                "rmsd",
                "mean_plddt",
            ]:
                if left.get(k) is not None and right.get(k) is not None:
                    row[k] = (left[k] + right[k]) / 2
        elif left["status"] == right["status"] == "invalid":
            row.update(status="invalid", score=0.0)
        out.append(row)
    return out


def review_comparison(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[(row["case"], row["model"])][row["condition"]] = row["score"]
    out = []
    for (case, model), scores in groups.items():
        if all(scores.get(k) is not None for k in ["geometry", "review", "restart"]):
            out.append(
                {
                    "case": case,
                    "model": model,
                    "revision_gain": scores["review"] - scores["geometry"],
                    "review_minus_restart": scores["review"] - scores["restart"],
                }
            )
    return out


def condition_comparison(rows):
    by_case = defaultdict(dict)
    for row in rows:
        by_case[row["case"], row["model"]][row["condition"]] = row
    result = []
    for (case, model), arms in by_case.items():
        if "geometry" not in arms or "length" not in arms:
            continue
        shape, length = arms["geometry"], arms["length"]
        if shape["score"] is None or length["score"] is None:
            continue
        result.append(
            {
                "case": case,
                "model": model,
                "geometry_minus_length": shape["score"] - length["score"],
                "margin_gain": shape["margin"] - length["margin"]
                if shape.get("margin") is not None and length.get("margin") is not None
                else None,
            }
        )
    return result


def strata(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["condition"], row[field])].append(row)
    return [
        {
            "model": model,
            "condition": condition,
            field: level,
            "targets": len(group),
            "valid_graded": sum(r["status"] == "graded" for r in group),
            "mean_score": sum(r["score"] for r in group) / len(group)
            if all(r["score"] is not None for r in group)
            else None,
        }
        for (model, condition, level), group in sorted(groups.items())
    ]


def representation_comparisons(rows):
    """Compare input conditions on the same targets, keeping calibration separate."""
    comparisons = (
        ("annotated", "geometry"),
        ("annotated", "topology"),
        ("topology", "geometry"),
        ("backbone", "geometry"),
        ("backbone-annotated", "backbone"),
        ("backbone-annotated", "annotated"),
    )
    groups = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        suite = "calibration" if row["generator"] == "known" else "generated"
        groups[(suite, row["model"])][row["case"]][row["condition"]] = row

    result = []
    for (suite, model), cases in sorted(groups.items()):
        for first, second in comparisons:
            if not any(first in arms and second in arms for arms in cases.values()):
                continue
            deltas, pair_margins = [], defaultdict(list)
            invalid, missing = 0, 0
            for arms in cases.values():
                a, b = arms.get(first), arms.get(second)
                if a is None or b is None or a["score"] is None or b["score"] is None:
                    missing += 1
                    continue
                invalid += int("invalid" in (a["status"], b["status"]))
                deltas.append(a["score"] - b["score"])
                if (
                    a["pair"]
                    and a.get("margin") is not None
                    and b.get("margin") is not None
                ):
                    pair_margins[a["pair"]].append(a["margin"] - b["margin"])
            margins = [
                sum(pair) / 2 for pair in pair_margins.values() if len(pair) == 2
            ]
            result.append(
                {
                    "suite": suite,
                    "model": model,
                    "first": first,
                    "second": second,
                    "cases": len(cases),
                    "compared_cases": len(deltas),
                    "missing_cases": missing,
                    "compared_cases_with_invalid_answer": invalid,
                    "mean_score_difference": sum(deltas) / len(deltas)
                    if deltas
                    else None,
                    "complete_margin_pairs": len(margins),
                    "mean_pair_margin_difference": sum(margins) / len(margins)
                    if margins
                    else None,
                }
            )
    return result
