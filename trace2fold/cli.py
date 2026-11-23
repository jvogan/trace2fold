import argparse
import json
from pathlib import Path
import sys
from .cases import (
    PROMPT_VERSION,
    PROMPT_VERSIONS,
    V3_CONDITIONS,
    V4_VERSIONS,
    dump,
    input_condition,
    load_cases,
    prompt,
    v4_conditions,
)
from .trials import ANSWER_PARSER, controls, lock, revision_prompt, fold_inputs
from .report import score, summarize
from .player import MODEL
from .session import play, run_all
from .view import export
from .consensus import (
    consensus,
    review_comparison,
    condition_comparison,
    representation_comparisons,
    strata,
)
from .audit import panel, audit, CONDITIONS, DEFAULT_CONDITIONS
from .v4_eval import (
    grade_v4,
    load_cases_flexible,
    load_submissions,
    load_twin_map,
    verify_benchmarks,
)


def read(path):
    return json.loads(Path(path).read_text())


def rows(directory):
    records = [read(path) for path in sorted(Path(directory).glob("*.json"))]
    if not records:
        raise ValueError("no locked trials found")
    return records


def main():
    parser = argparse.ArgumentParser(prog="trace2fold")
    subs = parser.add_subparsers(dest="cmd", required=True)
    p = subs.add_parser("list")
    p.add_argument("--dataset-root", type=Path, help="Root of a separate frozen dataset")
    p = subs.add_parser("plan")
    p.add_argument("run", type=Path)
    p.add_argument("cases", nargs="*")
    p.add_argument(
        "--cases",
        dest="cases_opt",
        help="comma-separated list of frozen case IDs to include in the plan",
    )
    p.add_argument(
        "--suite",
        choices=["generated", "calibration"],
        help="select all frozen cases in a suite",
    )
    p.add_argument("--model", default=MODEL)
    p.add_argument("--endpoint", help="trusted loopback chat-completions endpoint")
    p.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS
    )
    p.add_argument(
        "--prompt-version",
        choices=PROMPT_VERSIONS,
        default=PROMPT_VERSION,
        help="Prompt format version (v4.3 for new runs; v4.2 matches Stage 1).",
    )
    p = subs.add_parser("run")
    p.add_argument("run", type=Path)
    p = subs.add_parser("audit")
    p.add_argument("run", type=Path)
    for name in ("prompt", "play"):
        p = subs.add_parser(name)
        p.add_argument("case")
        p.add_argument(
            "--condition",
            choices=CONDITIONS,
            default="geometry",
        )
        p.add_argument("--first", type=Path, help="locked first submission for review")
        if name == "play":
            p.add_argument("--run", type=Path, required=True)
        else:
            p.add_argument(
                "--prompt-version", choices=PROMPT_VERSIONS, default=PROMPT_VERSION
            )
            p.add_argument("--dataset-root", type=Path, help="Root of a separate frozen dataset")
    p = subs.add_parser("controls")
    p.add_argument("out", type=Path)
    p.add_argument("cases", nargs="*")
    p = subs.add_parser("fold-inputs")
    p.add_argument("records", type=Path)
    p.add_argument("out", type=Path)
    p = subs.add_parser("score")
    p.add_argument("records", type=Path)
    p.add_argument("folds", type=Path)
    p.add_argument("out", type=Path)
    p = subs.add_parser("combine")
    p.add_argument("esm_report", type=Path)
    p.add_argument("boltz_report", type=Path)
    p.add_argument("out", type=Path)
    p = subs.add_parser("view")
    p.add_argument("folds", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("trials", type=Path, nargs="+")

    p = subs.add_parser(
        "grade-v4",
        help="Parse submissions with length-aware-v2 and grade with spacefill-v4 and Twin Delta",
    )
    p.add_argument(
        "--cases-root",
        type=Path,
        default=None,
        help="Dataset root or data/ directory (defaults to bundled starter data)",
    )
    p.add_argument(
        "--submissions",
        type=Path,
        required=True,
        help="Locked trial directory, JSON file, or JSONL file of submissions",
    )
    p.add_argument(
        "--folds",
        type=Path,
        required=True,
        help="Directory containing <predictor>/<seq_id>.pdb (or .json) fold predictions",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON path for the spacefill-v4 summary",
    )
    p.add_argument(
        "--predictors",
        nargs="+",
        default=None,
        help="Optional explicit predictor directory names under --folds",
    )
    p.add_argument(
        "--condition",
        default="vectors",
        help="Default condition for simple submissions that omit 'condition'",
    )
    p.add_argument(
        "--bare-only",
        action="store_true",
        help="Require bare sequences with no surrounding text",
    )
    p.add_argument(
        "--twin-map",
        type=Path,
        default=None,
        help="Optional JSON mapping of case_id -> twin_case_id (for example, examples/starter_pilot_pairs.json)",
    )
    p.add_argument(
        "--allow-partial-judges",
        action="store_true",
        help="Allow exploratory single-judge grading instead of requiring both esmfold2 and boltz2",
    )

    p = subs.add_parser(
        "verify-benchmarks",
        help="Verify benchmark JSON schema, bounds consistency, twin symmetry, and zero holdout leaks",
    )
    p.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=Path("benchmarks"),
        help="Directory containing public benchmark JSON artifacts",
    )
    p.add_argument(
        "--strict-suite",
        action="store_true",
        help="Require all 5 canonical benchmark JSON archives and their expected target/cell counts",
    )

    args = parser.parse_args()
    if args.cmd == "verify-benchmarks":
        strict_suite = args.strict_suite or args.benchmarks_dir == Path("benchmarks")
        report = verify_benchmarks(
            args.benchmarks_dir, require_complete_suite=strict_suite
        )
        print(json.dumps(report, indent=2))
        if not report["ok"]:
            sys.exit(1)
        return
    if args.cmd == "grade-v4":
        cases = load_cases_flexible(args.cases_root)
        submissions = load_submissions(
            args.submissions, default_condition=args.condition
        )
        twin_map = load_twin_map(args.twin_map) if args.twin_map is not None else None
        report = grade_v4(
            cases,
            submissions,
            args.folds,
            predictors=args.predictors,
            default_condition=args.condition,
            bare_only=args.bare_only,
            allow_partial_judges=args.allow_partial_judges,
            twin_map=twin_map,
        )
        dump(args.out, report)
        return

    dataset_root = getattr(args, "dataset_root", None)
    cases = load_cases(dataset_root) if dataset_root is not None else load_cases()
    if args.cmd == "list":
        for case in cases.values():
            d = case.info
            print(
                f"{case.id:28s} {d['tier']:10s} {d['family']:15s} {'+'.join(map(str, case.lengths))}"
            )
    elif args.cmd == "plan":
        if args.conditions is None:
            args.conditions = (
                ["geometry"]
                if args.prompt_version in ("v3", *V4_VERSIONS)
                else list(DEFAULT_CONDITIONS)
            )
        if args.cases_opt:
            opt_cases = [c.strip() for c in args.cases_opt.split(",") if c.strip()]
            args.cases = list(args.cases) + opt_cases
        if args.suite:
            if args.cases:
                parser.error("select a suite or name individual cases")
            args.cases = [
                case.id
                for case in cases.values()
                if bool(case.info.get("generator")) == (args.suite == "generated")
            ]
        dump(
            args.run / "design.json",
            panel(
                cases,
                args.cases,
                args.conditions,
                args.model,
                args.endpoint,
                args.prompt_version,
                answer_parser=ANSWER_PARSER if args.prompt_version != "v1" else None,
            ),
        )
    elif args.cmd == "audit":
        dump(args.run / "audit.json", audit(cases, args.run))
    elif args.cmd == "run":
        run_all(cases, args.run)
        dump(args.run / "audit.json", audit(cases, args.run))
    elif args.cmd == "play":
        if args.first and (
            args.condition != "review"
            or args.first.resolve()
            != (args.run / "records" / (args.case + "-geometry.json")).resolve()
        ):
            parser.error("--first must be the original locked submission for this run")
        play(cases, args.run, args.case, args.condition)
    elif args.cmd == "prompt":
        case = cases[args.case]
        if args.prompt_version == "v3" and args.condition not in V3_CONDITIONS:
            parser.error("v3 supports only geometry and length conditions")
        if args.prompt_version in V4_VERSIONS and args.condition not in v4_conditions(args.prompt_version):
            parser.error(f"{args.prompt_version} supports only {', '.join(v4_conditions(args.prompt_version))} conditions")
        if args.condition == "review":
            if not args.first:
                parser.error("previewing a review prompt requires --first")
            first = read(args.first)
            if first["case"] != case.id or first["condition"] != "geometry":
                parser.error("review requires the first answer for the same case")
            print(revision_prompt(case, first["raw"], args.prompt_version))
        else:
            if args.first:
                parser.error("--first applies only to review")
            task = prompt(case, input_condition(args.condition), args.prompt_version)
            print(task)
    elif args.cmd == "controls":
        for case_id in args.cases or cases:
            case = cases[case_id]
            for condition, sequence in controls(case).items():
                lock(
                    case,
                    args.out / (case_id + "-" + condition + ".json"),
                    condition,
                    sequence,
                    {"model": "control"},
                )
    elif args.cmd == "fold-inputs":
        entries = rows(args.records)
        if any(r["protocol"]["model"] != "control" for r in entries):
            audit(cases, args.records.parent)
        dump(args.out, fold_inputs(entries))
    elif args.cmd == "score":
        entries = rows(args.records)
        design = args.records.parent / "design.json"
        if design.is_file() and read(design).get("prompt_version") in V4_VERSIONS:
            parser.error("Use 'trace2fold grade-v4' for a frozen v4 study")
        if any(r["protocol"]["model"] != "control" for r in entries):
            audit(cases, args.records.parent)
        result = score(cases, entries, args.folds)
        dump(
            args.out,
            {
                "judge": read(args.folds / "manifest.json")["protocol"],
                "rows": result,
                "summary": summarize(result),
            },
        )
    elif args.cmd == "combine":
        result = consensus(read(args.esm_report), read(args.boltz_report))
        dump(
            args.out,
            {
                "rows": result,
                "summary": summarize(result),
                "review": review_comparison(result),
                "geometry_control": condition_comparison(result),
                "representation_comparisons": representation_comparisons(result),
                "by_tier": strata(result, "tier"),
                "by_generator": strata(result, "generator"),
            },
        )
    elif args.cmd == "view":
        records = [read(p) for p in args.trials]
        export(cases[records[0]["case"]], records, args.folds, args.out)


if __name__ == "__main__":
    main()
