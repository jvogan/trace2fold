"""Public v4 evaluation engine (spacefill-v4) and benchmark integrity verifier."""

from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re

import numpy as np

from .cases import (
    AA,
    ROOT,
    THREE,
    condition_case,
    digest,
    load_cases,
    read_ca,
)
from .metrics import SPACEFILL_PROFILE, measure
from .report import paired_statistics
from .trials import ANSWER_PARSER, answer_sequence, decoys

SCHEMA_VERSION = "trace2fold-v4-eval-v1"
TIGHT_RMSD_THRESHOLD = 2.0
PRIMARY_PREDICTORS = ("esmfold2", "boltz2")

HOLDOUT_PREFIXES = ("release-", "stage2-", "genie-")
HOLDOUT_FORBIDDEN_KEYS = frozenset(
    {"xyz", "target_xyz", "aligned", "backbone", "ca_coords", "ca", "coordinates", "sequence"}
)
PUBLIC_3D_ALLOWED_FILENAMES = frozenset({"public-development-3d.json"})
CANONICAL_BENCHMARK_FILES = (
    "stage-one-scores.json",
    "stage-two-scores.json",
    "length-scaling-24-shapes.json",
    "variations-and-simplefold.json",
    "public-development-3d.json",
)

# Generic structural hygiene patterns (disclose no private names).
GENERIC_PRIVACY_PATTERNS = (
    ("macos_users_path", re.compile(r"/[U]sers/")),
    ("macos_volumes_path", re.compile(r"/[V]olumes/")),
    ("windows_users_path", re.compile(r"[C]:\\[U]sers\\", re.IGNORECASE)),
    ("holdout_zip", re.compile(r"[t]argets\.zip", re.IGNORECASE)),
    ("oauth_mod", re.compile(r"[o]auth_player", re.IGNORECASE)),
    ("secret_key_prefix", re.compile(r"[s]k-[A-Za-z0-9]{20,}")),
)

GUARD_SALT = b"trace2fold-public-guard-v1:"
TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{3,}")
# One-way salted SHA-256 digests of forbidden internal tokens.
FORBIDDEN_TOKEN_HASHES = frozenset(
    {
        "1928a10235fffa725e48760a0aa7043a14ec8e1c7ba032c7bb94694a3405f0a9",
        "391edc47a5fea43ff6334b94805bdc5c135ab3f734597faab513d80dfe983559",
        "417d8efccad2d7b0538c3c4a6d8fa89d4741d5765dc71ff75acab6a7f0b9272c",
        "58d2beefdc41978faaa65bc014fac547a6a577fdf8c426c9c42cc2eb411a0f80",
        "6665e4ba9399e68ad6043b200f0661935ce7276df5a3954808cf83b019afad53",
        "7a72583b29764618c5851e0cac3749503ef2106bd54469d9105d03420041e31d",
        "7f6a9da97e5fddd65e5ac9b27557edac1b099f3bb31c1377161dbe0b02fafd05",
        "8edcae9a458f7f802d024c62efd745994ea2bcfeb0c971216dc49eb97e7717e5",
        "98c9173757f199aab912037728b35a36b3a9d5856772515a6ccb3e1d19867fab",
        "9effd53ea8364767fa4987decc21e3d8e3246d45722d9de8f1209807c9b60577",
        "ab3c75445b00066fff675356f4df8e35bc7f73ee177dd49a9aa25c23622ed383",
        "b5f50ac56153195e8c88628172989e61061488fc1c81ff3ca66b22daf7ea78d7",
        "b7292ef0341a06781bd9b992513f2abffca5a8b5a20cd4f3d154ed5c2920a886",
        "b843e1fbdf91dc73646e8d95662f2bf4e79f4cfc1341fa4b61f23b1d75571691",
        "c9be5f976814429d2feb18124cc16316b47f367002b4c27456ee922bf8706b76",
        "d1a5d119a3130691bd91b9b4c85cf22294706d83692b32dbfd5918034d513349",
        "e3372daee51914892d7fcfefb244b1f476bb4debde206585c079836b64481553",
        "eb32535a49f25852719eb2f91d081a80a4bf98393845d09d0539acb6ce8c2835",
    }
)

LONG_AA_SEQ_RE = re.compile(r"^[A-Z]{40,}$")
STRICT_AA_SET = frozenset(AA)


def _iter_candidate_tokens(text):
    """Yield lowercase tokens and adjacent hyphen/underscore 1- and 2-grams."""
    for match in TOKEN_RE.finditer(text):
        tok = match.group(0).lower()
        yield tok
        if "-" in tok:
            parts = [p for p in tok.split("-") if p]
            for part in parts:
                if len(part) >= 3:
                    yield part
            for i in range(len(parts) - 1):
                joined = f"{parts[i]}-{parts[i + 1]}"
                if len(joined) >= 3:
                    yield joined
        if "_" in tok:
            parts = [p for p in tok.split("_") if p]
            for part in parts:
                if len(part) >= 3:
                    yield part
            for i in range(len(parts) - 1):
                joined = f"{parts[i]}_{parts[i + 1]}"
                if len(joined) >= 3:
                    yield joined


def scan_privacy_text(text):
    """Return a list of privacy violation labels found in text."""
    hits = []
    for label, pattern in GENERIC_PRIVACY_PATTERNS:
        if pattern.search(text):
            hits.append(label)
    seen_hashes = set()
    for tok in _iter_candidate_tokens(text):
        h = sha256(GUARD_SALT + tok.encode("utf-8")).hexdigest()
        if h in FORBIDDEN_TOKEN_HASHES and h not in seen_hashes:
            seen_hashes.add(h)
            hits.append(f"token_hash:{h[:12]}")
    return hits


def read_atoms(pdb, sequence):
    """Read the first model in sequence order, retaining each resolved heavy atom once."""
    residues = {}
    atom_keys = set()
    positions, elements, points, trace = [], [], [], []
    text = pdb.decode("ascii") if isinstance(pdb, (bytes, bytearray)) else str(pdb)
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM  ") or line[16] not in " A1":
            continue
        code = THREE.get(line[17:20].strip())
        key = line[21:27]
        i = residues.setdefault(key, len(residues) + 1)
        if i > len(sequence) or code != sequence[i - 1]:
            raise ValueError("Viewer atom and submitted sequence disagree")
        atom = line[12:16].strip()
        element = line[76:78].strip().upper() or atom[0]
        if element in {"H", "D"} or (key, atom) in atom_keys:
            continue
        atom_keys.add((key, atom))
        point = [float(line[k : k + 8]) for k in (30, 38, 46)]
        if atom == "CA":
            trace.append((i, point))
        positions.append(i)
        elements.append(element)
        points.append(point)
    if [i for i, _ in trace] != list(range(1, len(sequence) + 1)):
        raise ValueError("Viewer PDB does not have one ordered Cα per residue")
    xyz = np.asarray(points, dtype=float)
    if not np.isfinite(xyz).all():
        raise ValueError("Viewer PDB contains nonfinite coordinates")
    return np.asarray([p for _, p in trace], dtype=float), positions, elements, xyz


def load_cases_flexible(cases_root=None):
    """Load cases from a repo root, a dataset root containing data/, or data/ itself."""
    if cases_root is None:
        return load_cases(ROOT)
    root = Path(cases_root).resolve()
    if (root / "data").is_dir() and list((root / "data").glob("*/index.json")):
        return load_cases(root)
    if list(root.glob("*/index.json")):
        return load_cases(root.parent)
    raise FileNotFoundError(
        f"No benchmark data/*/index.json found under {root} or {root / 'data'}"
    )


def load_twin_map(path):
    """Load and validate a reciprocal JSON mapping of case_id -> twin_case_id."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Twin map JSON must be a non-empty object mapping case_id -> twin_id")
    twin_map = {}
    for case_id, twin_id in payload.items():
        if not isinstance(case_id, str) or not isinstance(twin_id, str):
            raise ValueError("Twin map keys and values must be strings")
        if case_id == twin_id:
            raise ValueError(f"Twin map cannot pair case {case_id!r} with itself")
        if payload.get(twin_id) != case_id:
            raise ValueError(
                f"Twin map must be reciprocal: {case_id!r} -> {twin_id!r}, "
                f"but reverse is {payload.get(twin_id)!r}"
            )
        twin_map[case_id] = twin_id
    return twin_map


def _resolve_mates_and_pairs(cases, twin_map=None):
    """Resolve decoy mates and pair IDs, applying optional reciprocal twin_map overrides."""
    mates = dict(decoys(cases))
    pair_overrides = {}
    if twin_map is None:
        return mates, pair_overrides

    if isinstance(twin_map, (str, Path)):
        twin_map = load_twin_map(twin_map)
    elif not isinstance(twin_map, dict):
        raise TypeError("twin_map must be a mapping or path to a JSON file")

    for case_id, twin_id in twin_map.items():
        if case_id == twin_id:
            raise ValueError(f"Twin map cannot pair case {case_id!r} with itself")
        if twin_map.get(twin_id) != case_id:
            raise ValueError(
                f"Twin map must be reciprocal: {case_id!r} -> {twin_id!r}, "
                f"but reverse is {twin_map.get(twin_id)!r}"
            )
        if case_id not in cases:
            raise KeyError(f"Twin map references unknown case ID: {case_id}")
        if twin_id not in cases:
            raise KeyError(f"Twin map references unknown twin case ID: {twin_id}")
        case_a = cases[case_id]
        case_b = cases[twin_id]
        if case_a.lengths != case_b.lengths or case_a.template != case_b.template:
            raise ValueError(
                f"Twin map pair ({case_id}, {twin_id}) differs in length or fixed template"
            )

    # Clear stale default mates for any case overridden by twin_map.
    for case_id in twin_map:
        old_mate = mates.pop(case_id, None)
        if old_mate is not None and old_mate.id not in twin_map:
            mates.pop(old_mate.id, None)

    for case_id, twin_id in twin_map.items():
        mates[case_id] = cases[twin_id]
        a, b = sorted((case_id, twin_id))
        pair_overrides[case_id] = f"twin-map:{a}:{b}"

    return mates, pair_overrides


def _extract_submission_list(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("rows", "submissions", "records", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
        if "case" in payload:
            return [payload]
    raise ValueError("Unsupported submission JSON format")


def load_submissions(path, *, default_condition="vectors"):
    """Load locked trace2fold trial records or simple JSON/JSONL submission mappings."""
    source = Path(path)
    raw_entries = []
    if source.is_dir():
        files = sorted(
            p for p in source.iterdir() if p.is_file() and p.suffix in {".json", ".jsonl"}
        )
        if not files:
            raise ValueError(f"No .json or .jsonl submission files found in {source}")
        for fp in files:
            raw_entries.extend(load_submissions(fp, default_condition=default_condition))
        return raw_entries
    if source.suffix == ".jsonl":
        for line_no, line in enumerate(source.read_text().splitlines(), 1):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_no} in {source} is not a JSON object")
            raw_entries.append(item)
    else:
        raw_entries = _extract_submission_list(json.loads(source.read_text()))

    normalized = []
    for entry in raw_entries:
        if not isinstance(entry, dict) or "case" not in entry:
            raise ValueError("Each submission must be a JSON object with a 'case' field")
        protocol = dict(entry.get("protocol") or {})
        model = entry.get("model") or protocol.get("model")
        if not model:
            raise ValueError(f"Submission for case {entry['case']!r} is missing 'model'")
        protocol.setdefault("model", model)
        raw = entry.get("raw")
        if raw is None and "sequence" in entry and entry["sequence"] is not None:
            raw = str(entry["sequence"])
        if raw is None:
            raw = ""
        record = {
            "case": entry["case"],
            "model": model,
            "condition": entry.get("condition", default_condition),
            "raw": str(raw),
            "protocol": protocol,
        }
        for optional in ("trial_id", "target_sha256", "valid", "sequence", "answer_parser"):
            if optional in entry:
                record[optional] = entry[optional]
        normalized.append(record)
    return normalized


def discover_predictors(folds_dir, predictors=None):
    """Discover predictor directories under folds_dir (or a single flat fold directory)."""
    root = Path(folds_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Folds directory does not exist: {root}")
    if predictors:
        found = {}
        for name in predictors:
            sub = root / name
            if sub.is_dir():
                found[name] = sub
            elif len(predictors) == 1 and any(
                p.suffix in {".pdb", ".json"} and p.name != "manifest.json"
                for p in root.iterdir()
                if p.is_file()
            ):
                found[name] = root
            else:
                raise FileNotFoundError(f"Predictor directory not found: {sub}")
        return found

    subdirs = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        has_folds = any(
            p.suffix in {".pdb", ".json"} and p.name != "manifest.json"
            for p in entry.iterdir()
            if p.is_file()
        ) or (entry / "structures").is_dir()
        if has_folds:
            subdirs[entry.name] = entry
    if subdirs:
        return subdirs
    has_flat = any(
        p.suffix in {".pdb", ".json"} and p.name != "manifest.json"
        for p in root.iterdir()
        if p.is_file()
    )
    if has_flat:
        return {root.name: root}
    raise ValueError(f"No predictor fold files (.pdb or .json) found under {root}")


def _read_pdb_ca_for_sequence(pdb_path, sequence):
    chains = list(
        dict.fromkeys(
            line[21]
            for line in pdb_path.read_text().splitlines()
            if line.startswith("ATOM") and line[12:16].strip() == "CA"
        )
    )
    if not chains:
        raise ValueError(f"No C-alpha ATOM records in {pdb_path}")
    xyz, seqs = read_ca(pdb_path, sorted(chains))
    if ":".join(seqs) != sequence:
        raise ValueError(f"Fold sequence mismatch in {pdb_path}")
    return xyz


def read_predictor_fold(pred_dir, sequence, case):
    """Read predicted Cα coordinates and optional sidecar metadata for one sequence."""
    pred_dir = Path(pred_dir)
    seq_sha = digest(sequence)
    seq_id = seq_sha[:32]
    search_bases = [pred_dir]
    if (pred_dir / "structures").is_dir():
        search_bases.append(pred_dir / "structures")

    pdb_path = None
    json_path = None
    for base in search_bases:
        for stem in (seq_id, seq_sha):
            cand_pdb = base / f"{stem}.pdb"
            cand_json = base / f"{stem}.json"
            if pdb_path is None and cand_pdb.is_file():
                pdb_path = cand_pdb
            if json_path is None and cand_json.is_file():
                json_path = cand_json

    if pdb_path is None and json_path is None:
        return None, {}

    meta = {}
    sidecar = json.loads(json_path.read_text()) if json_path is not None else None
    if pdb_path is not None:
        xyz = _read_pdb_ca_for_sequence(pdb_path, sequence)
        if sidecar is not None and "ca" in sidecar:
            if not np.allclose(np.asarray(sidecar["ca"], dtype=float), xyz, atol=1e-3):
                raise ValueError(f"Sidecar C-alpha coordinates disagree with {pdb_path}")
    else:
        if "ca" not in sidecar:
            raise ValueError(f"Fold JSON {json_path} lacks 'ca' coordinates")
        if "sequence" in sidecar and sidecar["sequence"] != sequence:
            raise ValueError(f"Fold JSON {json_path} sequence mismatch")
        xyz = np.asarray(sidecar["ca"], dtype=float)

    if sidecar is not None:
        if "plddt" in sidecar and sidecar["plddt"] is not None:
            plddt = np.asarray(sidecar["plddt"], dtype=float)
            if plddt.shape == case.mutable.shape and np.isfinite(plddt).all():
                if not np.all((0 <= plddt) & (plddt <= 100)):
                    raise ValueError(f"pLDDT out of [0, 100] in {json_path}")
                meta["mean_plddt"] = float(np.mean(plddt[case.mutable]))
        if "ptm" in sidecar and sidecar["ptm"] is not None:
            ptm = float(sidecar["ptm"])
            if not (0.0 <= ptm <= 1.0):
                raise ValueError(f"pTM out of [0, 1] in {json_path}")
            meta["ptm"] = ptm
    return xyz, meta


def _score_predictor(pred_path, sequence, eff_case, case_id, condition, mates):
    xyz, meta = read_predictor_fold(pred_path, sequence, eff_case)
    if xyz is None:
        return None
    terms, _ = measure(xyz, eff_case, profile=SPACEFILL_PROFILE)
    judge_rec = {
        **terms,
        "within_2a": bool(terms["rmsd"] <= TIGHT_RMSD_THRESHOLD),
        **meta,
    }
    if case_id in mates and condition != "masked":
        decoy_terms, _ = measure(xyz, mates[case_id], profile=SPACEFILL_PROFILE)
        delta = float(terms["score"] - decoy_terms["score"])
        judge_rec["decoy_score"] = float(decoy_terms["score"])
        judge_rec["decoy_rmsd"] = float(decoy_terms["rmsd"])
        judge_rec["twin_delta"] = delta
        judge_rec["margin"] = delta
    return judge_rec


def grade_v4(
    cases,
    submissions,
    folds_dir,
    *,
    predictors=None,
    default_condition="vectors",
    bare_only=False,
    allow_partial_judges=False,
    twin_map=None,
):
    """Parse submissions with length-aware-v2 and grade with spacefill-v4 + Twin Δ."""
    pred_dirs = discover_predictors(folds_dir, predictors=predictors)
    primary_dirs = {
        name: pred_dirs[name] for name in PRIMARY_PREDICTORS if name in pred_dirs
    }
    exploratory_dirs = {
        name: path for name, path in pred_dirs.items() if name not in PRIMARY_PREDICTORS
    }

    if not allow_partial_judges:
        missing_primary = [p for p in PRIMARY_PREDICTORS if p not in primary_dirs]
        if missing_primary:
            raise ValueError(
                f"Official spacefill-v4 grading requires both primary predictors "
                f"{list(PRIMARY_PREDICTORS)}; missing {missing_primary}. "
                f"Pass allow_partial_judges=True (--allow-partial-judges) for "
                f"exploratory single-judge runs."
            )
    elif not primary_dirs:
        raise ValueError(
            f"No primary predictor directory ({', '.join(PRIMARY_PREDICTORS)}) found under {folds_dir}"
        )

    official_protocol = not allow_partial_judges
    mates, pair_overrides = _resolve_mates_and_pairs(cases, twin_map=twin_map)
    seen_slots = set()
    rows = []

    for entry in submissions:
        case_id = entry["case"]
        if case_id not in cases:
            raise KeyError(f"Unknown case ID: {case_id}")
        base_case = cases[case_id]
        condition = entry.get("condition", default_condition)
        slot = (entry["model"], condition, case_id)
        if slot in seen_slots:
            raise ValueError(
                f"Duplicate submission slot for (model={entry['model']!r}, "
                f"condition={condition!r}, case={case_id!r})"
            )
        seen_slots.add(slot)

        eff_case = condition_case(base_case, condition)
        locked_hash = digest(eff_case.xyz.astype("<f8").tobytes().hex() + eff_case.template)
        if "target_sha256" in entry and entry["target_sha256"] != locked_hash:
            raise ValueError(f"target_sha256 mismatch for case {case_id}")

        protocol = dict(entry.get("protocol") or {"model": entry["model"]})
        raw = entry.get("raw", "")
        sequence = answer_sequence(
            eff_case, raw, protocol, bare_only=bare_only, parser=ANSWER_PARSER
        )
        seq_sha = digest(sequence) if sequence is not None else None
        seq_id = seq_sha[:32] if seq_sha is not None else None

        row = {
            "case": case_id,
            "model": entry["model"],
            "condition": condition,
            "pair": pair_overrides.get(case_id, base_case.info.get("pair")),
            "twin": mates[case_id].id if case_id in mates else None,
            "family": base_case.info.get("family"),
            "generator": base_case.info.get("generator", "known"),
            "length": sum(eff_case.lengths),
            "valid": sequence is not None,
            "sequence_id": seq_id,
            "sequence_sha256": seq_sha,
            "status": "invalid",
            "score": 0.0,
            "rmsd": None,
            "aligned_coverage": None,
            "ca_lddt": None,
            "contact_f1": None,
            "decoy_score": None,
            "twin_delta": None,
            "margin": None,
            "both_within_2a": False,
            "all_within_2a": False,
            "judges": {},
            "exploratory_predictors": {},
        }
        if "trial_id" in entry:
            row["trial_id"] = entry["trial_id"]

        if sequence is not None:
            judges = {}
            for pred_name, pred_path in primary_dirs.items():
                judge_rec = _score_predictor(
                    pred_path, sequence, eff_case, case_id, condition, mates
                )
                if judge_rec is not None:
                    judges[pred_name] = judge_rec

            exploratory = {}
            for pred_name, pred_path in exploratory_dirs.items():
                exp_rec = _score_predictor(
                    pred_path, sequence, eff_case, case_id, condition, mates
                )
                if exp_rec is not None:
                    exploratory[pred_name] = exp_rec

            row["judges"] = judges
            row["exploratory_predictors"] = exploratory

            missing_row_primary = [p for p in PRIMARY_PREDICTORS if p not in judges]
            is_complete = (
                (not missing_row_primary) if official_protocol else bool(judges)
            )
            if not is_complete:
                row["status"] = "ungraded"
                row["score"] = None
                row["both_within_2a"] = None
                row["all_within_2a"] = None
                if missing_row_primary:
                    row["missing_primary_predictors"] = missing_row_primary
            else:
                vals = [judges[p] for p in PRIMARY_PREDICTORS if p in judges]
                row["status"] = "graded"
                row["score"] = float(np.mean([j["score"] for j in vals]))
                row["rmsd"] = float(np.mean([j["rmsd"] for j in vals]))
                row["aligned_coverage"] = float(
                    np.mean([j["aligned_coverage"] for j in vals])
                )
                lddts = [j["ca_lddt"] for j in vals if j["ca_lddt"] is not None]
                row["ca_lddt"] = float(np.mean(lddts)) if lddts else None
                f1s = [j["contact_f1"] for j in vals if j["contact_f1"] is not None]
                row["contact_f1"] = float(np.mean(f1s)) if f1s else None
                if all("decoy_score" in j for j in vals):
                    row["decoy_score"] = float(np.mean([j["decoy_score"] for j in vals]))
                    delta = float(np.mean([j["twin_delta"] for j in vals]))
                    row["twin_delta"] = delta
                    row["margin"] = delta
                row["all_within_2a"] = all(j["within_2a"] for j in vals)
                row["both_within_2a"] = (
                    all(p in judges for p in PRIMARY_PREDICTORS)
                    and all(judges[p]["within_2a"] for p in PRIMARY_PREDICTORS)
                )
        rows.append(row)

    return {
        "schema": SCHEMA_VERSION,
        "scoring_profile": SPACEFILL_PROFILE,
        "answer_parser": ANSWER_PARSER,
        "official_protocol": official_protocol,
        "primary_predictors": list(primary_dirs),
        "exploratory_predictors": list(exploratory_dirs),
        "predictors": list(primary_dirs),
        "rows": rows,
        "summaries": summarize_v4(rows),
    }


def summarize_v4(rows):
    """Aggregate spacefill-v4 rows by (model, condition)."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["condition"])].append(row)

    summaries = []
    for (model, condition), group in sorted(groups.items()):
        graded = [r for r in group if r["status"] == "graded"]
        invalid = [r for r in group if r["status"] == "invalid"]
        ungraded = [r for r in group if r["status"] == "ungraded"]
        scored = [r for r in group if r["score"] is not None]
        deltas = [r["twin_delta"] for r in graded if r["twin_delta"] is not None]
        rmsds = [r["rmsd"] for r in graded if r["rmsd"] is not None]

        twin_slots = [
            r for r in group if r.get("twin") is not None and r["condition"] != "masked"
        ]
        ungraded_twin = [r for r in twin_slots if r["status"] == "ungraded"]
        assigned_deltas = [
            (r["twin_delta"] if r["status"] == "graded" else 0.0)
            for r in twin_slots
            if r["status"] == "invalid"
            or (r["status"] == "graded" and r["twin_delta"] is not None)
        ]

        pred_names = sorted({p for r in graded for p in r["judges"]})
        by_predictor = {}
        for pred in pred_names:
            p_rows = [r["judges"][pred] for r in graded if pred in r["judges"]]
            by_predictor[pred] = {
                "graded": len(p_rows),
                "mean_score": float(np.mean([j["score"] for j in p_rows])),
                "mean_rmsd": float(np.mean([j["rmsd"] for j in p_rows])),
                "within_2a_hits": sum(1 for j in p_rows if j["within_2a"]),
            }

        paired_rows = [
            {"pair": r["pair"], "margin": r["margin"]}
            for r in graded
            if r.get("pair") and r.get("margin") is not None
        ]
        paired = paired_statistics(paired_rows)
        tight_hits = sum(1 for r in graded if r["both_within_2a"])
        graded_twin_mean = float(np.mean(deltas)) if deltas else None
        assigned_twin_mean = (
            None
            if (ungraded_twin or not assigned_deltas)
            else float(np.mean(assigned_deltas))
        )
        summaries.append(
            {
                "model": model,
                "condition": condition,
                "attempts": len(group),
                "graded": len(graded),
                "invalid": len(invalid),
                "ungraded": len(ungraded),
                "mean_score": None
                if ungraded
                else (float(np.mean([r["score"] for r in scored])) if scored else None),
                "mean_graded_score": float(np.mean([r["score"] for r in graded]))
                if graded
                else None,
                "mean_rmsd": float(np.mean(rmsds)) if rmsds else None,
                "mean_twin_delta": graded_twin_mean,
                "twin_margin_graded_mean": graded_twin_mean,
                "twin_margin_graded_n": len(deltas),
                "twin_margin_assigned_mean": assigned_twin_mean,
                "twin_margin_assigned_n": len(twin_slots),
                "one_sided_sign_p": paired["one_sided_sign_p"],
                "two_sided_sign_p": paired["two_sided_sign_p"],
                "both_within_2a_hits": tight_hits,
                "both_within_2a_rate": (tight_hits / len(group)) if group else 0.0,
                "by_predictor": by_predictor,
                "paired": paired,
            }
        )
    return summaries


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_signed_interval_key(key):
    lower = key.lower()
    return any(
        token in lower
        for token in (
            "margin",
            "delta",
            "gain",
            "diff",
            "minus",
            "bootstrap",
            "matched_comparison",
            "generator_sensitivity.cells",
            "pearson",
            "spearman",
            "ci95",
            "tie_matrix.pairs",
            "tie_matrix.matrix",
        )
    )


def _is_bound_key(key):
    lower = key.lower()
    return any(
        token in lower
        for token in ("bound", "interval", "lopo_range", "_range", "score_range", "ci")
    )


def _is_holdout_id(value):
    return isinstance(value, str) and value.startswith(HOLDOUT_PREFIXES)


def _strict_json_loads(text):
    def reject_constant(token):
        raise ValueError(f"Non-standard JSON constant not allowed: {token}")

    return json.loads(text, parse_constant=reject_constant)


def _check_bounds_and_leaks(obj, filename, path, errors, *, allow_public_3d=False, in_holdout=False):
    if isinstance(obj, dict):
        obj_id = obj.get("id") or obj.get("case") or obj.get("target") or obj.get("target_id")
        is_holdout_obj = in_holdout or _is_holdout_id(obj_id)

        # Check lo/hi or min/max dicts
        for lo_k, hi_k in (("lo", "hi"), ("low", "high"), ("min", "max")):
            if lo_k in obj and hi_k in obj and _is_number(obj[lo_k]) and _is_number(obj[hi_k]):
                lo, hi = float(obj[lo_k]), float(obj[hi_k])
                if not (math.isfinite(lo) and math.isfinite(hi)):
                    errors.append(f"{filename}:{path}: non-finite interval [{lo}, {hi}]")
                elif lo > hi + 1e-9:
                    errors.append(f"{filename}:{path}: inverted interval {lo_k}={lo} > {hi_k}={hi}")
                elif not _is_signed_interval_key(path) and (
                    "score" in path.lower() or "lopo" in path.lower() or "bound" in path.lower()
                ):
                    if not (0.0 <= lo <= hi <= 100.0):
                        errors.append(
                            f"{filename}:{path}: score interval [{lo}, {hi}] outside [0, 100]"
                        )

        for key, value in obj.items():
            sub_path = f"{path}.{key}" if path else str(key)
            key_holdout = is_holdout_obj or _is_holdout_id(key)

            # Holdout coordinate / sequence leak check
            if key in HOLDOUT_FORBIDDEN_KEYS:
                if not allow_public_3d:
                    if key != "sequence" or key_holdout or (
                        isinstance(value, str) and len(value) >= 40 and set(value) <= STRICT_AA_SET
                    ):
                        errors.append(
                            f"{filename}:{sub_path}: forbidden coordinate/sequence key '{key}' outside public-development-3d.json"
                        )
                elif key_holdout:
                    errors.append(
                        f"{filename}:{sub_path}: holdout coordinate/sequence key '{key}' inside {filename}"
                    )

            # Interval 2-tuple / 2-list check
            if (
                _is_bound_key(key)
                and isinstance(value, list)
                and len(value) == 2
                and all(_is_number(x) for x in value)
            ):
                lo, hi = float(value[0]), float(value[1])
                if not (math.isfinite(lo) and math.isfinite(hi)):
                    errors.append(f"{filename}:{sub_path}: non-finite bounds {value}")
                elif lo > hi + 1e-9:
                    errors.append(f"{filename}:{sub_path}: inverted bounds [{lo}, {hi}]")
                elif _is_signed_interval_key(sub_path):
                    if not (-100.0 - 1e-9 <= lo <= hi <= 100.0 + 1e-9):
                        errors.append(
                            f"{filename}:{sub_path}: signed interval [{lo}, {hi}] outside [-100, 100]"
                        )
                else:
                    if not (0.0 - 1e-9 <= lo <= hi <= 100.0 + 1e-9):
                        errors.append(
                            f"{filename}:{sub_path}: bounds [{lo}, {hi}] outside [0, 100]"
                        )

            # Scalar metric range checks
            if _is_number(value):
                val = float(value)
                if not math.isfinite(val):
                    errors.append(f"{filename}:{sub_path}: non-finite numeric value")
                elif key in {"score", "mean_score", "seed0_score", "decoy_score", "mean_plddt"}:
                    if not (0.0 <= val <= 100.0):
                        errors.append(f"{filename}:{sub_path}: {key}={val} outside [0, 100]")
                elif key in {"aligned_coverage", "ca_lddt", "contact_f1", "ptm"}:
                    if not (0.0 <= val <= 1.0 + 1e-9):
                        errors.append(f"{filename}:{sub_path}: {key}={val} outside [0, 1]")
                elif key in {"rmsd", "decoy_rmsd"} and val < 0.0:
                    errors.append(f"{filename}:{sub_path}: negative {key}={val}")

            _check_bounds_and_leaks(
                value,
                filename,
                sub_path,
                errors,
                allow_public_3d=allow_public_3d,
                in_holdout=key_holdout,
            )

    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            _check_bounds_and_leaks(
                item,
                filename,
                f"{path}[{idx}]",
                errors,
                allow_public_3d=allow_public_3d,
                in_holdout=in_holdout,
            )

    elif isinstance(obj, str):
        if not allow_public_3d and LONG_AA_SEQ_RE.match(obj) and set(obj) <= STRICT_AA_SET:
            if len(set(obj)) >= 5:
                errors.append(
                    f"{filename}:{path}: potential amino-acid sequence leak (length {len(obj)})"
                )


def _extract_target_tables(obj, path=""):
    """Find target metadata collections declaring twin/pair relationships."""
    tables = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            sub_path = f"{path}.{key}" if path else str(key)
            if key in {"targets", "cases", "shapes"}:
                if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
                    by_id = {}
                    for item in value:
                        tid = item.get("id") or item.get("case") or item.get("target")
                        if tid:
                            by_id[tid] = item
                    if by_id:
                        tables.append((sub_path, by_id))
                elif isinstance(value, dict) and value and all(
                    isinstance(v, dict) for v in value.values()
                ):
                    tables.append((sub_path, value))
            tables.extend(_extract_target_tables(value, sub_path))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            tables.extend(_extract_target_tables(item, f"{path}[{idx}]"))
    return tables


def _check_twin_symmetry(obj, filename, errors):
    checked_pairs = 0
    for table_path, table in _extract_target_tables(obj):
        by_pair = defaultdict(list)
        for tid, meta in table.items():
            twin = (
                meta.get("twin")
                or meta.get("twin_id")
                or meta.get("mate")
                or meta.get("paired_decoy")
            )
            if twin is not None:
                if twin == tid:
                    errors.append(f"{filename}:{table_path}.{tid}: target cannot be its own twin")
                elif twin in table:
                    other = table[twin]
                    back = (
                        other.get("twin")
                        or other.get("twin_id")
                        or other.get("mate")
                        or other.get("paired_decoy")
                    )
                    if back != tid:
                        errors.append(
                            f"{filename}:{table_path}: asymmetric twin link {tid} -> {twin} (reverse={back!r})"
                        )
                    if (
                        "length" in meta
                        and "length" in other
                        and meta["length"] != other["length"]
                    ):
                        errors.append(
                            f"{filename}:{table_path}: twin length mismatch {tid} ({meta['length']}) vs {twin} ({other['length']})"
                        )
                    checked_pairs += 1
            pair_id = meta.get("pair") or meta.get("pair_id")
            if pair_id:
                by_pair[pair_id].append((tid, meta))

        for pair_id, members in by_pair.items():
            if len(members) != 2:
                errors.append(
                    f"{filename}:{table_path}: pair {pair_id!r} has {len(members)} member(s), expected 2"
                )
            else:
                (id_a, meta_a), (id_b, meta_b) = members
                if id_a == id_b:
                    errors.append(f"{filename}:{table_path}: pair {pair_id!r} repeats {id_a}")
                if (
                    "length" in meta_a
                    and "length" in meta_b
                    and meta_a["length"] != meta_b["length"]
                ):
                    errors.append(
                        f"{filename}:{table_path}: pair {pair_id!r} length mismatch ({meta_a['length']} vs {meta_b['length']})"
                    )
                checked_pairs += 1
    return checked_pairs


def _check_canonical_suite_counts(parsed_by_name, errors):
    """Verify expected canonical archive filenames and target/cell counts."""
    missing = [name for name in CANONICAL_BENCHMARK_FILES if name not in parsed_by_name]
    if missing:
        errors.append(f"Canonical benchmark suite missing required file(s): {missing}")
        return

    s1 = parsed_by_name["stage-one-scores.json"]
    s1_targets = s1.get("targets")
    s1_cells = s1.get("cells")
    if not isinstance(s1_targets, list) or len(s1_targets) != 18:
        errors.append(
            f"stage-one-scores.json: expected 18 targets, found {len(s1_targets) if isinstance(s1_targets, list) else type(s1_targets).__name__}"
        )
    if not isinstance(s1_cells, list) or len(s1_cells) != 342:
        errors.append(
            f"stage-one-scores.json: expected 342 cells, found {len(s1_cells) if isinstance(s1_cells, list) else type(s1_cells).__name__}"
        )

    s2 = parsed_by_name["stage-two-scores.json"]
    s2_targets = s2.get("targets")
    s2_cells = s2.get("cells")
    if not isinstance(s2_targets, list) or len(s2_targets) != 6:
        errors.append(
            f"stage-two-scores.json: expected 6 targets, found {len(s2_targets) if isinstance(s2_targets, list) else type(s2_targets).__name__}"
        )
    if not isinstance(s2_cells, list) or len(s2_cells) != 114:
        errors.append(
            f"stage-two-scores.json: expected 114 cells, found {len(s2_cells) if isinstance(s2_cells, list) else type(s2_cells).__name__}"
        )

    scaling = parsed_by_name["length-scaling-24-shapes.json"]
    if scaling.get("total_shapes") != 24:
        errors.append(
            f"length-scaling-24-shapes.json: expected total_shapes=24, found {scaling.get('total_shapes')!r}"
        )
    mel = scaling.get("mixed_effort_ladder")
    if mel is not None:
        if not isinstance(mel, dict) or mel.get("coordinate_free") is not True:
            errors.append("length-scaling-24-shapes.json: mixed_effort_ladder must set coordinate_free=true")
        elif not (1 <= len(mel.get("new_arms") or []) <= 4) or not (1 <= len(mel.get("cells") or []) <= 32):
            errors.append(
                "length-scaling-24-shapes.json: mixed_effort_ladder must contain 1..4 new_arms and 1..32 evaluated cells"
            )

    variations = parsed_by_name["variations-and-simplefold.json"]
    for section in ("variations_60call", "simplefold_3b", "pdb100_novelty", "cpu_diagnostics"):
        if not isinstance(variations.get(section), dict):
            errors.append(
                f"variations-and-simplefold.json: missing required section '{section}'"
            )

    pub3d = parsed_by_name["public-development-3d.json"]
    pub_cases = pub3d.get("cases")
    if not isinstance(pub_cases, (dict, list)) or len(pub_cases) != 6:
        errors.append(
            f"public-development-3d.json: expected 6 cases, found {len(pub_cases) if isinstance(pub_cases, (dict, list)) else type(pub_cases).__name__}"
        )


def verify_benchmarks(benchmarks_dir, *, require_complete_suite=False):
    """Verify schema integrity, bounds consistency, twin symmetry, and zero holdout leaks."""
    root = Path(benchmarks_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmarks directory does not exist: {root}")
    json_files = sorted(root.glob("*.json"))
    if not json_files:
        raise ValueError(f"No .json files found in {root}")

    errors = []
    checked = []
    twin_checks = 0
    parsed_by_name = {}

    for fp in json_files:
        text = fp.read_text(encoding="utf-8")
        for label in scan_privacy_text(text):
            errors.append(f"{fp.name}: forbidden private token matched ({label})")
        try:
            payload = _strict_json_loads(text)
        except Exception as exc:
            errors.append(f"{fp.name}: invalid JSON ({exc})")
            continue
        if not isinstance(payload, dict) or not payload:
            errors.append(f"{fp.name}: top-level JSON must be a non-empty object")
            continue
        parsed_by_name[fp.name] = payload
        checked.append(fp.name)

        allow_3d = fp.name in PUBLIC_3D_ALLOWED_FILENAMES
        _check_bounds_and_leaks(payload, fp.name, "", errors, allow_public_3d=allow_3d)
        twin_checks += _check_twin_symmetry(payload, fp.name, errors)

    has_all_canonical = all(name in parsed_by_name for name in CANONICAL_BENCHMARK_FILES)
    complete_suite_checked = bool(require_complete_suite or has_all_canonical)
    if complete_suite_checked:
        _check_canonical_suite_counts(parsed_by_name, errors)

    # Optional manifest SHA-256 verification
    manifest = parsed_by_name.get("manifest.json")
    if isinstance(manifest, dict) and isinstance(manifest.get("files"), (dict, list)):
        entries = manifest["files"]
        if isinstance(entries, dict):
            items = entries.items()
        else:
            items = [
                (e.get("file") or e.get("path") or e.get("name"), e)
                for e in entries
                if isinstance(e, dict)
            ]
        for fname, info in items:
            if not fname:
                continue
            target = root / Path(fname).name
            if not target.is_file():
                errors.append(f"manifest.json: listed file missing: {fname}")
                continue
            expected_sha = info.get("sha256") if isinstance(info, dict) else None
            if expected_sha:
                actual_sha = sha256(target.read_bytes()).hexdigest()
                if actual_sha != expected_sha:
                    errors.append(
                        f"manifest.json: sha256 mismatch for {fname} ({actual_sha} != {expected_sha})"
                    )

    return {
        "ok": not errors,
        "benchmarks_dir": str(root),
        "files_checked": checked,
        "complete_suite_checked": complete_suite_checked,
        "twin_checks": twin_checks,
        "error_count": len(errors),
        "errors": errors,
    }
