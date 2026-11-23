"""Freeze matched, sequence-masked test backbones from completed generation receipts."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from biotite.structure import AtomArray, PSeaElement, annotate_sse
from biotite.structure.io.pdbx import CIFFile, get_structure

GENERATORS = ("genie3", "rfd4-proteina")
LENGTHS = (48, 72, 96)
FAMILIES = ("beta", "mixed", "helical")
AA3 = "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()
AA1 = "ACDEFGHIKLMNPQRSTVWY"
AA_CODES = dict(zip(AA3, AA1, strict=True))
PAIR_TARGET = 3
MPNN_SEED = 20261118
MPNN_MODEL = "v_48_020"
RELEASE_PAIR_SEED = "trace2fold-v4-release-20261119"


@dataclass
class Shape:
    generator: str
    source: str
    source_sha256: str
    sequence: str | None
    xyz: np.ndarray
    secondary_structure: str
    helix: float
    strand: float
    radius: float
    contact_coverage: float
    contacts: np.ndarray

    @property
    def length(self) -> int:
        return len(self.xyz)

    @property
    def family(self) -> str:
        if self.helix >= 0.45 and self.strand < 0.15:
            return "helical"
        if self.strand >= 0.30 and self.helix < 0.15:
            return "beta"
        if self.helix >= 0.15 and self.strand >= 0.15:
            return "mixed"
        return "other"


def read_ca(path: Path) -> tuple[np.ndarray, list[str]]:
    if path.suffix == ".pdb":
        with path.open() as stream:
            rows = [
                (line[21:27], line[17:20],
                 tuple(float(line[start:start + 8]) for start in (30, 38, 46)))
                for line in stream if line.startswith("ATOM  ")
                and line[12:16].strip() == "CA" and line[16] in " A"
            ]
    else:
        array = get_structure(CIFFile.read(path), model=1)
        ca = array[(array.atom_name == "CA") & ~array.hetero]
        rows = [
            ((chain, index, insertion), name, coordinates)
            for chain, index, insertion, name, coordinates in zip(
                ca.chain_id, ca.res_id, ca.ins_code, ca.res_name, ca.coord, strict=True
            )
        ]
    residue_ids = [residue for residue, _, _ in rows]
    if len({residue[0] for residue in residue_ids}) != 1:
        raise ValueError("expected one C-alpha chain")
    if len(set(residue_ids)) != len(residue_ids):
        raise ValueError("duplicate C-alpha")
    xyz = np.array([coordinates for _, _, coordinates in rows], dtype=float)
    if not np.isfinite(xyz).all():
        raise ValueError("nonfinite coordinates")
    return xyz, [name for _, name, _ in rows]


def make_pdb(xyz: np.ndarray) -> str:
    lines = [
        f"ATOM  {index:5d}  CA  GLY A{index:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}           C"
        for index, (x, y, z) in enumerate(xyz, start=1)
    ]
    lines.extend([f"TER   {len(xyz) + 1:5d}      GLY A{len(xyz):4d}", "END"])
    return "\n".join(lines) + "\n"


def annotate_ca(xyz: np.ndarray) -> str:
    atoms = AtomArray(len(xyz))
    atoms.coord = xyz
    atoms.chain_id[:] = "A"
    atoms.res_id = np.arange(1, len(xyz) + 1)
    atoms.res_name[:] = "GLY"
    atoms.atom_name[:] = "CA"
    symbols = PSeaElement.to_symbols(annotate_sse(atoms))
    return "".join(symbols).translate(str.maketrans("abc", "HEC"))


def characterize(xyz: np.ndarray, names: list[str], generator: str, source: str, digest: str, length: int) -> Shape:
    if len(xyz) != length:
        raise ValueError("incorrect residue count")
    ca = xyz
    ca_steps = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    if np.any((ca_steps < 3.2) | (ca_steps > 4.4)):
        raise ValueError("C-alpha chain discontinuity")
    radius = float(np.sqrt(np.mean(np.sum((ca - ca.mean(axis=0)) ** 2, axis=1))))
    if radius > 3.2 * length ** (1 / 3):
        raise ValueError("radius exceeds compactness limit")

    separation = np.abs(np.arange(length)[:, None] - np.arange(length)[None, :])
    distances = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=2)
    if np.any((distances < 2.9) & (separation >= 4)):
        raise ValueError("nonlocal C-alpha clash")
    contacts = (distances <= 8.0) & (separation >= 6)
    coverage = float(np.mean(contacts.any(axis=0)))
    if coverage < 0.55:
        raise ValueError("low nonlocal contact coverage")

    assignment = annotate_ca(ca)
    if len(assignment) != length or not set(assignment) <= {"H", "E", "C"}:
        raise ValueError("incomplete P-SEA assignment")
    sequence = None
    if generator == "rfd4-proteina":
        if any(name not in AA_CODES for name in names):
            raise ValueError("nonstandard native sequence")
        sequence = "".join(AA_CODES[name] for name in names)
    return Shape(generator, source, digest, sequence, xyz, assignment, assignment.count("H") / length,
                 assignment.count("E") / length, radius, coverage, contacts)


def pair_geometry(left: Shape, right: Shape) -> tuple[float, float]:
    left_ca = left.xyz - left.xyz.mean(axis=0)
    right_ca = right.xyz - right.xyz.mean(axis=0)
    u, _, vh = np.linalg.svd(left_ca.T @ right_ca)
    parity = np.diag([1.0, 1.0, np.linalg.det(u @ vh)])
    residual = left_ca @ u @ parity @ vh - right_ca
    rmsd = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    shared = np.count_nonzero(left.contacts & right.contacts)
    union = np.count_nonzero(left.contacts | right.contacts)
    return rmsd, shared / union


def different(left: Shape, right: Shape) -> bool:
    rmsd, jaccard = pair_geometry(left, right)
    return rmsd >= 3.0 and jaccard <= 0.55


def balanced_pair(left: Shape, right: Shape) -> bool:
    return (
        left.length == right.length
        and left.family == right.family and left.family in FAMILIES
        and abs(left.helix - right.helix) <= 0.20
        and abs(left.strand - right.strand) <= 0.20
        and abs(left.radius - right.radius) <= 0.22 * min(left.radius, right.radius)
    )


def eligible_pair(left: Shape, right: Shape) -> bool:
    return balanced_pair(left, right) and different(left, right)


def pair_rank(pair: tuple[Shape, Shape]) -> tuple:
    left, right = pair
    return (
        left.generator == right.generator,
        abs(left.helix - right.helix) + abs(left.strand - right.strand)
        + abs(math.log(left.radius / right.radius)),
        -(left.contact_coverage + right.contact_coverage),
        left.source, right.source,
    )


def select_pairs(
    shapes: list[Shape], lengths: tuple[int, ...] = LENGTHS
) -> list[tuple[Shape, Shape]]:
    selected = []
    for length in lengths:
        pool = [shape for shape in shapes if shape.length == length]
        options = sorted((pair for pair in combinations(pool, 2) if eligible_pair(*pair)), key=pair_rank)
        by_family = {family: [pair for pair in options if pair[0].family == family] for family in FAMILIES}
        length_pairs = []

        def compatible(pair):
            return all(different(shape, previous) for shape in pair for old in length_pairs for previous in old)

        # One pair per available family first, rarest first. Then prefer
        # cross-generator pairs in the remaining pool until the target is met.
        for family in sorted(FAMILIES, key=lambda family: len(by_family[family])):
            candidate = next((pair for pair in by_family[family] if compatible(pair)), None)
            if candidate:
                length_pairs.append(candidate)
        for pair in options:
            if len(length_pairs) >= PAIR_TARGET:
                break
            if compatible(pair):
                length_pairs.append(pair)
        selected.extend(length_pairs)
    return selected


def select_release_pairs(shapes: list[Shape], lengths: tuple[int, ...], count: int, *, ranked: bool):
    """Find a maximum set of balanced, disjoint RFD4 pairs in each stratum."""
    from networkx import Graph, max_weight_matching

    selected, coverage, reserve = [], [], []
    for length in lengths:
        for family in FAMILIES:
            pool = sorted(
                (shape for shape in shapes if shape.length == length and shape.family == family),
                key=lambda shape: shape.source,
            )
            options = []
            for i, j in combinations(range(len(pool)), 2):
                left, right = pool[i], pool[j]
                agreement = sum(a == b for a, b in zip(
                    left.secondary_structure, right.secondary_structure, strict=True,
                ))
                if agreement * 10 >= length * 7 and eligible_pair(left, right):
                    contact_difference = abs(int(left.contacts.sum()) - int(right.contacts.sum()))
                    tie = sha256(f"{RELEASE_PAIR_SEED}\n{left.source}\n{right.source}".encode()).hexdigest()
                    options.append((-agreement, contact_difference, tie, i, j))
            options.sort()
            graph = Graph()
            max_pairs = len(pool) // 2
            contact_weight = max_pairs * len(options) + 1
            max_contact_difference = max((option[1] for option in options), default=0)
            agreement_weight = contact_weight * (max_pairs * max_contact_difference + 1)
            for rank, (negative_agreement, difference, _, i, j) in enumerate(options):
                quality = -negative_agreement * agreement_weight - difference * contact_weight
                graph.add_edge(i, j, weight=quality + len(options) - rank)
            matched = max_weight_matching(graph, maxcardinality=True)
            available = len(matched)
            if ranked:
                reserve_edges = set()
                if available > count:
                    # Pair each unused real vertex with a dummy. A perfect matching then
                    # contains exactly the requested number of real edges.
                    capped = graph.copy()
                    dummies = range(len(pool), len(pool) + len(graph) - 2 * count)
                    capped.add_edges_from((i, j, {"weight": 0}) for i in graph for j in dummies)
                    matched = {(i, j) for i, j in max_weight_matching(capped, maxcardinality=True)
                               if i < len(pool) and j < len(pool)}
                    if len(matched) != count:
                        raise ValueError("could not recover a quota-constrained matching")
                    used = {vertex for edge in matched for vertex in edge}
                    remainder = graph.subgraph(graph.nodes - used)
                    reserve_edges = max_weight_matching(remainder, maxcardinality=True)
                matched = {(min(i, j), max(i, j)) for i, j in matched}
                reserve_edges = {(min(i, j), max(i, j)) for i, j in reserve_edges}
                selected.extend((pool[i], pool[j]) for _, _, _, i, j in options if (i, j) in matched)
                reserve.extend([pool[i].source, pool[j].source]
                               for _, _, _, i, j in options if (i, j) in reserve_edges)
            coverage.append({
                "length": length,
                "family": family,
                "qualified_shapes": len(pool),
                "eligible_pair_edges": len(options),
                "available_disjoint_pairs": available,
                "required_pairs": count,
            })
    return selected, coverage, reserve


def specialist_sequence(path: Path, program: Path, output: Path) -> str:
    # Ties favor the first sample so repeated runs give the same control.
    command = [
        sys.executable, str(program), "--pdb_path", str(path), "--pdb_path_chains", "A",
        "--out_folder", str(output), "--ca_only", "--model_name", MPNN_MODEL,
        "--seed", str(MPNN_SEED), "--num_seq_per_target", "4", "--batch_size", "4",
        "--sampling_temp", "0.1", "--suppress_print", "1",
    ]
    subprocess.run(command, check=True, env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    lines = (output / "seqs" / f"{path.stem}.fa").read_text().splitlines()
    samples = []
    for index in range(2, len(lines), 2):
        match = re.search(r"(?:^|, )score=([0-9.]+)", lines[index])
        if match:
            samples.append((float(match[1]), index, lines[index + 1]))
    if len(samples) != 4:
        raise ValueError(f"Expected four ProteinMPNN control draws for {path.stem}")
    return min(samples)[2]


def metadata(shape: Shape) -> dict:
    return {
        "generator": shape.generator, "source": shape.source, "source_sha256": shape.source_sha256,
        "length": shape.length, "family": shape.family, "secondary_structure": shape.secondary_structure,
        "helix_fraction": shape.helix, "strand_fraction": shape.strand,
        "ca_radius_angstrom": shape.radius, "nonlocal_contact_coverage": shape.contact_coverage,
    }


def curate(
    raw: Path,
    output: Path,
    dataset_root: Path,
    proteinmpnn: Path | None,
    workspace: Path,
    lengths: tuple[int, ...],
    *,
    include_beta: bool,
    rfd4_lanes: tuple[str, ...] = (),
    pairs_per_stratum: int = 8,
    coverage_only: bool = False,
    holdout: tuple[Path, Path, Path] | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    mpnn_output = workspace / "proteinmpnn"
    shapes = []
    rejects = []
    receipts = {}
    emitted_sequences = set()
    mpnn_metadata = None
    mpnn_program = proteinmpnn.resolve() if proteinmpnn else None
    if mpnn_program:
        mpnn_path = mpnn_program.parent
        mpnn_metadata = {
            "revision": subprocess.check_output(["git", "-C", str(mpnn_path), "rev-parse", "HEAD"], text=True).strip(),
            "model": MPNN_MODEL,
            "checkpoint_sha256": sha256((mpnn_path / "ca_model_weights" / f"{MPNN_MODEL}.pt").read_bytes()).hexdigest(),
            "seed": MPNN_SEED,
            "sampling_temp": 0.1,
            "draws_per_target": 4,
            "selection": "minimum ProteinMPNN score",
        }
    if rfd4_lanes:
        from backends.freeze_backbones import _cif_atom_rows, _pdb_precision
        from trace2fold.backbone import _virtual_cb, atom_coordinates

        lanes = [(name, "rfd4-proteina") for name in rfd4_lanes]
    else:
        lanes = [(name, name) for name in GENERATORS]
    if include_beta and "rfd4-proteina-beta" not in rfd4_lanes:
        lanes.append(("rfd4-proteina-beta", "rfd4-proteina"))
    supplement = []
    frozen_shapes = set()
    for lane, generator in lanes:
        directory = raw / lane
        receipt = json.loads((directory / "receipt.json").read_text())
        if receipt["generator"] != generator or len(receipt["hashes"]) != receipt["emitted"]:
            raise ValueError(f"Inconsistent generation receipt for {lane}")
        receipts[lane] = {key: value for key, value in receipt.items() if key not in {"hashes", "phase_configs"}}
        for source, digest in sorted(receipt["hashes"].items()):
            path = directory / source
            if not path.resolve().is_relative_to(directory) or sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Artifact hash mismatch for {generator}/{source}")
            length = int(path.stem.split("-")[0][1:])
            if length not in lengths:
                raise ValueError(f"Unexpected artifact length {length} in {lane}: {source}")
            if lane != generator:
                source = f"{lane}/{source}"
            try:
                xyz, names = read_ca(path)
                if rfd4_lanes:
                    xyz = _pdb_precision(xyz)
                if generator == "rfd4-proteina" and all(name in AA_CODES for name in names):
                    emitted_sequences.add("".join(AA_CODES[name] for name in names))
                shape = characterize(xyz, names, generator, source, digest, length)
                if rfd4_lanes:
                    if path.suffix != ".cif":
                        raise ValueError("expected original RFD4 CIF")
                    atoms = _pdb_precision(atom_coordinates(_cif_atom_rows(path.read_bytes())))
                    if not np.array_equal(atoms[:, 1], xyz):
                        raise ValueError("four-atom backbone differs from the C-alpha trace")
                    _virtual_cb(atoms)
            except ValueError as exc:
                rejects.append({"generator": generator, "source": source, "reason": str(exc)})
                continue
            if shape.family == "other":
                rejects.append({"generator": generator, "source": source, "reason": "insufficient classifiable secondary structure"})
                continue
            if rfd4_lanes:
                shape_hash = sha256(make_pdb(xyz).encode()).hexdigest()
                if shape_hash in frozen_shapes:
                    rejects.append({"generator": generator, "source": source, "reason": "duplicate frozen C-alpha geometry"})
                    continue
                frozen_shapes.add(shape_hash)
            if lane == generator or rfd4_lanes:
                shapes.append(shape)
            else:
                supplement.append(shape)

    release_coverage = holdout_evidence = None
    reserve = []
    if rfd4_lanes:
        if holdout:
            from backends.holdout_similarity import split_holdout

            shapes, holdout_evidence = split_holdout(shapes, make_pdb, *holdout)
        selection, release_coverage, reserve = select_release_pairs(
            shapes, lengths, pairs_per_stratum, ranked=not coverage_only,
        )
        short = any(row["available_disjoint_pairs"] < pairs_per_stratum for row in release_coverage)
        report = {
            "screened": sum(row["emitted"] for row in receipts.values()),
            "qualified_before_holdout": holdout_evidence["qualified_before_split"] if holdout_evidence else len(shapes),
            "eligible": len(shapes),
            "selected": 0,
            "coverage_stage": "holdout" if holdout_evidence else "geometry",
            "release_coverage": release_coverage,
        }
        if coverage_only or short:
            print(json.dumps(report), flush=True)
        if coverage_only:
            return
        if short:
            raise ValueError("Insufficient qualified RFD4 pairs for a complete release dataset")
    else:
        selection = select_pairs(shapes, lengths)
    if include_beta and not rfd4_lanes:
        for length in sorted({shape.length for shape in supplement}):
            previous = [shape for pair in selection for shape in pair if shape.length == length]
            beta_mates = (
                (left, right) for left in shapes for right in supplement
                if right.length == length and left.generator == "genie3"
                and left.family == "beta" and eligible_pair(left, right)
                and all(different(shape, prior)
                        for shape in (left, right) for prior in previous)
            )
            best = min(beta_mates, key=pair_rank, default=None)
            if best:
                selection.append(best)
        shapes.extend(supplement)
    by_length_generator = Counter((shape.length, shape.generator) for pair in selection for shape in pair)
    requested_generators = ("rfd4-proteina",) if rfd4_lanes else GENERATORS
    missing = [f"{length}/{generator}" for length in lengths for generator in requested_generators
               if by_length_generator[length, generator] < 2]
    rows = []
    details = []
    selected_names = set()
    pair_numbers = Counter()
    for left, right in selection:
        pair_numbers[left.length] += 1
        prefix = "release" if rfd4_lanes else "gen"
        pair_id = f"{prefix}-l{left.length}-{pair_numbers[left.length]:02d}"
        matched_rmsd, matched_jaccard = pair_geometry(left, right)
        for suffix, shape in zip(("a", "b"), (left, right), strict=True):
            name = f"{pair_id}-{suffix}"
            path = output / f"{name}.pdb"
            pdb = make_pdb(shape.xyz)
            path.write_text(pdb)
            if shape.sequence:
                sequence = shape.sequence
            else:
                if mpnn_program is None:
                    raise ValueError(f"Missing native sequence and ProteinMPNN program for {name}")
                sequence = specialist_sequence(path, mpnn_program, mpnn_output)
            if len(sequence) != shape.length or not set(sequence) <= set(AA1):
                raise ValueError(f"Invalid specialist sequence for {name}")
            sequence_source = "rfd4-native" if shape.sequence else "proteinmpnn-ca"
            rows.append({
                "id": name, "path": path.relative_to(dataset_root).as_posix(), "chain": "A",
                "generator": shape.generator, "length": shape.length,
                "tier": f"L{shape.length}-{shape.family}", "family": shape.family, "pair": pair_id,
                "sequence": sequence, "sequence_source": sequence_source,
            })
            details.append({
                "id": name, **metadata(shape), "backbone_sha256": sha256(pdb.encode()).hexdigest(),
                "mate_ca_rmsd_angstrom": matched_rmsd, "mate_contact_jaccard": matched_jaccard,
            })
            selected_names.add((shape.generator, shape.source))

    native_sequences = {shape.sequence for shape in shapes if shape.sequence}
    audit = {
        "generation": receipts, "proteinmpnn": mpnn_metadata,
        "secondary_structure_method": "P-SEA on C-alpha coordinates (Biotite 2.0.0)", "screened": sum(row["emitted"] for row in receipts.values()),
        "unique_native_rfd4_sequences": len(emitted_sequences),
        "eligible": len(shapes), "eligible_native_unique_rfd4_sequences": len(native_sequences),
        "selected": len(rows), "pairs": len(selection),
        "eligible_unselected": len(shapes) - len(rows), "missing_multiple_shapes": missing,
        "selected_details": details,
        "skipped_eligible": [metadata(shape) for shape in shapes if (shape.generator, shape.source) not in selected_names],
        "rejected": rejects,
    }
    if release_coverage is not None:
        import networkx

        audit["release_coverage"] = release_coverage
        audit["release_pair_tie_seed"] = RELEASE_PAIR_SEED
        audit["networkx_version"] = networkx.__version__
        audit["reserve_pair_sources"] = reserve
    if holdout_evidence is not None:
        audit["holdout_similarity"] = {
            key: value for key, value in holdout_evidence.items()
            if key not in {"candidate_coordinate_sha256", "development_coordinate_sha256", "near_duplicate_edges", "restricted_clusters"}
        }
        holdout_text = json.dumps(holdout_evidence, indent=2) + "\n"
        (dataset_root / "holdout-similarity.json").write_text(holdout_text)
        audit["holdout_similarity"]["evidence_sha256"] = sha256(holdout_text.encode()).hexdigest()
    (workspace / "curation-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    index_stage = output / "index.json.in-progress"
    index_stage.write_text(json.dumps(rows, indent=2) + "\n")
    index_stage.replace(output / "index.json")
    (output / "provenance.json").write_text(json.dumps(
        {key: value for key, value in audit.items() if key not in {"skipped_eligible", "rejected"}}, indent=2
    ) + "\n")
    print(json.dumps({
        "screened": audit["screened"], "eligible": len(shapes), "selected": len(rows), "pairs": len(selection),
        "selected_groups": {f"{length}/{generator}": count for (length, generator), count in by_length_generator.items()},
        "missing_multiple_shapes": missing,
        **({"release_coverage": release_coverage} if release_coverage is not None else {}),
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--proteinmpnn", type=Path, help="Path to protein_mpnn_run.py for Genie 3 controls")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--output", type=Path, help="Output directory inside this repository")
    output_mode.add_argument("--dataset-root", type=Path, help="Build an independent dataset under a new directory")
    parser.add_argument("--lengths", default=",".join(map(str, LENGTHS)), help="Comma-separated residue counts")
    parser.add_argument("--include-beta", action="store_true", help="Append the eligible primarily-beta supplement")
    parser.add_argument("--rfd4-only", action="store_true", help="Build a balanced panel from complete RFD4 backbones")
    parser.add_argument("--rfd4-lanes", help="Comma-separated artifact directories for the RFD4-only panel")
    parser.add_argument("--pairs-per-stratum", type=int, default=8, help="RFD4-only pairs required for every length and family")
    parser.add_argument("--coverage-only", action="store_true", help="Report RFD4 cell coverage without publishing any dataset")
    parser.add_argument("--usalign", type=Path, help="Pinned US-align binary for ranked holdout screening")
    parser.add_argument("--similarity-cache", type=Path, help="Directory for the reusable US-align comparison cache")
    parser.add_argument("--development-root", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="Previous Trace2Fold dataset root for holdout exclusion")
    args = parser.parse_args()
    lengths = tuple(int(value) for value in args.lengths.split(","))
    if tuple(sorted(set(lengths))) != lengths or any(n < 1 for n in lengths):
        parser.error("specify distinct positive lengths in increasing order")
    if args.rfd4_only and (args.proteinmpnn or (not args.dataset_root and not args.coverage_only) or args.output):
        parser.error("RFD4-only curation needs --coverage-only or a new --dataset-root")
    if not args.rfd4_only and (not args.proteinmpnn or args.rfd4_lanes):
        parser.error("the original panel needs --proteinmpnn and uses the standard generator lanes")
    if bool(args.usalign) != bool(args.similarity_cache):
        parser.error("specify --usalign and --similarity-cache together")
    if args.rfd4_only and not args.coverage_only and not args.usalign:
        parser.error("publishing a ranked panel requires --usalign and --similarity-cache")
    if args.coverage_only and (not args.rfd4_only or args.dataset_root):
        parser.error("--coverage-only reports RFD4 coverage without a --dataset-root")
    if args.pairs_per_stratum < 1:
        parser.error("--pairs-per-stratum must be positive")
    lanes = tuple((args.rfd4_lanes or "rfd4-proteina").split(",")) if args.rfd4_only else ()
    if len(set(lanes)) != len(lanes) or any(Path(lane).name != lane or lane == ".." for lane in lanes):
        parser.error("RFD4 artifact lanes must be distinct directory names")
    raw = args.raw.resolve()
    holdout = (args.development_root.resolve(), args.usalign, args.similarity_cache) if args.usalign else None
    if args.coverage_only:
        with TemporaryDirectory(dir=raw.parent, prefix=".trace2fold-coverage-") as workspace:
            workspace = Path(workspace)
            curate(
                raw, workspace / "data/generated", workspace, None, workspace, lengths,
                include_beta=args.include_beta, rfd4_lanes=lanes, pairs_per_stratum=args.pairs_per_stratum,
                coverage_only=True, holdout=holdout,
            )
        return
    if args.dataset_root:
        from backends.freeze_backbones import freeze_panel
        from trace2fold.cases import dump

        destination = args.dataset_root.resolve()
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=destination.parent, prefix=".trace2fold-curation-") as workspace:
            workspace = Path(workspace)
            stage = workspace / "dataset"
            output = stage / "data/generated"
            curate(
                raw, output, stage, args.proteinmpnn, workspace, lengths,
                include_beta=args.include_beta, rfd4_lanes=lanes, pairs_per_stratum=args.pairs_per_stratum,
                holdout=holdout,
            )
            audit_detail = workspace / "curation-audit.json"
            audit_detail.replace(stage / audit_detail.name)
            frozen = freeze_panel(raw, stage)
            if args.rfd4_only:
                index = json.loads((output / "index.json").read_text())
                if frozen.keys() != {record["id"] for record in index}:
                    raise ValueError("The ranked dataset is missing original four-atom rows")
            dump(output / "backbone_atoms.json", frozen)
            if destination.exists():
                raise FileExistsError(destination)
            stage.rename(destination)
        return

    repo_root = Path(__file__).resolve().parent.parent
    output = args.output.resolve() if args.output else repo_root / "data/generated"
    if not output.is_relative_to(repo_root):
        raise ValueError("Frozen index paths must be relative to the repository")
    curate(raw, output, repo_root, args.proteinmpnn, raw, lengths, include_beta=args.include_beta)


if __name__ == "__main__":
    main()
