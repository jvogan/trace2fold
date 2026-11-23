"""Freeze the four experimental calibration backbones from local wwPDB PDB files."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKBONE = ("N", "CA", "C", "O")
AA = dict(zip(
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
    "ACDEFGHIKLMNPQRSTVWY",
    strict=True,
))

# Spans use deposited PDB author residue IDs. They are independent of the
# one-based flattened coordinate indices used in the task.
RECIPES = [
    {
        "id": "trp-cage",
        "pdb_id": "1L2Y",
        "spans": {"A": (1, 20)},
        "model": 1,
        "doi": "10.1038/nsb798",
        "method": "solution NMR",
        "family": "miniprotein",
        "tier": "easy",
        "tier_reason": "20-residue, single-chain compact fold with no disulfides.",
    },
    {
        "id": "insulin",
        "pdb_id": "1TRZ",
        "spans": {"A": (1, 21), "B": (1, 30)},
        "model": 1,
        "doi": "10.1021/bi00172a030",
        "method": "X-ray diffraction",
        "resolution_angstrom": 1.6,
        "family": "insulin",
        "tier": "medium",
        "tier_reason": "Two associated chains and three disulfides in a compact 51-residue target.",
        "chemistry": "Mature human insulin A and B chains from one protomer of the deposited assembly.",
    },
    {
        "id": "gfp",
        "pdb_id": "1QY3",
        "spans": {"A": (3, 229)},
        "model": 1,
        "doi": "10.1073/pnas.2133463100",
        "method": "X-ray diffraction",
        "resolution_angstrom": 2.0,
        "family": "gfp",
        "tier": "hard",
        "tier_reason": "227-residue beta-barrel with an enclosed central helix.",
        "chemistry": "Precyclization R96A GFP variant. Thr65, Tyr66 and Gly67 are three distinct canonical residues in the deposit.",
        "chromophore_precursors": [63, 64, 65],
    },
    {
        "id": "vhh-cdr3",
        "pdb_id": "1I3V",
        "spans": {"A": (1, 129)},
        "model": 1,
        "doi": "10.1006/jmbi.2001.4856",
        "method": "X-ray diffraction",
        "resolution_angstrom": 2.03,
        "family": "vhh",
        "tier": "hard",
        "tier_reason": "Match the geometry of the masked 17-residue loop with the framework fixed.",
        "mutable": list(range(102, 119)),
        "pair_status": "unpaired",
        "chemistry": "Llama VHH A52 domain, chain A. SAbDab2 places CDR-H3 at author positions A102 to A118.",
    },
]


def freeze(recipe: dict, download: bytes) -> tuple[dict, str]:
    """Extract one deposited model and retain its declared sequence mapping."""
    pdb_id = recipe["pdb_id"]
    spans = recipe["spans"]
    coordinates: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    declared: dict[str, list[str]] = defaultdict(list)
    source_bonds = []
    model = 1
    for line in download.decode("ascii").splitlines():
        if line.startswith("SEQRES"):
            declared[line[11]].extend(line[19:70].split())
        elif line.startswith("SSBOND"):
            source_bonds.append(line)
        elif line.startswith("MODEL "):
            model = int(line[10:14])
        elif line.startswith("ENDMDL") and model == recipe["model"]:
            break
        elif line.startswith("ATOM  ") and model == recipe["model"] and line[21] in spans:
            chain = line[21]
            position = int(line[22:26])
            start, end = spans[chain]
            atom = line[12:16].strip()
            if start <= position <= end and atom in (*BACKBONE, "SG"):
                if line[16] != " " or line[26] != " " or atom in coordinates[chain, position]:
                    raise ValueError(f"{pdb_id} has ambiguous atoms at {chain}:{position}")
                coordinates[chain, position][atom] = line

    ordered_positions = [(chain, i) for chain, (start, end) in spans.items() for i in range(start, end + 1)]
    indices = {position: i for i, position in enumerate(ordered_positions, start=1)}
    bonds = []
    bond_records = []
    sulfur_positions = set()
    for line in source_bonds:
        a = (line[15], int(line[17:21]))
        b = (line[29], int(line[31:35]))
        if a not in indices or b not in indices:
            continue
        bonds.append([indices[a], indices[b]])
        bond_records.append(f"SSBOND {len(bonds):>3}{line[10:].rstrip()}")
        sulfur_positions.update((a, b))

    chain_lengths = []
    sequence = []
    output = [f"REMARK   1 BACKBONE EXTRACTED FROM PDB {pdb_id}, MODEL {recipe['model']}", *bond_records]
    serial = 0
    for chain, (start, end) in spans.items():
        chain_lengths.append(end - start + 1)
        for number in range(start, end + 1):
            residue = coordinates[chain, number]
            atoms = (*BACKBONE, "SG") if (chain, number) in sulfur_positions else BACKBONE
            if any(atom not in residue for atom in atoms):
                raise ValueError(f"{pdb_id} lacks required coordinates at {chain}:{number}")
            names = {residue[atom][17:20] for atom in atoms}
            if names != {declared[chain][number - 1]}:
                raise ValueError(f"{pdb_id} ATOM/SEQRES mismatch at {chain}:{number}")
            name = names.pop()
            if "SG" in atoms and name != "CYS":
                raise ValueError(f"{pdb_id} has non-cysteine disulfide at {chain}:{number}")
            sequence.append(AA[name])
            for atom in atoms:
                serial += 1
                output.append(f"ATOM  {serial:>5}{residue[atom][11:].rstrip()}")
        serial += 1
        output.append(f"TER   {serial:>5}      {name} {chain}{end:>4}")
    output.append("END")
    pdb = "\n".join(output) + "\n"

    record = {key: value for key, value in recipe.items() if key not in {"spans", "doi"}}
    if len(spans) == 1:
        record["chain"] = next(iter(spans))
    else:
        record["chains"] = list(spans)
    record.update(
        path=f"data/known/{recipe['id']}.pdb",
        length=len(sequence),
        chain_lengths=chain_lengths,
        author_residue_ranges=spans,
        sequence="".join(sequence),
        pdb_url=f"https://www.rcsb.org/structure/{pdb_id}",
        coordinate_url=f"https://files.rcsb.org/download/{pdb_id}.pdb",
        article_url=f"https://doi.org/{recipe['doi']}",
        source_sha256=hashlib.sha256(download).hexdigest(),
        fixture_sha256=hashlib.sha256(pdb.encode("ascii")).hexdigest(),
    )
    if bonds:
        record["disulfides"] = bonds
        record["fixed"] = sorted(indices[position] for position in sulfur_positions)
    if pdb_id == "1I3V":
        record["numbering_source"] = {
            "database": "SAbDab2",
            "instance": "pdb_00001i3v-A",
            "field": "CDR-H3",
            "reference_url": "https://doi.org/10.1093/nar/gkag1091",
        }
    return record, pdb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/known")
    args = parser.parse_args()
    frozen = [
        freeze(recipe, (args.source_dir / f"{recipe['pdb_id']}.pdb").read_bytes())
        for recipe in RECIPES
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for record, pdb in frozen:
        (args.output_dir / Path(record["path"]).name).write_text(pdb, encoding="ascii")
    (args.output_dir / "index.json").write_text(
        json.dumps([record for record, _ in frozen], indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
