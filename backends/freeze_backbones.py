"""Freeze anonymous N/CA/C/O coordinates from the generated panel's original output files."""

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np

from trace2fold.backbone import ATOMS, atom_coordinates, pdb_atom_rows
from trace2fold.cases import ROOT, dump, load_cases


def _cif_atom_rows(payload):
    from biotite.structure.io.pdbx import CIFFile, get_structure

    structure = get_structure(CIFFile.read(io.StringIO(payload.decode())), model=1)
    rows = {}
    for chain, position, insertion, atom, coordinates, hetero in zip(
        structure.chain_id,
        structure.res_id,
        structure.ins_code,
        structure.atom_name,
        structure.coord,
        structure.hetero,
        strict=True,
    ):
        if hetero or atom not in ATOMS:
            continue
        residue = rows.setdefault(chain, {}).setdefault((position, insertion), {})
        if atom in residue:
            raise ValueError("Ambiguous backbone atom coordinates")
        residue[atom] = coordinates
    return rows


def _pdb_precision(coordinates):
    return np.fromiter(
        (float(f"{v:.3f}") for v in coordinates.flat), dtype=float
    ).reshape(coordinates.shape)


def freeze_panel(raw_root, root=ROOT):
    """Verify the original files for the existing frozen panel before retaining coordinates."""
    raw_root = Path(raw_root).resolve()
    provenance = json.loads((Path(root) / "data/generated/provenance.json").read_text())
    cases = load_cases(root)
    frozen = {}
    for record in provenance["selected_details"]:
        case = cases[record["id"]]
        source_name = Path(record["source"])
        # Sources outside the generator's default lane carry its folder prefix.
        if len(source_name.parts) <= 2:
            source_name = Path(record["generator"]) / source_name
        source = (raw_root / source_name).resolve()
        if not source.is_relative_to(raw_root) or source.suffix not in {".pdb", ".cif"}:
            raise ValueError(f"Invalid original backbone path for {case.id}")
        payload = source.read_bytes()
        if sha256(payload).hexdigest() != record["source_sha256"]:
            raise ValueError(f"Backbone source hash mismatch for {case.id}")
        ca_hash = sha256((Path(root) / case.info["path"]).read_bytes()).hexdigest()
        if ca_hash != record["backbone_sha256"]:
            raise ValueError(f"Backbone C-alpha hash mismatch for {case.id}")

        rows = (
            pdb_atom_rows(payload)
            if source.suffix == ".pdb"
            else _cif_atom_rows(payload)
        )
        lengths = [len(chain) for chain in rows.values()]
        alpha = _pdb_precision(atom_coordinates(rows, ("CA",))[:, 0])
        if lengths != case.lengths or not np.array_equal(alpha, case.xyz):
            raise ValueError(
                f"Original residues do not match frozen C-alpha rows for {case.id}"
            )
        if all(
            set(residue) == {"CA"}
            for chain in rows.values()
            for residue in chain.values()
        ):
            continue
        coordinates = _pdb_precision(atom_coordinates(rows))
        frozen[case.id] = {
            "source_sha256": record["source_sha256"],
            "ca_sha256": ca_hash,
            "chain_lengths": lengths,
            "coordinates": coordinates.tolist(),
        }
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "raw_root",
        type=Path,
        help="Root containing the original generator output directories",
    )
    parser.add_argument("--dataset-root", type=Path, default=ROOT)
    args = parser.parse_args()
    frozen = freeze_panel(args.raw_root, args.dataset_root)
    output = args.dataset_root / "data/generated/backbone_atoms.json"
    dump(output, frozen)
    print(f"Froze N/CA/C/O coordinates for {len(frozen)} cases in {output}")


if __name__ == "__main__":
    main()
