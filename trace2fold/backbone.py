"""Render only backbone geometry for cases with a complete four-atom source."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .cases import local_frame_axes

ATOMS = ("N", "CA", "C", "O")


def pdb_atom_rows(source, chains=None):
    """Keep only backbone atom coordinates in residue and chain order."""
    rows = {} if chains is None else {chain: {} for chain in chains}
    for line in source.decode("ascii").splitlines():
        if line.startswith("ENDMDL"):
            break
        if (
            line[:6].strip() not in {"ATOM", "HETATM"}
            or line[12:16].strip() not in ATOMS
        ):
            continue
        chain = line[21]
        if (chains is not None and chain not in rows) or line[16] not in " A1":
            continue
        residue = rows.setdefault(chain, {}).setdefault(line[22:27], {})
        atom = line[12:16].strip()
        if atom in residue:
            raise ValueError("Ambiguous backbone atom coordinates")
        residue[atom] = [float(line[i : i + 8]) for i in (30, 38, 46)]
    return rows


def atom_coordinates(rows, names=ATOMS):
    try:
        atoms = [
            [row[atom] for atom in names]
            for chain in rows.values()
            for row in chain.values()
        ]
    except KeyError as exc:
        raise ValueError(f"Incomplete {'/'.join(names)} backbone") from exc
    coordinates = np.array(atoms, dtype=float).reshape(-1, len(names), 3)
    if not np.isfinite(coordinates).all():
        raise ValueError("Nonfinite backbone coordinates")
    return coordinates


@lru_cache(maxsize=8)
def _json_for_file(path, identity):
    return json.loads(path.read_text())


def _read_coordinates_metadata(path):
    stat = path.stat()
    return _json_for_file(path, (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))


def _backbone_coordinates(case):
    path = Path(case.root) / case.info["path"]
    expected = case.info.get("fixture_sha256")
    if expected:
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError(f"Backbone source hash mismatch for {case.id}")
        chains = case.info.get("chains") or [case.info.get("chain", "A")]
        rows = pdb_atom_rows(source, chains)
        atoms = atom_coordinates(rows)
        lengths = [len(chain) for chain in rows.values()]
    else:
        sidecar = path.parent / "backbone_atoms.json"
        record = (
            _read_coordinates_metadata(sidecar).get(case.id) if sidecar.is_file() else None
        )
        if record is None:
            raise ValueError(f"No frozen N/CA/C/O backbone for {case.id}")
        provenance = _read_coordinates_metadata(path.parent / "provenance.json")
        original = next(
            (row for row in provenance["selected_details"] if row["id"] == case.id), None,
        )
        if original is None:
            raise ValueError(f"Missing original backbone provenance for {case.id}")
        ca_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if ca_hash != record["ca_sha256"] or ca_hash != original["backbone_sha256"]:
            raise ValueError(f"Backbone C-alpha hash mismatch for {case.id}")
        if record["source_sha256"] != original["source_sha256"]:
            raise ValueError(f"Backbone source hash mismatch for {case.id}")
        atoms = np.asarray(record["coordinates"], dtype=float)
        lengths = record["chain_lengths"]
    if lengths != case.lengths or atoms.shape != (len(case.xyz), len(ATOMS), 3):
        raise ValueError(f"Incomplete N/CA/C/O backbone for {case.id}")
    if not np.isfinite(atoms).all() or not np.array_equal(atoms[:, 1], case.xyz):
        raise ValueError(
            f"N/CA/C/O rows do not match the frozen C-alpha coordinates for {case.id}"
        )
    return atoms


def _dihedrals(a, b, c, d):
    axis = c - b
    norm = np.linalg.norm(axis, axis=1, keepdims=True)
    axis = np.divide(axis, norm, out=np.zeros_like(axis), where=norm > 1e-8)
    before, after = a - b, d - c
    before -= np.sum(before * axis, axis=1, keepdims=True) * axis
    after -= np.sum(after * axis, axis=1, keepdims=True) * axis
    x = np.sum(before * after, axis=1)
    y = np.sum(np.cross(axis, before) * after, axis=1)
    angles = np.degrees(np.arctan2(y, x))
    return np.where((norm[:, 0] > 1e-8) & (np.hypot(x, y) > 1e-8), angles, np.nan)


def _torsions(coordinates, lengths):
    nitrogen, alpha, carbon, _ = coordinates.transpose(1, 0, 2)
    phi, psi = np.full((2, len(coordinates)), np.nan)
    ends = np.cumsum(lengths)[:-1]
    # A peptide bond is approximately 1.33 Å. Leave torsions undefined at a break.
    connected = np.linalg.norm(carbon[:-1] - nitrogen[1:], axis=1) <= 1.8
    connected[ends - 1] = False
    left = np.flatnonzero(connected)
    right = left + 1
    phi[right] = _dihedrals(carbon[left], nitrogen[right], alpha[right], carbon[right])
    psi[left] = _dihedrals(nitrogen[left], alpha[left], carbon[left], nitrogen[right])
    return phi, psi


def _virtual_cb(coordinates):
    nitrogen, alpha, carbon, _ = coordinates.transpose(1, 0, 2)
    n_bond, c_bond = nitrogen - alpha, carbon - alpha
    n_lengths = np.linalg.norm(n_bond, axis=1, keepdims=True)
    c_lengths = np.linalg.norm(c_bond, axis=1, keepdims=True)
    normal = np.cross(n_bond, c_bond)
    normal_lengths = np.linalg.norm(normal, axis=1, keepdims=True)
    if np.any(normal_lengths < 1e-8):
        raise ValueError(
            "Cannot infer a virtual C-beta direction from collinear N/CA/C atoms"
        )
    bisector = n_bond / n_lengths + c_bond / c_lengths
    bisector_lengths = np.linalg.norm(bisector, axis=1, keepdims=True)
    # The projection of an ideal tetrahedral bond on either backbone bond is -1/3.
    # Choose the normal consistent with L-amino-acid chirality for every position.
    plane_fraction = 2 / (3 * bisector_lengths)
    if np.any(plane_fraction > 1):
        raise ValueError("Cannot infer a tetrahedral direction from the N/CA/C angle")
    return -plane_fraction * bisector / bisector_lengths + (
        np.sqrt(1 - plane_fraction**2) * normal / normal_lengths
    )


def _nearby_counts(alpha, lengths):
    dist = np.linalg.norm(alpha[:, None] - alpha[None, :], axis=2)
    positions = np.arange(len(alpha))
    chain = np.repeat(np.arange(len(lengths)), lengths)
    neighbors = (chain[:, None] == chain[None, :]) & (
        np.abs(positions[:, None] - positions) <= 2
    )
    return ((dist <= 8) & ~neighbors).sum(axis=1)


def side_chain_directions(case):
    """Return virtual C-beta unit directions in the frozen C-alpha frame."""
    atoms = _backbone_coordinates(case)
    _, axes = local_frame_axes(case.xyz)
    return _virtual_cb(atoms) @ axes


def backbone_text(case, *, derived=False):
    """Return indexed N, C-alpha, C and O coordinates in the shared C-alpha frame."""
    atoms = _backbone_coordinates(case)
    origin, axes = local_frame_axes(case.xyz)
    frame = (atoms - origin) @ axes
    lines = ["Position N_x N_y N_z CA_x CA_y CA_z C_x C_y C_z O_x O_y O_z (angstroms)"]
    annotations = [""] * len(frame)
    if derived:
        phi, psi = _torsions(atoms, case.lengths)
        cb = _virtual_cb(atoms) @ axes
        nearby = _nearby_counts(case.xyz, case.lengths)
        lines.insert(
            0,
            "Phi and psi are in degrees, with NA where undefined. Virtual CB is a unit direction from N/CA/C geometry.",
        )
        lines.insert(
            1,
            "CA neighbors are within 8 angstroms, excluding the same chain within two positions.",
        )
        lines[-1] += " phi psi virtual_CB_x virtual_CB_y virtual_CB_z CA_neighbors"
        for i, (p, q, direction, count) in enumerate(
            zip(phi, psi, cb, nearby, strict=True)
        ):
            angles = [
                f"{angle:.0f}" if np.isfinite(angle) else "NA" for angle in (p, q)
            ]
            annotations[i] = " " + " ".join(
                [*angles, *(f"{v:.2f}" for v in direction), str(count)]
            )
    start = 0
    for chain, length in enumerate(case.lengths, 1):
        if len(case.lengths) > 1:
            lines.append(f"Chain {chain}")
        for i in range(start, start + length):
            lines.append(
                f"{i + 1} "
                + " ".join(f"{v:.2f}" for v in frame[i].flat)
                + annotations[i]
            )
        start += length
    return "\n".join(lines)
