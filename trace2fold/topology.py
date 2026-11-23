"""Build sequence-free descriptions from frozen C-alpha topology."""

from dataclasses import dataclass
from functools import cache
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

SECONDARY_STRUCTURE_METHOD = "P-SEA on C-alpha coordinates (Biotite 2.0.0)"
CONTACT_DISTANCE = 8.0
MIN_SAME_CHAIN_SEPARATION = 6


@dataclass(frozen=True)
class SecondaryStructure:
    chains: tuple[str, ...]
    coordinate_sha256: str


def _coordinate_digest(xyz, lengths):
    chain_lengths = np.asarray(lengths, dtype="<i8")
    digest = sha256(len(chain_lengths).to_bytes(8, "little"))
    digest.update(chain_lengths.tobytes())
    digest.update(np.asarray(xyz, dtype="<f8", order="C").tobytes())
    return digest.hexdigest()


def load_annotations(root=None):
    """Load versioned P-SEA labels tied to the indexed, hashed PDB fixtures."""
    from .cases import ROOT

    return _load_annotations(Path(ROOT if root is None else root).resolve())


@cache
def _load_annotations(root):
    from .cases import read_ca

    data = root / "data"
    sources = (
        ("generated", "provenance.json", "selected_details", "backbone_sha256"),
        ("known", "secondary_structure.json", "annotations", "fixture_sha256"),
    )
    annotations = {}
    for subset, filename, rows_key, hash_key in sources:
        path = data / subset / filename
        if not path.is_file():
            continue
        source = json.loads(path.read_text())
        if source.get("secondary_structure_method") != SECONDARY_STRUCTURE_METHOD:
            raise ValueError(f"unsupported secondary structure method in {path}")
        index = {
            record["id"]: record
            for record in json.loads(path.with_name("index.json").read_text())
        }
        rows = source[rows_key]
        if sorted(row["id"] for row in rows) != sorted(index):
            raise ValueError(f"secondary structure coverage mismatch in {path}")
        for row in rows:
            case_id = row["id"]
            record = index[case_id]
            fixture = root / record["path"]
            if sha256(fixture.read_bytes()).hexdigest() != row[hash_key]:
                raise ValueError(
                    f"secondary structure fixture hash mismatch for {case_id}"
                )
            chains = record.get("chains") or [record.get("chain", "A")]
            xyz, sequences = read_ca(fixture, chains)
            lengths = list(map(len, sequences))
            labels = tuple(row["secondary_structure"].split(":"))
            if list(map(len, labels)) != lengths or set("".join(labels)) - set("HEC"):
                raise ValueError(f"invalid secondary structure labels for {case_id}")
            annotations[case_id] = SecondaryStructure(
                labels, _coordinate_digest(xyz, lengths)
            )
    return MappingProxyType(annotations)


def contact_pairs(xyz, lengths):
    """Return sorted, global 1-based C-alpha contact pairs.

    Apply the inclusive cutoff to the stored coordinates before any prompt
    rounding. Same-chain positions must differ by at least six indices.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    chain_ids = np.repeat(np.arange(len(lengths)), lengths)
    left, right = np.triu_indices(len(xyz), k=1)
    eligible = (right - left >= MIN_SAME_CHAIN_SEPARATION) | (
        chain_ids[left] != chain_ids[right]
    )
    left, right = left[eligible], right[eligible]
    displacement = xyz[left] - xyz[right]
    within_cutoff = (displacement**2).sum(axis=1) <= CONTACT_DISTANCE**2
    return list(
        zip(
            (left[within_cutoff] + 1).tolist(),
            (right[within_cutoff] + 1).tolist(),
        )
    )


def topology_text(case, annotation):
    """Format one topology block after checking its chain lengths and coordinates."""
    if annotation is None:
        raise ValueError(f"missing frozen secondary structure for {case.id}")
    if tuple(map(len, annotation.chains)) != tuple(case.lengths) or set(
        "".join(annotation.chains)
    ) - set("HEC"):
        raise ValueError(f"invalid secondary structure labels for {case.id}")
    if _coordinate_digest(case.xyz, case.lengths) != annotation.coordinate_sha256:
        raise ValueError(f"secondary structure coordinate hash mismatch for {case.id}")

    labels = ":".join(annotation.chains)
    contacts = ", ".join(f"{i}-{j}" for i, j in contact_pairs(case.xyz, case.lengths))
    return "\n".join(
        (
            "Secondary structure in sequence order (H helix, E strand, C coil)"
            + (", with a colon between chains" if len(case.lengths) > 1 else ""),
            labels,
            "Nonlocal C-alpha contacts (1-based residue pairs within 8 angstroms, "
            "separated by at least 6 positions on the same chain or on different chains)",
            contacts or "None",
        )
    )
