"""Read evaluator records and format anonymous structure prompts."""

from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import json
import re
import os
import tempfile

import numpy as np

PACKAGE = Path(__file__).resolve().parent
ROOT = (
    PACKAGE / "assets"
    if (PACKAGE / "assets/data/known/index.json").is_file()
    else PACKAGE.parent
)
AA = "ACDEFGHIKLMNPQRSTVWY"
THREE = dict(
    zip(
        "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
        AA,
    )
)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    # A second invocation can verify an artifact but cannot overwrite a submission.
    if path.exists() and path.read_text() == payload:
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        try:
            os.link(stream.name, path)
        except FileExistsError:
            if path.read_text() != payload:
                raise


def read_ca(path, chains):
    """Use the first model and one C-alpha per residue, preserving chain boundaries."""
    rows = {chain: {} for chain in chains}
    for line in Path(path).read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if line[:6].strip() not in {"ATOM", "HETATM"} or line[12:16].strip() != "CA":
            continue
        chain = line[21]
        if chain not in rows or line[16] not in " A1":
            continue
        key = line[22:27]
        if key in rows[chain]:
            if line[16] == " ":
                raise ValueError(f"duplicate C-alpha in {path} chain {chain} {key}")
            continue
        rows[chain][key] = (
            THREE.get(line[17:20], "X"),
            [float(line[i : i + 8]) for i in (30, 38, 46)],
        )
    if any(not residues for residues in rows.values()):
        raise ValueError(f"missing chain in {path}")
    seqs = ["".join(aa for aa, _ in rows[c].values()) for c in chains]
    xyz = np.asarray([p for c in chains for _, p in rows[c].values()])
    if not np.isfinite(xyz).all():
        raise ValueError(f"nonfinite coordinates in {path}")
    return xyz, seqs


@dataclass
class Case:
    info: dict
    xyz: np.ndarray
    lengths: list
    template: str
    mutable: np.ndarray
    root: Path = ROOT

    @property
    def id(self):
        return self.info["id"]

    def split(self, flat):
        offsets = np.cumsum([0, *self.lengths])
        return ":".join(flat[a:b] for a, b in zip(offsets, offsets[1:]))

    def validate(self, answer):
        if not isinstance(answer, str):
            return False
        if not re.fullmatch(":".join(f"[{AA}]{{{n}}}" for n in self.lengths), answer):
            return False
        return all(
            a == "?" or a == b for a, b in zip(self.template, answer.replace(":", ""))
        )


def load_cases(root=ROOT):
    cases = {}
    for path in sorted((Path(root) / "data").glob("*/index.json")):
        for record in json.loads(path.read_text()):
            chains = record.get("chains") or [record.get("chain", "A")]
            xyz, seqs = read_ca(Path(root) / record["path"], chains)
            n = len(xyz)
            if record["length"] != n or record["id"] in cases:
                raise ValueError(f"inconsistent case {record['id']}")
            positive = record.get("sequence") or ""
            positive = positive.replace(":", "")
            if positive and (len(positive) != n or set(positive) - set(AA)):
                raise ValueError(f"invalid control sequence in {record['id']}")
            mutable = np.ones(n, dtype=bool)
            if "mutable" in record:
                mutable[:] = False
            for field, value in [("mutable", True), ("fixed", False)]:
                for i in record.get(field, []):
                    if not 1 <= i <= n:
                        raise ValueError(f"invalid position in {record['id']}")
                    mutable[i - 1] = value
            if not mutable.any() or (not mutable.all() and not positive):
                raise ValueError(f"invalid design mask in {record['id']}")
            template = "".join("?" if m else positive[i] for i, m in enumerate(mutable))
            cases[record["id"]] = Case(
                record, xyz, list(map(len, seqs)), template, mutable, Path(root).resolve()
            )
    return cases


def local_frame_axes(xyz):
    # All chains share this rigid frame so inter-chain packing is preserved.
    delta = xyz - xyz[0]
    x = delta[1] / np.linalg.norm(delta[1])
    y = delta - np.outer(delta @ x, x)
    anchor = y[np.argmax(np.linalg.norm(y, axis=1))]
    if np.linalg.norm(anchor) < 1e-6:
        raise ValueError("collinear backbone")
    y = anchor / np.linalg.norm(anchor)
    return xyz[0], np.stack([x, y, np.cross(x, y)], axis=1)


def local_frame(xyz):
    origin, axes = local_frame_axes(xyz)
    return (xyz - origin) @ axes


PROMPT_VERSION = "v2"
V4_VERSIONS = ("v4", "v4.1", "v4.2", "v4.3")
PROMPT_VERSIONS = ("v1", "v2", "v3", *V4_VERSIONS)
V3_CONDITIONS = ("geometry", "length")
V4_CONDITIONS = ("geometry", "vectors", "vectors-contacts", "length", "masked")
STRUCTURE_CONDITIONS = ("geometry", "topology", "annotated", "backbone", "backbone-annotated")


def v4_conditions(version):
    return ("geometry", "vectors") if version == "v4.3" else V4_CONDITIONS


def input_condition(condition):
    # The two follow-ups reuse the primary geometry input.
    return "geometry" if condition in ("review", "restart") else condition


def prompt(case, condition="geometry", version=PROMPT_VERSION):
    """Build one anonymous contestant prompt from the frozen case."""
    if version == "v1":
        return prompt_v1(case, condition)
    if version == "v3":
        return prompt_v3(case, condition)
    if version in V4_VERSIONS:
        return prompt_v4(case, condition, version=version)
    if version != "v2":
        raise ValueError("unknown prompt version")
    if condition not in (*STRUCTURE_CONDITIONS, "length"):
        raise ValueError("unknown condition")
    chains = len(case.lengths)
    if chains == 1:
        size = f"{case.lengths[0]} residues"
    else:
        counts = ", ".join(map(str, case.lengths[:-1])) + f" and {case.lengths[-1]}"
        size = f"{chains} chains ({counts} residues)"
    lines = [f"Write an amino-acid sequence for a protein with {size}"]
    if condition == "length":
        lines[0] += "."
    else:
        lines[0] += " that folds into this backbone."
    if condition in ("geometry", "annotated"):
        lines.append(
            "Each coordinate line gives one residue: its number, then the x, y, z "
            "position of its C-alpha atom in angstroms."
            + (" Numbers continue across chains." if chains > 1 else "")
        )
    if not case.mutable.all():
        lines.append("Template: " + case.split(case.template))
        lines.append("Keep each letter in the template and replace each ? with an amino acid.")
    if case.info.get("disulfides"):
        pairs = ", ".join(f"{a}-{b}" for a, b in case.info["disulfides"])
        lines.append(
            "Disulfide bonds join these pairs of residue numbers, counted across all chains: "
            + pairs
            + "."
        )
    lines.append(
        f"Reply with the sequence only, in one-letter codes ({AA})"
        + (", with a colon between chains." if chains > 1 else ".")
    )
    if condition in ("topology", "annotated", "backbone-annotated"):
        from .topology import load_annotations, topology_text

        annotation = load_annotations(case.root).get(case.id)
        lines.extend(["", topology_text(case, annotation)])
    if condition in ("backbone", "backbone-annotated"):
        from .backbone import backbone_text

        lines.extend(["", backbone_text(case, derived=condition == "backbone-annotated")])
    if condition in ("geometry", "annotated"):
        lines.append("")
        frame = local_frame(case.xyz)
        start = 0
        for chain, length in enumerate(case.lengths, 1):
            if chains > 1:
                lines.append(f"Chain {chain}")
            for i in range(start, start + length):
                lines.append(f"{i + 1} " + " ".join(f"{v:.2f}" for v in frame[i]))
            start += length
    return "\n".join(lines)


def prompt_v3(case, condition="geometry"):
    """Render the plain single-chain pilot text."""
    if condition not in V3_CONDITIONS:
        raise ValueError("v3 supports only geometry and length conditions")
    if len(case.lengths) != 1 or not case.mutable.all() or case.info.get("disulfides"):
        raise ValueError("v3 requires one chain with all positions editable and no specified disulfides")
    n = case.lengths[0]
    instruction = f"Choose one amino acid for each of the {n} positions so the chain is likely to fold"
    lines = [
        instruction + (" into the shape shown." if condition == "geometry" else "."),
        f"Return one {n}-letter sequence. Use only {AA}.",
    ]
    if condition == "geometry":
        lines.extend(["", "Cα coordinates in Å (position x y z)"])
        for i, row in enumerate(local_frame(case.xyz), 1):
            lines.append(f"{i} " + " ".join(f"{v:.2f}" for v in row))
    return "\n".join(lines)


def _require_generated_monomer(case):
    path = Path(case.info["path"])
    if (
        case.info.get("generator") not in {"genie3", "rfd4-proteina"}
        or path.parent != Path("data/generated")
        or len(case.lengths) != 1
        or not case.mutable.all()
        or case.info.get("disulfides")
    ):
        raise ValueError("v4 requires a generated single chain with all positions editable")


def condition_case(case, condition):
    """Use a geometry-selected interior mask on the generated parent case."""
    if condition != "masked":
        return case

    from .topology import contact_pairs

    _require_generated_monomer(case)
    n = case.lengths[0]
    sequence = case.info.get("sequence")
    if n < 48 or not case.validate(sequence):
        raise ValueError("masked v4 requires a generated chain of at least 48 residues and a reference sequence")
    contacts = contact_pairs(case.xyz, case.lengths)
    degrees = np.bincount(np.asarray(contacts, dtype=int).ravel(), minlength=n + 1)[1:]
    radial = np.linalg.norm(case.xyz - case.xyz.mean(axis=0), axis=1)
    candidates = [i for i in range(3, n - 3) if degrees[i]]
    count = min(18, max(12, n // 5))
    if len(candidates) < count:
        raise ValueError(f"Not enough interior positions with nonlocal contacts for {case.id}")
    ranked = sorted(candidates, key=lambda i: (-degrees[i], radial[i], i))
    editable = np.zeros(n, dtype=bool)
    editable[ranked[:count]] = True
    template = "".join("?" if edit else aa for edit, aa in zip(editable, sequence, strict=True))
    return replace(
        case,
        info={**case.info, "mutable": (np.flatnonzero(editable) + 1).tolist()},
        template=template,
        mutable=editable,
    )


def prompt_v4(case, condition="geometry", *, version="v4"):
    """Format comparable prompts for the generated single-chain shape benchmark."""
    from .backbone import side_chain_directions
    from .topology import CONTACT_DISTANCE, MIN_SAME_CHAIN_SEPARATION, contact_pairs

    permitted = v4_conditions(version)
    if condition not in permitted:
        raise ValueError(f"{version} supports only {', '.join(permitted)} conditions")
    _require_generated_monomer(case)
    case = condition_case(case, condition)
    n = case.lengths[0]
    if version == "v4.3":
        objective = (
            f"Choose one amino acid for each of the {n} positions in this single protein chain. "
            "We will predict its structure from your sequence alone and compare its Cα trace "
            "with the supplied coordinates at corresponding positions."
        )
    elif version == "v4.2":
        objective = (
            f"Choose one amino acid for each of the {n} positions. "
            "If a target backbone is supplied, choose a sequence whose predicted Cα trace "
            "matches the target as closely as possible at corresponding residue positions. "
            "Otherwise, choose a sequence likely to fold."
        )
    elif version == "v4.1":
        objective = (
            f"Choose one amino acid for each of the {n} positions. "
            "If a target shape is supplied, choose a protein sequence whose predicted fold matches it as closely as possible."
        )
    else:
        objective = f"Choose one amino acid for each of the {n} positions so the protein chain is likely to fold. Use any shape information supplied."
    lines = [objective]
    if version in {"v4.2", "v4.3"}:
        lines.append("Work directly from this message and reply once. Do not use tools.")
    answer_format = f"Return one {n}-letter sequence. Use only {AA}."
    if version == "v4.3":
        answer_format = f"Return exactly {n} uppercase ASCII letters from {AA} on one line, with no spaces or explanation."
    lines.extend([answer_format, ""])
    if condition == "length":
        lines.extend(["Positions in sequence order", " ".join(map(str, range(1, n + 1)))])
        return "\n".join(lines)

    if condition == "masked":
        lines.extend([
            "Keep each given amino acid and choose an amino acid for each ?. Return the full sequence.",
            f"Template with {int(case.mutable.sum())} editable interior positions",
            case.template,
            "",
        ])
    frame = local_frame(case.xyz)
    header = "Cα coordinates in Å (position x y z)"
    if condition in {"vectors", "vectors-contacts", "masked"}:
        if version in {"v4.1", "v4.2", "v4.3"}:
            lines.append(
                "The vector (dx, dy, dz) gives the estimated unit direction from the backbone toward the side chain, "
                "in the same axes as x, y, z. It is derived from the backbone N, Cα and C atoms."
            )
            header = "Cα coordinates in Å and side-chain directions (position x y z dx dy dz)"
        else:
            lines.append("Each virtual Cβ unit direction starts at Cα and uses the same axes as the coordinates. It is derived from the backbone N, Cα and C atoms.")
            header = "Cα coordinates in Å and virtual Cβ unit directions (position x y z dx dy dz)"
        frame = np.column_stack([frame, side_chain_directions(case)])
    lines.append(header)
    for i, row in enumerate(frame, 1):
        lines.append(f"{i} " + " ".join(f"{v:.2f}" for v in row))

    if condition == "vectors-contacts":
        contacts = contact_pairs(case.xyz, case.lengths)
        lines.extend([
            "",
            f"Cα contact pairs within {CONTACT_DISTANCE:g} Å and separated by at least {MIN_SAME_CHAIN_SEPARATION} positions",
            ", ".join(f"{i}-{j}" for i, j in contacts) or "None",
        ])
    return "\n".join(lines)


def prompt_v1(case, condition="geometry"):
    """Frozen text of the 2026-11-18 pilot. Audits of runs planned with it need this exact text."""
    lines = [
        "Choose amino acids for a compact protein. Return only the complete sequence in uppercase one-letter codes.",
        "Use a colon between chains. Keep each specified residue. Each ? marks one residue you choose.",
        "Chain lengths: " + ", ".join(map(str, case.lengths)),
        "Sequence mask: " + case.split(case.template),
    ]
    if case.info.get("disulfides"):
        pairs = ", ".join(f"{a}-{b}" for a, b in case.info["disulfides"])
        lines.append("Fixed disulfide pairs (positions across all chains): " + pairs)
    if condition == "geometry":
        lines.append(
            "Target C-alpha coordinates in angstroms, in sequence order. Columns: position x y z."
        )
        for i, p in enumerate(local_frame(case.xyz), 1):
            lines.append(f"{i} " + " ".join(f"{v:.2f}" for v in p))
    elif condition != "length":
        raise ValueError("unknown condition")
    return "\n".join(lines)
