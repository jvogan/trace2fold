"""Export an anonymous target and aligned all-atom submissions to PyMOL and ChimeraX."""

from pathlib import Path
import re
import numpy as np
from .cases import digest
from .metrics import measure, rigid_fit
from .report import read_fold


def target_trace(path, coords, lengths):
    rows, i = [], 0
    for chain, length in enumerate(lengths):
        for j in range(length):
            x, y, z = coords[i]
            rows.append(
                f"ATOM  {i + 1:5d}  CA  GLY {chr(65 + chain)}{j + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
            )
            i += 1
        rows.append("TER")
    path.write_text("\n".join(rows + ["END", ""]))


def aligned_pdb(original, destination, rotation, offset):
    lines = []
    for line in original.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            point = (
                np.array([float(line[k : k + 8]) for k in (30, 38, 46)]) @ rotation
                + offset
            )
            line = line[:30] + "".join(f"{v:8.3f}" for v in point) + line[54:]
        lines.append(line)
    destination.write_text("\n".join(lines) + "\n")


def export(case, records, folds, out, *, profile=None):
    out = Path(out)
    layers = []
    for i, row in enumerate(records, 1):
        if row["case"] != case.id or not row["valid"]:
            raise ValueError("views require valid submissions for the selected case")
        xyz = read_fold(folds, row["sequence"])
        if xyz is None:
            raise ValueError("missing folded submission")
        metrics, aligned = measure(xyz, case, profile=profile)
        rotation, offset = rigid_fit(xyz, aligned, np.ones(len(xyz), dtype=bool))
        name = f"trial_{i}"
        if profile is not None:
            role = re.sub(r"[^a-zA-Z0-9_]", "_", row["protocol"]["model"])
            label = re.sub(r"[^a-zA-Z0-9_]", "_", row["condition"])
            name = f"{name}_{role}_{label}"
        layers.append((name, row, metrics, rotation, offset))
    out.mkdir(parents=True, exist_ok=False)
    target_trace(out / "target.pdb", case.xyz, case.lengths)
    colors = ["marine", "orange", "forest", "violet"]
    cmds = [
        "bg_color white",
        "set orthoscopic, on",
        "load target.pdb, target",
        "set cartoon_trace_atoms, 1, target",
        "cartoon tube, target",
        "color gray65, target",
        "set cartoon_tube_radius, 0.35, target",
        "hide everything",
        "show cartoon, target",
        "set specular, 0.2",
        "set ray_shadows, off",
        "set antialias, 2",
    ]
    chimerax = [
        "set bgColor white",
        "open target.pdb",
        "show #1 atoms",
        "style #1 sphere",
        "size #1 atomRadius 0.45",
        "color #1 gray",
    ]
    details = [
        "Target in gray. In PyMOL, use the saved scenes to step through submissions.",
        "",
    ]
    for i, (name, row, metrics, rotation, offset) in enumerate(layers):
        pdb = Path(folds) / (digest(row["sequence"])[:32] + ".pdb")
        aligned_pdb(pdb, out / (name + ".pdb"), rotation, offset)
        color = colors[i % len(colors)]
        cmds += [
            f"load {name}.pdb, {name}",
            f"hide everything, {name}",
            f"color {color}, {name}",
            f"show cartoon, {name}",
        ]
        chimerax += [
            f"open {name}.pdb",
            *([f"rename #{i + 2} {name}"] if profile is not None else []),
            f"hide #{i + 2} atoms",
            f"cartoon #{i + 2}",
            f"color #{i + 2} {['blue', 'orange', 'green', 'violet'][i % 4]}",
        ]
        details.append(
            f"{name} ({color})  {row['protocol']['model']}  {row['condition']}  score {metrics['score']:.1f}/100  RMSD {metrics['rmsd']:.2f} Å"
        )
        details.append(row["sequence"])
    cmds += [
        "set seq_view, on",
        "viewport 1200, 900",
        "orient all",
        "zoom all, 7",
        "disable trial_*",
        "scene target_only, store",
    ]
    for name, *_ in layers:
        cmds += ["disable trial_*", f"enable {name}", f"scene {name}, store"]
    cmds += ["enable trial_*", "scene overlay, store"]
    if profile is not None and layers:
        first = next((i for i, layer in enumerate(layers) if layer[1]["condition"] == "vectors"), 0)
        cmds.append(f"scene {layers[first][0]}, recall")
        for i in range(len(layers)):
            if i != first:
                chimerax.append(f"hide #{i + 2} models")
    cmds.append("save overlay.pse")
    chimerax += ["view", "save overlay.cxs"]
    (out / "overlay.pml").write_text("\n".join(cmds) + "\n")
    (out / "overlay.cxc").write_text("\n".join(chimerax) + "\n")
    (out / "legend.txt").write_text("\n".join(details) + "\n")
