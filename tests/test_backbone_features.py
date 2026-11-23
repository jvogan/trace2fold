"""Check coordinate alignment, feature definitions, and source provenance."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from backends.freeze_backbones import freeze_panel
from trace2fold.backbone import backbone_text
from trace2fold.cases import load_cases, local_frame


def _rows(text):
    return [line.split() for line in text.splitlines() if line[:1].isdigit()]


def _pdb(atoms, positions):
    xyz = [(0, 0, 0), (3, 0, 0), (4, 2, 0)]
    rows = []
    for i in positions:
        for j, atom in enumerate(atoms):
            x, y, z = xyz[i]
            rows.append(
                f"ATOM  {len(rows) + 1:5d} {atom:>4s} GLY A{i + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z + (j if atom != 'CA' else 0):8.3f}"
            )
    return "\n".join(rows).encode("ascii") + b"\nEND\n"


class BackboneFeatures(unittest.TestCase):
    def test_known_raw_rows_match_geometry_frame(self):
        cases = load_cases()
        for name in ("trp-cage", "insulin", "gfp", "vhh-cdr3"):
            with self.subTest(case=name):
                case = cases[name]
                text = backbone_text(case)
                self.assertNotIn(name, text)
                self.assertNotIn("SG", text)
                self.assertNotIn("CB", text)
                rows = _rows(text)
                self.assertEqual(
                    [int(row[0]) for row in rows], list(range(1, len(case.xyz) + 1))
                )
                self.assertTrue(all(len(row) == 13 for row in rows))
                np.testing.assert_allclose(
                    [list(map(float, row[4:7])) for row in rows],
                    local_frame(case.xyz),
                    atol=0.005001,
                )

    def test_derived_rows_follow_chain_boundaries_and_geometry(self):
        cases = load_cases()
        text = backbone_text(cases["insulin"], derived=True)
        self.assertIn("Chain 1", text)
        self.assertIn("Chain 2", text)
        rows = _rows(text)
        self.assertEqual([int(row[0]) for row in rows if row[13] == "NA"], [1, 22])
        self.assertEqual([int(row[0]) for row in rows if row[14] == "NA"], [21, 51])
        directions = np.array([list(map(float, row[15:18])) for row in rows])
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1, atol=0.009)
        self.assertEqual(
            _rows(backbone_text(cases["trp-cage"], derived=True))[2][13:15],
            ["-66", "-31"],
        )

    def test_missing_generated_data_and_mismatched_known_source(self):
        cases = load_cases()
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No frozen N/CA/C/O backbone"):
                backbone_text(replace(cases["gen-l48-01-a"], root=Path(directory)))
        case = cases["trp-cage"]
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            backbone_text(replace(case, info={**case.info, "fixture_sha256": "wrong"}))
        with self.assertRaisesRegex(
            ValueError, "do not match the frozen C-alpha coordinates"
        ):
            backbone_text(replace(case, xyz=case.xyz + 1))

    def test_freezer_keeps_verified_atoms_and_skips_ca_only(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            generated = root / "data/generated"
            generated.mkdir(parents=True)
            raw = root / "raw"
            production = raw / "genie3/production"
            production.mkdir(parents=True)
            ca = _pdb(("CA",), range(3))
            full = _pdb(("N", "CA", "C", "O", "CB"), range(3))
            index, provenance = [], []
            for name, source in [("full", full), ("alpha", ca)]:
                fixture = generated / f"{name}.pdb"
                fixture.write_bytes(ca)
                (production / fixture.name).write_bytes(source)
                index.append(
                    {"id": name, "path": f"data/generated/{name}.pdb", "length": 3}
                )
                provenance.append(
                    {
                        "id": name,
                        "generator": "genie3",
                        "source": f"production/{name}.pdb",
                        "source_sha256": sha256(source).hexdigest(),
                        "backbone_sha256": sha256(ca).hexdigest(),
                    }
                )
            (generated / "index.json").write_text(json.dumps(index))
            manifest = generated / "provenance.json"
            manifest.write_text(json.dumps({"selected_details": provenance}))
            frozen = freeze_panel(raw, root)
            self.assertEqual(set(frozen), {"full"})
            self.assertEqual(np.shape(frozen["full"]["coordinates"]), (3, 4, 3))
            (generated / "backbone_atoms.json").write_text(json.dumps(frozen))
            self.assertEqual(len(_rows(backbone_text(load_cases(root)["full"]))), 3)

            reordered = _pdb(("N", "CA", "C", "O"), [0, 2, 1])
            (production / "full.pdb").write_bytes(reordered)
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                freeze_panel(raw, root)
            provenance[0]["source_sha256"] = sha256(reordered).hexdigest()
            manifest.write_text(json.dumps({"selected_details": provenance}))
            with self.assertRaisesRegex(ValueError, "do not match frozen C-alpha rows"):
                freeze_panel(raw, root)
