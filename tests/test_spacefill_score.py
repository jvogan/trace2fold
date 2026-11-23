"""Unit tests for spacefill-v4 scoring terms, chirality enforcement, and multi-judge evaluation."""

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from trace2fold.cases import THREE, condition_case, digest, dump, load_cases
from trace2fold.metrics import SPACEFILL_PROFILE, measure, rigid_fit
from trace2fold.v4_eval import grade_v4


def _write_pdb(path, sequence, xyz, plddt=85.0):
    residue_name = {aa: name for name, aa in THREE.items()}
    lines = [
        f"ATOM  {i:5d}  CA  {residue_name[aa]} A{i:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1:6.2f}{plddt:6.2f}           C\n"
        for i, (aa, (x, y, z)) in enumerate(zip(sequence, xyz, strict=True), 1)
    ]
    payload = "".join(lines).encode("ascii")
    path.write_bytes(payload)
    return payload


class SpaceFillScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = load_cases()
        cls.a = cls.library["gen-l48-02-b"]
        cls.b = cls.library["gen-l48-03-b"]

    def test_sharpened_terms_are_opt_in_and_fixed_alignment_uses_editable_positions(self):
        case = condition_case(self.a, "masked")
        shifted = case.xyz.copy()
        shifted[case.mutable] += [3.0, 0, 0]
        old, _ = measure(shifted, case)
        new, _ = measure(shifted, case, profile=SPACEFILL_PROFILE)
        self.assertAlmostEqual(new["rmsd"], 3.0)
        self.assertEqual(new["positions_scored"], int(case.mutable.sum()))
        self.assertEqual(old["aligned_coverage"], 0.5)
        self.assertEqual(new["aligned_coverage"], 0.25)
        expected = 40 * 0.25 + 25 * new["ca_lddt"] + 35 * new["contact_f1"]
        self.assertAlmostEqual(new["score"], expected)
        with self.assertRaisesRegex(ValueError, "one chain"):
            insulin = self.library["insulin"]
            measure(insulin.xyz, insulin, profile=SPACEFILL_PROFILE)

    def test_kabsch_rigid_fit_preserves_chirality_det_plus_one(self):
        mirrored = self.a.xyz.copy()
        mirrored[:, 2] *= -1.0
        rotation, _ = rigid_fit(mirrored, self.a.xyz, np.ones(len(mirrored), dtype=bool))
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=6)
        mirrored_metrics, _ = measure(mirrored, self.a, profile=SPACEFILL_PROFILE)
        self.assertGreater(mirrored_metrics["rmsd"], 2.0)

    def test_v4_multi_judge_scoring_with_twin_delta_and_masked_and_invalid(self):
        # Use a genuine paired pair (gen-l48-01-a and gen-l48-01-b)
        pair_a = self.library["gen-l48-01-a"]
        pair_b = self.library["gen-l48-01-b"]
        seq_a = "A" * len(pair_a.xyz)
        seq_mask = pair_a.info["sequence"]

        submissions = [
            {"case": pair_a.id, "model": "demo", "condition": "vectors", "raw": seq_a},
            {"case": pair_b.id, "model": "demo", "condition": "vectors", "raw": "bad-reply"},
            {"case": pair_a.id, "model": "demo", "condition": "masked", "raw": seq_mask},
        ]
        with TemporaryDirectory() as root_str:
            root = Path(root_str)
            for pred_name, conf in (("esmfold2", 30.0), ("boltz2", 95.0)):
                pdir = root / pred_name
                pdir.mkdir()
                for seq, xyz in ((seq_a, pair_a.xyz), (seq_mask, pair_a.xyz)):
                    stem = digest(seq)[:32]
                    pdb_bytes = _write_pdb(pdir / f"{stem}.pdb", seq, xyz, plddt=conf)
                    dump(
                        pdir / f"{stem}.json",
                        {
                            "id": stem,
                            "sequence": seq,
                            "ca": xyz.tolist(),
                            "plddt": [conf] * len(seq),
                            "ptm": conf / 100.0,
                            "pdb_sha256": sha256(pdb_bytes).hexdigest(),
                        },
                    )

            report = grade_v4(self.library, submissions, root)

        vec_a, vec_b, masked_a = report["rows"]
        self.assertEqual(vec_a["status"], "graded")
        self.assertAlmostEqual(vec_a["score"], 100.0, places=5)
        self.assertTrue(vec_a["both_within_2a"])
        self.assertGreater(vec_a["twin_delta"], 0.0)
        self.assertEqual(vec_a["judges"]["esmfold2"]["mean_plddt"], 30.0)
        self.assertEqual(vec_a["judges"]["boltz2"]["mean_plddt"], 95.0)
        self.assertEqual(vec_b["status"], "invalid")
        self.assertEqual(vec_b["score"], 0.0)
        self.assertFalse(vec_b["both_within_2a"])
        self.assertEqual(masked_a["status"], "graded")
        self.assertIsNone(masked_a["twin_delta"])
        self.assertEqual(
            masked_a["judges"]["esmfold2"]["positions_scored"],
            int(condition_case(pair_a, "masked").mutable.sum()),
        )
