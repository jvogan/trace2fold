"""Unit tests for diagnostic grading invariants (invalid=0, ungraded preserved, target hash integrity)."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from trace2fold.cases import THREE, digest, load_cases, prompt
from trace2fold.trials import lock
from trace2fold.v4_eval import grade_v4, load_submissions


def _write_ca_pdb(path, seq, xyz):
    res = {aa: name for name, aa in THREE.items()}
    text = "".join(
        f"ATOM  {i:5d}  CA  {res[aa]} A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 80.00           C\n"
        for i, (aa, (x, y, z)) in enumerate(zip(seq, xyz, strict=True), 1)
    )
    path.write_text(text, encoding="ascii")


class DiagnosticScoringTests(unittest.TestCase):
    def test_invalid_counts_zero_ungraded_preserved_and_target_hash_checked(self):
        cases = load_cases()
        case_a = cases["gen-l48-01-a"]
        case_b = cases["gen-l48-01-b"]
        with TemporaryDirectory() as directory:
            base = Path(directory)
            recs = base / "records"
            recs.mkdir()
            folds_esm = base / "folds" / "esmfold2"
            folds_esm.mkdir(parents=True)
            folds_boltz = base / "folds" / "boltz2"
            folds_boltz.mkdir(parents=True)

            seq_valid_graded = case_a.info["sequence"]
            seq_valid_ungraded = "A" * len(case_b.xyz)
            _write_ca_pdb(
                folds_esm / f"{digest(seq_valid_graded)[:32]}.pdb",
                seq_valid_graded,
                case_a.xyz,
            )
            _write_ca_pdb(
                folds_boltz / f"{digest(seq_valid_graded)[:32]}.pdb",
                seq_valid_graded,
                case_a.xyz,
            )

            lock(
                case_a,
                recs / "01-graded.json",
                "vectors",
                seq_valid_graded,
                {"model": "m1"},
                task=prompt(case_a, "vectors", "v4.2"),
                version="v4.2",
            )
            lock(
                case_b,
                recs / "02-invalid.json",
                "vectors",
                "wrong length",
                {"model": "m1"},
                task=prompt(case_b, "vectors", "v4.2"),
                version="v4.2",
            )
            lock(
                case_b,
                recs / "03-ungraded.json",
                "vectors",
                seq_valid_ungraded,
                {"model": "m2"},
                task=prompt(case_b, "vectors", "v4.2"),
                version="v4.2",
            )

            submissions = load_submissions(recs)
            result = grade_v4(cases, submissions, base / "folds")
            by_model = {s["model"]: s for s in result["summaries"]}

            # m1: 1 graded (100.0) + 1 invalid (0.0) -> mean_score = 50.0
            self.assertEqual(by_model["m1"]["graded"], 1)
            self.assertEqual(by_model["m1"]["invalid"], 1)
            self.assertAlmostEqual(by_model["m1"]["mean_score"], 50.0, places=4)

            # m2: 1 ungraded -> mean_score is None (never silently dropped or zeroed)
            self.assertEqual(by_model["m2"]["ungraded"], 1)
            self.assertIsNone(by_model["m2"]["mean_score"])

            # Tampered target_sha256 must raise
            tampered = [dict(submissions[0], target_sha256="0" * 64)]
            with self.assertRaisesRegex(ValueError, "target_sha256 mismatch"):
                grade_v4(cases, tampered, base / "folds")
