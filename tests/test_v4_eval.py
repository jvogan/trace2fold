"""Unit and CLI tests for trace2fold.v4_eval (grade-v4 and verify-benchmarks)."""

from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from trace2fold.cases import THREE, digest, dump, load_cases
from trace2fold.cli import main
from trace2fold.v4_eval import (
    grade_v4,
    load_submissions,
    load_twin_map,
    verify_benchmarks,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_pdb(path, seq, xyz, plddt=90.0):
    res = {aa: name for name, aa in THREE.items()}
    lines = [
        f"ATOM  {i:5d}  CA  {res[aa]} A{i:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1:6.2f}{plddt:6.2f}           C\n"
        for i, (aa, (x, y, z)) in enumerate(zip(seq, xyz, strict=True), 1)
    ]
    path.write_text("".join(lines), encoding="ascii")


class V4EvalGradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()
        cls.a = cls.cases["gen-l48-01-a"]
        cls.b = cls.cases["gen-l48-01-b"]

    def test_cli_grade_v4_official_two_primary_judges_and_exploratory_isolation(self):
        seq_a = self.a.info["sequence"]
        seq_b = self.b.info["sequence"]
        rng = np.random.default_rng(7)
        perturbed_a = self.a.xyz + rng.normal(0.0, 0.4, size=self.a.xyz.shape)
        far_b = self.b.xyz + rng.normal(0.0, 4.5, size=self.b.xyz.shape)
        # Deliberately bad SimpleFold coordinates for case A to verify isolation
        bad_sf_a = self.a.xyz + rng.normal(0.0, 6.0, size=self.a.xyz.shape)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            subs_path = root / "submissions.jsonl"
            subs_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "case": self.a.id,
                                "model": "demo-llm",
                                "raw": f"Final sequence:\n```\n{seq_a}\n```",
                            }
                        ),
                        json.dumps(
                            {
                                "case": self.b.id,
                                "model": "demo-llm",
                                "raw": seq_b,
                            }
                        ),
                    ]
                )
                + "\n"
            )
            folds = root / "folds"
            # Primary Predictor 1: esmfold2
            esm = folds / "esmfold2"
            esm.mkdir(parents=True)
            _write_pdb(esm / f"{digest(seq_a)[:32]}.pdb", seq_a, perturbed_a, plddt=88.0)
            dump(
                esm / f"{digest(seq_a)[:32]}.json",
                {"sequence": seq_a, "ca": perturbed_a.tolist(), "plddt": [88.0] * len(seq_a), "ptm": 0.89},
            )
            _write_pdb(esm / f"{digest(seq_b)[:32]}.pdb", seq_b, far_b, plddt=42.0)

            # Primary Predictor 2: boltz2
            boltz = folds / "boltz2"
            boltz.mkdir(parents=True)
            _write_pdb(boltz / f"{digest(seq_a)[:32]}.pdb", seq_a, perturbed_a, plddt=90.0)
            _write_pdb(boltz / f"{digest(seq_b)[:32]}.pdb", seq_b, far_b, plddt=40.0)

            # Exploratory Predictor: simplefold (must NOT affect consensus score or both_within_2a)
            sf = folds / "simplefold" / "structures"
            sf.mkdir(parents=True)
            _write_pdb(sf / f"{digest(seq_a)}.pdb", seq_a, bad_sf_a, plddt=0.0)
            _write_pdb(sf / f"{digest(seq_b)}.pdb", seq_b, far_b, plddt=0.0)

            out_path = root / "report.json"
            with patch(
                "sys.argv",
                [
                    "trace2fold",
                    "grade-v4",
                    "--cases-root",
                    str(self.a.root / "data"),
                    "--submissions",
                    str(subs_path),
                    "--folds",
                    str(folds),
                    "--out",
                    str(out_path),
                ],
            ):
                main()

            report = json.loads(out_path.read_text())
            self.assertTrue(report["official_protocol"])
            self.assertEqual(report["scoring_profile"], "spacefill-v4")
            self.assertEqual(report["answer_parser"], "length-aware-v2")
            self.assertEqual(report["primary_predictors"], ["esmfold2", "boltz2"])
            self.assertEqual(report["exploratory_predictors"], ["simplefold"])
            row_a, row_b = report["rows"]
            self.assertTrue(row_a["both_within_2a"])
            self.assertGreater(row_a["score"], 85.0)
            self.assertGreater(row_a["twin_delta"], 30.0)
            self.assertAlmostEqual(row_a["judges"]["esmfold2"]["ptm"], 0.89)
            self.assertIn("simplefold", row_a["exploratory_predictors"])
            self.assertLess(row_a["exploratory_predictors"]["simplefold"]["score"], 50.0)
            self.assertFalse(row_b["both_within_2a"])
            self.assertGreater(row_b["rmsd"], 2.0)
            summary = report["summaries"][0]
            self.assertEqual(summary["model"], "demo-llm")
            self.assertEqual(summary["both_within_2a_hits"], 1)
            self.assertEqual(summary["paired"]["complete_pairs"], 1)
            self.assertEqual(summary["twin_margin_graded_n"], 2)
            self.assertEqual(summary["twin_margin_assigned_n"], 2)
            self.assertAlmostEqual(summary["one_sided_sign_p"], 0.5)
            self.assertAlmostEqual(summary["two_sided_sign_p"], 1.0)

    def test_official_protocol_requires_both_primary_judges_and_partial_flag_allows_single(self):
        seq_a = self.a.info["sequence"]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdir = root / "folds" / "boltz2"
            pdir.mkdir(parents=True)
            dump(
                pdir / f"{digest(seq_a)[:32]}.json",
                {"sequence": seq_a, "ca": self.a.xyz.tolist(), "plddt": [92.0] * len(seq_a), "ptm": 0.91},
            )
            subs = [{"case": self.a.id, "model": "m", "condition": "vectors", "raw": seq_a}]

            with self.assertRaisesRegex(ValueError, "requires both primary predictors"):
                grade_v4(self.cases, subs, root / "folds")

            # Also fails official protocol if only esmfold2 + simplefold are present
            sf_dir = root / "folds" / "simplefold"
            sf_dir.mkdir(parents=True)
            _write_pdb(sf_dir / f"{digest(seq_a)[:32]}.pdb", seq_a, self.a.xyz)
            with self.assertRaisesRegex(ValueError, "requires both primary predictors"):
                grade_v4(self.cases, subs, root / "folds")

            # Allowed with allow_partial_judges=True, and marks official_protocol=False
            report = grade_v4(
                self.cases,
                subs,
                root / "folds",
                allow_partial_judges=True,
            )
            self.assertFalse(report["official_protocol"])
            self.assertAlmostEqual(report["rows"][0]["score"], 100.0, places=5)
            self.assertFalse(report["rows"][0]["both_within_2a"])
            self.assertAlmostEqual(report["rows"][0]["judges"]["boltz2"]["mean_plddt"], 92.0)
            self.assertIn("simplefold", report["rows"][0]["exploratory_predictors"])

    def test_duplicate_slot_rejected(self):
        seq_a = self.a.info["sequence"]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for pred in ("esmfold2", "boltz2"):
                pdir = root / "folds" / pred
                pdir.mkdir(parents=True)
                _write_pdb(pdir / f"{digest(seq_a)[:32]}.pdb", seq_a, self.a.xyz)
            subs = [
                {"case": self.a.id, "model": "dup-model", "condition": "vectors", "raw": seq_a},
                {"case": self.a.id, "model": "dup-model", "condition": "vectors", "raw": seq_a},
            ]
            with self.assertRaisesRegex(ValueError, "Duplicate submission slot"):
                grade_v4(self.cases, subs, root / "folds")

    def test_twin_map_matches_starter_pilot_pairs_and_assigned_vs_graded_margin(self):
        pairs_path = REPO_ROOT / "examples" / "starter_pilot_pairs.json"
        twin_map = load_twin_map(pairs_path)
        self.assertEqual(twin_map["gen-l48-02-b"], "gen-l48-03-b")
        c2b = self.cases["gen-l48-02-b"]
        c3b = self.cases["gen-l48-03-b"]
        seq_2b = c2b.info["sequence"]
        rng = np.random.default_rng(20261122)
        synth_2b = c2b.xyz + rng.normal(0.0, 0.6, size=c2b.xyz.shape)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            subs_path = root / "subs.json"
            dump(
                subs_path,
                [
                    {"case": c2b.id, "model": "pilot-m", "condition": "vectors", "raw": seq_2b},
                    # Invalid reply on twin mate to test graded vs assigned Twin Delta denominators
                    {"case": c3b.id, "model": "pilot-m", "condition": "vectors", "raw": "INVALID"},
                ],
            )
            for pred in ("esmfold2", "boltz2"):
                pdir = root / "folds" / pred
                pdir.mkdir(parents=True)
                _write_pdb(pdir / f"{digest(seq_2b)[:32]}.pdb", seq_2b, synth_2b)

            out_path = root / "report.json"
            with patch(
                "sys.argv",
                [
                    "trace2fold",
                    "grade-v4",
                    "--submissions",
                    str(subs_path),
                    "--folds",
                    str(root / "folds"),
                    "--twin-map",
                    str(pairs_path),
                    "--out",
                    str(out_path),
                ],
            ):
                main()

            report = json.loads(out_path.read_text())
            row_2b, row_3b = report["rows"]
            self.assertEqual(row_2b["twin"], "gen-l48-03-b")
            self.assertEqual(row_3b["twin"], "gen-l48-02-b")
            self.assertEqual(row_3b["status"], "invalid")
            summary = report["summaries"][0]
            self.assertEqual(summary["twin_margin_graded_n"], 1)
            self.assertEqual(summary["twin_margin_assigned_n"], 2)
            self.assertAlmostEqual(
                summary["twin_margin_assigned_mean"],
                summary["twin_margin_graded_mean"] / 2.0,
                places=5,
            )

    def test_load_submissions_accepts_list_and_dict_wrappers(self):
        with TemporaryDirectory() as tmp:
            fp = Path(tmp) / "subs.json"
            dump(fp, {"submissions": [{"case": self.a.id, "model": "m", "raw": "AAAA"}]})
            rows = load_submissions(fp)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["condition"], "vectors")

    def test_cli_plan_supports_comma_separated_cases_flag(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "pilot"
            with patch(
                "sys.argv",
                [
                    "trace2fold",
                    "plan",
                    str(run_dir),
                    "--cases",
                    "gen-l48-02-b,gen-l48-03-b",
                    "--model",
                    "demo-model",
                    "--endpoint",
                    "http://127.0.0.1:8000/v1/chat/completions",
                    "--conditions",
                    "vectors",
                    "--prompt-version",
                    "v4.3",
                ],
            ):
                main()
            design = json.loads((run_dir / "design.json").read_text())
            self.assertEqual(design["case_ids"], ["gen-l48-02-b", "gen-l48-03-b"])
            self.assertEqual(design["condition_order"], ["vectors"])
            self.assertEqual(design["calls_planned"], 2)


class VerifyBenchmarksTests(unittest.TestCase):
    def _write_valid_bundle(self, bdir):
        scores = {
            "schema": "stage-one-v1",
            "targets": [
                {"id": "release-l48-01-a", "length": 48, "pair": "p48-01", "twin_id": "release-l48-01-b"},
                {"id": "release-l48-01-b", "length": 48, "pair": "p48-01", "twin_id": "release-l48-01-a"},
            ],
            "models": [
                {
                    "model": "Gemini 4 Neon High",
                    "mean_score": 53.35,
                    "score_bounds": [50.1, 56.4],
                    "twin_margin_bounds": [8.2, 22.5],
                    "lopo_range": [45.0, 58.0],
                }
            ],
        }
        pub3d = {
            "schema": "public-3d-v1",
            "cases": [
                {
                    "id": "gen-l48-02-b",
                    "sequence": "M" * 48,
                    "xyz": [[0.0, 0.0, 0.0]],
                }
            ],
        }
        dump(bdir / "stage-one-scores.json", scores)
        dump(bdir / "public-development-3d.json", pub3d)
        manifest = {
            "schema": "manifest-v1",
            "files": {
                "stage-one-scores.json": {
                    "sha256": sha256((bdir / "stage-one-scores.json").read_bytes()).hexdigest()
                },
                "public-development-3d.json": {
                    "sha256": sha256((bdir / "public-development-3d.json").read_bytes()).hexdigest()
                },
            },
        }
        dump(bdir / "manifest.json", manifest)

    def test_valid_benchmarks_pass_cli_and_api(self):
        with TemporaryDirectory() as tmp:
            bdir = Path(tmp) / "benchmarks"
            bdir.mkdir()
            self._write_valid_bundle(bdir)
            result = verify_benchmarks(bdir)
            self.assertTrue(result["ok"], result["errors"])
            self.assertEqual(len(result["files_checked"]), 3)
            self.assertGreaterEqual(result["twin_checks"], 2)

            buf = StringIO()
            with (
                patch("sys.argv", ["trace2fold", "verify-benchmarks", "--benchmarks-dir", str(bdir)]),
                redirect_stdout(buf),
            ):
                main()
            parsed = json.loads(buf.getvalue())
            self.assertTrue(parsed["ok"])

    def test_require_complete_suite_rejects_partial_dir_and_passes_real_benchmarks(self):
        with TemporaryDirectory() as tmp:
            bdir = Path(tmp) / "benchmarks"
            bdir.mkdir()
            self._write_valid_bundle(bdir)
            strict_res = verify_benchmarks(bdir, require_complete_suite=True)
            self.assertFalse(strict_res["ok"])
            self.assertTrue(
                any("missing required file" in e for e in strict_res["errors"])
            )

        real_bdir = REPO_ROOT / "benchmarks"
        real_res = verify_benchmarks(real_bdir, require_complete_suite=True)
        self.assertTrue(real_res["ok"], real_res["errors"])
        self.assertTrue(real_res["complete_suite_checked"])

    def test_catches_bad_bounds_asymmetric_twin_and_holdout_leaks(self):
        with TemporaryDirectory() as tmp:
            bdir = Path(tmp) / "benchmarks"
            bdir.mkdir()
            # 1. Out-of-range / inverted bounds
            dump(
                bdir / "bad-bounds.json",
                {"models": [{"score_bounds": [60.0, 50.0]}, {"lopo_range": [-5.0, 105.0]}]},
            )
            res = verify_benchmarks(bdir)
            self.assertFalse(res["ok"])
            self.assertTrue(any("inverted bounds" in e for e in res["errors"]))
            self.assertTrue(any("outside [0, 100]" in e for e in res["errors"]))

        with TemporaryDirectory() as tmp:
            bdir = Path(tmp) / "benchmarks"
            bdir.mkdir()
            # 2. Asymmetric twin link
            dump(
                bdir / "bad-twin.json",
                {
                    "targets": [
                        {"id": "release-l48-01-a", "length": 48, "twin_id": "release-l48-01-b"},
                        {"id": "release-l48-01-b", "length": 48, "twin_id": "release-l48-02-a"},
                    ]
                },
            )
            res = verify_benchmarks(bdir)
            self.assertFalse(res["ok"])
            self.assertTrue(any("asymmetric twin link" in e for e in res["errors"]))

        with TemporaryDirectory() as tmp:
            bdir = Path(tmp) / "benchmarks"
            bdir.mkdir()
            # 3. Holdout coordinate & sequence leak
            dump(
                bdir / "leak.json",
                {
                    "cells": [
                        {
                            "id": "stage2-l176-01-a",
                            "xyz": [[1.0, 2.0, 3.0]],
                            "sequence": "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",
                        }
                    ]
                },
            )
            res = verify_benchmarks(bdir)
            self.assertFalse(res["ok"])
            self.assertTrue(any("forbidden coordinate/sequence key" in e for e in res["errors"]))
            self.assertTrue(any("amino-acid sequence leak" in e for e in res["errors"]))
