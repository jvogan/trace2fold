import json
import tempfile
import unittest
from pathlib import Path


from trace2fold.cases import digest, load_cases, prompt
from trace2fold.metrics import measure
from trace2fold.player import audit_muse
from trace2fold.report import paired_statistics
from trace2fold.trials import controls, fold_inputs, lock


class BenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()

    def test_geometry_only(self):
        for case in self.cases.values():
            task = prompt(case)
            self.assertNotIn(case.id, task)
            self.assertNotIn(case.info["path"], task)
            if case.mutable.all() and case.info.get("sequence"):
                self.assertNotIn(case.info["sequence"], task)
            self.assertEqual(
                len([s for s in task.splitlines() if s[:1].isdigit()]),
                sum(case.lengths),
            )
            self.assertNotIn("coordinates", prompt(case, "length"))
            if case.info.get("sequence"):
                for seq in controls(case).values():
                    self.assertTrue(case.validate(seq))

    def test_prompt_is_three_plain_lines_and_the_coordinates(self):
        case = self.cases["gen-l48-01-a"]
        lines = prompt(case).splitlines()
        self.assertEqual(
            lines[0],
            "Write an amino-acid sequence for a protein with 48 residues that folds into this backbone.",
        )
        self.assertEqual(len(lines), 3 + 1 + 48)
        self.assertEqual(lines[3], "")
        self.assertNotIn("Template", "\n".join(lines))
        self.assertNotIn("?", "\n".join(lines))
        self.assertEqual(
            prompt(case, "length").splitlines(),
            [
                "Write an amino-acid sequence for a protein with 48 residues.",
                "Reply with the sequence only, in one-letter codes (ACDEFGHIKLMNPQRSTVWY).",
            ],
        )
        # The length-only control is one prompt for both members of a pair.
        self.assertEqual(
            prompt(case, "length"), prompt(self.cases["gen-l48-01-b"], "length")
        )

    def test_fixed_residues_and_chains_add_only_what_the_answer_needs(self):
        insulin = prompt(self.cases["insulin"])
        for text in ("Template: ", "Chain 1", "Chain 2", "colon", "6-11, 7-28, 20-40"):
            self.assertIn(text, insulin)
        trp = prompt(self.cases["trp-cage"])
        self.assertNotIn("Template", trp)
        self.assertNotIn("colon", trp)
        vhh = prompt(self.cases["vhh-cdr3"], "length")
        self.assertIn("Template: QVQLQESGGG", vhh)
        self.assertNotIn("coordinate", vhh)

    def test_pilot_prompt_text_is_frozen(self):
        pinned = {
            ("trp-cage", "geometry"): "d5ff311ceb9d1dd8",
            ("trp-cage", "length"): "6504a6ee51f4f4d7",
            ("insulin", "geometry"): "3144226ad21b8630",
            ("insulin", "length"): "4fd687196618647a",
            ("vhh-cdr3", "geometry"): "cb2efe85e3379404",
            ("vhh-cdr3", "length"): "0e059a47ef533185",
            ("gfp", "geometry"): "ac85c2a77fed0757",
            ("gen-l48-01-a", "geometry"): "1ab2dd8487883bec",
            ("gen-l48-01-b", "geometry"): "e27899b137d8dc12",
            ("gen-l48-01-a", "length"): "6b4564490be4d17b",
        }
        for (case_id, condition), expected in pinned.items():
            text = prompt(self.cases[case_id], condition, version="v1")
            self.assertEqual(digest(text)[:16], expected, (case_id, condition))
        with self.assertRaises(ValueError):
            prompt(self.cases["trp-cage"], version="v0")

    def test_chain_boundaries_and_loop_framework(self):
        case = self.cases["insulin"]
        seq = controls(case)["positive"]
        self.assertTrue(case.validate(seq))
        self.assertFalse(case.validate(seq.replace(":", "")))
        self.assertFalse(case.validate(seq.replace("C", "A")))
        vhh = self.cases["vhh-cdr3"]
        seq = controls(vhh)["positive"]
        self.assertLess(vhh.mutable.sum(), len(vhh.xyz))
        self.assertFalse(vhh.validate(("A" if seq[0] != "A" else "G") + seq[1:]))

    def test_rigid_fit_and_reflection(self):
        case = self.cases["gfp"]
        expected, _ = measure(case.xyz, case)
        self.assertAlmostEqual(expected["score"], 100.0)
        transformed = case.xyz[:, [1, 2, 0]] + [10, -2, 28]
        metrics, _ = measure(transformed, case)
        self.assertLess(metrics["rmsd"], 1e-8)
        reflected = case.xyz * [-1, 1, 1]
        self.assertGreater(measure(reflected, case)[0]["rmsd"], 2.0)
        with self.assertRaises(ValueError):
            measure(case.xyz[:-1], case)

    def test_framework_anchored_loop(self):
        case = self.cases["vhh-cdr3"]
        shifted = case.xyz.copy()
        shifted[case.mutable] += [3.0, 0.0, 0.0]
        result, _ = measure(shifted, case)
        self.assertAlmostEqual(result["rmsd"], 3.0)
        self.assertEqual(result["positions_scored"], 17)

    def test_pair_unit_and_invalid_answers(self):
        rows = [
            {"pair": "p", "margin": 3.0},
            {"pair": "p", "margin": -1.0},
            {"pair": "q"},
        ]
        result = paired_statistics(rows)
        self.assertEqual(result["complete_pairs"], 1)
        self.assertEqual(result["incomplete_pairs"], 1)
        self.assertEqual(result["one_sided_sign_p"], 0.5)
        case = self.cases["trp-cage"]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "trial"
            r = lock(case, p, "geometry", "maybe AA", {"model": "test"})
            self.assertFalse(r["valid"])
            self.assertEqual(fold_inputs([r]), {"items": []})
            with self.assertRaises(FileExistsError):
                lock(case, p, "geometry", controls(case)["positive"], {"model": "test"})

    def test_anonymous_fold_queue(self):
        case = self.cases["insulin"]
        s = controls(case)["positive"]
        queue = fold_inputs([{"sequence": s, "valid": True}] * 2)["items"]
        self.assertEqual(len(queue), 1)
        self.assertEqual(set(queue[0]), {"id", "sequence"})
        self.assertNotIn(case.id, queue[0]["id"])

    def test_muse_rejects_tool_event(self):
        with self.assertRaisesRegex(ValueError, "unexpected Muse event"):
            audit_muse(
                json.dumps({"payload_type": "tool.execute", "payload": {}}), "prompt"
            )


if __name__ == "__main__":
    unittest.main()
