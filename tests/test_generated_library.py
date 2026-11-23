"""The frozen generated targets preserve coverage, anonymous inputs, and control pairing."""
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import unittest

import numpy as np

from trace2fold.cases import AA, load_cases, prompt, read_ca
from trace2fold.metrics import rigid_fit


DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "generated"
REPO_ROOT = DATA_ROOT.parents[1]


class GeneratedLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = json.loads((DATA_ROOT / "index.json").read_text())
        cls.provenance = json.loads((DATA_ROOT / "provenance.json").read_text())

    def test_coverage_and_controls(self):
        coverage = Counter((row["generator"], row["length"]) for row in self.index)
        self.assertEqual(set(coverage), {
            (generator, length)
            for generator in ("genie3", "rfd4-proteina")
            for length in (48, 72, 96)
        })
        self.assertGreaterEqual(min(coverage.values()), 2)
        self.assertEqual({row["family"] for row in self.index}, {"helical", "beta", "mixed"})
        self.assertEqual(len(self.index), len({row["id"] for row in self.index}))

        for row in self.index:
            with self.subTest(target=row["id"]):
                self.assertEqual(len(row["sequence"]), row["length"])
                self.assertFalse(set(row["sequence"]) - set(AA))
                expected_source = {"genie3": "proteinmpnn-ca", "rfd4-proteina": "rfd4-native"}
                self.assertEqual(row["sequence_source"], expected_source[row["generator"]])
                path = REPO_ROOT / row["path"]
                _, names = read_ca(path, ["A"])
                self.assertEqual(names, ["G" * row["length"]])
                lines = path.read_text().splitlines()
                self.assertEqual({line[:6].strip() for line in lines}, {"ATOM", "TER", "END"})
                atoms = [line for line in lines if line.startswith("ATOM  ")]
                self.assertEqual(len(atoms), row["length"])
                self.assertEqual({line[12:16].strip() for line in atoms}, {"CA"})
                self.assertEqual({line[21] for line in atoms}, {"A"})
                self.assertEqual({line[17:20] for line in atoms}, {"GLY"})

    def test_prompt_contains_only_the_mask_and_geometric_input(self):
        cases = load_cases(REPO_ROOT)
        for row in self.index:
            case = cases[row["id"]]
            self.assertEqual(case.template, "?" * row["length"])
            text = prompt(case)
            self.assertNotIn(row["sequence"], text)
            self.assertNotIn(row["id"], text)
            self.assertNotIn(row["generator"], text)

    def test_pairs_match_length_secondary_structure_family_and_have_distinct_coordinates(self):
        pairs = defaultdict(list)
        coordinates = defaultdict(list)
        provenance_by_id = {row["id"]: row for row in self.provenance["selected_details"]}
        for row in self.index:
            pairs[row["pair"]].append(row)
            path = REPO_ROOT / row["path"]
            self.assertEqual(sha256(path.read_bytes()).hexdigest(), provenance_by_id[row["id"]]["backbone_sha256"])
            xyz, _ = read_ca(path, [row["chain"]])
            coordinates[row["length"]].append(xyz)

        for group in pairs.values():
            left, right = group
            self.assertEqual((left["length"], left["family"], left["tier"]),
                             (right["length"], right["family"], right["tier"]))
            first, second = (provenance_by_id[row["id"]] for row in group)
            for field in ("helix_fraction", "strand_fraction"):
                self.assertLessEqual(abs(first[field] - second[field]), 0.20)
            self.assertLessEqual(first["mate_contact_jaccard"], 0.55)
        for length, shapes in coordinates.items():
            for first, second in combinations(shapes, 2):
                rotation, offset = rigid_fit(first, second, np.ones(length, dtype=bool))
                aligned = first @ rotation + offset
                rmsd = np.sqrt(np.mean(np.sum((aligned - second)**2, axis=1)))
                self.assertGreaterEqual(rmsd, 2.999)


if __name__ == "__main__":
    unittest.main()
