from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from trace2fold.cases import ROOT, load_cases
from trace2fold.topology import contact_pairs, load_annotations, topology_text


class TopologyTests(unittest.TestCase):
    def test_contacts_use_exact_cutoff_and_chain_numbering(self):
        xyz = np.zeros((11, 3))
        xyz[:, 0] = [0, 90, 100, 110, 120, 0, 8, 8.001, 130, 130, 300]

        self.assertEqual(contact_pairs(xyz, [9, 2]), [(1, 7), (9, 10)])

    def test_frozen_annotations_cover_every_case(self):
        cases = load_cases()
        labels = load_annotations()
        self.assertEqual(set(labels), set(cases))
        for case_id, case in cases.items():
            with self.subTest(case=case_id):
                _, secondary, _, contacts = topology_text(
                    case, labels[case_id]
                ).splitlines()
                self.assertRegex(secondary, r"^[HEC]+(?::[HEC]+)*$")
                self.assertEqual(list(map(len, secondary.split(":"))), case.lengths)
                self.assertRegex(contacts, r"^(?:None|\d+-\d+(?:, \d+-\d+)*)$")

    def test_rejects_stale_coordinates_or_labels(self):
        case = load_cases()["insulin"]
        labels = load_annotations()[case.id]
        changed = case.xyz.copy()
        changed[0, 0] += 0.001
        with self.assertRaisesRegex(ValueError, "coordinate hash mismatch"):
            topology_text(replace(case, xyz=changed), labels)
        with self.assertRaisesRegex(ValueError, "invalid secondary structure labels"):
            topology_text(case, replace(labels, chains=("H" * 51,)))
        with self.assertRaisesRegex(ValueError, "missing frozen secondary structure"):
            topology_text(case, None)

    def test_rejects_changed_frozen_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "data" / "known", root / "data" / "known")
            fixture = root / "data" / "known" / "trp-cage.pdb"
            with fixture.open("a") as stream:
                stream.write("REMARK changed fixture\n")
            with self.assertRaisesRegex(ValueError, "fixture hash mismatch"):
                load_annotations(root)


if __name__ == "__main__":
    unittest.main()
