"""Verify the frozen calibrators and their task-index mappings without downloads."""

from collections import defaultdict
import hashlib
import json
from math import dist
from pathlib import Path
import unittest

from backends.prepare_known import AA, BACKBONE

ROOT = Path(__file__).resolve().parents[1]
RECORDS = {
    record["id"]: record
    for record in json.loads((ROOT / "data/known/index.json").read_text())
}


def read_atoms(record):
    atoms = defaultdict(dict)
    for line in (ROOT / record["path"]).read_text().splitlines():
        if line.startswith("ATOM  "):
            key = (line[21], int(line[22:26]), AA[line[17:20]])
            atom = line[12:16].strip()
            if atom in atoms[key]:
                raise ValueError(f"Duplicate atom {atom} in {key}")
            atoms[key][atom] = tuple(float(line[i:i + 8]) for i in (30, 38, 46))
    return atoms


class KnownFixtures(unittest.TestCase):
    def test_backbones_and_coordinate_sequence(self):
        self.assertEqual(set(RECORDS), {"trp-cage", "insulin", "gfp", "vhh-cdr3"})
        for record in RECORDS.values():
            with self.subTest(track=record["id"]):
                pdb = (ROOT / record["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(pdb).hexdigest(), record["fixture_sha256"])
                self.assertNotIn(b"HETATM", pdb)
                atoms = read_atoms(record)
                positions = [
                    (chain, number)
                    for chain, (start, end) in record["author_residue_ranges"].items()
                    for number in range(start, end + 1)
                ]
                chains = record.get("chains") or [record["chain"]]
                self.assertEqual(len(atoms), record["length"])
                self.assertEqual([(chain, number) for chain, number, _ in atoms], positions)
                self.assertEqual("".join(aa for _, _, aa in atoms), record["sequence"])
                self.assertEqual(
                    [sum(atom_chain == chain for atom_chain, _, _ in atoms) for chain in chains],
                    record["chain_lengths"],
                )
                self.assertEqual([chr(line[21]) for line in pdb.splitlines() if line.startswith(b"TER")], chains)
                for residue_atoms in atoms.values():
                    self.assertTrue(set(BACKBONE) <= residue_atoms.keys())

    def test_mature_insulin_disulfides_and_chain_boundaries(self):
        record = RECORDS["insulin"]
        self.assertEqual(record["chains"], ["A", "B"])
        self.assertEqual(record["chain_lengths"], [21, 30])
        self.assertEqual(record["disulfides"], [[6, 11], [7, 28], [20, 40]])
        self.assertEqual(record["fixed"], [6, 7, 11, 20, 28, 40])
        pdb = (ROOT / record["path"]).read_text()
        self.assertEqual(sum(line.startswith("SSBOND") for line in pdb.splitlines()), 3)
        atoms = list(read_atoms(record).items())
        for (a, b), expected_chains in zip(record["disulfides"], ("AA", "AB", "AB"), strict=True):
            (key_a, residue_a), (key_b, residue_b) = atoms[a - 1], atoms[b - 1]
            self.assertEqual(key_a[0] + key_b[0], expected_chains)
            self.assertEqual(key_a[2] + key_b[2], "CC")
            self.assertLess(dist(residue_a["SG"], residue_b["SG"]), 2.2)

    def test_gfp_uses_observed_separate_precursor_residues(self):
        record = RECORDS["gfp"]
        residues = list(read_atoms(record))
        self.assertEqual(record["length"], 227)
        self.assertEqual(record["chromophore_precursors"], [63, 64, 65])
        self.assertEqual([residues[i - 1] for i in record["chromophore_precursors"]],
                         [("A", 65, "T"), ("A", 66, "Y"), ("A", 67, "G")])

    def test_vhh_has_apo_loop_and_fixed_scaffold(self):
        record = RECORDS["vhh-cdr3"]
        self.assertEqual(record["pdb_id"], "1I3V")
        self.assertEqual(record["pair_status"], "unpaired")
        self.assertEqual(record["mutable"], list(range(102, 119)))
        self.assertEqual(record["disulfides"], [[22, 101]])
        self.assertTrue(set(record["mutable"]).isdisjoint(record["fixed"]))
        self.assertEqual(record["length"], 129)
