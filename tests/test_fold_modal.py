import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if importlib.util.find_spec("modal") is None:
    raise unittest.SkipTest("the Modal SDK is required for backend tests")

from backends import fold_modal as backend


def ca_line(serial, residue, chain, number, x=1.0, plddt=89.0):
    return (
        f"ATOM  {serial:5d}  CA  {residue} {chain}{number:4d}    "
        f"{x:8.3f}{2.0:8.3f}{3.0:8.3f}{1.0:6.2f}{plddt:6.2f}           C  \n"
    )


class InputTests(unittest.TestCase):
    def test_protein_only_input_and_unique_ids(self):
        first = {"id": "Control-1", "sequence": "AGS:YWF"}
        self.assertEqual(backend.parse_items({"items": [first]}), [first])
        for items in ([{**first, "target": "template"}],
                      [first, {**first, "id": "control-1"}],
                      [{**first, "id": "../outside"}],
                      [{**first, "sequence": "AGS::YWF"}],
                      [{**first, "sequence": "AGXS"}]):
            with self.subTest(items=items), self.assertRaises(ValueError):
                backend.parse_items({"items": items})


class PredictionTests(unittest.TestCase):
    def test_reads_coordinates_and_confidence_in_sequence_order(self):
        text = ca_line(1, "GLY", "B", 1, 9, 51) + ca_line(2, "ALA", "A", 1, 6, 99)
        coords, plddt = backend.read_ca(text, "A:G")
        self.assertEqual(coords, [[6.0, 2.0, 3.0], [9.0, 2.0, 3.0]])
        self.assertEqual(plddt, [99.0, 51.0])

    def test_rejects_partial_mismatched_duplicate_and_invalid_confidence(self):
        ala = ca_line(1, "ALA", "A", 1)
        bad_predictions = (
            (ala, "AG"), (ala, "G"), (ala + ala, "A"),
            (ca_line(1, "ALA", "A", 1, plddt=101), "A"),
        )
        for pdb, sequence in bad_predictions:
            with self.subTest(sequence=sequence), self.assertRaises(ValueError):
                backend.read_ca(pdb, sequence)


class BatchTests(unittest.TestCase):
    def test_partial_result_resumes_only_missing_items(self):
        first = {"id": "first", "sequence": "A"}
        second = {"id": "second", "sequence": "G"}
        protocol = {"model": "test-fold"}
        response = {
            "id": "first", "pdb": ca_line(1, "ALA", "A", 1),
            "runtime": {}, "elapsed_seconds": 2.0, "ptm": 0.7, "iptm": None,
        }
        with tempfile.TemporaryDirectory() as tmp, patch.object(backend, "FoldWorker") as worker:
            source = Path(tmp) / "inputs.json"
            target = Path(tmp) / "output"
            source.write_text(json.dumps({"items": [first, second]}))
            remote = worker.return_value.fold.remote_gen
            remote.return_value = iter([response, {"id": "second", "error": "RuntimeError", "detail": "timeout"}])
            with self.assertRaisesRegex(RuntimeError, "batch incomplete"):
                backend.run_batch(source, target, protocol=protocol)
            manifest = json.loads((target / "manifest.json").read_text())
            self.assertEqual(manifest["counts"], {"completed": 1, "reused": 0, "failed": 1, "pending": 0})
            sidecar = json.loads((target / "first.json").read_text())
            self.assertEqual(sidecar["pdb_sha256"], backend.sha256((target / "first.pdb").read_bytes()))
            self.assertEqual(sidecar["provenance"]["protocol"], protocol)
            remote.return_value = iter([{**response, "id": "second", "pdb": ca_line(1, "GLY", "A", 1)}])
            backend.run_batch(source, target, protocol=protocol)
            remote.assert_called_with([second])
            manifest = json.loads((target / "manifest.json").read_text())
            self.assertEqual(manifest["counts"], {"completed": 1, "reused": 1, "failed": 0, "pending": 0})
            source.write_text(json.dumps({"items": [{**first, "sequence": "G"}]}))
            with self.assertRaisesRegex(ValueError, "different sequence"):
                backend.run_batch(source, target, protocol=protocol)


if __name__ == "__main__":
    unittest.main()
