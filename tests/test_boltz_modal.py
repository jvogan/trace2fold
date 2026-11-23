"""Check the Boltz-2 protocol and translation to the shared fold artifact schema."""

import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

if importlib.util.find_spec("modal") is None:
    raise unittest.SkipTest("the Modal SDK is required for backend tests")

from backends import boltz_modal as boltz
from backends import fold_modal as shared


class BoltzProtocol(unittest.TestCase):
    def test_spec_only_contains_protein_sequences_and_empty_msa(self):
        self.assertEqual(boltz.sequence_spec("AG:VS:AG"), {
            "sequences": [
                {"protein": {"id": ["A", "C"], "sequence": "AG", "msa": "empty"}},
                {"protein": {"id": "B", "sequence": "VS", "msa": "empty"}},
            ],
        })
        self.assertFalse(boltz.PROTOCOL["templates"])
        self.assertFalse(boltz.PROTOCOL["msa"])

    def test_parallel_chunks_preserve_every_item_and_bound_call_size(self):
        items = [{"id": str(i), "sequence": "A"} for i in range(20)]
        with patch.object(boltz, "BoltzWorker") as worker:
            generator = worker.return_value.fold.remote_gen
            generator.side_effect = iter
            actual = list(boltz.fold_parallel(items, workers=3))
        self.assertCountEqual(actual, items)
        self.assertEqual(sorted(len(call.args[0]) for call in generator.call_args_list), [6, 7, 7])

    def test_native_multichain_output_matches_shared_contract(self):
        pdb = (
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 82.00           C  \n"
            "TER       2      ALA A   1\n"
            "ATOM      3  CA  GLY B   1       4.000   5.000   6.000  1.00 91.00           C  \n"
            "END\n"
        )
        item = {"id": "test", "sequence": "A:G"}
        start = time.monotonic()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subdir = root / "boltz_results_test" / "predictions" / "test"
            subdir.mkdir(parents=True)
            (subdir / "test_model_0.pdb").write_text(pdb)
            confidence = subdir / "confidence_test_model_0.json"
            confidence.write_text('{"ptm":0.7, "iptm":0.6}')
            result = boltz.read_result(root, item, {"gpu": "test"}, start)
            checksum = shared.save_result(root, item, result, boltz.PROTOCOL)
            self.assertEqual(shared.prior_result(root, item, boltz.PROTOCOL), checksum)
            sidecar = json.loads((root / "test.json").read_text())
            self.assertEqual(sidecar["plddt"], [82.0, 91.0])
            self.assertEqual(sidecar["chains"], [{"id": "A", "length": 1}, {"id": "B", "length": 1}])
            self.assertEqual(sidecar["iptm"], 0.6)
            self.assertEqual(sidecar["provenance"]["protocol"], boltz.PROTOCOL)
            confidence.write_text('{"ptm":1.1, "iptm":0.6}')
            with self.assertRaisesRegex(ValueError, "pTM is out of range"):
                boltz.read_result(root, item, {}, start)
