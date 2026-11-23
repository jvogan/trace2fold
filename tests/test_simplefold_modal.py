"""Exercise the SimpleFold gates and evidence without cloud compute or model weights."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

if importlib.util.find_spec("modal") is None:
    raise unittest.SkipTest("Modal SDK is required")

from backends import simplefold_modal as backend


CANARY = [
    {"id": item_id, "sequence": seq}
    for item_id, seq in zip(backend.CANARY_IDS, ("A" * 48, "C" * 112, "D" * 176), strict=True)
]


def mock_ledgers(root):
    full = root / "full.json"
    canary = root / "canary.json"
    extra = [{"id": f"other-{i:02d}", "sequence": "W" * 48} for i in range(69)]
    backend.write_json(full, {"items": [{**i, "private_metadata": "excluded"} for i in CANARY + extra]})
    backend.write_json(canary, {"items": CANARY, "parent_ledger_sha256": backend.sha256(full.read_bytes())})
    return full, canary


def mock_cif(path, sequence):
    import gemmi

    structure = gemmi.Structure()
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    names = {value: key for key, value in backend.RESIDUES.items()}
    for i, aa in enumerate(sequence, 1):
        residue = gemmi.Residue()
        residue.name = names[aa]
        residue.seqid = gemmi.SeqId(i, " ")
        residue.het_flag = "A"
        for name, shift in (("N", -1), ("CA", 0), ("C", 1), ("O", 2)):
            atom = gemmi.Atom()
            atom.name = name
            atom.pos = gemmi.Position(i * 3.19432 + shift, 2.32128, 3.87871)
            atom.element = gemmi.Element(name[0])
            atom.b_iso = 100
            residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    structure.make_mmcif_document().write_file(str(path))
    return path


class Planning(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.full, self.canary = mock_ledgers(self.root)

    def test_plan_is_bound_to_both_inputs_and_strips_metadata(self):
        plan, items = backend.build_plan(self.full, self.canary, 500, True)
        self.assertEqual(len(plan["items"]["full"]), 72)
        self.assertEqual(len(plan["items"]["canary"]), 3)
        self.assertEqual(items["canary"], CANARY)
        self.assertEqual(set(items["full"][0]), {"id", "sequence"})
        self.assertNotIn(CANARY[0]["sequence"], json.dumps(plan))
        self.assertFalse(plan["protocol"]["confidence_available"])
        self.assertFalse(plan["protocol"]["pae_available"])
        self.assertEqual(plan["budget"]["max_total_reservation_usd"], 204)
        self.full.write_bytes(self.full.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "exact parent ledger"):
            backend.build_plan(self.full, self.canary, 500, True)

    def test_plan_requires_license_order_membership_and_budget(self):
        with self.assertRaisesRegex(ValueError, "research-only"):
            backend.build_plan(self.full, self.canary, 500, False)
        with self.assertRaisesRegex(ValueError, "planning capacity"):
            backend.build_plan(self.full, self.canary, 90, True)
        for rows, error in (
            (list(reversed(CANARY)), "residue rows in order"),
            ([{**CANARY[0], "sequence": "C" * 48}, *CANARY[1:]], "exactly match"),
        ):
            backend.write_json(self.canary, {"items": rows, "parent_ledger_sha256": backend.sha256(self.full.read_bytes())})
            with self.assertRaisesRegex(ValueError, error):
                backend.build_plan(self.full, self.canary, 500, True)

    def test_freeze_canary_binds_parent_exactly(self):
        frozen = self.root / "frozen.json"
        backend.freeze_canary(self.full, frozen)
        self.assertEqual(backend.read_json(frozen)["items"], CANARY)
        parent_sha = backend.sha256(self.full.read_bytes())
        self.assertEqual(backend.read_json(frozen)["parent_ledger_sha256"], parent_sha)
        self.full.write_bytes(self.full.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "existing canary ledger differs"):
            backend.freeze_canary(self.full, frozen)

    def test_supervisor_treats_builtin_timeout_as_an_empty_poll(self):
        class Call:
            polls = 0

            def get(self, *, timeout):
                self.assert_timeout = timeout
                self.polls += 1
                if self.polls < 3:
                    raise TimeoutError()
                return {"status": "complete"}

        call = Call()
        with patch.object(backend, "volume_record", return_value=None):
            self.assertEqual(backend.supervise(call, "canary", 1), {"status": "complete"})
        self.assertEqual(call.polls, 3)
        self.assertEqual(call.assert_timeout, 20)


@unittest.skipUnless(importlib.util.find_spec("gemmi"), "Gemmi is required for structure fixtures")
class Execution(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        full, canary = mock_ledgers(self.root)
        self.plan, self.items = backend.build_plan(full, canary, 500, True)
        backend.write_json(self.root / "plan.json", self.plan)
        patcher = patch.object(backend, "acquire_assets")
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_batch(self, stage, attempt, gate=None, fail_at=None):
        backend.write_json(self.root / stage / "attempts" / f"{attempt}.reservation.json", {
            "plan_sha256": backend.sha256(backend.canonical(self.plan)),
            "app_id": "ap-synthetic", "attempt": attempt, "stage": stage,
        })
        calls = {"load": 0, "preprocess": [], "sample": []}

        class Model:
            def __init__(self, _assets):
                calls["load"] += 1

            def prepare(self, rows, _work):
                calls["preprocess"].extend(row["id"] for row in rows)

            def fold(self, item, work, _heartbeat):
                calls["sample"].append(item["id"])
                if item["id"] == fail_at:
                    raise RuntimeError("demonstrate failure and resume")
                return mock_cif(work / f"{item['id']}.cif", item["sequence"])

        receipt = backend.execute_batch(
            self.root, self.root / "assets", self.plan, stage, self.items[stage], attempt,
            gate, lambda: None, model_factory=Model,
        )
        return receipt, calls

    def test_canary_full_gate_and_hash_keyed_deduplication(self):
        canary, calls = self.run_batch("canary", 1)
        self.assertEqual(canary["status"], "complete")
        self.assertEqual(calls, {"load": 1, "preprocess": [i["id"] for i in CANARY],
                                 "sample": [i["id"] for i in CANARY]})
        with self.assertRaisesRegex(ValueError, "canary gate"):
            self.run_batch("full", 1)
        gate, export = self.root / "gate.json", self.root / "structures.json"
        backend.write_json(self.root / "egress.json", {
            "stage": "canary", "status": "complete", "plan_sha256": canary["plan_sha256"],
            "files": {"canary/receipt.json": {"sha256": backend.sha256(backend.canonical(canary))}},
            "closeout": {"state": "stopped", "tasks": 0, "running_containers": 0},
        })
        backend.local_artifact_check(self.root, self.plan, "canary", self.items["canary"], export, gate)
        self.assertEqual([i["id"] for i in backend.read_json(export)["items"]], [i["id"] for i in CANARY])
        full, calls = self.run_batch("full", 1, backend.read_json(gate))
        self.assertEqual(full["counts"]["completed"], 1)
        self.assertEqual(full["counts"]["reused"], 71)
        self.assertEqual(calls, {"load": 1, "preprocess": ["other-00"], "sample": ["other-00"]})
        self.assertEqual(len(list((self.root / "structures").glob("*.pdb"))), 4)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            self.run_batch("full", 1, backend.read_json(gate))

    def test_partial_retry_reuses_only_verified_outputs(self):
        failed_id = CANARY[1]["id"]
        receipt, _ = self.run_batch("canary", 1, fail_at=failed_id)
        self.assertEqual(receipt["status"], "incomplete")
        self.assertEqual(receipt["counts"]["valid"], 2)
        self.assertFalse((self.root / "canary" / "DONE.json").exists())
        with self.assertRaisesRegex(RuntimeError, "replay"):
            self.run_batch("canary", 1)
        receipt, calls = self.run_batch("canary", 2)
        self.assertEqual(receipt["counts"]["reused"], 2)
        self.assertEqual(calls["sample"], [failed_id])
        _, pdb, _ = backend.structure_paths(self.root, backend.identity(CANARY[0])["input_sha256"])
        pdb.write_bytes(pdb.read_bytes() + b"REMARK unauthorized change\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            backend.verify_complete(self.root, self.plan, "canary", CANARY)

    def test_egress_binds_volume_attempt_plan_and_closeout(self):
        receipt, _ = self.run_batch("canary", 1)
        root = self.root
        controller = {"app_id": "ap-synthetic", "attempt": 1, "stage": "canary",
                      "remote_receipt_sha256": backend.sha256(backend.canonical(receipt))}

        class Volume:
            object_id = "vo-synthetic"

            def hydrate(self):
                pass

            def listdir(self, prefix, *, recursive):
                return [SimpleNamespace(path=f"{prefix}/{path.relative_to(root)}", size=path.stat().st_size,
                                        type=SimpleNamespace(name="FILE")) for path in root.rglob("*") if path.is_file()]

            def read_file(self, path):
                yield (root / path.removeprefix("/campaign/")).read_bytes()

        # The fake Volume serves committed records only.
        for unmounted in (root / "full.json", root / "canary.json"):
            unmounted.unlink()
        calls = []

        def closeout(app_id):
            calls.append(app_id)
            return {"app_id": app_id, "state": "stopped", "tasks": 0, "running_containers": 0}

        with TemporaryDirectory() as destination_parent:
            destination = Path(destination_parent) / "campaign"
            args = (Volume(), destination, self.plan, "canary", self.items["canary"])
            with self.assertRaisesRegex(ValueError, "controller receipt"):
                backend.download_snapshot(*args, {**controller, "remote_receipt_sha256": "0" * 64}, closeout=closeout)
            self.assertFalse(destination.exists())
            backend.download_snapshot(*args, controller, closeout=closeout)
            self.assertEqual(calls, ["ap-synthetic"] * 3)
            output = backend.read_json(destination / "egress.json")
            self.assertEqual(output["volume_id"], "vo-synthetic")
            self.assertEqual(output["closeout"]["running_containers"], 0)
            self.assertEqual(output["files"]["canary/receipt.json"]["sha256"], controller["remote_receipt_sha256"])
            with self.assertRaises(FileExistsError):
                backend.download_snapshot(*args, controller, closeout=closeout)
