"""Guarded, sequence-only Apple SimpleFold 3B inference on Modal."""

from argparse import ArgumentParser, Namespace
from copy import deepcopy
from datetime import UTC, datetime
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen

import modal

COMMIT = "c7a5570a6be9f5c695126e27c804e77567209934"
REPOSITORY = "https://github.com/apple/ml-simplefold.git"
LICENSE = f"https://github.com/apple/ml-simplefold/blob/{COMMIT}/LICENSE_MODEL"
SOURCE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ESM_REVISION = "87d45333963df5923f29fe23a9a15cbcb6c9c71a"
CCD_REVISION = "7c1d83b779e4c65ecc37dfdf0c6b2788076f31e1"
ASSETS = {
    "simplefold_3B.ckpt": (
        "https://ml-site.cdn-apple.com/models/simplefold/simplefold_3B.ckpt",
        11_460_066_894, "88d4c7a240bf3815cb35342b4ddc1128ac243a2ea0256eb8a4df1209125868b5",
    ),
    "esm2_t36_3B_UR50D.pt": (
        "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt",
        5_678_116_398, "7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500",
    ),
    "esm2_t36_3B_UR50D-contact-regression.pt": (
        "https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t36_3B_UR50D-contact-regression.pt",
        6_759, "4da500eab246481dc9c8c95bc7b1d02f2803d761c380b0e95186d4a07d0fc84e",
    ),
    "ccd.pkl": (
        f"https://huggingface.co/boltz-community/boltz-1/resolve/{CCD_REVISION}/ccd.pkl",
        345_859_128, "2d3b2f03a3c5665944adba51e33263511e51b21c9cd05d902f9c4b7c1e58d2f4",
    ),
}
SETTINGS = {"num_steps": 500, "tau": 0.01, "seed": 0, "samples_per_sequence": 1}
PROTOCOL = {
    "model": "simplefold_3B", "simplefold_commit": COMMIT, "fair_esm_version": "2.0.0",
    "fair_esm_revision": ESM_REVISION, "ccd_revision": CCD_REVISION, "torch_version": "2.7.1",
    "gemmi_version": "0.7.3", "backend": "torch", "input": "single_chain_sequence",
    "msa": False, "templates": False, "confidence_available": False, "pae_available": False,
    "settings": SETTINGS, "license_url": LICENSE, "use": "noncommercial_scientific_research",
    "assets": {name: {"bytes": size, "sha256": digest} for name, (_, size, digest) in ASSETS.items()},
}
VOLUME_NAME = "trace2fold-simplefold3b-variations-v2"
MOUNT = Path("/simplefold-data")
CAMPAIGN_DIR = "campaign"
TIMEOUTS = {"canary": 3600, "full": 43_200}
COUNTS = {"canary": 3, "full": 72}
CANARY_LENGTHS = [48, 112, 176]
CANARY_IDS = ["51c3146fa23fcddbd085abaa401a4c5c", "8e760b594b31fd6c574c8ef2e4f42ef6",
              "bc3047dd93cf05710bc4bd80851e4ae1"]
MAX_ATTEMPTS = 2
MAX_SPEND_USD = 500
PLANNING_RATE_USD_PER_HOUR = 6
RESERVE_USD_PER_ATTEMPT = 10
STARTUP_SECONDS = 1200
SILENCE_SECONDS = 900
HEARTBEAT_SECONDS = 60
MAX_LEDGER_BYTES = 2 * 1024 * 1024
MAX_STRUCTURE_BYTES = 2 * 1024 * 1024
SHA = re.compile(r"[0-9a-f]{64}")
ITEM_ID = re.compile(r"[a-zA-Z0-9_-][a-zA-Z0-9_.-]{0,127}")
SEQUENCE = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{1,512}")
RESIDUES = dict(zip(
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
    "ACDEFGHIKLMNPQRSTVWY", strict=True,
))


def canonical(data):
    return (json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


PROTOCOL_SHA = sha256(canonical(PROTOCOL))


def identity(item):
    sequence_sha = sha256(item["sequence"].encode("ascii"))
    return {
        "id": item["id"], "sequence_sha256": sequence_sha, "residue_count": len(item["sequence"]),
        "input_sha256": sha256(canonical({"protocol_sha256": PROTOCOL_SHA, "sequence_sha256": sequence_sha})),
    }


def read_ledger(path, stage):
    with path.open("rb") as handle:
        raw = handle.read(MAX_LEDGER_BYTES + 1)
    if len(raw) > MAX_LEDGER_BYTES:
        raise ValueError(f"{stage} ledger exceeds the size limit")
    ledger = json.loads(raw)
    if not isinstance(ledger, dict):
        raise ValueError(f"{stage} ledger must be an object")
    items = ledger.get("items")
    if not isinstance(items, list) or len(items) != COUNTS[stage]:
        raise ValueError(f"{stage} ledger must hold exactly {COUNTS[stage]} rows")
    sanitized = []
    for row in items:
        if not isinstance(row, dict):
            raise ValueError(f"{stage} ledger row must be an object")
        item = {"id": row.get("id"), "sequence": row.get("sequence")}
        # Check any supplied sequence digest before discarding metadata.
        if isinstance(item["sequence"], str) and "sequence_sha256" in row:
            if sha256(item["sequence"].encode()) != row["sequence_sha256"]:
                raise ValueError(f"{stage} ledger sequence hash disagrees with its row")
        sanitized.append(item)
    return sha256(raw), ledger.get("parent_ledger_sha256"), sanitized


def validate_items(items, stage):
    if len(items) != COUNTS[stage]:
        raise ValueError(f"unexpected {stage} item count")
    ids = set()
    for item in items:
        name, sequence = item["id"], item["sequence"]
        if set(item) != {"id", "sequence"} or not isinstance(name, str) or not ITEM_ID.fullmatch(name):
            raise ValueError("predictor input requires a safe ID and sequence")
        if name.casefold() in ids:
            raise ValueError("ledger has duplicate IDs")
        ids.add(name.casefold())
        if not isinstance(sequence, str) or not SEQUENCE.fullmatch(sequence):
            raise ValueError(f"invalid single-chain canonical sequence for {name}")
    if stage == "canary" and [len(row["sequence"]) for row in items] != CANARY_LENGTHS:
        raise ValueError("canary must contain the predeclared 48/112/176-residue rows in order")
    if stage == "canary" and [row["id"] for row in items] != CANARY_IDS:
        raise ValueError("canary IDs differ from the three predeclared campaign rows")


def freeze_canary(full_path, canary_path):
    parent_sha, _, full = read_ledger(full_path, "full")
    validate_items(full, "full")
    by_id = {row["id"]: row for row in full}
    if not all(item_id in by_id for item_id in CANARY_IDS):
        raise ValueError("the parent ledger is missing a predeclared canary row")
    selected = [by_id[item_id] for item_id in CANARY_IDS]
    validate_items(selected, "canary")
    data = canonical({"schema_version": 1, "parent_ledger_sha256": parent_sha, "items": selected})
    if canary_path.exists() and canary_path.read_bytes() != data:
        raise ValueError("existing canary ledger differs from the predeclared parent rows")
    atomic_write(canary_path, data)


def reserve_cost(stage):
    return (TIMEOUTS[stage] + STARTUP_SECONDS) * PLANNING_RATE_USD_PER_HOUR / 3600 + RESERVE_USD_PER_ATTEMPT


def build_plan(full_path, canary_path, max_spend_usd, license_ack):
    if license_ack is not True:
        raise ValueError(f"acknowledge the Apple research-only model license at {LICENSE}")
    if not math.isfinite(max_spend_usd) or not 0 < max_spend_usd <= MAX_SPEND_USD:
        raise ValueError(f"planning spend cap must be positive and at most ${MAX_SPEND_USD}")
    parent_sha, _, full = read_ledger(full_path, "full")
    canary_sha, canary_parent_sha, canary = read_ledger(canary_path, "canary")
    validate_items(full, "full")
    validate_items(canary, "canary")
    if canary_parent_sha != parent_sha:
        raise ValueError("canary must bind the exact parent ledger SHA-256")
    full_by_id = {row["id"]: row for row in full}
    if any(full_by_id.get(row["id"]) != row for row in canary):
        raise ValueError("canary rows must exactly match their parent ledger ID and sequence")
    initial_reservation = sum(map(reserve_cost, COUNTS))
    if initial_reservation > max_spend_usd:
        raise ValueError(f"both initial runs require ${initial_reservation:.2f} of planning capacity")
    items = {"canary": canary, "full": full}
    plan = {
        "schema_version": 1, "protocol": PROTOCOL, "protocol_sha256": PROTOCOL_SHA,
        "adapter_sha256": SOURCE_SHA, "research_license_acknowledged": True,
        "inputs_sha256": {"canary": canary_sha, "full": parent_sha},
        "items": {stage: [identity(row) for row in rows] for stage, rows in items.items()},
        "budget": {
            "max_spend_usd": max_spend_usd, "planning_usd_per_hour": PLANNING_RATE_USD_PER_HOUR,
            "reservation_usd": {stage: reserve_cost(stage) for stage in COUNTS},
            "max_total_reservation_usd": min(max_spend_usd, MAX_ATTEMPTS * initial_reservation),
            "max_attempts_per_stage": MAX_ATTEMPTS, "timeout_seconds": TIMEOUTS,
            "startup_seconds": STARTUP_SECONDS, "silence_seconds": SILENCE_SECONDS,
        },
    }
    return plan, items


def now():
    return datetime.now(UTC).isoformat()


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_bytes(data)
    partial.replace(path)


def write_json(path, data):
    atomic_write(path, canonical(data))


def read_json(path):
    return json.loads(path.read_bytes())


def structure_paths(root, input_sha):
    folder = root / "structures"
    return [folder / f"{input_sha}{ext}" for ext in (".cif", ".pdb", ".json")]


def structure_atoms(path, sequence, *, neutral=False):
    import gemmi

    if not 0 < path.stat().st_size <= MAX_STRUCTURE_BYTES:
        raise ValueError("structure file exceeds the artifact size bound or is empty")
    structure = gemmi.read_structure(str(path))
    if len(structure) != 1 or len(structure[0]) != 1 or structure[0][0].name != "A":
        raise ValueError("expected exactly one model and protein chain A")
    residues = list(structure[0][0])
    if "".join(RESIDUES.get(residue.name, "?") for residue in residues) != sequence:
        raise ValueError("structure residue identity differs from the input sequence")
    atoms = {}
    for index, residue in enumerate(residues, 1):
        if residue.het_flag != "A" or residue.seqid.num != index or residue.seqid.icode != " ":
            raise ValueError("expected consecutive protein residues numbered from one")
        names = set()
        for atom in residue:
            xyz = tuple(atom.pos)
            if atom.name in names or atom.has_altloc() or not all(map(math.isfinite, xyz)):
                raise ValueError("expected unique atoms with finite coordinates")
            if neutral and atom.b_iso != 0.0:
                raise ValueError("neutral PDB must have zero B factors")
            names.add(atom.name)
            atoms[index, atom.name] = xyz
        if not {"N", "CA", "C", "O"} <= names:
            raise ValueError("structure lacks a complete backbone")
    return structure, atoms


def structure_record(cif, pdb, item, root):
    _, raw_atoms = structure_atoms(cif, item["sequence"])
    _, pdb_atoms = structure_atoms(pdb, item["sequence"], neutral=True)
    if raw_atoms.keys() != pdb_atoms.keys() or any(
        max(abs(a - b) for a, b in zip(xyz, pdb_atoms[key], strict=True)) > 0.00051
        for key, xyz in raw_atoms.items()
    ):
        raise ValueError("neutral PDB differs from the raw mmCIF coordinates")
    meta = identity(item)
    del meta["id"]
    return {
        **meta, "protocol_sha256": PROTOCOL_SHA, "adapter_sha256": SOURCE_SHA,
        "chain_id": "A", "atom_count": len(raw_atoms), "complete_backbone": True,
        "finite_coordinates": True, "confidence_available": False, "pae_available": False,
        "files": {
            key: {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                  "sha256": sha256(path.read_bytes())}
            for key, path in (("cif", cif), ("pdb", pdb))
        },
    }


def prior_result(root, item):
    cif, pdb, sidecar = structure_paths(root, identity(item)["input_sha256"])
    if not sidecar.is_file():
        return None
    record = structure_record(cif, pdb, item, root)
    if read_json(sidecar) != record:
        raise ValueError("committed structure record or artifact differs from its SHA-256 evidence")
    return record


def persist_result(root, item, raw_cif):
    key = identity(item)["input_sha256"]
    cif, pdb, sidecar = structure_paths(root, key)
    structure, _ = structure_atoms(raw_cif, item["sequence"])
    for residue in structure[0][0]:
        for atom in residue:
            atom.b_iso = 0.0
    atomic_write(cif, raw_cif.read_bytes())
    neutral = "REMARK   1 SIMPLEFOLD SAMPLE. B FACTORS ZERO. CONFIDENCE UNAVAILABLE.\n"
    atomic_write(pdb, (neutral + structure.make_pdb_string()).encode("ascii"))
    record = structure_record(cif, pdb, item, root)
    write_json(sidecar, record)
    return record


def acquire_assets(asset_dir, heartbeat):
    asset_dir.mkdir(parents=True, exist_ok=True)
    for filename, (url, expected_bytes, expected_sha) in ASSETS.items():
        heartbeat("asset_check", filename, force=True)
        path = asset_dir / filename
        digest = hashlib.sha256()
        count = 0
        if path.is_file():
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                    digest.update(block)
                    count += len(block)
                    heartbeat("asset_check", filename)
        else:
            # Stage outside the Volume so heartbeat commits never upload partially acquired weights.
            with tempfile.TemporaryDirectory(prefix="simplefold-assets-") as directory:
                staged = Path(directory) / filename
                request = Request(url, headers={"User-Agent": "Trace2Fold-SimpleFold/1"})
                with urlopen(request, timeout=75) as response, staged.open("wb") as output:
                    for block in iter(lambda: response.read(16 * 1024 * 1024), b""):
                        count += len(block)
                        if count > expected_bytes:
                            raise ValueError(f"asset {filename} exceeded its pinned byte count")
                        digest.update(block)
                        output.write(block)
                        heartbeat("asset_download", filename)
                if count != expected_bytes or digest.hexdigest() != expected_sha:
                    raise ValueError(f"asset {filename} does not match its pinned size and SHA-256")
                partial = path.with_name(f".{path.name}.partial")
                try:
                    with staged.open("rb") as source, partial.open("wb") as output:
                        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
                            output.write(block)
                    partial.replace(path)
                finally:
                    partial.unlink(missing_ok=True)
        if count != expected_bytes or digest.hexdigest() != expected_sha:
            raise ValueError(f"cached asset {filename} does not match its pinned size and SHA-256")
        heartbeat("asset_verified", filename, force=True)


def blocked_download(*_args, **_kwargs):
    raise RuntimeError("upstream downloads are disabled; use the pinned local assets")


class FoldingModel:
    """One set of weights, one ESM2 encoder and one sampler per batch attempt."""

    def __init__(self, assets):
        import esm
        import lightning.pytorch as lightning
        import numpy
        import simplefold
        import torch

        installed = importlib.metadata.distribution("simplefold").read_text("direct_url.json")
        direct_url = json.loads(installed or "{}")
        if direct_url.get("url") != REPOSITORY or direct_url.get("vcs_info", {}).get("commit_id") != COMMIT:
            raise RuntimeError("installed SimpleFold distribution does not identify the pinned source commit")
        for package, version in (("torch", "2.7.1"), ("fair-esm", "2.0.0"), ("gemmi", "0.7.3")):
            if importlib.metadata.version(package).split("+")[0] != version:
                raise RuntimeError(f"unexpected installed version of {package}")
        sys.path.insert(0, str(Path(simplefold.__file__).resolve().parent))
        from simplefold import inference

        esm_path = assets / "esm2_t36_3B_UR50D.pt"
        inference.esm_registry["esm2_3B"] = lambda: esm.pretrained.load_model_and_alphabet_local(str(esm_path))
        # Upstream's folding checkpoint loader calls os.system only if its checkpoint is missing.
        # Replace its os binding without changing the process-wide os module.
        inference.os = Namespace(path=os.path, makedirs=os.makedirs, system=blocked_download)
        inference.download_fasta_utilities = blocked_download
        torch.serialization.add_safe_globals([Namespace])
        self.args = Namespace(
            simplefold_model=PROTOCOL["model"], ckpt_dir=str(assets), num_steps=SETTINGS["num_steps"],
            tau=SETTINGS["tau"], nsample_per_protein=1, no_log_timesteps=False, backend="torch", plddt=False,
        )
        self.torch, self.lightning, self.numpy = torch, lightning, numpy
        self.inference = inference
        self.model, self.device = inference.initialize_folding_model(self.args)
        self.esm_model, self.esm_dict, self.af2_to_esm = inference.initialize_esm_model(self.args, self.device)
        self.tokenizer, self.featurizer, self.processor, self.flow, self.sampler = (
            inference.initialize_others(self.args, self.device)
        )
        self.ccd = assets / "ccd.pkl"

    def prepare(self, items, work):
        paths = []
        for item in items:
            path = work / f"{identity(item)['input_sha256']}.fasta"
            path.write_text(f">A\n{item['sequence']}\n", encoding="ascii")
            paths.append(path)
        self.inference.process_fastas(data=paths, out_dir=work, ccd_path=self.ccd)

    def fold(self, item, work, heartbeat):
        key = identity(item)["input_sha256"]
        self.lightning.seed_everything(SETTINGS["seed"], workers=True)
        self.numpy.random.seed(SETTINGS["seed"])
        random.seed(SETTINGS["seed"])
        self.torch.manual_seed(SETTINGS["seed"])
        self.torch.cuda.manual_seed_all(SETTINGS["seed"])
        original_step = self.sampler.euler_maruyama_step

        def heartbeat_step(*args):
            heartbeat("sampling", item["id"])
            return original_step(*args)

        with self.torch.no_grad():
            batch, structure, record = self.inference.process_one_inference_structure(
                work / "structures" / f"{key}.npz", work / "records" / f"{key}.json",
                self.tokenizer, self.featurizer, self.processor, self.esm_model, self.esm_dict, self.af2_to_esm,
            )
            self.sampler.euler_maruyama_step = heartbeat_step
            try:
                coords, pad_mask, confidence = self.inference.generate_structure(
                    self.args, batch, self.sampler, self.flow, self.processor, self.model, None, None, self.device,
                )
            finally:
                self.sampler.euler_maruyama_step = original_step
        if coords.shape[0] != 1 or pad_mask.shape[0] != 1 or confidence is not None:
            raise RuntimeError("inference did not return exactly one sample without confidence")
        structure = self.inference.process_structure(deepcopy(structure), coords[0], pad_mask[0], record, backend="torch")
        self.inference.save_structure(structure, work, key, output_format="mmcif", plddts=None)
        return work / f"{key}.cif"


def verify_complete(root, plan, stage, items):
    folder = root / stage
    receipt_raw = (folder / "receipt.json").read_bytes()
    receipt = json.loads(receipt_raw)
    terminal = read_json(folder / "DONE.json")
    if terminal != {"status": "complete", "receipt_sha256": sha256(receipt_raw)}:
        raise ValueError(f"{stage} terminal marker does not match the receipt")
    if (receipt["plan_sha256"], receipt["stage"], receipt["status"]) != (
        sha256(canonical(plan)), stage, "complete",
    ) or receipt["counts"].get("valid") != len(items):
        raise ValueError(f"{stage} receipt is incomplete or belongs to a different plan")
    if set(receipt["items"]) != {row["id"] for row in items}:
        raise ValueError(f"{stage} receipt does not account for the frozen ledger")
    records = {}
    for item in items:
        state_path = folder / "items" / f"{item['id']}.json"
        state_raw = state_path.read_bytes()
        state = json.loads(state_raw)
        if receipt["items"][item["id"]] != sha256(state_raw):
            raise ValueError("per-item receipt hash does not match the batch receipt")
        if state["status"] not in {"completed", "reused"} or state["identity"] != identity(item):
            raise ValueError("per-item receipt does not match the frozen input")
        record = prior_result(root, item)
        if record is None or state["artifacts"] != record["files"]:
            raise ValueError("per-item receipt does not match validated output artifacts")
        records[item["id"]] = record
    return terminal, records


def check_canary_gate(gate, plan, terminal):
    if gate != {
        "schema_version": 1, "plan_sha256": sha256(canonical(plan)),
        "canary_receipt_sha256": terminal["receipt_sha256"],
        "scorer_consumer": "analysis.spacefill_display.read_atoms", "items": plan["items"]["canary"],
        "pass": True,
    }:
        raise ValueError("canary gate is missing or does not match the verified canary outputs")


def execute_batch(root, assets, plan, stage, items, attempt, gate, commit, model_factory=FoldingModel):
    """Remote worker body, also exercisable with synthetic structures and a fake model."""
    validate_items(items, stage)
    if [identity(row) for row in items] != plan["items"][stage]:
        raise ValueError("worker received rows that differ from the frozen plan")
    if (plan["adapter_sha256"], plan["protocol"], plan["research_license_acknowledged"]) != (
        SOURCE_SHA, PROTOCOL, True,
    ) or plan != read_json(root / "plan.json"):
        raise ValueError("worker plan fails the protocol, license, source or campaign gate")
    folder = root / stage
    reservation = folder / "attempts" / f"{attempt}.reservation.json"
    if not 1 <= attempt <= MAX_ATTEMPTS or read_json(reservation)["plan_sha256"] != sha256(canonical(plan)):
        raise ValueError("worker run has no valid local reservation")
    if stage == "full":
        by_id = {row["id"]: row for row in items}
        canary = [by_id[row["id"]] for row in plan["items"]["canary"]]
        terminal, _ = verify_complete(root, plan, "canary", canary)
        check_canary_gate(gate, plan, terminal)
    started_marker = folder / "attempts" / f"{attempt}.started.json"
    if started_marker.exists() or (folder / "DONE.json").exists():
        raise RuntimeError("duplicate worker invocation or replay suppressed")
    write_json(started_marker, {"started_at": now(), "reservation_sha256": sha256(reservation.read_bytes())})
    commit()

    started = time.monotonic()
    stamp = 0.0
    processed = 0

    def heartbeat(operation, current=None, *, force=False):
        nonlocal stamp
        wall = time.monotonic()
        if wall - started >= TIMEOUTS[stage] - 60:
            raise TimeoutError("worker reached its internal deadline")
        if force or wall - stamp >= HEARTBEAT_SECONDS:
            write_json(folder / "heartbeat.json", {
                "operation": operation, "current": current, "attempt": attempt,
                "processed": processed, "total": len(items),
                "elapsed_seconds": round(wall - started, 2), "updated_at": now(),
            })
            commit()
            stamp = wall
            print(f"{stage} attempt {attempt}: {operation} ({processed}/{len(items)})", flush=True)

    runtime = {}
    states = {row["id"]: {"status": "pending", "identity": identity(row)} for row in items}
    try:
        heartbeat("resume_check", force=True)
        needed = {}
        for item in items:
            record = prior_result(root, item)
            key = states[item["id"]]["identity"]["input_sha256"]
            if record is None:
                needed.setdefault(key, item)
        with tempfile.TemporaryDirectory(prefix="simplefold-batch-") as directory:
            work = Path(directory)
            if needed:
                acquire_assets(assets, heartbeat)
                heartbeat("model_load", force=True)
                model = model_factory(assets)
                if model_factory is FoldingModel:
                    runtime = {"gpu": model.torch.cuda.get_device_name(0), "cuda": model.torch.version.cuda}
                heartbeat("input_preprocessing", force=True)
                model.prepare(list(needed.values()), work)
            failed = {}
            for item in items:
                item_id = item["id"]
                state = states[item_id]
                key = state["identity"]["input_sha256"]
                heartbeat("sequence", item_id, force=True)
                item_started = time.monotonic()
                record = prior_result(root, item)
                if record is not None:
                    state.update(status="reused", artifacts=record["files"])
                elif key in failed:
                    state.update(status="duplicate_failed", error_type=failed[key])
                else:
                    try:
                        cif = model.fold(item, work, heartbeat)
                        record = persist_result(root, item, cif)
                        state.update(status="completed", artifacts=record["files"])
                    except TimeoutError:
                        raise
                    except Exception as exc:
                        # Persist the exception class without embedding a private sequence or URL.
                        failed[key] = type(exc).__name__
                        state.update(status="failed", error_type=failed[key])
                state["elapsed_seconds"] = round(time.monotonic() - item_started, 2)
                write_json(folder / "items" / f"{item_id}.json", state)
                processed += 1
                commit()
    except Exception as exc:
        runtime["batch_error_type"] = type(exc).__name__
    counts = {status: sum(state["status"] == status for state in states.values())
              for status in ("completed", "reused", "failed", "duplicate_failed", "pending")}
    counts["valid"] = counts["completed"] + counts["reused"]
    status = "complete" if counts["valid"] == len(items) and "batch_error_type" not in runtime else "incomplete"
    item_hashes = {}
    for item_id, state in states.items():
        raw = canonical(state)
        atomic_write(folder / "items" / f"{item_id}.json", raw)
        item_hashes[item_id] = sha256(raw)
    elapsed = time.monotonic() - started
    receipt = {
        "schema_version": 1, "stage": stage, "attempt": attempt, "status": status,
        "plan_sha256": sha256(canonical(plan)), "items": item_hashes, "counts": counts, "runtime": runtime,
        "elapsed_seconds": round(elapsed, 2), "finished_at": now(),
        "cost_estimate_usd": round(elapsed * PLANNING_RATE_USD_PER_HOUR / 3600, 4),
        "reserved_cost_usd": reserve_cost(stage),
    }
    write_json(folder / "attempts" / f"{attempt}.receipt.json", receipt)
    write_json(folder / "receipt.json", receipt)
    terminal = {"status": status, "receipt_sha256": sha256(canonical(receipt))}
    write_json(folder / "TERMINAL.json", terminal)
    if status == "complete":
        write_json(folder / "DONE.json", terminal)
    commit()
    return receipt


app = modal.App("trace2fold-simplefold3b-variations")
volume = modal.Volume.from_name(VOLUME_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "fair-esm==2.0.0", "gemmi==0.7.3",
                 f"simplefold @ git+{REPOSITORY}@{COMMIT}")
    .env({"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    .add_local_file(Path(__file__).resolve(), "/root/backends/simplefold_modal.py")
)
WORKER = {
    "image": image, "gpu": "H100!", "cpu": (4, 4), "memory": (65536, 65536),
    # Modal's current H100 contract enforces a 512 GiB minimum ephemeral disk
    # request even though this workload uses substantially less local scratch.
    "ephemeral_disk": 524_288, "volumes": {MOUNT: volume}, "min_containers": 0,
    "max_containers": 1, "scaledown_window": 2, "single_use_containers": True,
    "retries": 0, "startup_timeout": STARTUP_SECONDS, "include_source": False,
}


def remote_batch(stage, plan, items, attempt, gate):
    import torch

    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or "H100" not in torch.cuda.get_device_name(0).upper()):
        raise RuntimeError("worker requires exactly one H100 CUDA device")
    volume.reload()
    return execute_batch(MOUNT / CAMPAIGN_DIR, MOUNT / "assets", plan, stage, items, attempt, gate, volume.commit)


@app.function(**WORKER, timeout=TIMEOUTS["canary"])
def run_canary(plan, items, attempt, gate):
    return remote_batch("canary", plan, items, attempt, gate)


@app.function(**WORKER, timeout=TIMEOUTS["full"])
def run_full(plan, items, attempt, gate):
    return remote_batch("full", plan, items, attempt, gate)


def volume_record(relative):
    try:
        return json.loads(b"".join(volume.read_file(f"/{CAMPAIGN_DIR}/{relative}")))
    except FileNotFoundError:
        return None


def store_volume_record(relative, record):
    with volume.batch_upload(force=False) as upload:
        upload.put_file(io.BytesIO(canonical(record)), f"/{CAMPAIGN_DIR}/{relative}")


def require_zero_containers(app_id):
    def modal_json(*arguments):
        result = subprocess.run(["modal", *arguments, "--json"], capture_output=True,
                                text=True, timeout=60, check=True)
        return json.loads(result.stdout)

    matches = [row for row in modal_json("app", "list") if row["App ID"] == app_id]
    if len(matches) != 1 or matches[0]["State"] != "stopped" or str(matches[0]["Tasks"]) != "0":
        raise RuntimeError("Modal must report the application stopped with zero tasks")
    if modal_json("container", "list", "--app-id", app_id) != []:
        raise RuntimeError("Modal reports running containers for the application")
    return {"app_id": app_id, "state": "stopped", "tasks": 0, "running_containers": 0}


def reserve_attempt(plan, stage, resume):
    prior_plan = volume_record("plan.json")
    if prior_plan is None:
        if stage != "canary":
            raise ValueError("no frozen canary campaign exists on the Volume")
        store_volume_record("plan.json", plan)
    elif prior_plan != plan:
        raise ValueError("Volume already belongs to a different frozen plan; run refused")
    if volume_record(f"{stage}/DONE.json") is not None:
        raise ValueError(f"{stage} is already complete; download and validate the existing artifacts")
    prior = {phase: [] for phase in COUNTS}
    for phase in COUNTS:
        for index in range(1, MAX_ATTEMPTS + 1):
            record = volume_record(f"{phase}/attempts/{index}.reservation.json")
            if record is not None:
                require_zero_containers(record["app_id"])
                prior[phase].append(record)
    attempt = len(prior[stage]) + 1
    if attempt > MAX_ATTEMPTS or (attempt > 1 and not resume) or (attempt == 1 and resume):
        raise ValueError("attempts exhausted or resume flag does not match existing attempts")
    spending = sum(record["reserved_cost_usd"] for records in prior.values() for record in records)
    remaining = reserve_cost("full") if stage == "canary" else 0
    if spending + reserve_cost(stage) + remaining > plan["budget"]["max_spend_usd"]:
        raise ValueError("aggregate reservation would exceed the planning budget")
    store_volume_record(f"{stage}/attempts/{attempt}.reservation.json", {
        "app_id": app.app_id, "stage": stage, "attempt": attempt, "reserved_at": now(),
        "reserved_cost_usd": reserve_cost(stage), "plan_sha256": sha256(canonical(plan)),
    })
    return attempt


def supervise(call, stage, attempt):
    started = last_update = time.monotonic()
    last_heartbeat = None
    while time.monotonic() - started < STARTUP_SECONDS + TIMEOUTS[stage]:
        try:
            return call.get(timeout=20)
        # Modal 1.4.x can surface an empty poll as either its compatibility
        # exception or Python's built-in TimeoutError, depending on the SDK
        # runtime. Both mean "no result yet" for this bounded get call.
        except (modal.exception.TimeoutError, TimeoutError):
            pass
        heartbeat = volume_record(f"{stage}/heartbeat.json")
        if heartbeat and heartbeat["attempt"] == attempt and heartbeat != last_heartbeat:
            last_update, last_heartbeat = time.monotonic(), heartbeat
        limit = STARTUP_SECONDS if last_heartbeat is None else SILENCE_SECONDS
        if time.monotonic() - last_update >= limit:
            raise TimeoutError(f"{stage} exceeded its startup or committed heartbeat deadline")
    raise TimeoutError(f"{stage} exceeded its launch-to-finish deadline")


@app.local_entrypoint()
def launch(stage: str, full_inputs: str, canary_inputs: str, plan_file: str, controller_receipt: str,
           max_spend_usd: float, acknowledge_research_license: bool = False,
           canary_gate: str = "", resume: bool = False, remote_launch_allowed: bool = False):
    if not remote_launch_allowed or stage not in COUNTS:
        raise ValueError("explicit remote launch approval and a valid stage are required")
    require_guarded_profile()
    plan, items = build_plan(Path(full_inputs), Path(canary_inputs), max_spend_usd, acknowledge_research_license)
    if Path(plan_file).read_bytes() != canonical(plan):
        raise ValueError("local plan differs from the frozen ledgers, budget, source, or protocol")
    gate = None
    if stage == "full":
        terminal = volume_record("canary/DONE.json")
        if not terminal or not canary_gate:
            raise ValueError("full launch requires a complete canary and local scorer-consumption gate")
        gate = read_json(Path(canary_gate))
        check_canary_gate(gate, plan, terminal)

    lock = Path(tempfile.gettempdir()) / "trace2fold-simplefold3b-variations.lock"
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt = reserve_attempt(plan, stage, resume)
        record = {"stage": stage, "attempt": attempt, "app_id": app.app_id, "started_at": now()}
        path = Path(controller_receipt)
        write_json(path, record)
        call = None
        try:
            call = {"canary": run_canary, "full": run_full}[stage].spawn(plan, items[stage], attempt, gate)
            record["call_id"] = call.object_id
            write_json(path, record)
            receipt = supervise(call, stage, attempt)
            record.update(status=receipt["status"], remote_receipt_sha256=sha256(canonical(receipt)))
        except BaseException as exc:
            record.update(status="interrupted", error_type=type(exc).__name__)
            if call is not None:
                try:
                    call.cancel(terminate_containers=True)
                    record["cancellation_requested"] = True
                except Exception as cancel_error:
                    record["cancellation_error_type"] = type(cancel_error).__name__
            raise
        finally:
            record["finished_at"] = now()
            write_json(path, record)
    if record["status"] != "complete":
        raise RuntimeError(f"{stage} finished without all validated predictions; download the batch receipt")


def local_artifact_check(root, plan, stage, items, export, gate):
    from trace2fold.v4_eval import read_atoms

    if read_json(root / "plan.json") != plan:
        raise ValueError("egress plan does not match the frozen local plan")
    terminal, records = verify_complete(root, plan, stage, items)
    if gate is not None:
        egress = read_json(root / "egress.json")
        if (stage != "canary" or egress["stage"] != stage or egress["status"] != "complete"
                or egress["plan_sha256"] != sha256(canonical(plan))
                or egress["files"]["canary/receipt.json"]["sha256"] != terminal["receipt_sha256"]
                or egress["closeout"]["state"] != "stopped" or egress["closeout"]["tasks"] != 0
                or egress["closeout"]["running_containers"] != 0):
            raise ValueError("the full-launch gate requires verified canary egress and app closeout")
    outputs = []
    for item in items:
        record = records[item["id"]]
        pdb = record["files"]["pdb"]
        trace, _, _, _ = read_atoms((root / pdb["path"]).read_bytes(), item["sequence"])
        if trace.shape != (len(item["sequence"]), 3):
            raise ValueError("SpaceFill reader did not recover every residue")
        outputs.append({"id": item["id"], **record})
    write_json(export, {
        "schema_version": 1, "stage": stage, "protocol": PROTOCOL,
        "plan_sha256": sha256(canonical(plan)), "receipt_sha256": terminal["receipt_sha256"], "items": outputs,
    })
    if gate is not None:
        write_json(gate, {
            "schema_version": 1, "plan_sha256": sha256(canonical(plan)),
            "canary_receipt_sha256": terminal["receipt_sha256"],
            "scorer_consumer": "trace2fold.v4_eval.read_atoms", "items": plan["items"]["canary"],
            "pass": True,
        })


def egress_paths(plan):
    """Allow only the frozen campaign's bounded evidence and structure paths."""
    metadata_limit = 512 * 1024
    limits = {"plan.json": metadata_limit}
    for stage, rows in plan["items"].items():
        for name in ("receipt.json", "heartbeat.json", "TERMINAL.json", "DONE.json"):
            limits[f"{stage}/{name}"] = metadata_limit
        for attempt in range(1, MAX_ATTEMPTS + 1):
            for tag in ("reservation", "started", "receipt"):
                limits[f"{stage}/attempts/{attempt}.{tag}.json"] = metadata_limit
        for row in rows:
            limits[f"{stage}/items/{row['id']}.json"] = metadata_limit
            for ext, cap in (("cif", MAX_STRUCTURE_BYTES), ("pdb", MAX_STRUCTURE_BYTES), ("json", metadata_limit)):
                limits[f"structures/{row['input_sha256']}.{ext}"] = cap
    return limits


def download_snapshot(source, destination, plan, stage, items, controller, *, allow_partial=False,
                      closeout=require_zero_containers):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("egress destination already exists; preserve it and choose a new directory")
    plan_sha = sha256(canonical(plan))
    attempt = controller["attempt"]
    app_id = controller["app_id"]
    if controller["stage"] != stage or type(attempt) is not int or not 1 <= attempt <= MAX_ATTEMPTS:
        raise ValueError("controller receipt belongs to a different stage or attempt")
    closed = closeout(app_id)
    source.hydrate()
    prefix = f"/{CAMPAIGN_DIR}"
    limits = egress_paths(plan)
    partial_paths = {str(Path(path).with_name(f".{Path(path).name}.partial")) for path in limits}

    def remote_json(relative):
        try:
            data = bytearray()
            for block in source.read_file(f"{prefix}/{relative}"):
                if len(data) + len(block) > limits[relative]:
                    raise ValueError("remote record exceeds its egress size bound")
                data.extend(block)
            return json.loads(data)
        except FileNotFoundError:
            return None

    reservation = remote_json(f"{stage}/attempts/{attempt}.reservation.json")
    if remote_json("plan.json") != plan or not reservation or (
        reservation["app_id"], reservation["plan_sha256"], reservation["stage"], reservation["attempt"],
    ) != (app_id, plan_sha, stage, attempt):
        raise ValueError("remote Volume plan or reservation differs from the requested stage")
    terminal = remote_json(f"{stage}/TERMINAL.json")
    done = remote_json(f"{stage}/DONE.json")
    controller_hash = controller.get("remote_receipt_sha256")
    if done and (done != terminal or (controller_hash is not None or not allow_partial)
                 and done.get("receipt_sha256") != controller_hash):
        raise ValueError("remote terminal marker does not match the controller receipt")
    if not done and not allow_partial:
        raise ValueError("the requested stage has no matching completion marker")

    def listing():
        result = {}
        for entry in source.listdir(prefix, recursive=True):
            if entry.type.name == "DIRECTORY":
                continue
            path = entry.path.lstrip("/").removeprefix(f"{CAMPAIGN_DIR}/")
            if allow_partial and path in partial_paths:
                continue
            if path in result or path not in limits or not 0 < entry.size <= limits[path] or entry.type.name != "FILE":
                raise ValueError("Volume contains an unexpected, duplicate or oversized egress artifact")
            result[path] = entry.size
        return result

    before = listing()
    if "plan.json" not in before or sum(before.values()) > 300 * 1024 * 1024:
        raise ValueError("Volume inventory is missing its plan or exceeds the aggregate egress bound")
    with tempfile.TemporaryDirectory(prefix=".simplefold-egress-", dir=destination.parent) as directory:
        staged = Path(directory) / "campaign"
        files = {}
        for relative, expected_size in sorted(before.items()):
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            count, digest = 0, hashlib.sha256()
            with target.open("xb") as output:
                for block in source.read_file(f"{prefix}/{relative}"):
                    count += len(block)
                    if count > expected_size:
                        raise ValueError("remote file exceeded the inventoried byte count")
                    output.write(block)
                    digest.update(block)
            if count != expected_size:
                raise ValueError("remote file differs from the inventoried byte count")
            files[relative] = {"bytes": count, "sha256": digest.hexdigest()}
        if read_json(staged / "plan.json") != plan or remote_json("plan.json") != plan or listing() != before:
            raise ValueError("campaign plan or inventory changed during download")
        if remote_json(f"{stage}/TERMINAL.json") != terminal or remote_json(f"{stage}/DONE.json") != done:
            raise ValueError("stage terminal evidence changed during download")
        if terminal:
            if files[f"{stage}/receipt.json"]["sha256"] != terminal["receipt_sha256"]:
                raise ValueError("downloaded receipt differs from terminal evidence")
            if read_json(staged / stage / "receipt.json")["attempt"] != attempt:
                raise ValueError("downloaded receipt belongs to a different controller attempt")
        if done:
            verify_complete(staged, plan, stage, items)
        if closeout(app_id) != closed:
            raise ValueError("application closeout evidence changed during download")
        write_json(staged / "egress.json", {
            "schema_version": 1, "provider": "modal", "volume": VOLUME_NAME, "volume_id": source.object_id,
            "stage": stage, "attempt": attempt, "plan_sha256": plan_sha, "status": "complete" if done else "partial",
            "closeout": closed, "files": files, "finished_at": now(),
        })
        # An exclusive directory create prevents overwriting a concurrent egress target.
        destination.mkdir()
        try:
            staged.rename(destination)
        except BaseException:
            destination.rmdir()
            raise


def require_guarded_profile():
    result = subprocess.run(["modal", "profile", "current"], capture_output=True, text=True,
                            timeout=60, check=True)
    guarded = result.stdout.strip()
    if (not guarded or os.environ.get("MODAL_PROFILE") != guarded
            or "MODAL_TOKEN_ID" in os.environ or "MODAL_TOKEN_SECRET" in os.environ):
        raise RuntimeError("use the workspace profile returned by modal with token overrides cleared")


def main():
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-canary")
    freeze.add_argument("--full-inputs", type=Path, required=True)
    freeze.add_argument("--canary-inputs", type=Path, required=True)
    for name in ("plan", "verify", "download"):
        command = commands.add_parser(name)
        command.add_argument("--full-inputs", type=Path, required=True)
        command.add_argument("--canary-inputs", type=Path, required=True)
        command.add_argument("--plan-file", type=Path, required=True)
        command.add_argument("--max-spend-usd", type=float, required=True)
        command.add_argument("--acknowledge-research-license", action="store_true")
    verify = commands.choices["verify"]
    verify.add_argument("--stage", choices=COUNTS, required=True)
    verify.add_argument("--artifacts", type=Path, required=True)
    verify.add_argument("--export", type=Path, required=True)
    verify.add_argument("--canary-gate", type=Path)
    download = commands.choices["download"]
    download.add_argument("--stage", choices=COUNTS, required=True)
    download.add_argument("--destination", type=Path, required=True)
    download.add_argument("--controller-receipt", type=Path, required=True)
    download.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze-canary":
        freeze_canary(args.full_inputs, args.canary_inputs)
        print("Frozen three predeclared canary rows and their parent-ledger hash")
        return
    plan, items = build_plan(args.full_inputs, args.canary_inputs, args.max_spend_usd,
                             args.acknowledge_research_license)
    if args.command == "plan":
        if args.plan_file.exists() and args.plan_file.read_bytes() != canonical(plan):
            raise ValueError("existing plan differs; use a new path and preserve the original")
        write_json(args.plan_file, plan)
        print(f"Frozen plan SHA-256 {sha256(canonical(plan))}")
        return
    if args.plan_file.read_bytes() != canonical(plan):
        raise ValueError("local plan differs from its frozen inputs")
    if args.command == "verify":
        local_artifact_check(args.artifacts, plan, args.stage, items[args.stage], args.export, args.canary_gate)
        print(f"Verified {len(items[args.stage])} {args.stage} structures and their SpaceFill input")
    else:
        require_guarded_profile()
        download_snapshot(volume, args.destination, plan, args.stage, items[args.stage],
                          read_json(args.controller_receipt), allow_partial=args.allow_partial)
        print(f"Verified {args.stage} egress and Modal closeout")


if __name__ == "__main__":
    main()
