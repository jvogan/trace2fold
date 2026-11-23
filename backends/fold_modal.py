"""Run sequence-only ESMFold2-Fast prediction on Modal."""

import fcntl
import hashlib
import importlib.metadata
import json
import math
import re
import string
import time
from pathlib import Path

import modal

MODEL = "biohub/ESMFold2-Fast"
REVISION = "45fe8656f5b3ef493c17fcf9abe9a2968902e712"
CCD_REVISION = "69869f737beffec5294845ede23db5fc0b4f509e"
ESM_VERSION = "3.4.1.post1"
CACHE = Path("/models")
MODEL_PATH = CACHE / "hub" / "models--biohub--ESMFold2-Fast" / "snapshots" / REVISION
CCD_PATH = CACHE / "hub" / "models--biohub--ESMFold2" / "snapshots" / CCD_REVISION / "ccd.pkl"
SETTINGS = {"num_loops": 20, "num_sampling_steps": 200, "num_diffusion_samples": 1,
            "seed": 0, "lm_dropout": 0.3}
PROTOCOL = {
    "schema_version": 1,
    "model": MODEL,
    "model_revision": REVISION,
    "ccd_revision": CCD_REVISION,
    "esm_version": ESM_VERSION,
    "mode": "protein-sequence-only",
    "msa": False,
    "templates": False,
    "settings": SETTINGS,
}
RESIDUES = dict(zip(
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
    "ACDEFGHIKLMNPQRSTVWY",
    strict=True,
))
ITEM_ID = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}\Z")
SEQUENCE = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]+(?::[ACDEFGHIKLMNPQRSTVWY]+)*\Z")

app = modal.App("trace2fold-esmfold2")
image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install(f"esm=={ESM_VERSION}", "gemmi==0.7.5")
    .env({"HF_HOME": str(CACHE), "HF_HUB_OFFLINE": "1", "ESMCFOLD_CCD_PATH": str(CCD_PATH)})
)
weights = modal.Volume.from_name("esmfold2-models").with_mount_options(read_only=True)


def parse_items(data):
    if not isinstance(data, dict) or set(data) != {"items"}:
        raise ValueError("input must contain just an items array")
    if not isinstance(data["items"], list) or not data["items"]:
        raise ValueError("items must be a nonempty array")
    seen = set()
    for item in data["items"]:
        if not isinstance(item, dict) or set(item) != {"id", "sequence"}:
            raise ValueError("each item must contain just id and sequence")
        item_id, sequence = item["id"], item["sequence"]
        if not isinstance(item_id, str) or not ITEM_ID.fullmatch(item_id):
            raise ValueError("id must use letters, digits, dots, underscores or hyphens")
        # Avoid filename collisions on case-insensitive filesystems.
        normalized_id = item_id.casefold()
        if normalized_id == "manifest" or normalized_id in seen:
            raise ValueError(f"duplicate or reserved id: {item_id}")
        seen.add(normalized_id)
        if not isinstance(sequence, str) or not SEQUENCE.fullmatch(sequence):
            raise ValueError(f"{item_id}: use uppercase canonical amino acids and colon-delimited chains")
        chains = sequence.split(":")
        if len(chains) > len(string.ascii_uppercase) or sum(map(len, chains)) > 9999:
            raise ValueError(f"{item_id}: too long for PDB output")
    return data["items"]


def read_ca(pdb, sequence):
    """Check chain/residue identity and extract sequence-ordered PDB values."""
    chains = sequence.split(":")
    chain_ids = string.ascii_uppercase[:len(chains)]
    expected = {
        (chain_id, index): residue
        for chain_id, subsequence in zip(chain_ids, chains, strict=True)
        for index, residue in enumerate(subsequence, 1)
    }
    atoms = {}
    for line in pdb.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if len(line) < 66 or line[21] not in chain_ids or line.startswith("HETATM"):
            raise ValueError("unexpected atom record or chain in prediction")
        if line[12:16].strip() != "CA":
            continue
        key = (line[21], int(line[22:26]))
        if key in atoms or line[16] != " " or line[26] != " ":
            raise ValueError("duplicate or alternate C-alpha in prediction")
        residue = RESIDUES.get(line[17:20])
        if key not in expected or expected[key] != residue:
            raise ValueError("prediction residues do not match the input sequence")
        values = [float(line[start:stop]) for start, stop in ((30, 38), (38, 46), (46, 54), (60, 66))]
        if not all(map(math.isfinite, values)) or not 0 <= values[3] <= 100:
            raise ValueError("non-finite coordinates or pLDDT outside 0–100")
        atoms[key] = values
    if atoms.keys() != expected.keys():
        raise ValueError("prediction is missing one or more C-alpha atoms")
    return [atoms[key][:3] for key in expected], [atoms[key][3] for key in expected]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_atomic(path, data):
    partial = path.with_name(f".{path.name}.tmp")
    partial.write_bytes(data)
    partial.replace(path)


def write_json(path, data):
    write_atomic(path, (json.dumps(data, indent=2, allow_nan=False) + "\n").encode())


def prior_result(root, item, protocol=PROTOCOL):
    item_id = item["id"]
    json_path = root / f"{item_id}.json"
    pdb_path = root / f"{item_id}.pdb"
    if not json_path.exists():
        # A PDB without its sidecar is safe to regenerate after interruption.
        return None
    previous = json.loads(json_path.read_text())
    if previous["sequence"] != item["sequence"] or previous["provenance"]["protocol"] != protocol:
        raise ValueError(f"{item_id}: output already belongs to a different sequence or protocol")
    if not pdb_path.is_file() or sha256(pdb_path.read_bytes()) != previous["pdb_sha256"]:
        return None
    ca, plddt = read_ca(pdb_path.read_text(), item["sequence"])
    if (previous["id"], previous["ca"], previous["plddt"]) != (item_id, ca, plddt):
        raise ValueError(f"{item_id}: existing sidecar disagrees with its PDB")
    return previous["pdb_sha256"]


@app.cls(
    image=image, gpu="H100!", cpu=4, memory=32768, volumes={CACHE: weights},
    block_network=True, max_containers=1, retries=0, startup_timeout=600, timeout=1800,
)
class FoldWorker:
    @modal.enter()
    def load(self):
        import torch
        from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model

        self.model = EsmFold2Model.from_pretrained(str(MODEL_PATH), device="cuda").eval()
        self.builder = ESMFold2InputBuilder()
        self.runtime = {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "gemmi": importlib.metadata.version("gemmi"),
        }

    @modal.method(is_generator=True)
    def fold(self, items):
        import gemmi
        from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

        for item in items:
            start = time.monotonic()
            chains = item["sequence"].split(":")
            spec = StructurePredictionInput(sequences=[
                ProteinInput(id=chain_id, sequence=sequence, msa=None)
                for chain_id, sequence in zip(string.ascii_uppercase[:len(chains)], chains, strict=True)
            ])
            try:
                prediction = self.builder.fold(self.model, spec, **SETTINGS)
                cif = gemmi.cif.read_string(prediction.complex.to_mmcif())
                pdb = gemmi.make_structure_from_block(cif.sole_block()).make_pdb_string()
                yield {
                    "id": item["id"], "pdb": pdb, "runtime": self.runtime,
                    "ptm": prediction.ptm, "iptm": prediction.iptm if len(chains) > 1 else None,
                    "elapsed_seconds": round(time.monotonic() - start, 3),
                }
            except Exception as exc:
                yield {"id": item["id"], "error": type(exc).__name__, "detail": str(exc)}


def save_result(root, item, prediction, protocol=PROTOCOL):
    pdb = prediction["pdb"]
    ca, plddt = read_ca(pdb, item["sequence"])
    pdb_bytes = pdb.encode()
    checksum = sha256(pdb_bytes)
    chains = item["sequence"].split(":")
    record = {
        **item,
        "chains": [
            {"id": string.ascii_uppercase[index], "length": len(chain)}
            for index, chain in enumerate(chains)
        ],
        "ca": ca,
        "plddt": plddt,
        "mean_plddt": sum(plddt) / len(plddt),
        "ptm": prediction["ptm"],
        "iptm": prediction["iptm"],
        "pdb_sha256": checksum,
        "sequence_sha256": sha256(item["sequence"].encode()),
        "elapsed_seconds": prediction["elapsed_seconds"],
        "provenance": {"protocol": protocol, "runtime": prediction["runtime"]},
    }
    write_atomic(root / f"{item['id']}.pdb", pdb_bytes)
    write_json(root / f"{item['id']}.json", record)
    return checksum


def run_batch(source, root, *, protocol=PROTOCOL, worker_factory=None, batch_size=32, prediction_iterator=None):
    data = source.read_bytes()
    items = parse_items(json.loads(data))
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".fold.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        states = {}
        outstanding = []
        for item in items:
            item_id = item["id"]
            previous = prior_result(root, item, protocol)
            if previous is None:
                states[item_id] = {"status": "pending"}
                outstanding.append(item)
            else:
                states[item_id] = {"status": "reused", "pdb_sha256": previous}
        manifest = {"protocol": protocol, "inputs_sha256": sha256(data), "items": states}
        start = time.monotonic()

        def sequential_predictions():
            worker = (worker_factory or FoldWorker)()
            for offset in range(0, len(outstanding), batch_size):
                yield from worker.fold.remote_gen(outstanding[offset:offset + batch_size])

        try:
            if outstanding:
                lookup = {item["id"]: item for item in outstanding}
                results = sequential_predictions() if prediction_iterator is None else prediction_iterator(outstanding)
                for result in results:
                    item_id = result["id"]
                    if "error" in result:
                        states[item_id] = {"status": "failed", "error": result["error"], "detail": result["detail"]}
                        continue
                    try:
                        checksum = save_result(root, lookup[item_id], result, protocol)
                    except ValueError as exc:
                        states[item_id] = {"status": "failed", "error": type(exc).__name__, "detail": str(exc)}
                    else:
                        states[item_id] = {"status": "completed", "pdb_sha256": checksum}
        finally:
            counts = {key: sum(state["status"] == key for state in states.values())
                      for key in ("completed", "reused", "failed", "pending")}
            manifest["counts"] = counts
            manifest["elapsed_seconds"] = round(time.monotonic() - start, 3)
            write_json(root / "manifest.json", manifest)
        if counts["failed"] or counts["pending"]:
            raise RuntimeError(f"fold batch incomplete: {counts}; see {root / 'manifest.json'}")
        print(f"Folded {counts['completed']}, reused {counts['reused']} validated predictions")


@app.local_entrypoint()
def main(inputs: str, output: str):
    run_batch(Path(inputs), Path(output))
