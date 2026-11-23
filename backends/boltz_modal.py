"""Run the independent sequence-only Boltz-2 grader on Modal."""

import hashlib
import importlib.metadata
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
import string
import subprocess
import tempfile
import time

import modal

BOLTZ_VERSION = "2.2.1"
TORCH_VERSION = "2.5.1"
WEIGHTS_REVISION = "6fdef46d763fee7fbb83ca5501ccceff43b85607"
WEIGHTS_SHA256 = "090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1"
CCD_SHA256 = "39e076d96dbec6b4e86982bbda16f3a53a2a60c9bdc17828d88f6f9a0c7d1fd7"
CACHE = Path("/models")
CACHE_VOLUME = os.environ.get("TRACE2FOLD_BOLTZ_VOLUME", "trace2fold-boltz2-cache")
ITEM_TIMEOUT = 180
METHOD_TIMEOUT = 1800
CHUNK_SIZE = 8
WORKER_LIMIT = 8
PROTOCOL = {
    "schema_version": 1,
    "model": "boltz-community/boltz-2",
    "model_revision": WEIGHTS_REVISION,
    "weights_sha256": WEIGHTS_SHA256,
    "ccd_sha256": CCD_SHA256,
    "boltz_version": BOLTZ_VERSION,
    "mode": "protein-sequence-only",
    "msa": False,
    "templates": False,
    "settings": {
        "recycling_steps": 3,
        "sampling_steps": 200,
        "diffusion_samples": 1,
        "seed": 0,
        "use_kernels": False,
        "use_potentials": False,
    },
}

app = modal.App("trace2fold-boltz2")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(f"torch=={TORCH_VERSION}", f"boltz=={BOLTZ_VERSION}")
    .env({"OMP_NUM_THREADS": "2", "PYTHONHASHSEED": "0"})
)
weights = modal.Volume.from_name(CACHE_VOLUME).with_mount_options(read_only=True)


def sequence_spec(sequence):
    # Boltz describes identical chains with a single entity and multiple IDs.
    groups = {}
    for chain_id, chain in zip(string.ascii_uppercase, sequence.split(":")):
        groups.setdefault(chain, []).append(chain_id)
    return {"sequences": [
        {"protein": {"id": ids[0] if len(ids) == 1 else ids, "sequence": chain, "msa": "empty"}}
        for chain, ids in groups.items()
    ]}


def checkpoint_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_result(output, item, runtime, start):
    item_id = item["id"]
    output = output / f"boltz_results_{item_id}" / "predictions" / item_id
    pdb = (output / f"{item_id}_model_0.pdb").read_text()
    confidence = json.loads((output / f"confidence_{item_id}_model_0.json").read_text())
    ptm = float(confidence["ptm"])
    iptm = float(confidence["iptm"]) if ":" in item["sequence"] else None
    if not math.isfinite(ptm) or not 0 <= ptm <= 1:
        raise ValueError("Boltz-2 pTM is out of range")
    if iptm is not None and (not math.isfinite(iptm) or not 0 <= iptm <= 1):
        raise ValueError("Boltz-2 ipTM is out of range")
    return {
        "id": item_id, "pdb": pdb, "runtime": runtime, "ptm": ptm, "iptm": iptm,
        "elapsed_seconds": round(time.monotonic() - start, 3),
    }


@app.cls(
    image=image,
    gpu="H100!",
    cpu=4,
    memory=32768,
    volumes={CACHE: weights},
    block_network=True,
    max_containers=WORKER_LIMIT,
    retries=0,
    scaledown_window=2,
    startup_timeout=600,
    timeout=METHOD_TIMEOUT,
)
class BoltzWorker:
    @modal.enter()
    def verify(self):
        import torch

        if checkpoint_digest(CACHE / "boltz2_conf.ckpt") != WEIGHTS_SHA256:
            raise ValueError("Cached Boltz-2 structure checkpoint differs from the upstream revision")
        if checkpoint_digest(CACHE / "mols.tar") != CCD_SHA256:
            raise ValueError("Cached Boltz-2 molecule dictionary differs from the upstream revision")
        self.runtime = {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "boltz": importlib.metadata.version("boltz"),
        }

    @modal.method(is_generator=True)
    def fold(self, items):
        for item in items:
            start = time.monotonic()
            try:
                with tempfile.TemporaryDirectory(prefix="protein-fold-") as directory:
                    root = Path(directory)
                    spec = root / f"{item['id']}.yaml"
                    spec.write_text(json.dumps(sequence_spec(item["sequence"])))
                    options = [
                        "boltz", "predict", str(spec), "--out_dir", str(root), "--cache", str(CACHE),
                        "--model", "boltz2", "--output_format", "pdb", "--devices", "1",
                        "--accelerator", "gpu", "--num_workers", "0", "--preprocessing-threads", "1",
                        "--no_kernels",
                    ]
                    for name in ("recycling_steps", "sampling_steps", "diffusion_samples", "seed"):
                        options.extend((f"--{name}", str(PROTOCOL["settings"][name])))
                    result = subprocess.run(options, capture_output=True, text=True, timeout=ITEM_TIMEOUT)
                    if result.returncode != 0:
                        diagnostic = result.stderr.strip().splitlines() or result.stdout.strip().splitlines()
                        raise RuntimeError(diagnostic[-1][:500] if diagnostic else f"Boltz-2 exited {result.returncode}")
                    yield read_result(root, item, self.runtime, start)
            except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as exc:
                yield {"id": item["id"], "error": type(exc).__name__, "detail": str(exc)}


def fold_chunk(worker, items):
    results = []
    try:
        results.extend(worker.fold.remote_gen(items))
    except Exception as exc:
        received = {result["id"] for result in results}
        results.extend(
            {"id": item["id"], "error": type(exc).__name__, "detail": str(exc)[:1000]}
            for item in items if item["id"] not in received
        )
    return results


def fold_parallel(items, workers):
    worker = BoltzWorker()
    chunk_size = min(CHUNK_SIZE, math.ceil(len(items) / workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [
            pool.submit(fold_chunk, worker, items[offset:offset + chunk_size])
            for offset in range(0, len(items), chunk_size)
        ]
        for job in as_completed(jobs):
            yield from job.result()


@app.local_entrypoint()
def main(inputs: str, output: str, workers: int = 1, max_items: int = CHUNK_SIZE * WORKER_LIMIT):
    from backends.fold_modal import parse_items, run_batch

    source = Path(inputs)
    if not 1 <= workers <= WORKER_LIMIT:
        raise ValueError(f"Use one to {WORKER_LIMIT} concurrent workers")
    if max_items < 1 or len(parse_items(json.loads(source.read_text()))) > max_items:
        raise ValueError(f"Judge queue exceeds the declared limit of {max_items} items")
    run_batch(
        source, Path(output), protocol=PROTOCOL,
        prediction_iterator=partial(fold_parallel, workers=workers),
    )
