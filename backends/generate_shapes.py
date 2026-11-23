"""Generate single-chain backbones with stock Genie 3 and RFD4-Proteina."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import modal

LENGTHS = (48, 72, 96)
# Confirm both models emit exact-length monomers at batch size 4 before opening
# production. 32 attempts per length and model allow for uncommon beta folds
# and geometry/compactness exclusions before selecting two matched pairs.
PHASES = (("pilot", 4, 81711), ("production", 28, 81712))
BATCH_SIZE = 4
TIMEOUT_SECONDS = 1500
GENIE_SOURCE = "d77ae5ac04212ff1e8b29b585859a3244c614804"
GENIE_WEIGHTS = "9ae31ebb8c56eebdc05ab282a8fd3f6a6d2a03a2"
RFD_SOURCE = "274c366a0981d680bf3ba79924d99da5e91ae107"
RFD_WEIGHTS = "48377e45eb25a4fce8d2aff56c1a659b726d2106"
RFD_MODEL_SHA = "5b549dd61adb2f6c6b5cb95bda13f40f5e31556b4e6c755edc0ca6f011c68fb0"
RFD_CONFIG_SHA = "154ea927ab8005b7eac2d047a792147fb4606ab4878e496aa8414b6b40dad368"
RFD_FIXTURE_SHA = "5e66073220a3314d3426ca5d7828c02492e61d24321966d4d73dff08481cfa37"
RFD_BETA_FIXTURE_SHA = "deef29e7a117ea8797fc000fec1c935b5825ed0aff5e9de5500a3f89c33ded2a"
BETA_PHASES = (("pilot", 4, 81713), ("production", 12, 81714))
app = modal.App("trace2fold-generated-shapes")

genie_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "ca-certificates", "curl", "git", "libgomp1")
    .pip_install("torch==2.7.1", "numpy==2.2.6", "scipy==1.15.3", "lightning==2.6.5", "pydantic==2.12.3")
    .run_commands(
        f"git init /opt/genie3 && git -C /opt/genie3 fetch --depth 1 https://github.com/aqlaboratory/genie3.git {GENIE_SOURCE} && git -C /opt/genie3 checkout --detach FETCH_HEAD && python -m pip install --no-cache-dir --no-build-isolation -e /opt/genie3"
    )
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "OMP_NUM_THREADS": "4", "PYTHONHASHSEED": "0"})
)

rfd_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "gcc", "g++", "libgomp1", "libxrender1", "libxext6", "libsm6")
    .pip_install(
        "torch==2.14.1",
        "bionemo-ir==0.2.0",
        "rfd4-proteina[all-cuda13] @ "
        f"https://github.com/RosettaCommons/RFD4-Proteina/archive/{RFD_SOURCE}.tar.gz",
        extra_index_url="https://download.pytorch.org/whl/cu132",
    )
    .env(
        {"HF_HOME": "/workspace/hf", "DATA_PATH": "/workspace/data", "ALLOW_BIOTITE_CCD": "true",
         "HF_HUB_DISABLE_TELEMETRY": "1", "OMP_NUM_THREADS": "4", "CLUSTER": "local"}
    )
)

GPU = {"gpu": "H100", "cpu": 4, "memory": 32768, "timeout": TIMEOUT_SECONDS,
       "startup_timeout": 600, "retries": 0, "max_containers": 1, "scaledown_window": 2}


def sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(
    files: list[tuple[str, bytes]], counter: Counter, length: int, path: Path,
    lengths: tuple[int, ...] = LENGTHS,
) -> None:
    if length not in lengths:
        raise ValueError(f"Unexpected native length {length} in {path.name}")
    counter[length] += 1
    files.append((f"l{length}-{counter[length]:03d}{path.suffix}", path.read_bytes()))


def check_counts(counter: Counter, phase: str, count: int, lengths: tuple[int, ...] = LENGTHS) -> None:
    if counter != Counter({length: count for length in lengths}):
        raise ValueError(f"Unexpected {phase} counts {dict(counter)}")
    print(json.dumps({"phase": phase, "emitted": dict(counter)}), flush=True)


def genie_length_step(lengths: tuple[int, ...]) -> int:
    steps = {b - a for a, b in zip(lengths, lengths[1:])}
    if len(steps) > 1:
        raise ValueError("Genie 3 needs equally spaced requested lengths")
    return next(iter(steps), 1)


def ca_count_from_pdb(path: Path) -> int:
    positions = {
        line[21:27] for line in path.read_text().splitlines()
        if line.startswith("ATOM  ") and line[12:16].strip() == "CA"
    }
    if len({position[0] for position in positions}) != 1:
        raise ValueError(f"Expected one chain in {path}")
    return len(positions)


def result(root: Path, files: list) -> dict:
    return {
        "files": files,
        "hashes": {name: sha256(content) for name, content in files},
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip(),
        "phase_configs": {
            p.name: p.read_text() for p in root.glob("*.json") if p.is_file()
        },
    }


@app.function(image=genie_image, **GPU)
def genie3(lengths: tuple[int, ...] = LENGTHS) -> dict:
    from huggingface_hub import snapshot_download

    start = time.monotonic()
    length_step = genie_length_step(lengths)
    weights = Path(snapshot_download(
        "yeqinglin/genie3", revision=GENIE_WEIGHTS, allow_patterns=["pretrained/v1/**"]
    )) / "pretrained/v1"
    native_config = weights / "config.yaml"
    checkpoint = weights / "checkpoints/step=600000.ckpt"
    with tempfile.TemporaryDirectory() as tempfile_path:
        root = Path(tempfile_path)
        all_files = []
        for phase, count, seed in PHASES:
            out = root / phase
            config = {
                "experiment": {"name": f"trace2fold_{phase}", "seed": seed},
                "paths": {"rootdir": str(out)},
                "generation": {
                    "base": {"checkpoint": str(checkpoint), "config": str(native_config)},
                    "dataset": {"source": "unconditional", "min_length": lengths[0],
                                "max_length": lengths[-1], "length_step": length_step,
                                "batch_size": BATCH_SIZE, "n_sample": count},
                    "sampler": {"sampler": {"direction_scale": 0.8}},
                },
            }
            config_path = root / f"{phase}.json"
            config_path.write_text(json.dumps(config, indent=2))
            with (root / f"{phase}.log").open("w") as log:
                command = ["genie3", "generate", "-c", str(config_path), "--num-devices", "1",
                           "--log-dir", str(root / "logs")]
                completed = subprocess.run(command, cwd="/opt/genie3", stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError((root / f"{phase}.log").read_text()[-8000:])
            files = []
            counts = Counter()
            for pdb in sorted((out / "pdbs").glob("*.pdb")):
                record(files, counts, ca_count_from_pdb(pdb), pdb, lengths)
            check_counts(counts, phase, count, lengths)
            all_files.extend((f"{phase}/{name}", content) for name, content in files)
        return {
            **result(root, all_files), "elapsed_seconds": time.monotonic() - start,
            "source_revision": GENIE_SOURCE, "weights_revision": GENIE_WEIGHTS,
            "checkpoint_sha256": file_sha256(checkpoint), "config_sha256": file_sha256(native_config),
        }


def monomer_input(template: str, length: int) -> str:
    bound_fields = ("_atom_site.annotation_expsegmin_1_residue", "_atom_site.annotation_expsegmax_1_residue")
    seen = set()
    lines = []
    for line in template.replace("100-100", f"{length}-{length}").splitlines():
        field, _, value = line.partition(" ")
        if field in bound_fields:
            if value.strip() != "100" or field in seen:
                raise ValueError(f"Unexpected pinned fixture length for {field}")
            line = f"{field} {length}"
            seen.add(field)
        lines.append(line)
    if seen != set(bound_fields):
        raise ValueError("Pinned fixture lacks expandable-segment bounds")
    return "\n".join(lines) + "\n"


@app.function(image=rfd_image, **GPU)
def rfd4_proteina(
    mainly_beta: bool = False, lengths: tuple[int, ...] | None = None,
    family: str | None = None, samples_per_length: int | None = None, first_seed: int = 0,
) -> dict:
    import io
    import urllib.request

    from biotite.structure.io.pdbx import CIFFile, get_structure
    from huggingface_hub import hf_hub_download
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import rfproteina
    from rfproteina.generate import load_ckpt_n_configure_inference, run_generation
    import zstandard

    start = time.monotonic()
    for name, expected in (("config.json", RFD_CONFIG_SHA), ("model.safetensors", RFD_MODEL_SHA)):
        path = Path(hf_hub_download("nvidia/RFD4-Proteina", name, revision=RFD_WEIGHTS))
        if file_sha256(path) != expected:
            raise ValueError(f"RFD4 model bundle hash differs for {name}")
    class_codes = {"helical": "1", "beta": "2", "mixed": "3"}
    if family is not None and (family not in class_codes or mainly_beta):
        raise ValueError("Select one broad single-chain shape family")
    class_conditioned = mainly_beta or family is not None
    fixture_path, fixture_sha = (
        ("adherence/monomer_ig", RFD_BETA_FIXTURE_SHA) if class_conditioned else ("monomer", RFD_FIXTURE_SHA)
    )
    if lengths is None:
        lengths = (72,) if mainly_beta else LENGTHS
    phases = BETA_PHASES if mainly_beta else PHASES
    if samples_per_length is not None:
        if samples_per_length < 8 or samples_per_length % BATCH_SIZE:
            raise ValueError("A release batch needs a four-sample pilot and whole production batches")
        phases = (("pilot", 4, first_seed), ("production", samples_per_length - 4, first_seed + 1))
    url = (f"https://raw.githubusercontent.com/RosettaCommons/RFD4-Proteina/{RFD_SOURCE}/"
           f"data/benchmarks/{fixture_path}/monomer-short.cif.zst")
    with urllib.request.urlopen(url, timeout=60) as response:
        fixture = response.read()
    if sha256(fixture) != fixture_sha:
        raise ValueError("Unexpected RFD4 monomer fixture hash")
    with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(fixture)) as reader:
        template = reader.read().decode("utf-8")
    if class_conditioned:
        # This fixture contains an empty expandable segment. Condition only on
        # a broad structural class; no fixed residues, coordinates or partner.
        if "2.60.40.x" not in template:
            raise ValueError("Pinned fixture is missing its class field")
        code = class_codes[family] if family else "2"
        template = template.replace("2.60.40.x", f"{code}.x.x.x")

    config_dir = Path(rfproteina.__file__).parent / "configs"
    with tempfile.TemporaryDirectory() as tempfile_path:
        root = Path(tempfile_path)
        inputs = root / "monomers"
        inputs.mkdir()
        for length in lengths:
            # The pinned fixture is a single empty expandable segment, with
            # exact length repeated in the segment's identifiers and bounds.
            cif = monomer_input(template, length)
            (inputs / f"monomer-{length}.cif").write_text(cif)

        all_files = []
        model = None
        for phase, count, seed in phases:
            out = root / phase
            overrides = [
                f"inputs={inputs}", f"out_dir={out}", "inference_backend=native",
                "hf_ckpt_path=nvidia/RFD4-Proteina", f"hf_ckpt_revision={RFD_WEIGHTS}",
                "recycle_strategy=thinking_medium", f"num_replicates={count}",
                f"batch_size={BATCH_SIZE}", f"seed={seed}", "num_workers=0",
                "~metrics", "compile_lengths=null", "skip_existing=false",
                "compress_outputs=false", "delete_failing_pre_oracle=false",
                f"run_name_={phase}",
            ]
            with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
                cfg = compose(config_name="inference", overrides=overrides)
            (root / f"{phase}.json").write_text(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
            if model is None:
                model = load_ckpt_n_configure_inference(cfg)
            run_generation(cfg, model=model)

            counts = Counter()
            files = []
            for cif in sorted(out.glob("inference_samples/**/*_generated.cif")):
                atoms = get_structure(CIFFile.read(cif), model=1)
                ca = atoms[atoms.atom_name == "CA"]
                if len(set(ca.chain_id)) != 1:
                    raise ValueError(f"Expected one chain in {cif.name}")
                record(files, counts, len(ca), cif, lengths)
            check_counts(counts, phase, count, lengths)
            all_files.extend((f"{phase}/{name}", content) for name, content in files)
        return {
            **result(root, all_files), "elapsed_seconds": time.monotonic() - start,
            "source_revision": RFD_SOURCE, "weights_revision": RFD_WEIGHTS,
            "checkpoint_sha256": RFD_MODEL_SHA, "config_sha256": RFD_CONFIG_SHA,
            "input_template_sha256": fixture_sha,
            "cath_class": f"{code}.x.x.x" if class_conditioned else None,
        }


@app.local_entrypoint()
def main(generator: str, output: str, lengths: str = ""):
    if generator not in {"genie3", "rfd4-proteina", "rfd4-proteina-beta"}:
        raise ValueError(f"Unknown generation lane {generator}")
    mainly_beta = generator == "rfd4-proteina-beta"
    requested = tuple(int(value) for value in lengths.split(",")) if lengths else (
        (72,) if mainly_beta else LENGTHS
    )
    if tuple(sorted(set(requested))) != requested or any(n < 1 for n in requested):
        raise ValueError("Specify distinct positive lengths in increasing order")
    if generator == "genie3":
        genie_length_step(requested)
    destination = Path(output).resolve() / generator
    destination.mkdir(parents=True, exist_ok=False)
    response = (
        genie3.remote(requested) if generator == "genie3"
        else rfd4_proteina.remote(mainly_beta, requested)
    )
    files = response.pop("files")
    for name, content in files:
        if sha256(content) != response["hashes"][name]:
            raise ValueError(f"Artifact hash mismatch for {name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    receipt = {
        **response, "app_id": app.app_id, "lane": generator,
        "generator": "rfd4-proteina" if mainly_beta else generator,
        "lengths": requested,
        "attempts": sum(count for _, count, _ in (BETA_PHASES if mainly_beta else PHASES)) * len(requested),
        "emitted": len(files),
    }
    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("app_id", "generator", "attempts", "emitted", "elapsed_seconds")}))
