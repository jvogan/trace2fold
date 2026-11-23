"""Bounded Modal adapters for pure-Cα Genie 3 backbone generation and PDB100 screening."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
from hashlib import file_digest, sha256
from math import isfinite
import os
from pathlib import Path
import re
import subprocess
import tarfile
from tempfile import TemporaryDirectory
from time import monotonic

import modal

from backends.generate_shapes import (
    BATCH_SIZE,
    GENIE_SOURCE,
    GENIE_WEIGHTS,
    ca_count_from_pdb,
    genie_image,
)
from backends.holdout_similarity import USALIGN_REVISION, align_query, length_allows_match
from trace2fold.cases import dump

LENGTHS = (80, 112)
PHASES = (("pilot", 4, 202611211), ("production", 60, 202611212))
GENIE_CHECKPOINT_SHA = "57039afcd4e5a66e58c52ba1ece256bc311a4294a31c37fca7301dd3a4c4479e"
GENIE_CONFIG_SHA = "a01d7ad8b908e4380f45a11c8d691ccf27af3f50a9dfb209823ee99989a343e5"
PDB100_BASE = "https://steineggerlab.s3.amazonaws.com/foldseek"
PDB100_VERSION_RECORD = (
    "aefd75e0a5d6acbe8a2b7791b53eb479 pdb100.tar.gz 250101 PDB_DATE "
    "1815f0d76d7b5807e63b13f9d446dcef43c1f3b1 FOLDSEEK_COMMIT"
)
PDB100_SHA = "2ec2b8b83c5f9ccadc1d88d680cd3a40c04e2bcb4d67ba366cebbf6e8156b090"
PDB100_CHAINS = 324204
PDB100_HIT_LIMIT = 1000
FOLDSEEK_ARCHIVE_SHA = "fd942cf05f922600f20c9a56d2f63c5bf41810106039752c5f806f922b342e84"
FOLDSEEK_BINARY_SHA = "5191c3e9301179ea913a46e6be90e65803364018a9ce0e022575d848b56d4c49"
FOLDSEEK_REVISION = "4f5c00ccb5590f08a261b9144fd9a5d28a71fa0e"
GENERATION = {"gpu": "H100!", "cpu": 4, "memory": 32768, "timeout": 1500, "startup_timeout": 600}
FAMILIARITY = {"cpu": 16, "memory": 32768, "timeout": 3600, "startup_timeout": 300}
GUARDS = {"retries": 0, "max_containers": 1, "scaledown_window": 2}
THREADS = 16
MAX_EGRESS_BYTES = 32 * 1024 * 1024
CORE_SOURCES = (
    "backends.generate_shapes",
    "backends.generate_curate",
    "backends.holdout_similarity",
    "backends.variations_genie_modal",
    "trace2fold",
)

app = modal.App("trace2fold-genie3-variations-v3")
generation_image = genie_image.pip_install("biotite==2.0.0", "networkx==3.7").add_local_python_source(*CORE_SOURCES)
familiarity_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "ca-certificates", "curl", "git", "libgomp1")
    .pip_install("numpy==2.2.6", "biotite==2.0.0", "networkx==3.7")
    .run_commands(
        "git init /opt/USalign && "
        "git -C /opt/USalign fetch --depth 1 https://github.com/pylelab/USalign.git "
        f"{USALIGN_REVISION} && git -C /opt/USalign checkout --detach FETCH_HEAD && "
        "g++ -O3 -std=c++11 -o /opt/USalign/USalign /opt/USalign/USalign.cpp"
    )
    .add_local_python_source(*CORE_SOURCES)
)
if archive := os.environ.get("TRACE2FOLD_FOLDSEEK_ARCHIVE"):
    familiarity_image = familiarity_image.add_local_file(archive, remote_path="/opt/foldseek.tar.gz")


def hash_file(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def attempts():
    return [
        {
            "artifact": f"{phase}/l{length}-{draw:03d}.pdb",
            "phase": phase,
            "phase_seed": seed,
            "length": length,
            "ordinal_within_phase_and_length": draw,
        }
        for phase, count, seed in PHASES
        for length in LENGTHS
        for draw in range(1, count + 1)
    ]


def rows_from_prefilter(text, lengths):
    rows, seen, counts = [], set(), Counter()
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError("Unexpected Foldseek result row")
        query, target, qlen_text, tlen_text = fields
        qlen, tlen = int(qlen_text), int(tlen_text)
        key = query, target
        counts[query] += 1
        if (
            query not in lengths
            or qlen != lengths[query]
            or tlen <= 0
            or key in seen
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", target)
            or counts[query] > PDB100_HIT_LIMIT
        ):
            raise ValueError("Invalid Foldseek PDB100 result row")
        seen.add(key)
        rows.append((query, target, qlen, tlen))
    return rows


def transfer(paths, **metadata):
    outputs = {name: path.read_bytes() for name, path in paths.items()}
    if sum(map(len, outputs.values())) > MAX_EGRESS_BYTES:
        raise ValueError("Modal result exceeds the bounded egress size")
    return {
        "files": outputs,
        "hashes": {name: sha256(payload).hexdigest() for name, payload in outputs.items()},
        **metadata,
    }


def publish(work, lane, plan_sha, result):
    if not app.app_id or not re.fullmatch(r"ap-[A-Za-z0-9]+", app.app_id):
        raise ValueError("Modal app ID is unavailable")
    files = result.pop("files")
    if (
        files.keys() != result["hashes"].keys()
        or sum(map(len, files.values())) > MAX_EGRESS_BYTES
        or not isfinite(result["elapsed_seconds"])
    ):
        raise ValueError("Invalid or oversized Modal egress")
    with TemporaryDirectory(dir=work, prefix=f".{lane}-egress-") as temporary:
        stage = Path(temporary) / lane
        stage.mkdir()
        for name, content in files.items():
            target = stage / name
            if (
                not target.resolve().is_relative_to(stage)
                or sha256(content).hexdigest() != result["hashes"][name]
            ):
                raise ValueError(f"Invalid Modal egress path or hash for {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        dump(
            stage / "receipt.json",
            {"lane": lane, "plan_sha256": plan_sha, "app_id": app.app_id, **result},
        )
        stage.rename(work / lane)


@app.function(image=generation_image, **GENERATION, **GUARDS)
def generate():
    from huggingface_hub import snapshot_download
    import torch

    started = monotonic()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
    if torch.cuda.device_count() != 1 or "H100" not in gpu.upper():
        raise ValueError(
            f"Expected exactly one H100, observed {torch.cuda.device_count()} device(s): {gpu!r}"
        )
    weights = (
        Path(
            snapshot_download(
                "yeqinglin/genie3",
                revision=GENIE_WEIGHTS,
                allow_patterns=["pretrained/v1/**"],
            )
        )
        / "pretrained/v1"
    )
    config, checkpoint = weights / "config.yaml", weights / "checkpoints/step=600000.ckpt"
    if (hash_file(config), hash_file(checkpoint)) != (
        GENIE_CONFIG_SHA,
        GENIE_CHECKPOINT_SHA,
    ):
        raise ValueError("Genie 3 checkpoint or config hash differs from the pinned build")
    start, stop = LENGTHS
    with TemporaryDirectory(prefix="genie-variations-") as directory:
        root = Path(directory)
        files = {}
        for phase, count, seed in PHASES:
            config_path = root / f"{phase}.json"
            phase_output = root / phase
            dump(
                config_path,
                {
                    "experiment": {"name": f"trace2fold_variations_{phase}", "seed": seed},
                    "paths": {"rootdir": str(phase_output)},
                    "generation": {
                        "base": {"checkpoint": str(checkpoint), "config": str(config)},
                        "dataset": {
                            "source": "unconditional",
                            "min_length": start,
                            "max_length": stop,
                            "length_step": stop - start,
                            "batch_size": BATCH_SIZE,
                            "n_sample": count,
                        },
                        "sampler": {"sampler": {"direction_scale": 0.8}},
                    },
                },
            )
            log = root / f"{phase}.log"
            with log.open("w") as stream:
                subprocess.run(
                    [
                        "genie3",
                        "generate",
                        "-c",
                        str(config_path),
                        "--num-devices",
                        "1",
                        "--log-dir",
                        str(root / "logs"),
                    ],
                    cwd="/opt/genie3",
                    check=True,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=GENERATION["timeout"] - (monotonic() - started),
                )
            counts = Counter()
            for native in sorted((phase_output / "pdbs").glob("*.pdb")):
                length = ca_count_from_pdb(native)
                counts[length] += 1
                files[f"{phase}/l{length}-{counts[length]:03d}.pdb"] = native
            if counts != Counter({length: count for length in LENGTHS}):
                raise ValueError(
                    f"Incorrect Genie 3 emission count in phase {phase}: {counts}"
                )
            files[config_path.name] = config_path
            files[log.name] = log
        return transfer(
            files,
            attempts=len(attempts()),
            elapsed_seconds=monotonic() - started,
            gpu=gpu,
            source_revision=GENIE_SOURCE,
            weights_revision=GENIE_WEIGHTS,
            checkpoint_sha256=GENIE_CHECKPOINT_SHA,
            config_sha256=GENIE_CONFIG_SHA,
        )


def run(command, seconds, logs):
    output = subprocess.run(command, capture_output=True, check=False, timeout=seconds)
    logs.write(output.stdout)
    logs.write(output.stderr)
    output.check_returncode()
    return output.stdout


def snapshot(root, logs):
    version = root / "pdb100.version"
    archive = root / "pdb100.tar.gz"
    for path, seconds in ((version, 30), (archive, 600)):
        run(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--location",
                "--connect-timeout",
                "15",
                "--max-time",
                str(seconds),
                "--output",
                str(path),
                f"{PDB100_BASE}/{path.name}",
            ],
            seconds + 10,
            logs,
        )
        if path == version and version.read_text().split() != PDB100_VERSION_RECORD.split():
            raise ValueError("PDB100 source version has changed")
    if hash_file(archive) != PDB100_SHA:
        raise ValueError("Downloaded PDB100 archive does not match the frozen snapshot SHA-256")
    database_root = root / "database"
    with tarfile.open(archive) as packed:
        packed.extractall(database_root, filter="data")
    database = database_root / "pdb"
    indexed = len(database.with_suffix(".index").read_bytes().splitlines())
    if indexed != PDB100_CHAINS:
        raise ValueError("Unexpected PDB100 chain count")
    return database, version


def reference_models(model_stream, keys, fixtures):
    """Write canonical single-chain, all-glycine US-align fixtures from DB keys."""
    emitted = {}
    model = None
    coords = []
    for line in model_stream:
        if line.startswith("MODEL "):
            if model is not None:
                raise ValueError("Nested PDB100 models")
            model = int(line.split()[1])
            if model not in keys or model in emitted:
                raise ValueError("Unexpected or repeated PDB100 model key")
        elif line.startswith("ATOM  ") and model is not None:
            if line[12:16].strip() != "CA" or line[21] != "A":
                raise ValueError("Expected one C-alpha chain for the PDB100 reference")
            xyz = [float(line[offset : offset + 8]) for offset in (30, 38, 46)]
            if not all(isfinite(value) for value in xyz):
                raise ValueError("Nonfinite PDB100 coordinates")
            coords.append(f"{line[:17]}GLY{line[20:54]}\n")
        elif line.startswith("ENDMDL"):
            if model is None or len(coords) != keys[model]:
                raise ValueError("PDB100 conversion changed the indexed chain length")
            alias = f"p{model:07d}"
            path = fixtures / f"{alias}.pdb"
            path.write_text("".join(coords) + "TER\nEND\n", encoding="ascii")
            emitted[model] = alias
            model, coords = None, []
    if model is not None or emitted.keys() != keys.keys():
        raise ValueError("PDB100 model set differs from length-compatible Foldseek hits")
    return emitted


def align_references(executable, database, binary, root, fixtures, hits, logs):
    ranges = {
        (query, target): (qlen, tlen)
        for query, target, qlen, tlen in hits
        if length_allows_match(qlen, tlen)
    }
    if not ranges:
        return "", {}
    target_lengths = {target: tlen for (_, target), (_, tlen) in ranges.items()}
    db_keys = {}
    for row in database.with_suffix(".lookup").read_text().splitlines():
        key, name, _ = row.split("\t", 2)
        if name in target_lengths:
            if name in db_keys:
                raise ValueError("PDB100 chain name is ambiguous in the snapshot lookup")
            db_keys[name] = int(key)
    if db_keys.keys() != target_lengths.keys() or len(set(db_keys.values())) != len(db_keys):
        raise ValueError("Foldseek reference IDs did not resolve to unique PDB100 keys")
    keys = {key: target_lengths[name] for name, key in db_keys.items()}
    selected = root / "selected-keys.txt"
    selected.write_text("\n".join(map(str, sorted(keys))) + "\n")
    subset = root / "subset"
    run([executable, "createsubdb", selected, database, subset], 180, logs)
    run(
        [executable, "base:createsubdb", selected, str(database) + "_h", str(subset) + "_h"],
        180,
        logs,
    )
    models = root / "models.pdb"
    run(
        [
            executable,
            "convert2pdb",
            subset,
            models,
            "--threads",
            str(THREADS),
            "--pdb-output-mode",
            "0",
        ],
        420,
        logs,
    )
    with models.open() as stream:
        emitted = reference_models(stream, keys, fixtures)
    name_by_alias = {emitted[key]: name for name, key in db_keys.items()}
    lengths = {alias: target_lengths[name] for alias, name in name_by_alias.items()}
    targets = defaultdict(list)
    for (query, target), (qlen, _) in sorted(ranges.items()):
        lengths[query] = qlen
        targets[query].append(emitted[db_keys[target]])

    def align(item):
        query, rights = item
        return align_query(binary, fixtures, lengths, query, rights)

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        rows = [
            f"{query}\t{name_by_alias[alias]}\t{tm_a}\t{tm_b}\t{aligned}\n"
            for result in pool.map(align, sorted(targets.items()))
            for query, alias, tm_a, tm_b, aligned in result
        ]
    hashes = {
        name: hash_file(fixtures / f"{alias}.pdb")
        for alias, name in sorted(name_by_alias.items())
    }
    return "".join(rows), hashes


@app.function(image=familiarity_image, **FAMILIARITY, **GUARDS)
def screen(queries, qualification_sha):
    started = monotonic()
    if not 1 <= len(queries) <= len(attempts()):
        raise ValueError("PDB100 query count is outside the bounded cardinality")
    archive = Path("/opt/foldseek.tar.gz")
    if hash_file(archive) != FOLDSEEK_ARCHIVE_SHA:
        raise ValueError("Foldseek archive changed after egress preparation")
    binary = Path("/opt/USalign/USalign")
    with TemporaryDirectory(prefix="genie-pdb100-") as directory:
        root = Path(directory)
        with tarfile.open(archive) as packed:
            packed.extractall(root, filter="data")
        executable = root / "foldseek/bin/foldseek"
        if hash_file(executable) != FOLDSEEK_BINARY_SHA:
            raise ValueError("Foldseek executable differs from the archived PDB100 tool")
        queries_root = root / "queries"
        queries_root.mkdir()
        lengths, hashes = {}, {}
        for alias, (length, expected, content) in sorted(queries.items()):
            if (
                not re.fullmatch(r"q\d{4}", alias)
                or length not in LENGTHS
                or len(content) > 10000
                or sha256(content).hexdigest() != expected
            ):
                raise ValueError("Invalid PDB100 query ID, length, size, or hash")
            path = queries_root / f"{alias}.pdb"
            path.write_bytes(content)
            if ca_count_from_pdb(path) != length:
                raise ValueError("PDB100 query length does not match its roster")
            lengths[alias], hashes[alias] = length, expected
        log = root / "screen.log"
        with log.open("wb") as logs:
            version = run([executable, "version"], 30, logs).decode().strip()
            if version != FOLDSEEK_REVISION:
                raise ValueError("Foldseek revision changed")
            revision = run(["git", "-C", str(binary.parent), "rev-parse", "HEAD"], 30, logs).decode().strip()
            if revision != USALIGN_REVISION:
                raise ValueError("US-align revision changed")
            database, record = snapshot(root, logs)
            qdb, hits, temporary, table = (
                root / name for name in ("qdb", "hits", "tmp", "foldseek.tsv")
            )
            run([executable, "createdb", queries_root, qdb, "--threads", str(THREADS)], 120, logs)
            if len(qdb.with_suffix(".index").read_bytes().splitlines()) != len(queries):
                raise ValueError("Foldseek did not ingest the complete C-alpha roster")
            run(
                [
                    executable,
                    "search",
                    qdb,
                    database,
                    hits,
                    temporary,
                    "--threads",
                    str(THREADS),
                    "--alignment-type",
                    "1",
                    "-a",
                    "-s",
                    "9.5",
                    "--max-seqs",
                    str(PDB100_HIT_LIMIT),
                    "--split-memory-limit",
                    "20G",
                ],
                1200,
                logs,
            )
            run(
                [
                    executable,
                    "convertalis",
                    qdb,
                    database,
                    hits,
                    table,
                    "--threads",
                    str(THREADS),
                    "--format-output",
                    "query,target,qlen,tlen",
                ],
                180,
                logs,
            )
            lookup = qdb.with_suffix(".lookup")
            if {row.split("\t")[1] for row in lookup.read_text().splitlines()} != set(queries):
                raise ValueError("Foldseek query IDs do not match the complete anonymous roster")
            parsed = rows_from_prefilter(table.read_text(), lengths)
            if not parsed:
                raise ValueError("PDB100 search returned no hits for the entire roster")
            aligned, references = align_references(
                executable, database, binary, root, queries_root, parsed, logs
            )
        artifacts = {"screen.log": log, "pdb100.version": record, "query.lookup": lookup}
        reference_manifest = root / "reference-hashes.json"
        dump(reference_manifest, references)
        artifacts[reference_manifest.name] = reference_manifest
        for name, content in (("foldseek", table.read_bytes()), ("usalign", aligned.encode())):
            path = root / f"{name}.tsv.gz"
            path.write_bytes(gzip.compress(content, mtime=0))
            artifacts[path.name] = path
        return transfer(
            artifacts,
            elapsed_seconds=monotonic() - started,
            provenance_scope="returned Foldseek hits only",
            qualification_sha256=qualification_sha,
            queries=hashes,
            database_archive_sha256=PDB100_SHA,
            database_entries=PDB100_CHAINS,
            foldseek_revision=version,
            foldseek_binary_sha256=FOLDSEEK_BINARY_SHA,
            usalign_revision=revision,
            usalign_binary_sha256=hash_file(binary),
        )


@app.local_entrypoint()
def generation(work: str):
    root = Path(work)
    root.mkdir(parents=True, exist_ok=True)
    publish(root, "generation", None, generate.remote())


@app.local_entrypoint()
def pdb100(work: str):
    root = Path(work)
    queries_dir = root / "queries"
    queries = {}
    for idx, pdb_path in enumerate(sorted(queries_dir.glob("*.pdb")), 1):
        alias = f"q{idx:04d}"
        content = pdb_path.read_bytes()
        length = ca_count_from_pdb(pdb_path)
        queries[alias] = (length, sha256(content).hexdigest(), content)
    publish(root, "pdb100", None, screen.remote(queries, None))
