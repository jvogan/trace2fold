"""Single bounded Modal job for Foldseek + US-align PDB100 structural novelty screening."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
from hashlib import file_digest, sha256
import json
from math import isfinite
import os
from pathlib import Path
import re
import subprocess
import tarfile
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import monotonic

import modal

PDB100_BASE = "https://steineggerlab.s3.amazonaws.com/foldseek"
PDB100_VERSION_RECORD = (
    "aefd75e0a5d6acbe8a2b7791b53eb479 pdb100.tar.gz 250101 PDB_DATE "
    "1815f0d76d7b5807e63b13f9d446dcef43c1f3b1 FOLDSEEK_COMMIT"
)
PDB100_SHA256 = "2ec2b8b83c5f9ccadc1d88d680cd3a40c04e2bcb4d67ba366cebbf6e8156b090"
PDB100_CHAINS = 324204
FOLDSEEK_ARCHIVE_SHA256 = "fd942cf05f922600f20c9a56d2f63c5bf41810106039752c5f806f922b342e84"
FOLDSEEK_BINARY_SHA256 = "5191c3e9301179ea913a46e6be90e65803364018a9ce0e022575d848b56d4c49"
FOLDSEEK_REVISION = "4f5c00ccb5590f08a261b9144fd9a5d28a71fa0e"
FOLDSEEK_SENSITIVITY = 9.5
FOLDSEEK_MAX_HITS = 1000
USALIGN_REVISION = "23f14ffcf8647f2dc1c0e01043db3a7f3a661c28"
USALIGN_OPTIONS = ("-mol", "prot", "-mm", "0", "-outfmt", "2")
THREADS = 16
MAX_EGRESS_BYTES = 32 * 1024 * 1024
APP_NAME = "trace2fold-pdb100-sidecar-v1"
SOURCES = ("backends.pdb100_sidecar_modal", "trace2fold")

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "ca-certificates", "curl", "git", "libgomp1")
    .run_commands(
        "git init /opt/USalign && "
        "git -C /opt/USalign fetch --depth 1 https://github.com/pylelab/USalign.git "
        f"{USALIGN_REVISION} && git -C /opt/USalign checkout --detach FETCH_HEAD && "
        "g++ -O3 -std=c++11 -o /opt/USalign/USalign /opt/USalign/USalign.cpp"
    )
    .add_local_python_source(*SOURCES)
)
if archive := os.environ.get("TRACE2FOLD_FOLDSEEK_ARCHIVE"):
    image = image.add_local_file(archive, remote_path="/opt/foldseek.tar.gz")


def hash_file(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def length_compatible(a, b):
    return 100 * min(a, b) >= 85 * max(a, b)


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
            or counts[query] > FOLDSEEK_MAX_HITS
        ):
            raise ValueError("Invalid Foldseek PDB100 result row")
        seen.add(key)
        rows.append((query, target, qlen, tlen))
    return rows


def run(command, seconds, logs):
    result = subprocess.run(command, capture_output=True, check=False, timeout=seconds)
    logs.write(result.stdout)
    logs.write(result.stderr)
    result.check_returncode()
    return result.stdout


def align_query(binary, fixtures, lengths, left, rights):
    """Batch one query through pinned global US-align."""
    with NamedTemporaryFile(dir=fixtures.parent, mode="w", encoding="ascii") as batch:
        batch.write("\n".join(rights) + "\n")
        batch.flush()
        result = subprocess.run(
            [
                str(binary),
                str(fixtures / f"{left}.pdb"),
                "-dir2",
                str(fixtures) + "/",
                batch.name,
                "-suffix",
                ".pdb",
                *USALIGN_OPTIONS,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    rows = {}
    for line in result.stdout.splitlines():
        if line.startswith("#PDBchain1\t"):
            continue
        fields = line.split("\t")
        if len(fields) != 11:
            raise ValueError(f"Unexpected US-align output for {left}: {line!r}")
        query, other = (Path(field.removesuffix(":A")).stem for field in fields[:2])
        tm1, tm2 = map(float, fields[2:4])
        l1, l2, aligned = map(int, fields[8:])
        if (
            query != left
            or other not in rights
            or other in rows
            or l1 != lengths[left]
            or l2 != lengths[other]
            or not 0 <= aligned <= min(l1, l2)
            or not (0 <= tm1 <= 1 and 0 <= tm2 <= 1)
        ):
            raise ValueError(f"Invalid US-align row for {left}/{other}")
        rows[other] = tm1, tm2, aligned
    if rows.keys() != set(rights):
        raise ValueError(f"US-align omitted comparisons for {left}")
    return [(left, other, *rows[other]) for other in rights]


def snapshot(root, logs):
    version, archive = root / "pdb100.version", root / "pdb100.tar.gz"
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
    if version.read_text().split() != PDB100_VERSION_RECORD.split():
        raise ValueError("PDB100 source version changed")
    if hash_file(archive) != PDB100_SHA256:
        raise ValueError("Downloaded PDB100 archive differs from the frozen snapshot")
    database_root = root / "database"
    with tarfile.open(archive) as packed:
        packed.extractall(database_root, filter="data")
    database = database_root / "pdb"
    if len(database.with_suffix(".index").read_bytes().splitlines()) != PDB100_CHAINS:
        raise ValueError("Unexpected PDB100 chain count")
    return database, version


def reference_models(stream, keys, fixtures):
    emitted, model, coords = {}, None, []
    for line in stream:
        if line.startswith("MODEL "):
            if model is not None:
                raise ValueError("Nested PDB100 models")
            model = int(line.split()[1])
            if model not in keys or model in emitted:
                raise ValueError("Unexpected PDB100 model key")
        elif line.startswith("ATOM  ") and model is not None:
            if line[12:16].strip() != "CA" or line[21] != "A":
                raise ValueError("Expected a single C-alpha chain")
            xyz = [float(line[offset : offset + 8]) for offset in (30, 38, 46)]
            if not all(isfinite(value) for value in xyz):
                raise ValueError("Nonfinite PDB100 coordinate")
            coords.append(f"{line[:17]}GLY{line[20:54]}\n")
        elif line.startswith("ENDMDL"):
            if model is None or len(coords) != keys[model]:
                raise ValueError("PDB100 conversion changed a chain length")
            alias = f"p{model:07d}"
            (fixtures / f"{alias}.pdb").write_text("".join(coords) + "TER\nEND\n", encoding="ascii")
            emitted[model] = alias
            model, coords = None, []
    if model is not None or emitted.keys() != keys.keys():
        raise ValueError("PDB100 converted model set is incomplete")
    return emitted


def align_references(foldseek, database, usalign, root, fixtures, hits, logs):
    ranges = {
        (q, t): (qlen, tlen)
        for q, t, qlen, tlen in hits
        if length_compatible(qlen, tlen)
    }
    if not ranges:
        return "", {}
    target_lengths = {target: tlen for (_, target), (_, tlen) in ranges.items()}
    db_keys = {}
    for row in database.with_suffix(".lookup").read_text().splitlines():
        key, name, _ = row.split("\t", 2)
        if name in target_lengths:
            if name in db_keys:
                raise ValueError("PDB100 chain name is ambiguous")
            db_keys[name] = int(key)
    if db_keys.keys() != target_lengths.keys() or len(set(db_keys.values())) != len(db_keys):
        raise ValueError("Foldseek reference IDs did not resolve uniquely")
    keys = {key: target_lengths[name] for name, key in db_keys.items()}
    selected = root / "selected-keys.txt"
    selected.write_text("\n".join(map(str, sorted(keys))) + "\n")
    subset = root / "subset"
    run([foldseek, "createsubdb", selected, database, subset], 180, logs)
    run([foldseek, "base:createsubdb", selected, str(database) + "_h", str(subset) + "_h"], 180, logs)
    models = root / "models.pdb"
    run(
        [
            foldseek,
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
        return align_query(usalign, fixtures, lengths, query, rights)

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        rows = [
            f"{query}\t{name_by_alias[alias]}\t{tm1}\t{tm2}\t{aligned}\n"
            for result in pool.map(align, sorted(targets.items()))
            for query, alias, tm1, tm2, aligned in result
        ]
    hashes = {
        name: hash_file(fixtures / f"{alias}.pdb")
        for alias, name in sorted(name_by_alias.items())
    }
    return "".join(rows), hashes


@app.function(
    image=image,
    cpu=16,
    memory=32768,
    timeout=3600,
    startup_timeout=300,
    retries=0,
    max_containers=1,
    scaledown_window=2,
)
def screen(queries, plan_sha256):
    started = monotonic()
    if not queries:
        raise ValueError("Expected at least one anonymous PDB100 query")
    archive = Path("/opt/foldseek.tar.gz")
    if hash_file(archive) != FOLDSEEK_ARCHIVE_SHA256:
        raise ValueError("Foldseek archive differs from the frozen tool")
    usalign = Path("/opt/USalign/USalign")
    with TemporaryDirectory(prefix="trace2fold-pdb100-") as directory:
        root = Path(directory)
        with tarfile.open(archive) as packed:
            packed.extractall(root, filter="data")
        foldseek = root / "foldseek/bin/foldseek"
        if hash_file(foldseek) != FOLDSEEK_BINARY_SHA256:
            raise ValueError("Foldseek binary differs from the frozen tool")
        fixtures = root / "queries"
        fixtures.mkdir()
        lengths, hashes = {}, {}
        for alias, row in sorted(queries.items()):
            payload = row["payload"]
            if (
                not re.fullmatch(r"q\d{4}", alias)
                or len(payload) > 100000
                or sha256(payload).hexdigest() != row["sha256"]
            ):
                raise ValueError("Invalid PDB100 query payload")
            (fixtures / f"{alias}.pdb").write_bytes(payload)
            if payload.count(b" CA ") != row["length"]:
                raise ValueError("PDB100 query residue count changed")
            lengths[alias], hashes[alias] = row["length"], row["sha256"]
        log = root / "screen.log"
        with log.open("wb") as logs:
            version = run([foldseek, "version"], 30, logs).decode().strip()
            if version != FOLDSEEK_REVISION:
                raise ValueError("Foldseek revision changed")
            revision = run(["git", "-C", str(usalign.parent), "rev-parse", "HEAD"], 30, logs).decode().strip()
            if revision != USALIGN_REVISION:
                raise ValueError("US-align revision changed")
            database, version_file = snapshot(root, logs)
            qdb, hits, temporary, table = (
                root / name for name in ("qdb", "hits", "tmp", "foldseek.tsv")
            )
            run([foldseek, "createdb", fixtures, qdb, "--threads", str(THREADS)], 120, logs)
            if len(qdb.with_suffix(".index").read_bytes().splitlines()) != len(queries):
                raise ValueError("Foldseek did not ingest all queries")
            run(
                [
                    foldseek,
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
                    str(FOLDSEEK_SENSITIVITY),
                    "--max-seqs",
                    str(FOLDSEEK_MAX_HITS),
                    "--split-memory-limit",
                    "20G",
                ],
                1200,
                logs,
            )
            run(
                [
                    foldseek,
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
                raise ValueError("Foldseek query lookup differs from the roster")
            parsed = rows_from_prefilter(table.read_text(), lengths)
            if not parsed:
                raise ValueError("PDB100 returned no Foldseek hits")
            aligned, references = align_references(
                foldseek, database, usalign, root, fixtures, parsed, logs
            )
        reference_manifest = root / "reference-hashes.json"
        reference_manifest.write_text(json.dumps(references, indent=2, sort_keys=True) + "\n")
        outputs = {
            "screen.log": log.read_bytes(),
            "pdb100.version": version_file.read_bytes(),
            "query.lookup": lookup.read_bytes(),
            "reference-hashes.json": reference_manifest.read_bytes(),
            "foldseek.tsv.gz": gzip.compress(table.read_bytes(), mtime=0),
            "usalign.tsv.gz": gzip.compress(aligned.encode(), mtime=0),
        }
        if sum(map(len, outputs.values())) > MAX_EGRESS_BYTES:
            raise ValueError("PDB100 Modal egress exceeds its finite bound")
        return {
            "files": outputs,
            "artifact_sha256": {name: sha256(data).hexdigest() for name, data in outputs.items()},
            "plan_sha256": plan_sha256,
            "elapsed_seconds": monotonic() - started,
            "queries": hashes,
            "database_archive_sha256": PDB100_SHA256,
            "database_entries": PDB100_CHAINS,
            "foldseek_revision": version,
            "foldseek_binary_sha256": FOLDSEEK_BINARY_SHA256,
            "usalign_revision": revision,
            "usalign_binary_sha256": hash_file(usalign),
            "provenance_scope": "every length-compatible hit among at most 1000 Foldseek hits per query",
        }


@app.local_entrypoint()
def main(work: str):
    work = Path(work)
    destination = work / "pdb100"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    queries_dir = work / "queries"
    queries = {}
    for idx, pdb_path in enumerate(sorted(queries_dir.glob("*.pdb")), 1):
        alias = f"q{idx:04d}"
        payload = pdb_path.read_bytes()
        queries[alias] = {
            "length": payload.count(b" CA "),
            "sha256": sha256(payload).hexdigest(),
            "payload": payload,
        }
    plan_path = work / "plan.json"
    plan_sha = hash_file(plan_path) if plan_path.is_file() else None
    result = screen.remote(queries, plan_sha)
    files = result.pop("files")
    result["app_id"] = app.app_id
    if not re.fullmatch(r"ap-[A-Za-z0-9]+", result["app_id"] or ""):
        raise ValueError("Modal application ID is unavailable")
    with TemporaryDirectory(dir=work, prefix=".pdb100-egress-") as temporary:
        stage = Path(temporary) / "pdb100"
        stage.mkdir()
        for name, payload in files.items():
            path = stage / name
            if path.name != name or sha256(payload).hexdigest() != result["artifact_sha256"][name]:
                raise ValueError("Invalid PDB100 egress artifact")
            path.write_bytes(payload)
        (stage / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        stage.rename(destination)
    print(json.dumps({"app_id": result["app_id"], "elapsed_seconds": result["elapsed_seconds"]}, sort_keys=True))
