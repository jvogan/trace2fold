"""Split qualified C-alpha traces by rigid US-align structural similarity."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import sqlite3
import subprocess
from tempfile import NamedTemporaryFile

from networkx import Graph, connected_components

from trace2fold.cases import load_cases


USALIGN_REVISION = "23f14ffcf8647f2dc1c0e01043db3a7f3a661c28"
USALIGN_SOURCE = "https://github.com/pylelab/USalign"
OPTIONS = ("-mol", "prot", "-mm", "0", "-outfmt", "2")
TM_MIN = 0.85
COVERAGE_PERCENT = 85


def length_allows_match(a, b):
    return 100 * min(a, b) >= COVERAGE_PERCENT * max(a, b)


def align_query(binary, fixtures, lengths, left, rights):
    """Batch a query while retaining both length-normalized scores and coverage."""
    with NamedTemporaryFile(dir=fixtures.parent, mode="w", encoding="ascii") as batch:
        batch.write("\n".join(rights) + "\n")
        batch.flush()
        result = subprocess.run(
            [str(binary), str(fixtures / f"{left}.pdb"), "-dir2", str(fixtures) + "/",
             batch.name, "-suffix", ".pdb", *OPTIONS],
            check=True, capture_output=True, text=True, timeout=180,
        )

    rows = {}
    for line in result.stdout.splitlines():
        if line.startswith("#PDBchain1\t"):
            continue
        fields = line.split("\t")
        if len(fields) != 11:
            raise ValueError(f"Unexpected output in a US-align query for {left}: {line!r}")
        query, other = (Path(field.removesuffix(":A")).stem for field in fields[:2])
        tm1, tm2 = map(float, fields[2:4])
        l1, l2, aligned = map(int, fields[8:])
        if (
            query != left or other not in rights or other in rows
            or l1 != lengths[left] or l2 != lengths[other]
            or not 0 <= aligned <= min(l1, l2)
            or not (0 <= tm1 <= 1 and 0 <= tm2 <= 1)
        ):
            raise ValueError(f"Invalid US-align statistics for {left}/{other}")
        rows[other] = (tm1, tm2, aligned)
    if rows.keys() != set(rights):
        raise ValueError(f"US-align did not account for every comparison for {left}")
    return [(left, other, *rows[other]) for other in rights]


def split_holdout(shapes, render, development_root, binary, cache):
    """Exclude development clusters and retain one deterministic release representative."""
    binary = Path(binary).resolve(strict=True)
    revision = subprocess.check_output(
        ["git", "-C", str(binary.parent), "rev-parse", "HEAD"], text=True,
    ).strip()
    if revision != USALIGN_REVISION:
        raise ValueError(f"US-align checkout must be at {USALIGN_REVISION}")
    binary_hash = sha256(binary.read_bytes()).hexdigest()
    fixtures = Path(cache).resolve() / "ca"
    fixtures.mkdir(parents=True, exist_ok=True)
    lengths = {}

    def collect(xyz):
        payload = render(xyz).encode("ascii")
        key = sha256(payload).hexdigest()
        lengths[key] = len(xyz)
        path = fixtures / f"{key}.pdb"
        try:
            with path.open("xb") as stream:
                stream.write(payload)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f"US-align fixture hash mismatch for {path}") from None
        return key

    candidates = {shape.source: collect(shape.xyz) for shape in sorted(shapes, key=lambda s: s.source)}
    development = {
        case.id: collect(case.xyz)
        for case in load_cases(development_root).values()
        if len(case.lengths) == 1
    }
    if not development:
        raise ValueError("The development dataset contains no single-chain traces")
    release_keys = set(candidates.values())
    development_keys = set(development.values())
    pairs = {
        tuple(sorted((a, b)))
        for a, b in combinations(sorted(release_keys | development_keys), 2)
        if (a in release_keys or b in release_keys) and length_allows_match(lengths[a], lengths[b])
    }

    pending = defaultdict(list)
    database = sqlite3.connect(fixtures.parent / "alignments.sqlite3")
    try:
        database.execute("""
            CREATE TABLE IF NOT EXISTS alignment (
                binary TEXT, left_hash TEXT, right_hash TEXT,
                tm1 REAL, tm2 REAL, aligned INTEGER,
                PRIMARY KEY (binary, left_hash, right_hash)
            )
        """)
        cached = {
            (a, b): (tm1, tm2, aligned)
            for a, b, tm1, tm2, aligned in database.execute(
                "SELECT left_hash, right_hash, tm1, tm2, aligned FROM alignment WHERE binary = ?",
                (binary_hash,),
            )
            if (a, b) in pairs
        }
        for a, b in sorted(pairs - cached.keys()):
            pending[a].append(b)

        def worker(query):
            left, rights = query
            return align_query(binary, fixtures, lengths, left, rights)

        with ThreadPoolExecutor(max_workers=4) as pool:
            for rows in pool.map(worker, pending.items()):
                database.executemany(
                    "INSERT OR IGNORE INTO alignment VALUES (?, ?, ?, ?, ?, ?)",
                    [(binary_hash, *row) for row in rows],
                )
                database.commit()
                cached.update({(a, b): (tm1, tm2, n) for a, b, tm1, tm2, n in rows})
    finally:
        database.close()

    alike = Graph()
    alike.add_nodes_from(lengths)
    close_edges = []
    for (a, b), (tm1, tm2, aligned) in sorted(cached.items()):
        if (
            min(tm1, tm2) >= TM_MIN
            and 100 * aligned >= COVERAGE_PERCENT * max(lengths[a], lengths[b])
        ):
            alike.add_edge(a, b)
            close_edges.append({"sha256": [a, b], "tm_scores": [tm1, tm2],
                                "coverages": [aligned / lengths[a], aligned / lengths[b]]})

    source_ids = defaultdict(list)
    for source, key in candidates.items():
        source_ids[key].append(source)
    dev_ids = defaultdict(list)
    for case_id, key in development.items():
        dev_ids[key].append(case_id)
    keep, clusters = set(), []
    for component in connected_components(alike):
        group = sorted(source for key in component for source in source_ids[key])
        if not group:
            continue
        prior = sorted(case for key in component for case in dev_ids[key])
        representative = None if prior else group[0]
        if representative:
            keep.add(representative)
        if prior or len(group) > 1:
            clusters.append({"candidates": group, "development_cases": prior,
                             "representative": representative})

    return [shape for shape in shapes if shape.source in keep], {
        "method": "rigid US-align C-alpha monomer alignment",
        "source": USALIGN_SOURCE,
        "source_revision": revision,
        "binary_sha256": binary_hash,
        "arguments": list(OPTIONS),
        "tm_minimum_both_normalizations": TM_MIN,
        "aligned_fraction_minimum_both_chains": COVERAGE_PERCENT / 100,
        "partition": "exclude connected components touching development and keep the first candidate path in all others",
        "candidate_coordinate_sha256": candidates,
        "development_coordinate_sha256": dict(sorted(development.items())),
        "comparisons": len(pairs),
        "near_duplicate_edges": close_edges,
        "restricted_clusters": sorted(clusters, key=lambda group: group["candidates"]),
        "qualified_before_split": len(shapes),
        "qualified_after_split": len(keep),
    }
