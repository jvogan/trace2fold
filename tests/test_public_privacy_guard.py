"""Public repository privacy and holdout-protection guard.

Walks the entire public repository and asserts zero private harness identifiers,
local filesystem paths, secret keys, or holdout coordinates/sequences.
Public author attribution ("Jacob Vogan" / "jvogan") is explicitly allowed.
"""

import hashlib
import json
from pathlib import Path
import re
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = frozenset({".git", "__pycache__", ".venv", ".ruff_cache", ".pytest_cache"})
TEXT_SUFFIXES = frozenset(
    {
        ".py",
        ".md",
        ".txt",
        ".json",
        ".jsonl",
        ".toml",
        ".cff",
        ".yml",
        ".yaml",
        ".html",
        ".css",
        ".js",
        ".sh",
        ".pdb",
        ".svg",
        "",
    }
)

# Generic structural hygiene patterns (disclose no private identifiers).
GENERIC_PATTERNS = (
    ("users_dir", re.compile(r"/[U]sers/")),
    ("volumes_dir", re.compile(r"/[V]olumes/")),
    ("win_users_dir", re.compile(r"[C]:\\[U]sers\\", re.IGNORECASE)),
    ("holdout_zip", re.compile(r"[t]argets\.zip", re.IGNORECASE)),
    ("oauth_mod", re.compile(r"[o]auth_player", re.IGNORECASE)),
    ("secret_key_prefix", re.compile(r"[s]k-[A-Za-z0-9]{20,}")),
    ("modal_entity_id", re.compile(r"\b(?:ap|vo|wk|ta)-[A-Za-z0-9]{12,}\b")),
)

INFRA_METADATA_KEYS = frozenset(
    {
        "app_id",
        "volume_id",
        "workspace_id",
        "task_id",
        "modal_app",
        "gpu",
        "gpu_type",
        "elapsed_seconds",
        "wall_clock_seconds",
        "cost_usd",
    }
)
INFRA_HARDWARE_RE = re.compile(
    r"\b(?:NVIDIA\s+[A-Z0-9-]+|H100|A100|A10G|L40S|B200)\b", re.IGNORECASE
)

GUARD_SALT = b"trace2fold-public-guard-v1:"
TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{3,}")
# One-way salted SHA-256 digests of forbidden internal tokens.
FORBIDDEN_TOKEN_HASHES = frozenset(
    {
        "1928a10235fffa725e48760a0aa7043a14ec8e1c7ba032c7bb94694a3405f0a9",
        "391edc47a5fea43ff6334b94805bdc5c135ab3f734597faab513d80dfe983559",
        "417d8efccad2d7b0538c3c4a6d8fa89d4741d5765dc71ff75acab6a7f0b9272c",
        "58d2beefdc41978faaa65bc014fac547a6a577fdf8c426c9c42cc2eb411a0f80",
        "6665e4ba9399e68ad6043b200f0661935ce7276df5a3954808cf83b019afad53",
        "7a72583b29764618c5851e0cac3749503ef2106bd54469d9105d03420041e31d",
        "7f6a9da97e5fddd65e5ac9b27557edac1b099f3bb31c1377161dbe0b02fafd05",
        "8edcae9a458f7f802d024c62efd745994ea2bcfeb0c971216dc49eb97e7717e5",
        "98c9173757f199aab912037728b35a36b3a9d5856772515a6ccb3e1d19867fab",
        "9effd53ea8364767fa4987decc21e3d8e3246d45722d9de8f1209807c9b60577",
        "ab3c75445b00066fff675356f4df8e35bc7f73ee177dd49a9aa25c23622ed383",
        "b5f50ac56153195e8c88628172989e61061488fc1c81ff3ca66b22daf7ea78d7",
        "b7292ef0341a06781bd9b992513f2abffca5a8b5a20cd4f3d154ed5c2920a886",
        "b843e1fbdf91dc73646e8d95662f2bf4e79f4cfc1341fa4b61f23b1d75571691",
        "c9be5f976814429d2feb18124cc16316b47f367002b4c27456ee922bf8706b76",
        "d1a5d119a3130691bd91b9b4c85cf22294706d83692b32dbfd5918034d513349",
        "e3372daee51914892d7fcfefb244b1f476bb4debde206585c079836b64481553",
        "eb32535a49f25852719eb2f91d081a80a4bf98393845d09d0539acb6ce8c2835",
    }
)

FORBIDDEN_FILENAMES = (
    re.compile(r"^release-.*\.pdb$", re.IGNORECASE),
    re.compile(r"^stage2-.*\.pdb$", re.IGNORECASE),
    re.compile(r"^genie-.*\.pdb$", re.IGNORECASE),
    re.compile(r"^[t]argets\.zip$", re.IGNORECASE),
)

AA_SET = frozenset("ACDEFGHIKLMNPQRSTVWY")
LONG_AA_SEQ_RE = re.compile(r"^[A-Z]{40,}$")


def _iter_repo_files():
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def _iter_candidate_tokens(text):
    for match in TOKEN_RE.finditer(text):
        tok = match.group(0).lower()
        yield tok
        if "-" in tok:
            parts = [p for p in tok.split("-") if p]
            for part in parts:
                if len(part) >= 3:
                    yield part
            for i in range(len(parts) - 1):
                joined = f"{parts[i]}-{parts[i + 1]}"
                if len(joined) >= 3:
                    yield joined
        if "_" in tok:
            parts = [p for p in tok.split("_") if p]
            for part in parts:
                if len(part) >= 3:
                    yield part
            for i in range(len(parts) - 1):
                joined = f"{parts[i]}_{parts[i + 1]}"
                if len(joined) >= 3:
                    yield joined


def _scan_line(line):
    hits = []
    for label, pattern in GENERIC_PATTERNS:
        if pattern.search(line):
            hits.append(label)
    for tok in _iter_candidate_tokens(line):
        digest = hashlib.sha256(GUARD_SALT + tok.encode("utf-8")).hexdigest()
        if digest in FORBIDDEN_TOKEN_HASHES:
            hits.append(f"token_hash:{digest[:12]}")
    return hits


class PublicPrivacyGuardTests(unittest.TestCase):
    def test_zero_private_tokens_across_entire_repository(self):
        violations = []
        for path in _iter_repo_files():
            rel = path.relative_to(REPO_ROOT).as_posix()
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for line_no, line in enumerate(text.splitlines(), 1):
                for label in _scan_line(line):
                    violations.append(f"{rel}:{line_no} [{label}]: {line.strip()[:120]}")
        self.assertEqual(
            violations,
            [],
            "Private/internal references detected in public repository:\n"
            + "\n".join(violations),
        )

    def test_zero_holdout_archives_or_pdb_coordinates_present(self):
        forbidden_files = []
        for path in _iter_repo_files():
            for pattern in FORBIDDEN_FILENAMES:
                if pattern.match(path.name):
                    forbidden_files.append(path.relative_to(REPO_ROOT).as_posix())
        self.assertEqual(
            forbidden_files,
            [],
            f"Holdout coordinate/archive files must never be included: {forbidden_files}",
        )

    def test_embedded_benchmark_html_has_no_holdout_sequences_or_forbidden_tokens(self):
        html_path = REPO_ROOT / "docs" / "index.html"
        self.assertTrue(html_path.is_file(), "docs/index.html is missing")
        html = html_path.read_text(encoding="utf-8")
        match = re.search(
            r'<script[^>]*\bid=["\']t2f-benchmark-data["\'][^>]*>(.*?)</script>',
            html,
            re.DOTALL,
        )
        self.assertIsNotNone(
            match,
            'Expected <script type="application/json" id="t2f-benchmark-data"> in docs/index.html',
        )
        raw_json = match.group(1).strip()

        token_hits = _scan_line(raw_json)
        self.assertEqual(
            token_hits,
            [],
            f"Forbidden tokens found in embedded docs/index.html JSON: {token_hits}",
        )

        payload = json.loads(raw_json)
        seq_leaks = []

        def walk(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if path == "" and k == "dev3d":
                        continue
                    walk(v, f"{path}.{k}" if path else k)
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    walk(item, f"{path}[{idx}]")
            elif isinstance(obj, str):
                if (
                    LONG_AA_SEQ_RE.match(obj)
                    and set(obj) <= AA_SET
                    and len(set(obj)) >= 5
                ):
                    seq_leaks.append(f"{path} (len={len(obj)})")

        walk(payload)
        self.assertEqual(
            seq_leaks,
            [],
            f"Holdout amino-acid sequences found outside dev3d in docs/index.html: {seq_leaks}",
        )

    def test_public_starter_and_calibration_data_complete(self):
        gen_pdbs = sorted((REPO_ROOT / "data/generated").glob("gen-*.pdb"))
        known_pdbs = sorted((REPO_ROOT / "data/known").glob("*.pdb"))
        self.assertEqual(len(gen_pdbs), 20)
        self.assertEqual(
            [p.name for p in known_pdbs],
            ["gfp.pdb", "insulin.pdb", "trp-cage.pdb", "vhh-cdr3.pdb"],
        )

    def test_zero_run_infrastructure_metadata_in_public_data_and_benchmarks(self):
        leaks = []
        json_paths = sorted((REPO_ROOT / "data").rglob("*.json")) + sorted(
            (REPO_ROOT / "benchmarks").rglob("*.json")
        )

        def walk_infra(obj, rel, json_path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    cur = f"{json_path}.{k}" if json_path else str(k)
                    if str(k).lower() in INFRA_METADATA_KEYS:
                        leaks.append(f"{rel}:{cur} (forbidden key {k!r})")
                    walk_infra(v, rel, cur)
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    walk_infra(item, rel, f"{json_path}[{idx}]")
            elif isinstance(obj, str):
                if INFRA_HARDWARE_RE.search(obj):
                    leaks.append(f"{rel}:{json_path} (hardware string {obj!r})")

        for path in json_paths:
            rel = path.relative_to(REPO_ROOT).as_posix()
            walk_infra(json.loads(path.read_text(encoding="utf-8")), rel)

        self.assertEqual(
            leaks,
            [],
            "Run-specific infrastructure/GPU/Modal metadata found in public JSON:\n"
            + "\n".join(leaks),
        )
