#!/usr/bin/env python3
"""Zero-GPU Trace2Fold quickstart: inspect v4.3 prompts, parse a reply, and score Twin Delta."""

import json
from pathlib import Path
import sys
import time

import numpy as np

# Allow running directly from a source checkout without installing the wheel first.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trace2fold.cases import load_cases, prompt
from trace2fold.metrics import SPACEFILL_PROFILE, measure
from trace2fold.trials import ANSWER_PARSER, answer_sequence

STARTER_PAIRS_PATH = Path(__file__).resolve().parent / "starter_pilot_pairs.json"


def main():
    t0 = time.perf_counter()
    cases = load_cases(REPO_ROOT)
    twin_map = json.loads(STARTER_PAIRS_PATH.read_text(encoding="utf-8"))
    assigned_id = "gen-l48-02-b"
    twin_id = twin_map[assigned_id]
    assigned = cases[assigned_id]
    twin = cases[twin_id]

    print(f"Loaded starter pair manifest: {STARTER_PAIRS_PATH.relative_to(REPO_ROOT)}")
    print(f"  assigned = {assigned.id} ({assigned.lengths[0]} aa, {assigned.info['family']})")
    print(f"  twin     = {twin.id} ({twin.lengths[0]} aa, {twin.info['family']})")
    print()

    for condition in ("vectors", "geometry"):
        task = prompt(assigned, condition=condition, version="v4.3")
        preview = task.splitlines()[:8]
        print(f"=== v4.3 {condition} prompt (first 8 of {len(task.splitlines())} lines) ===")
        for line in preview:
            print("  " + line)
        print()

    # Validate a realistic LLM reply wrapped in prose + markdown fence with length-aware-v2.
    sample_seq = assigned.info["sequence"]
    raw_reply = (
        "Here is a compact helical sequence designed for the supplied backbone:\n"
        f"```\n{sample_seq}\n```\n"
    )
    extracted = answer_sequence(
        assigned, raw_reply, {"model": "demo-model"}, parser=ANSWER_PARSER
    )
    assert extracted == sample_seq, "length-aware-v2 failed to extract valid sequence"
    print(f"Parsed sequence with {ANSWER_PARSER}: valid={extracted is not None} (len={len(extracted)})")
    print()

    # Score a synthetic near-native Cα trace (target + 0.6 Å Gaussian noise) with spacefill-v4.
    rng = np.random.default_rng(20261122)
    synthetic_ca = assigned.xyz + rng.normal(0.0, 0.6, size=assigned.xyz.shape)

    assigned_metrics, _ = measure(synthetic_ca, assigned, profile=SPACEFILL_PROFILE)
    decoy_metrics, _ = measure(synthetic_ca, twin, profile=SPACEFILL_PROFILE)
    twin_delta = assigned_metrics["score"] - decoy_metrics["score"]

    print(f"=== {SPACEFILL_PROFILE} evaluation (synthetic control Cα trace) ===")
    print(f"  Assigned ({assigned.id}):")
    print(f"    Score              : {assigned_metrics['score']:6.2f} / 100")
    print(f"    GDT-HA / cov (40%) : {assigned_metrics['aligned_coverage']:6.3f}")
    print(f"    Cα lDDT      (25%) : {assigned_metrics['ca_lddt']:6.3f}")
    print(f"    Contact F1   (35%) : {assigned_metrics['contact_f1']:6.3f}")
    print(f"    Kabsch RMSD (det+1): {assigned_metrics['rmsd']:5.2f} Å  (<= 2.0 Å: {assigned_metrics['rmsd'] <= 2.0})")
    print(f"  Paired Twin Decoy ({twin.id}):")
    print(f"    Decoy score        : {decoy_metrics['score']:6.2f} / 100 (RMSD {decoy_metrics['rmsd']:.2f} Å)")
    print(f"  Twin Delta (Assigned - Decoy): {twin_delta:+6.2f} pts")
    print(f"\nCompleted in {(time.perf_counter() - t0) * 1000:.1f} ms (zero GPU required).")


if __name__ == "__main__":
    main()
