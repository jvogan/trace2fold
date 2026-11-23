"""Unit tests for matched-round consensus, condition comparison, and twin-pair statistics."""

import unittest

from trace2fold.consensus import condition_comparison, consensus, review_comparison
from trace2fold.report import paired_statistics


def _judge_report(model_name, rows):
    return {"judge": {"model": model_name}, "rows": rows}


def _row(case, condition, score, rmsd, margin=None, pair="p1"):
    return {
        "case": case,
        "condition": condition,
        "model": "sample",
        "pair": pair,
        "tier": "generated",
        "family": "helical",
        "generator": "rfd4-proteina",
        "length": 48,
        "status": "graded",
        "score": score,
        "rmsd": rmsd,
        "aligned_coverage": score / 100.0,
        "ca_lddt": score / 100.0,
        "contact_f1": score / 100.0,
        "margin": margin,
        "target_sha256": f"t-{case}",
        "sequence_sha256": f"s-{case}-{condition}",
    }


class MatchedRoundTests(unittest.TestCase):
    def test_consensus_and_condition_comparison_preserve_paired_gains(self):
        esm = _judge_report(
            "biohub/ESMFold2-Fast",
            [
                _row("a", "geometry", 80.0, 1.5, margin=30.0),
                _row("b", "geometry", 60.0, 2.5, margin=10.0),
                _row("a", "length", 40.0, 6.0, margin=-5.0),
                _row("b", "length", 30.0, 7.0, margin=5.0),
                _row("a", "review", 85.0, 1.2, margin=35.0),
                _row("a", "restart", 76.0, 1.8, margin=26.0),
            ],
        )
        boltz = _judge_report(
            "boltz-community/boltz-2",
            [
                _row("a", "geometry", 70.0, 1.8, margin=20.0),
                _row("b", "geometry", 50.0, 2.8, margin=10.0),
                _row("a", "length", 40.0, 6.5, margin=-5.0),
                _row("b", "length", 30.0, 7.5, margin=5.0),
                _row("a", "review", 75.0, 1.6, margin=25.0),
                _row("a", "restart", 74.0, 1.9, margin=24.0),
            ],
        )
        combined = consensus(esm, boltz)
        self.assertEqual(len(combined), 6)
        geo_a = next(r for r in combined if r["case"] == "a" and r["condition"] == "geometry")
        self.assertAlmostEqual(geo_a["score"], 75.0)
        self.assertAlmostEqual(geo_a["margin"], 25.0)
        self.assertTrue(geo_a["both_within_2a"])

        cond = {r["case"]: r for r in condition_comparison(combined)}
        self.assertAlmostEqual(cond["a"]["geometry_minus_length"], 35.0)
        self.assertAlmostEqual(cond["a"]["margin_gain"], 30.0)
        self.assertAlmostEqual(cond["b"]["geometry_minus_length"], 25.0)

        rev = review_comparison(combined)
        self.assertEqual(len(rev), 1)
        self.assertAlmostEqual(rev[0]["revision_gain"], 5.0)
        self.assertAlmostEqual(rev[0]["review_minus_restart"], 5.0)

    def test_paired_statistics_counts_complete_and_incomplete_pairs(self):
        rows = [
            {"pair": "p1", "margin": 20.0},
            {"pair": "p1", "margin": 10.0},
            {"pair": "p2", "margin": 15.0},
            {"pair": "p2", "margin": 5.0},
            {"pair": "p3", "margin": 8.0},
        ]
        stats = paired_statistics(rows)
        self.assertEqual(stats["complete_pairs"], 2)
        self.assertEqual(stats["incomplete_pairs"], 1)
        self.assertEqual(stats["wins"], 2)
        self.assertEqual(stats["losses"], 0)
        self.assertAlmostEqual(stats["mean_pair_margin"], 12.5)
        self.assertEqual(stats["one_sided_sign_p"], 0.25)
        self.assertIsNotNone(stats["bootstrap_95pct"])
        lo, hi = stats["bootstrap_95pct"]
        self.assertLessEqual(lo, hi)

    def test_incompatible_consensus_reports_rejected(self):
        esm = _judge_report("biohub/ESMFold2-Fast", [_row("a", "geometry", 80.0, 1.5)])
        boltz = _judge_report("boltz-community/boltz-2", [_row("b", "geometry", 80.0, 1.5)])
        with self.assertRaises(ValueError):
            consensus(esm, boltz)
