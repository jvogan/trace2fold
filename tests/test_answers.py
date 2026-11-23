from pathlib import Path
import tempfile
import unittest

from trace2fold.cases import load_cases
from trace2fold.trials import ANSWER_PARSER, LEGACY_PARSER, answer_sequence, controls, lock


class AnswerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()
        cls.case = cls.cases["gen-l48-01-a"]
        cls.seq = "ACDEFGHIKLMNPQRSTVWY" * 2 + "ACDEFGHI"

    def answer(self, text, case=None, **protocol):
        return answer_sequence(case or self.case, text, protocol)

    def test_a_bare_sequence_counts_whole(self):
        self.assertEqual(self.answer(self.seq), self.seq)
        self.assertEqual(self.answer("  " + self.seq + "\n"), self.seq)

    def test_wrappers_around_the_sequence_do_not_cost_the_answer(self):
        seq = self.seq
        for text in (
            f"```\n{seq}\n```",
            f"Sequence: {seq}",
            f"**Sequence:** {seq}",
            f"The helices pack against the strands.\n\n{seq}\n\nDone.",
            f'"{seq}"',
            f"{seq}.",
        ):
            self.assertEqual(self.answer(text), seq, text)

    def test_the_last_sequence_is_the_answer(self):
        other = self.seq[::-1]
        self.assertEqual(self.answer(f"{self.seq}\nOn reflection: {other}"), other)
        self.assertEqual(self.answer(f"{self.seq}\n{self.seq[:-1]}"), self.seq)

    def test_whitespace_and_wrapped_blocks_preserve_residue_order(self):
        seq = self.seq
        for text in (
            seq[:28] + " " + seq[28:],
            "```" + seq[:28] + " " + seq[28:] + "```",
            seq[:17] + "\r\n" + seq[17:],
            "Sequence:\n```text\r\n" + seq[:12] + "\t" + seq[12:] + "\r\n```\nLoop: NPDLSGKE",
            f"| Sequence | `{seq}` |\n| Helix 5 | `PELARLIEELARALEELLK` |",
        ):
            self.assertEqual(self.answer(text), seq)
            self.assertIsNone(answer_sequence(self.case, text, {}, parser=LEGACY_PARSER))
            self.assertIsNone(answer_sequence(self.case, text, {}, bare_only=True))
        other = seq[::-1]
        block = f"```\n{seq[:12]}\n{seq[12:]}\n```"
        self.assertEqual(self.answer(f"{other}\nFinal:\n{block}"), seq)
        self.assertEqual(self.answer(f"{block}\nFinal: {other}"), other)
        self.assertIsNone(self.answer(f"First segment: {seq[:28]}\nThen use {seq[28:]}"))
        self.assertIsNone(self.answer(seq[:23] + "-" + seq[23:]))

    def test_invalid_answers_stay_invalid(self):
        seq = self.seq
        for text in (
            "I cannot help with that.",
            seq[:-1],
            seq + "A",
            seq[:-1] + "B",
            seq.lower(),
            "",
        ):
            self.assertIsNone(self.answer(text), text)
        for status in ("refused", "truncated", "empty"):
            self.assertIsNone(self.answer(seq, answer_status=status))

    def test_fixed_residues_and_chain_breaks_are_still_enforced(self):
        vhh = self.cases["vhh-cdr3"]
        good = controls(vhh)["positive"]
        self.assertEqual(self.answer(f"Here it is: {good}", vhh), good)
        changed = ("A" if good[0] != "A" else "G") + good[1:]
        self.assertIsNone(self.answer(f"Here it is: {changed}", vhh))
        insulin = self.cases["insulin"]
        two = controls(insulin)["positive"]
        self.assertEqual(self.answer(f"Answer: {two}", insulin), two)
        self.assertIsNone(self.answer(two.replace(":", ""), insulin))

    def test_locked_record_says_whether_the_reply_was_bare(self):
        with tempfile.TemporaryDirectory() as td:
            rows = {
                name: lock(self.case, Path(td) / name, "geometry", text, {"model": "test"})
                for name, text in {
                    "bare": self.seq,
                    "wrapped": f"Sequence: {self.seq}",
                    "bad": "no sequence here",
                }.items()
            }
        self.assertEqual([rows[k]["format"] for k in ("bare", "wrapped", "bad")], ["bare", "extracted", None])
        self.assertEqual([rows[k]["valid"] for k in ("bare", "wrapped", "bad")], [True, True, False])
        self.assertEqual(rows["wrapped"]["sequence"], self.seq)
        self.assertIsNone(rows["bad"]["sequence"])
        self.assertEqual(rows["wrapped"]["raw"], f"Sequence: {self.seq}")
        self.assertEqual(rows["wrapped"]["answer_parser"], ANSWER_PARSER)


if __name__ == "__main__":
    unittest.main()
