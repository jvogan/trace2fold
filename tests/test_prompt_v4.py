"""Check the frozen v4/v4.1/v4.2/v4.3 input representations for the generated shape starter set."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from trace2fold.audit import panel
from trace2fold.backbone import _backbone_coordinates, side_chain_directions
from trace2fold.cases import AA, V4_CONDITIONS, condition_case, digest, load_cases, prompt
from trace2fold.cli import main
from trace2fold.topology import contact_pairs, load_annotations

PILOT = ("gen-l48-02-b", "gen-l48-03-b")


def coordinate_rows(task):
    return [
        line.split()
        for line in task.splitlines()
        if line[:1].isdigit() and len(line.split()) in {4, 7}
    ]


class PromptV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()

    def test_indexed_control_matches_across_equal_lengths(self):
        tasks = [prompt(self.cases[key], "length", "v4") for key in PILOT]
        self.assertEqual(*tasks)
        self.assertEqual(
            tasks[0].splitlines()[-1].split(), list(map(str, range(1, 49)))
        )
        self.assertNotIn("Cα", tasks[0])

    def test_conditions_share_instructions_and_add_only_their_input(self):
        for key in PILOT:
            with self.subTest(case=key):
                case = self.cases[key]
                tasks = {arm: prompt(case, arm, "v4") for arm in V4_CONDITIONS}
                self.assertEqual(
                    len({tuple(text.splitlines()[:2]) for text in tasks.values()}), 1
                )
                for text in tasks.values():
                    self.assertNotIn(key, text)
                    self.assertNotIn(case.info["sequence"], text)
                self.assertEqual(len(coordinate_rows(tasks["length"])), 0)
                xyz_rows = coordinate_rows(tasks["geometry"])
                self.assertEqual(len(xyz_rows), 48)
                vector_rows = coordinate_rows(tasks["vectors"])
                self.assertEqual([line[:4] for line in vector_rows], xyz_rows)
                self.assertEqual(coordinate_rows(tasks["masked"]), vector_rows)
                self.assertTrue(
                    tasks["vectors-contacts"].startswith(tasks["vectors"] + "\n")
                )
                expected = ", ".join(
                    f"{i}-{j}" for i, j in contact_pairs(case.xyz, case.lengths)
                )
                self.assertEqual(tasks["vectors-contacts"].splitlines()[-1], expected)
                printed = np.array([row[4:] for row in vector_rows], dtype=float)
                np.testing.assert_allclose(
                    printed, side_chain_directions(case), atol=0.005001
                )
                np.testing.assert_allclose(
                    np.linalg.norm(printed, axis=1), 1, atol=0.009
                )

    def test_mask_holds_shell_letters_and_scores_only_buried_positions(self):
        case = self.cases[PILOT[0]]
        effective = condition_case(case, "masked")
        self.assertTrue(case.mutable.all())
        self.assertIs(condition_case(case, "vectors"), case)
        self.assertIs(effective.xyz, case.xyz)
        self.assertEqual(effective.template.count("?"), 12)
        self.assertIn(effective.template, prompt(case, "masked", "v4"))
        self.assertTrue(effective.validate(case.info["sequence"]))
        self.assertFalse(effective.validate("A" * case.lengths[0]))
        residues_with_contacts = set(
            np.asarray(contact_pairs(case.xyz, case.lengths)).flat
        )
        self.assertTrue(set(effective.info["mutable"]) <= residues_with_contacts)
        self.assertTrue(set(effective.info["mutable"]) <= set(range(4, 46)))
        self.assertEqual(
            effective.template,
            "".join(
                "?" if bit else letter
                for bit, letter in zip(effective.mutable, case.info["sequence"])
            ),
        )

    def test_original_atom_coverage_and_protected_inputs(self):
        for case in self.cases.values():
            if case.info.get("generator") == "rfd4-proteina":
                atoms = _backbone_coordinates(case)
                np.testing.assert_array_equal(atoms[:, 1], case.xyz)
        with self.assertRaisesRegex(ValueError, "No frozen N/CA/C/O backbone"):
            prompt(self.cases["gen-l48-02-a"], "vectors", "v4")
        with self.assertRaisesRegex(ValueError, "v4 requires a generated single chain"):
            prompt(self.cases["trp-cage"], "length", "v4")
        annotations = load_annotations()
        left, right = (annotations[key].chains[0] for key in PILOT)
        self.assertEqual(sum(a == b for a, b in zip(left, right)), 41)

    def test_roster_freezes_each_v4_prompt(self):
        arms = list(V4_CONDITIONS)
        design = panel(self.cases, list(PILOT), arms, version="v4")
        self.assertEqual(design["calls_planned"], 10)
        for arm in arms:
            expected = [digest(prompt(self.cases[key], arm, "v4")) for key in PILOT]
            hashes = (
                design["target_prompts"]
                if arm == "geometry"
                else design["representation_prompts"][arm]
            )
            self.assertEqual(hashes, expected)

    def test_ranked_version_changes_words_and_preserves_the_pilot_and_payloads(self):
        expected_vectors = (
            "8fc59354756dd495e1370ebf5589dcfb9b99035b0cc80a32e0a11e90ba9e046d",
            "e8c25b3178aeb9bfd8e3136a89de5d5876aa8f5c56e3af27a126438bedfa6f3d",
        )
        expected_length = "47f32f49da5ed3ea10accfb95fe5ab4a5cd91acc33c5dd53c519bc06dba2c3aa"
        objective = (
            "Choose one amino acid for each of the 48 positions. "
            "If a target shape is supplied, choose a protein sequence whose predicted fold matches it as closely as possible."
        )
        for key, expected in zip(PILOT, expected_vectors, strict=True):
            case = self.cases[key]
            self.assertEqual(digest(prompt(case, "vectors", "v4")), expected)
            self.assertEqual(digest(prompt(case, "length", "v4")), expected_length)
            for arm in V4_CONDITIONS:
                previous = prompt(case, arm, "v4")
                ranked = prompt(case, arm, "v4.1")
                self.assertEqual(ranked.splitlines()[0], objective)
                self.assertEqual(ranked.splitlines()[1], previous.splitlines()[1])
                self.assertEqual(coordinate_rows(ranked), coordinate_rows(previous))
                self.assertEqual(ranked.splitlines()[-1], previous.splitlines()[-1])
            self.assertIn("estimated unit direction", prompt(case, "vectors", "v4.1"))
        ranked_design = panel(self.cases, list(PILOT), list(V4_CONDITIONS), version="v4.1")
        self.assertEqual(ranked_design["prompt_version"], "v4.1")
        self.assertEqual(
            ranked_design["representation_prompts"]["length"],
            [digest(prompt(self.cases[key], "length", "v4.1")) for key in PILOT],
        )

    def test_cli_previews_a_separate_dataset(self):
        case = self.cases[PILOT[0]]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / case.info["path"]
            source.parent.mkdir(parents=True)
            source.write_bytes((case.root / case.info["path"]).read_bytes())
            (source.parent / "index.json").write_text(
                json.dumps([{**case.info, "id": "another-case"}])
            )
            for command in (
                ["list"],
                [
                    "prompt",
                    "another-case",
                    "--condition",
                    "length",
                    "--prompt-version",
                    "v4",
                ],
            ):
                output = StringIO()
                with (
                    patch(
                        "sys.argv",
                        ["trace2fold", *command, "--dataset-root", directory],
                    ),
                    redirect_stdout(output),
                ):
                    main()
                expected = (
                    "another-case"
                    if command[0] == "list"
                    else prompt(case, "length", "v4")
                )
                self.assertIn(expected, output.getvalue())

    def test_v42_and_v43_instructions_and_cli_preview(self):
        case = self.cases[PILOT[0]]
        new_tasks = {arm: prompt(case, arm, "v4.2") for arm in V4_CONDITIONS}
        headers = {tuple(task.splitlines()[:3]) for task in new_tasks.values()}
        self.assertEqual(len(headers), 1)
        [(objective, constraints, answer_format)] = headers
        self.assertIn("If a target backbone is supplied", objective)
        self.assertIn("at corresponding residue positions", objective)
        self.assertIn("Otherwise, choose a sequence likely to fold.", objective)
        self.assertEqual(constraints, "Work directly from this message and reply once. Do not use tools.")
        self.assertEqual(answer_format, f"Return one 48-letter sequence. Use only {AA}.")
        for arm, task in new_tasks.items():
            self.assertEqual(
                task.splitlines()[3:], prompt(case, arm, "v4.1").splitlines()[2:]
            )
        self.assertEqual(new_tasks["length"], prompt(self.cases[PILOT[1]], "length", "v4.2"))
        self.assertEqual(coordinate_rows(new_tasks["length"]), [])

        # v4.3 supports vectors and geometry with explicit one-line ASCII format
        v43_vec = prompt(case, "vectors", "v4.3")
        v43_geo = prompt(case, "geometry", "v4.3")
        self.assertIn("uppercase ASCII letters", v43_vec)
        self.assertEqual(len(coordinate_rows(v43_vec)), 48)
        self.assertEqual(len(coordinate_rows(v43_geo)), 48)
        with self.assertRaisesRegex(ValueError, "v4.3 supports only"):
            prompt(case, "length", "v4.3")

        preview = StringIO()
        with (
            patch(
                "sys.argv",
                ["trace2fold", "prompt", case.id, "--condition", "vectors", "--prompt-version", "v4.2"],
            ),
            redirect_stdout(preview),
        ):
            main()
        self.assertEqual(preview.getvalue(), new_tasks["vectors"] + "\n")

    def test_generic_score_rejects_v4_before_using_legacy_metrics(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "records"
            records.mkdir()
            (records / "trial.json").write_text("{}")
            (root / "design.json").write_text('{"prompt_version": "v4"}')
            command = [
                "trace2fold",
                "score",
                str(records),
                str(root / "folds"),
                str(root / "scores.json"),
            ]
            error = StringIO()
            with (
                patch("sys.argv", command),
                redirect_stderr(error),
                self.assertRaises(SystemExit),
            ):
                main()
            self.assertIn("grade-v4", error.getvalue())
