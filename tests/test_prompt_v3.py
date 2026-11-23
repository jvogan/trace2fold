from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

from trace2fold.audit import CONDITIONS, DEFAULT_CONDITIONS, audit, panel
from trace2fold.cases import AA, digest, dump, load_cases, prompt
from trace2fold.cli import main
from trace2fold.session import run_all
from trace2fold.trials import revision_prompt

MODEL = "demo-model"
ANSWER = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "model": MODEL,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": ""},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
}


@contextmanager
def stub(payload):
    observed = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            observed.append(json.loads(self.rfile.read(length)))
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/v1/chat/completions", observed
    finally:
        server.shutdown()
        server.server_close()


class PromptV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()
        cls.pilot_ids = ["gen-l48-01-a", "gen-l48-01-b", "gen-l72-04-a", "gen-l72-04-b"]

    def test_v3_geometry_and_length_prompts_preserve_coordinates(self):
        for case_id in self.pilot_ids:
            with self.subTest(case=case_id):
                case = self.cases[case_id]
                geometry = prompt(case, version="v3")
                rows = geometry.splitlines()[4:]
                self.assertEqual(
                    rows, prompt(case, version="v2").split("\n\n")[-1].splitlines()
                )
                n = case.lengths[0]
                self.assertEqual(len(rows), n)
                self.assertEqual(
                    prompt(case, "length", "v3"),
                    f"Choose one amino acid for each of the {n} positions so the chain is likely to fold.\n"
                    f"Return one {n}-letter sequence. Use only {AA}.",
                )
        with patch(
            "sys.argv",
            ["trace2fold", "prompt", "gen-l48-01-a", "--prompt-version", "v3"],
        ):
            with redirect_stdout(StringIO()) as output:
                main()
        self.assertEqual(
            output.getvalue(), prompt(self.cases["gen-l48-01-a"], version="v3") + "\n"
        )

    def test_ineligible_conditions_and_cases_rejected(self):
        case = self.cases["gen-l48-01-a"]
        for condition in set(CONDITIONS) - {"geometry", "length"}:
            with self.subTest(condition=condition):
                with self.assertRaisesRegex(ValueError, "v3 supports only"):
                    panel(self.cases, [case.id], ["geometry", condition], version="v3")
                with self.assertRaisesRegex(ValueError, "v3 supports only"):
                    prompt(case, condition, "v3")
        with self.assertRaisesRegex(ValueError, "v3 supports only"):
            revision_prompt(case, "first", "v3")
        for case_id in ("insulin", "vhh-cdr3"):
            with self.assertRaisesRegex(ValueError, "v3 requires one chain"):
                panel(self.cases, [case_id], ["length"], version="v3")

    def test_v1_and_v2_panels_are_deterministic(self):
        design_v1 = panel(
            self.cases,
            ["gen-l48-01-a", "gen-l48-01-b"],
            list(DEFAULT_CONDITIONS),
            version="v1",
        )
        self.assertNotIn("prompt_version", design_v1)
        self.assertEqual(design_v1["calls_planned"], 8)
        design_v2 = panel(
            self.cases,
            ["gen-l48-01-a", "gen-l48-01-b"],
            list(DEFAULT_CONDITIONS),
            version="v2",
        )
        self.assertEqual(design_v2["prompt_version"], "v2")
        self.assertEqual(design_v2["calls_planned"], 8)

    def test_run_audit_and_replay(self):
        answer = deepcopy(ANSWER)
        answer["choices"][0]["message"]["content"] = "A" * 48
        with TemporaryDirectory() as temp, stub(answer) as (endpoint, observed):
            run = Path(temp)
            ids = list(self.pilot_ids)
            design = panel(
                self.cases, ids, ["geometry", "length"], MODEL, endpoint, "v3"
            )
            dump(run / "design.json", design)
            self.assertEqual(
                design["representation_prompts"]["length"],
                [digest(prompt(self.cases[i], "length", "v3")) for i in ids],
            )
            run_all(self.cases, run)
            run_all(self.cases, run)
            result = audit(self.cases, run)
            self.assertEqual(
                (len(observed), result["verified_calls"], result["valid"]), (8, 8, 4)
            )
