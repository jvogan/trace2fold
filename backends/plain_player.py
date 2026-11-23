"""Submit one tool-free Chat Completions request to a trusted loopback server."""
from __future__ import annotations

import hashlib
from http.client import HTTPConnection
import ipaddress
import json
import re
from pathlib import Path
from urllib.parse import urlsplit


ROUTE = "loopback-chat-completions"
API_PATH = "/v1/chat/completions"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
TIMEOUT_SECONDS = 900


def reported_cli_version(value: object) -> bool:
    """Recognize a printable version from a CLI without pinning a release."""
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9][A-Za-z0-9.+_-]{0,99}", value))


def route_address(endpoint: str) -> tuple[str, int]:
    if not isinstance(endpoint, str) or not endpoint.isascii() or any(c.isspace() or not c.isprintable() for c in endpoint):
        raise ValueError("endpoint must be an explicit loopback URL")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "http" or parsed.path != API_PATH
        or parsed.username is not None or parsed.password is not None
        or "?" in endpoint or "#" in endpoint or parsed.port is None
    ):
        raise ValueError("endpoint must use http, an explicit port, and /v1/chat/completions")
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
    except ValueError as exc:
        raise ValueError("endpoint must use a literal loopback address") from exc
    if not address.is_loopback or "%" in str(address) or not 1 <= parsed.port <= 65535:
        raise ValueError("endpoint must use a literal loopback address and a valid port")
    return str(address), parsed.port


def trial_request(prompt: str, model: str) -> dict:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("expected a nonempty trial prompt")
    if not isinstance(model, str) or not model or any(c.isspace() or not c.isprintable() for c in model):
        raise ValueError("expected an explicit model ID without whitespace")
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [],
        "tool_choice": "none",
        "n": 1,
        "stream": False,
    }


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(raw: bytes) -> object:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"invalid JSON constant {value!r}")

    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_pairs, parse_constant=invalid_constant)


def completion(raw: bytes, model: str) -> tuple[str, dict, str, str, str | None]:
    answer = read_json(raw)
    if not isinstance(answer, dict) or answer.get("object") != "chat.completion":
        raise ValueError("expected one nonstreaming chat.completion object")
    if answer.get("model") != model:
        raise ValueError("returned model differs from the requested model")
    choices = answer.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("expected exactly one completion choice")
    choice = choices[0]
    reason = choice.get("finish_reason")
    if type(choice.get("index")) is not int or choice["index"] != 0:
        raise ValueError("expected a first choice with index zero")
    if reason not in ("stop", "length", "content_filter", "refusal"):
        raise ValueError("unexpected completion finish reason")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("expected an assistant message")
    if any(
        row.get("tool_calls") not in (None, []) or row.get("function_call") is not None
        for row in (answer, choice, message)
    ):
        raise ValueError("unexpected tool or function call")
    if message.get("audio") or message.get("annotations"):
        raise ValueError("unexpected audio or augmented output")
    content = message.get("content")
    refusal = message.get("refusal")
    if content is not None and not isinstance(content, str):
        raise ValueError("expected text or null completion content")
    if refusal is not None and not isinstance(refusal, str):
        raise ValueError("expected text or null refusal")
    content = "" if content is None else content
    if reason == "length":
        status = "truncated"
    elif refusal or reason in ("content_filter", "refusal"):
        status = "refused"
    else:
        status = "complete" if content.strip() else "empty"
    usage = answer.get("usage")
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    if not isinstance(usage, dict) or any(
        type(usage.get(key)) is not int or usage[key] < 0 for key in keys
    ):
        raise ValueError("expected nonnegative token usage counters")
    return content, usage, reason, status, refusal


def audit_plain(trace_dir: str | Path, expected_prompt: str, *, endpoint: str, model: str) -> tuple[str, dict]:
    """Reconstruct the admitted request and validate the raw response and route."""
    host, port = route_address(endpoint)
    path = Path(trace_dir)
    request = (path / "request.json").read_bytes()
    if request != json_bytes(trial_request(expected_prompt, model)):
        raise ValueError("request differs from the frozen prompt or tool policy")
    response = (path / "response.json").read_bytes()
    transport_bytes = (path / "transport.json").read_bytes()
    transport = read_json(transport_bytes)
    if not isinstance(transport, dict) or transport != {
        "endpoint": endpoint,
        "peer": [host, port],
        "status": 200,
        "content_type": "application/json",
        "request_sha256": sha256(request),
        "response_sha256": sha256(response),
    }:
        raise ValueError("transport route, status, media type, or artifact hash mismatch")
    content, usage, reason, status, refusal = completion(response, model)
    return content, {
        "model": model,
        "requested_model": model,
        "returned_model": model,
        "route": ROUTE,
        "endpoint": endpoint,
        "model_calls": 1,
        "usage": usage,
        "finish_reason": reason,
        "answer_status": status,
        "refusal": refusal,
        "tool_calls_observed": 0,
        "trace_sha256": sha256(transport_bytes),
    }


def plain(task: str, out: str | Path, *, endpoint: str, model: str) -> tuple[str, dict]:
    """Record a fresh, single-response trial; an existing trace is never overwritten."""
    host, port = route_address(endpoint)
    request = json_bytes(trial_request(task, model))
    trace_dir = Path(out)
    trace_dir.mkdir(parents=True, exist_ok=False)
    with (trace_dir / "request.json").open("xb") as stream:
        stream.write(request)

    connection = HTTPConnection(host, port=port, timeout=TIMEOUT_SECONDS)
    try:
        # Connect directly by literal IP. http.client does not use environment
        # proxies or follow redirects.
        connection.connect()
        peer = connection.sock.getpeername()[:2]
        if (str(ipaddress.ip_address(peer[0])), peer[1]) != (host, port):
            raise ValueError("socket peer differs from the requested endpoint")
        connection.request("POST", API_PATH, body=request, headers={"Content-Type": "application/json"})
        incoming = connection.getresponse()
        status = incoming.status
        content_type = incoming.headers.get_content_type() if incoming.headers.get("Content-Type") else None
        response = incoming.read(MAX_RESPONSE_BYTES + 1)
        if len(response) > MAX_RESPONSE_BYTES:
            raise ValueError("response exceeds the trace size limit")
    finally:
        connection.close()

    with (trace_dir / "response.json").open("xb") as stream:
        stream.write(response)
    with (trace_dir / "transport.json").open("xb") as stream:
        stream.write(json_bytes({
            "endpoint": endpoint,
            "peer": [host, port],
            "status": status,
            "content_type": content_type,
            "request_sha256": sha256(request),
            "response_sha256": sha256(response),
        }))
    return audit_plain(trace_dir, task, endpoint=endpoint, model=model)
