"""A local stand-in for OpenRouter, so tests exercise DSPy's real call path and error types.

Importing this module turns DSPy's cache off: a fake reply stored under a real model's
name would be returned to a later real run.
"""

import json
import os
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

# isort: off
import openai  # noqa: F401  (must come before dspy, see gepa_toy/dspy_task.py)
import dspy
import dspy.clients.execution
import httpx
from dspy._vendor.lm15.transports import _sync
# isort: on

dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)


def chat(content: str | None, finish: str = "stop") -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 1,
        "model": "fake/resolved-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def error(status: int, message: str = "boom") -> dict:
    return {"error": {"message": message, "code": status}}


def answer(queue: str, type_: str, priority: str, tags: list[str]) -> str:
    """A reply in DSPy's ChatAdapter format."""
    return (
        f"[[ ## queue ## ]]\n{queue}\n\n[[ ## type ## ]]\n{type_}\n\n"
        f"[[ ## priority ## ]]\n{priority}\n\n[[ ## tags ## ]]\n{json.dumps(tags)}\n\n"
        "[[ ## completed ## ]]"
    )


@contextmanager
def fake_openrouter(handler):
    """Routes OpenRouter calls to `handler(body) -> (status, payload)`; payload may be bytes.

    Yields the list of request bodies received.
    """
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            requests.append(body)
            status, payload = handler(body)
            data = (
                payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            )
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()

    original_stream = _sync.StdlibTransport.stream
    original_send = httpx.Client.send

    # DSPy may call through its vendored transport or through LiteLLM's httpx client.
    def stream(self, request):
        request.url = request.url.replace("https://openrouter.ai", base)
        return original_stream(self, request)

    def send(self, request, *args, **kwargs):
        if "openrouter.ai" in str(request.url):
            request.url = httpx.URL(
                str(request.url).replace("https://openrouter.ai", base)
            )
            request.headers["host"] = base.removeprefix("http://")
        return original_send(self, request, *args, **kwargs)

    with (
        patch.object(_sync.StdlibTransport, "stream", stream),
        patch.object(httpx.Client, "send", send),
        # Skip DSPy's retry pauses only: patching time.sleep would also stop a fake
        # server from simulating a slow reply.
        patch.object(dspy.clients.execution, "_delay", lambda *_: 0),
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-fake"}),
    ):
        try:
            yield requests
        finally:
            server.shutdown()
