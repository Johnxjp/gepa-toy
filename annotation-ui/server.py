"""Minimal ticket annotation server. Run: python3 server.py [config.json]"""

import csv
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    # Relative paths resolve against the config file, so the server runs from any directory.
    base = path.parent
    return {
        "tickets": base / config["tickets"],
        "labels": base / config["labels"],
        "output": base / config["output"],
        "id_field": config.get("id_field", "id"),
        "show_fields": config.get("show_fields", ["subject", "body"]),
        "port": config.get("port", 8000),
    }


def load_tickets(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_annotations(path):
    return json.loads(path.read_text()) if path.exists() else {}


def save_annotation(path, ticket_id, annotation):
    annotations = load_annotations(path)
    annotations[ticket_id] = annotation
    # Write then rename so a crash mid-write can't corrupt earlier labels.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(annotations, indent=2))
    tmp.replace(path)


def make_handler(config):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self._send(200, (HERE / "index.html").read_bytes(), "text/html")
            elif self.path == "/api/data":
                tickets = [
                    {
                        "id": t[config["id_field"]],
                        "fields": {f: t.get(f, "") for f in config["show_fields"]},
                    }
                    for t in load_tickets(config["tickets"])
                ]
                self._json(
                    {
                        "labels": json.loads(config["labels"].read_text()),
                        "tickets": tickets,
                        "annotations": load_annotations(config["output"]),
                    }
                )
            else:
                self._send(404, b"Not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/annotations":
                self._send(404, b"Not found", "text/plain")
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            save_annotation(config["output"], body.pop("id"), body)
            self._json({"ok": True})

        def _json(self, data):
            self._send(200, json.dumps(data).encode(), "application/json")

        def _send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return Handler


def main():
    config = load_config(sys.argv[1] if len(sys.argv) > 1 else HERE / "config.json")
    # Single-threaded on purpose: saves read-modify-write the output file.
    server = HTTPServer(("127.0.0.1", config["port"]), make_handler(config))
    print(f"Annotating {config['tickets']} -> {config['output']}")
    print(f"Open http://127.0.0.1:{config['port']}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
