import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import HTTPServer
from pathlib import Path

from server import load_config, load_tickets, make_handler, save_annotation


class AnnotationServerTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        (self.dir / "data").mkdir()
        (self.dir / "data" / "tickets.tsv").write_text(
            "id\tsubject\tbody\textra\n7\tHello\tLine one\tx\n8\t\tNo subject\ty\n"
        )
        self.labels = {"queue": ["A", "B"], "priority": ["high"], "type": ["Incident"], "tags": ["t1", "t2"]}
        (self.dir / "data" / "labels.json").write_text(json.dumps(self.labels))
        (self.dir / "config.json").write_text(
            json.dumps({"tickets": "data/tickets.tsv", "labels": "data/labels.json", "output": "data/out.json"})
        )
        self.config = load_config(self.dir / "config.json")

    def test_config_paths_resolve_against_config_file(self):
        self.assertEqual(self.config["tickets"], (self.dir / "data" / "tickets.tsv").resolve())
        self.assertEqual(self.config["show_fields"], ["subject", "body"])

    def test_load_tickets_keeps_empty_fields(self):
        tickets = load_tickets(self.config["tickets"])
        self.assertEqual([t["id"] for t in tickets], ["7", "8"])
        self.assertEqual(tickets[1]["subject"], "")

    def test_save_annotation_overwrites_same_ticket_and_keeps_others(self):
        out = self.config["output"]
        save_annotation(out, "7", {"queue": "A"})
        save_annotation(out, "8", {"queue": "B"})
        save_annotation(out, "7", {"queue": "B"})
        self.assertEqual(json.loads(out.read_text()), {"7": {"queue": "B"}, "8": {"queue": "B"}})

    def test_http_round_trip(self):
        server = HTTPServer(("127.0.0.1", 0), make_handler(self.config))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"

        data = json.load(urllib.request.urlopen(f"{base}/api/data"))
        self.assertEqual(data["labels"], self.labels)
        self.assertEqual(data["tickets"][0], {"id": "7", "fields": {"subject": "Hello", "body": "Line one"}})
        self.assertEqual(data["annotations"], {})

        annotation = {"queue": "A", "priority": "high", "type": "Incident", "tags": ["t1", "t2"]}
        request = urllib.request.Request(
            f"{base}/api/annotations",
            data=json.dumps({"id": "7", **annotation}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request)

        data = json.load(urllib.request.urlopen(f"{base}/api/data"))
        self.assertEqual(data["annotations"], {"7": annotation})
        self.assertIn(b"Ticket annotation", urllib.request.urlopen(base).read())


if __name__ == "__main__":
    unittest.main()
