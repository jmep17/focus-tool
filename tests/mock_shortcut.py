"""Mock Shortcut API so tests run without a token or a network.

Deliberately a separate module and port from mock_llm: the suite asserts on
mock_llm.SEEN all over the place, and a ticket fetch must not show up there.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# (path, token) per request, newest last. Tests read it to assert both that the header
# went out and — for the cache-only guarantee — that no request happened at all.
SEEN = []

STORIES = {
    "12345": {
        "id": 12345,
        "name": "Retry payments on transient errors",
        "description": ("Customers see failed charges when the PSP blips.\n\n"
                        "## Acceptance criteria\n"
                        "- retries stop after three attempts\n"
                        "- no duplicate charges under any retry path\n"),
        "story_type": "feature",
        "started": True,
        "completed": False,
        "estimate": 3,
        "app_url": "https://app.shortcut.com/acme/story/12345",
        "tasks": [{"description": "Add the retry wrapper", "complete": True},
                  {"description": "Cover the double-charge case", "complete": False}],
        "comments": [{"created_at": "2026-07-20T10:00:00Z",
                      "text": "Finance need this before month end."}],
    },
    "777": {
        "id": 777,
        "name": "Tidy the rota spreadsheet",
        "description": "Nobody can find the current rota.",
        "story_type": "chore",
        "started": False,
        "completed": False,
        "app_url": "https://app.shortcut.com/acme/story/777",
    },
}

class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        token = self.headers.get("Shortcut-Token", "")
        SEEN.append((self.path, token))
        sid = self.path.rstrip("/").rsplit("/", 1)[-1]
        if not token:
            return self._send(401, {"message": "no token"})
        if sid not in STORIES:
            return self._send(404, {"message": "not found"})
        self._send(200, STORIES[sid])

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

def start(port=0):
    server = ThreadingHTTPServer(("127.0.0.1", port), MockHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, server.server_address[1]
