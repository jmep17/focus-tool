"""Mock OpenAI-compatible server so tests run without a real local model."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Every system prompt this mock has been sent, newest last. The mock runs in-process, so
# tests read it directly to assert what reached the model without decoding canned replies.
SEEN = []

def canned_reply(messages):
    full = messages[0]["content"]
    # Dispatch on the BASE prompt only. Everything after a suffix marker is arbitrary
    # user/repo text — this repo's own CLAUDE.md documents the string "brain-dump", so
    # dispatching on the whole thing answers a PR request with a task list.
    system = (full.split("\nMATCH THIS PERSON'S VOICE")[0]
              .split("\nPROJECT CONTEXT")[0]
              .split("\nUSER MEMORY")[0])
    if "compact project profile" in system:
        return json.dumps({"profile":
            "## Stack\nPython 3.9, stdlib only, no dependencies.\n"
            "## Conventions\nAll writes go through save_json (tmp then os.replace).\n"
            "## payments/\nEvery retry must carry an idempotency key.\n"
            "## frontend/\nReact, no state library.\n"})
    if "brain-dump" in system:
        return json.dumps({"tasks": [
            {"title": "Fix flaky auth test", "priority": 1, "estimate_min": 45},
            {"title": "Reply to Priya about rota", "priority": 2, "estimate_min": 10},
        ]})
    if "break a task" in system:
        return json.dumps({"subtasks": [
            {"text": "Open auth_test.py and run the failing test once", "estimate_min": 5},
            {"text": "Read the last 3 CI failure logs for the pattern", "estimate_min": 15},
            {"text": "Write the fix and re-run locally 5 times", "estimate_min": 25},
        ]})
    if "unified diff" in system:
        return json.dumps({
            "summary": "Adds retry logic to the payment client and covers it with tests.",
            "risks": ["Retry loop could double-charge if idempotency key is missing"],
            "checklist": [
                {"file": "payments/client.py", "item": "Verify idempotency key is sent on retries"},
                {"file": "tests/test_client.py", "item": "Check the retry test asserts call count"},
            ],
        })
    if "writing-voice profile" in system:
        return json.dumps({"profile":
            "Open Slack messages with 'hey'. Sign off emails 'Cheers, J'. "
            "Short sentences, one emoji max, never say 'circling back'. "
            "Example: 'hey — quick one: can you swap Thursday? Cheers, J'"})
    if "write work messages" in system:
        base = "Hi Priya — quick one: I can cover Tuesday but not Thursday. Could you swap? Happy either way."
        if "MATCH THIS PERSON'S VOICE" in full:
            return "[voiced] " + base
        return base
    return "ok"

class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.endswith("/models"):
            body = json.dumps({"data": [{"id": "mock-model-8b"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n).decode())
        SEEN.append(payload["messages"][0]["content"])
        content = canned_reply(payload["messages"])
        body = json.dumps({"choices": [{"message": {"role": "assistant",
                                                    "content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

def start(port=0):
    server = ThreadingHTTPServer(("127.0.0.1", port), MockHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, server.server_address[1]
