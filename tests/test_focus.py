"""End-to-end tests for focus.py against a mock local model. Run: python3 tests/test_focus.py"""
import io
import json
import os
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import mock_llm

# isolated data dir wired to the mock model BEFORE importing focus
TMP = tempfile.mkdtemp(prefix="focus-test-")
os.environ["FOCUS_HOME"] = TMP
server, port = mock_llm.start()

import focus  # noqa: E402

with open(os.path.join(TMP, "config.json"), "w") as f:
    json.dump({"endpoint": f"http://127.0.0.1:{port}/v1"}, f)

PASS = 0

def run(argv):
    out = io.StringIO()
    old = sys.stdout
    sys.stdout = out
    try:
        code = focus.main(argv)
    finally:
        sys.stdout = old
    return code, out.getvalue()

def check(name, cond, detail=""):
    global PASS
    assert cond, f"FAIL {name} {detail}"
    PASS += 1
    print(f"  ok  {name}")

# --- doctor sees the mock model
code, out = run(["doctor"])
check("doctor finds model", code == 0 and "mock-model-8b" in out, out)

# --- dump via AI
code, out = run(["dump", "fix flaky auth test and reply to priya"])
check("dump creates 2 tasks", code == 0 and "Captured 2 task(s)" in out, out)
check("dump parses priority", "Fix flaky auth test" in out)

# --- add without AI
code, out = run(["add", "Write", "standup", "notes", "-p", "3"])
check("add works", code == 0 and "Write standup notes" in out)

# --- ls
code, out = run(["ls"])
check("ls groups by status", "INBOX" in out and "#1" in out and "#3" in out)

# --- break task 1
code, out = run(["break", "1"])
check("break adds subtasks", code == 0 and "3 steps" in out.replace("0/3", "3") or "auth_test.py" in out, out)
store = focus.load_store()
t1 = focus.get_task(store, 1)
check("break persisted 3 subtasks", len(t1["subtasks"]) == 3)
check("subtask estimates saved", t1["subtasks"][0]["estimate_min"] == 5)

# --- next picks highest-priority task's first subtask
code, out = run(["next"])
check("next shows ONE thing", "DO THIS ONE THING" in out)
check("next picks task 1 first subtask", "Open auth_test.py" in out, out)
check("next gives done ref", "focus done 1.1" in out)

# --- energy low prefers the smallest task (priya, 10m)
code, out = run(["next", "--energy", "low"])
check("low energy picks small task", "Priya" in out or "priya" in out, out)

# --- done on a subtask advances, done on last completes
code, out = run(["done", "1.1"])
check("done subtask advances", "Next up on this task" in out, out)
run(["done", "1.2"])
code, out = run(["done", "1.3"])
check("last subtask completes task", "COMPLETE" in out, out)
store = focus.load_store()
check("task 1 status done", focus.get_task(store, 1)["status"] == "done")

# --- move + start
code, out = run(["move", "2", "later"])
check("move works", "LATER" in out)
code, out = run(["start", "3"])
check("start moves to now", "NOW" in out)

# --- pr from a diff file
diff = """--- a/payments/client.py
+++ b/payments/client.py
@@ -1,3 +1,9 @@
+def retry(fn):
+    for i in range(3):
+        try:
+            return fn()
+        except TransientError:
+            continue
"""
diff_path = os.path.join(TMP, "sample.diff")
with open(diff_path, "w") as f:
    f.write(diff)
code, out = run(["pr", "-f", diff_path, "--name", "test-pr"])
check("pr summary shown", "retry logic" in out, out)
check("pr risks shown", "double-charge" in out)
check("pr checklist has position marker", "<- you are here" in out)

# --- pr resume + check
code, out = run(["pr", "--check", "1", "--name", "test-pr"])
check("pr check ticks item", "[x] 1." in out, out)
check("pr progress counts", "1/2 done" in out)
code, out = run(["pr", "--resume"])
check("pr resume finds latest", "test-pr" in out)

# --- draft
code, out = run(["draft", "tell priya i can cover tuesday not thursday",
                 "--type", "slack", "--tone", "friendly"])
check("draft produces message", "Priya" in out and "Tuesday" in out, out)

# --- voice profile: setup from samples, injection, off
sample_path = os.path.join(TMP, "sample_msg.txt")
with open(sample_path, "w") as f:
    f.write("hey — quick one: auth tests are flaky again, on it. Cheers, J")
code, out = run(["voice", "setup", "--samples", sample_path])
check("voice setup builds profile", code == 0 and "Cheers, J" in out, out)
check("voice profile persisted", focus.load_voice().get("profile", "").startswith("Open Slack"))
code, out = run(["voice", "show"])
check("voice show prints profile", "YOUR VOICE PROFILE" in out)
code, out = run(["draft", "tell priya i can cover tuesday", "--type", "slack"])
check("draft uses voice profile", "[voiced]" in out, out)
code, out = run(["draft", "tell priya i can cover tuesday", "--no-voice"])
check("draft --no-voice skips profile", "[voiced]" not in out, out)
code, out = run(["voice", "learn", "--samples", sample_path])
check("voice learn updates profile", code == 0 and "Learned" in out)
check("voice samples capped list grows", len(focus.load_voice()["samples"]) == 2)

# --- extract_json is robust to chatter and fences
messy = 'Sure! Here you go:\n```json\n{"a": {"b": "brace } in string"}, "c": 1}\n``` hope that helps'
check("extract_json handles fences+nesting",
      focus.extract_json(messy) == {"a": {"b": "brace } in string"}, "c": 1})

# --- UI API round-trip
import threading
ui_server = focus.ThreadingHTTPServer(("127.0.0.1", 0), focus.UIHandler)
ui_port = ui_server.server_address[1]
threading.Thread(target=ui_server.serve_forever, daemon=True).start()

def api(path, payload=None):
    url = f"http://127.0.0.1:{ui_port}{path}"
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, r.read().decode()

status, body = api("/")
check("ui serves html", status == 200 and "Do this one thing" in body)
status, body = api("/api/state")
state = json.loads(body)
check("ui state has tasks+llm", state["llm"] is True and len(state["tasks"]) >= 3)
status, body = api("/api/add", {"title": "From the UI", "status": "next"})
check("ui add task", json.loads(body)["title"] == "From the UI")
tid = json.loads(body)["id"]
status, body = api("/api/break", {"id": tid})
check("ui break subtasks", len(json.loads(body)["subtasks"]) == 3)
status, body = api("/api/update", {"id": tid, "toggle_subtask": 0})
check("ui toggle subtask", json.loads(body)["subtasks"][0]["done"] is True)
status, body = api("/api/state")
check("ui state exposes voice flag", json.loads(body)["voice"] is True)
status, body = api("/api/draft", {"text": "cover tuesday", "type": "slack", "tone": "friendly"})
check("ui draft uses voice", "[voiced]" in json.loads(body)["message"])
status, body = api("/api/draft", {"text": "cover tuesday", "type": "slack",
                                  "tone": "friendly", "no_voice": True})
check("ui draft no_voice", "[voiced]" not in json.loads(body)["message"])
status, body = api("/api/dump", {"text": "one thing\nanother thing"})
check("ui dump", len(json.loads(body)["created"]) == 2)

# --- graceful no-model fallback: point config at a dead port
with open(os.path.join(TMP, "config.json"), "w") as f:
    json.dump({"endpoint": "http://127.0.0.1:9/v1"}, f)
code, out = run(["dump", "line one\nline two"])
check("dump falls back without model", "falling back" in out and "2 task(s)" in out, out)
code, out = run(["next"])
check("next works without model", code == 0 and "DO THIS ONE THING" in out)
code, out = run(["break", "2"])
check("break fails politely without model", code == 2)

print(f"\nAll {PASS} checks passed.")
