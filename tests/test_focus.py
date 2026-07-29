"""End-to-end tests for focus.py against a mock local model. Run: python3 tests/test_focus.py"""
import io
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
# project context resolves against the cwd, so pin it or the checks below depend on
# where you happened to invoke the suite from
os.chdir(os.path.dirname(HERE))

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
    old, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = out   # stderr too: friendly model errors land there
    try:
        code = focus.main(argv)
    except SystemExit as e:          # argparse --help, and cmd_* bail-outs
        code = e.code if isinstance(e.code, int) else 1
        if isinstance(e.code, str):
            print(e.code)
    finally:
        sys.stdout, sys.stderr = old, old_err
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

# --- pr resume + check, verb form
code, out = run(["pr", "check", "1", "--name", "test-pr"])
check("pr check ticks item", "[x] 1." in out, out)
check("pr progress counts", "1/2 done" in out)
code, out = run(["pr", "resume"])
check("pr resume finds latest", "test-pr" in out)
code, out = run(["pr", "check", "--name", "test-pr"])
check("pr check without a number explains itself", code == 1 and "Which item" in out, out)

# the pre-verb spellings still work, they're just hidden from --help
code, out = run(["pr", "--check", "2", "--name", "test-pr"])
check("legacy --check still ticks", "[x] 2." in out, out)
code, out = run(["pr", "--resume"])
check("legacy --resume still resumes", "test-pr" in out)
code, out = run(["pr", "--help"])
check("--resume is hidden from help", "--resume" not in out, out)
check("verbs are advertised in help", "resume" in out and "check" in out, out)

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
check("ui state exposes project source", json.loads(api("/api/state")[1])["project"] != "")
status, body = api("/api/dump", {"text": "and another", "no_project": True})
check("ui dump no_project", "PROJECT CONTEXT" not in mock_llm.SEEN[-1])

# --- project context
# pure helpers first: no model, no repo, no I/O
check("diff_paths finds changed files",
      focus.diff_paths(diff) == ["payments/client.py"], focus.diff_paths(diff))
toks = focus.query_tokens(["payments/StripeClient_v2.py"])
check("query_tokens splits paths and camelCase", {"payments", "stripe", "client"} <= toks, toks)
prof = ("intro\n## Stack\npython only\n## payments/\nidempotency key required\n"
        "## frontend/\nreact bits\n")
picked = focus.select_sections(prof, {"payments"}, 80)
check("select_sections keeps the matching section",
      "payments/" in picked and "frontend/" not in picked, picked)
check("select_sections pins the stack section", "## Stack" in picked, picked)
check("select_sections respects the budget", len(picked) <= 80, len(picked))
check("select_sections clips a heading-less doc",
      len(focus.select_sections("no headings here\n" + "x" * 500, set(), 100)) <= 100)
check("project_key is filename-safe", re.fullmatch(r"[\w.-]+", focus.project_key()))
check("project_key honours an explicit name", focus.project_key("acme/api") == "acme_api")

# zero-setup tier, in a throwaway dir so we never assert on this repo's real docs
ctx_repo = tempfile.mkdtemp(prefix="focus-ctxrepo-")
with open(os.path.join(ctx_repo, "CLAUDE.md"), "w") as f:
    f.write("## Conventions\nNever return a bare dict from a handler.\n")
back = os.getcwd()
os.chdir(ctx_repo)
try:
    code, out = run(["pr", "-f", diff_path, "--name", "ctx-auto"])
    check("repo docs reach the model with no setup",
          "PROJECT CONTEXT" in mock_llm.SEEN[-1] and "bare dict" in mock_llm.SEEN[-1],
          mock_llm.SEEN[-1][-200:])
    check("pr prints context provenance", "project context: CLAUDE.md" in out, out)
    code, out = run(["pr", "-f", diff_path, "--no-project", "--name", "ctx-off"])
    check("pr --no-project suppresses context", "PROJECT CONTEXT" not in mock_llm.SEEN[-1])
    check("pr --no-project prints no provenance", "project context:" not in out, out)
    code, out = run(["pr", "-f", diff_path, "--project", "none", "--name", "ctx-none"])
    check("--project none suppresses context", "PROJECT CONTEXT" not in mock_llm.SEEN[-1])
finally:
    os.chdir(back)

# --- project setup / show / ls, and a stored profile outranking repo docs
code, out = run(["project", "setup"])
check("project setup builds a profile", code == 0 and "idempotency" in out, out)
pkey = focus.project_key()
saved = focus.load_project(pkey)
check("project profile persisted", "idempotency" in saved.get("profile", ""))
check("project setup records provenance", saved.get("source") == "harvest", saved.get("source"))
code, out = run(["project", "show"])
check("project show prints the profile", "PROJECT PROFILE" in out and "idempotency" in out)
code, out = run(["project", "ls"])
check("project ls lists the current key", pkey in out and "<- here" in out, out)
code, out = run(["pr", "-f", diff_path, "--name", "ctx-stored"])
check("stored profile outranks repo docs", f"project context: {pkey}" in out, out)
check("stored profile reaches the model", "idempotency" in mock_llm.SEEN[-1])
session = focus.load_json(focus.pr_session_path("ctx-stored"), None)
check("session records which context was used", session["project"]["source"] == pkey)
code, out = run(["pr", "--resume", "--name", "ctx-stored"])
check("resume still shows provenance", "project context:" in out, out)

# oversized profiles get clipped, never allowed to eat the diff's budget
focus.save_project(pkey, {"profile": "## Stack\nfoo\n" + "".join(
    f"## area{i}/\n" + "y" * 400 + "\n" for i in range(40))})
ctx, ctx_src = focus.resolve_project_context(focus.diff_paths(diff))
check("oversized profile clipped to budget", len(ctx) <= focus.MAX_CONTEXT_CHARS, len(ctx))
check("clipped context keeps the pinned section", "## Stack" in ctx, ctx[:80])

# --- context reaches every AI command, and --no-project turns it off everywhere
run(["project", "setup"])
run(["dump", "fix the retry path in payments"])
check("dump carries project context", "PROJECT CONTEXT" in mock_llm.SEEN[-1])
run(["break", "1"])
check("break carries project context", "PROJECT CONTEXT" in mock_llm.SEEN[-1])
run(["break", "1", "--no-project"])
check("break --no-project suppresses context", "PROJECT CONTEXT" not in mock_llm.SEEN[-1])
code, out = run(["draft", "tell priya the payments retry is fixed"])
check("draft stacks voice and project context",
      "MATCH THIS PERSON'S VOICE" in mock_llm.SEEN[-1]
      and "PROJECT CONTEXT" in mock_llm.SEEN[-1])
check("draft still applies the voice profile", "[voiced]" in out, out)

# turning the profile off falls back to the repo's own docs, it does not disable context
code, out = run(["project", "off"])
check("project off clears the profile", "Cleared" in out and not focus.load_project(pkey), out)
code, out = run(["pr", "-f", diff_path, "--name", "ctx-after-off"])
check("after project off the repo's own docs still apply",
      "project context: CLAUDE.md" in out, out)

# --- task refs: .0 has no subtask to point at
code, out = run(["done", "3.0"])
check("subtask .0 rejected", code == 1 and "Bad task id" in out, out)

# --- add with estimate + notes; start/done stamp timestamps; calibration line
code, out = run(["add", "Calibrate", "estimates", "-e", "15", "--notes", "seed note"])
check("add takes an estimate", "~15m" in out, out)
store = focus.load_store()
cal = next(t for t in store["tasks"] if t["title"] == "Calibrate estimates")
check("add stores notes", cal["notes"] == "seed note")
run(["start", str(cal["id"])])
store = focus.load_store()
cal = focus.get_task(store, cal["id"])
check("start stamps started", bool(cal.get("started")))
# backdate the start so elapsed time is derivable
from datetime import datetime, timedelta, timezone
cal["started"] = (datetime.now(timezone.utc)
                  - timedelta(minutes=30)).isoformat(timespec="seconds")
focus.save_store(store)
code, out = run(["done", str(cal["id"])])
check("done prints calibration", "est ~15m" in out and "took ~30m" in out, out)
check("done stamps completed",
      bool(focus.get_task(focus.load_store(), cal["id"]).get("completed")))

# --- history log captured lifecycle events
kinds = {e["event"] for e in focus.load_history()}
check("history logs created/started/done", {"created", "started", "done"} <= kinds, kinds)

# --- today recap
code, out = run(["today"])
check("today lists the win", "DONE TODAY" in out and "Calibrate estimates" in out, out)

# --- note breadcrumbs, shown on start and next
code, out = run(["note", "2", "left off at the parser"])
check("note appends", code == 0 and "Noted" in out, out)
code, out = run(["note", "2"])
check("note shows notes", "left off at the parser" in out, out)
code, out = run(["start", "2"])
check("start shows last note", "left off at the parser" in out, out)
code, out = run(["next"])   # task 2: status now, priority 2 — beats task 3 (pri 3)
check("next shows last note", "left off at the parser" in out, out)

# --- timer (0 minutes: completes instantly; notification stubbed out)
notified = []
real_notify = focus._notify
focus._notify = lambda *a: notified.append(a)
try:
    code, out = run(["timer", "0"])
finally:
    focus._notify = real_notify
check("timer completes and notifies", code == 0 and "Time!" in out and notified, out)
check("timer logged to history",
      any(e["event"] == "timer_done" for e in focus.load_history()))

# --- triage: non-tty lists stale tasks; ls shows their age
store = focus.load_store()
stale_t = next(t for t in store["tasks"] if t["status"] == "inbox")
stale_t["updated"] = "2026-01-01T00:00:00+00:00"
focus.save_store(store)
old_stdin = sys.stdin
sys.stdin = io.StringIO("")
try:
    code, out = run(["triage"])
finally:
    sys.stdin = old_stdin
check("triage lists stale tasks non-interactively",
      "stale task(s)" in out and stale_t["title"] in out, out)
code, out = run(["ls"])
check("ls shows age marker on stale tasks", "d old" in out, out)

# --- memory: add / show / suffix injection / off
code, out = run(["memory", "add", "prefers tiny first steps under 5 minutes"])
check("memory add", code == 0 and "Remembered" in out, out)
code, out = run(["memory", "show"])
check("memory show lists the fact", "tiny first steps" in out, out)
run(["dump", "sort out the expenses"])
check("dump carries user memory", "USER MEMORY" in mock_llm.SEEN[-1]
      and "tiny first steps" in mock_llm.SEEN[-1])
run(["dump", "another thing entirely", "--no-memory"])
check("--no-memory suppresses memory", "USER MEMORY" not in mock_llm.SEEN[-1])
run(["break", "2"])
check("break carries user memory", "USER MEMORY" in mock_llm.SEEN[-1])
run(["draft", "tell priya thanks for covering"])
check("draft stacks memory with voice and project",
      "USER MEMORY" in mock_llm.SEEN[-1])
code, out = run(["memory", "off"])
check("memory off", code == 0 and "off" in out, out)
run(["dump", "yet another thing"])
check("memory off suppresses the suffix", "USER MEMORY" not in mock_llm.SEEN[-1])

# --- a garbage model reply fails politely, not with a traceback
real_chat = focus.chat
focus.chat = lambda *a, **k: "sorry, thinking out loud, no json here"
try:
    code, out = run(["break", "2"])
finally:
    focus.chat = real_chat
check("garbage model reply fails politely",
      code == 2 and "couldn't be used" in out, (code, out))

# --- UI: note, timer, done_today, drop, and origin hardening
status, body = api("/api/note", {"id": 2, "text": "ui breadcrumb"})
check("ui note appends", "ui breadcrumb" in json.loads(body)["notes"])
status, body = api("/api/timer", {"event": "timer_start", "minutes": 25})
check("ui timer accepted", status == 200)
check("ui timer logged", any(e["event"] == "timer_start" and e.get("source") == "ui"
                             for e in focus.load_history()))
state = json.loads(api("/api/state")[1])
check("state carries done_today",
      any(d["title"] == "Calibrate estimates" for d in state["done_today"]),
      state["done_today"])
status, body = api("/api/add", {"title": "Doomed task"})
doomed = json.loads(body)["id"]
status, body = api("/api/update", {"id": doomed, "delete": True})
check("ui drop deletes the task", json.loads(body).get("deleted") == doomed)
check("dropped task is gone", focus.get_task(focus.load_store(), doomed) is None)

def api_status(path, payload):
    """Status code even for error responses (api() raises on non-2xx)."""
    try:
        return api(path, payload)[0]
    except urllib.error.HTTPError as e:
        return e.code

# --- UI: state extensions (doctor detail, energy, week, calibration)
state = json.loads(api("/api/state")[1])
check("state exposes model detail", state["model"]["model"] == "mock-model-8b", state)
check("state has low-energy pick", "next_low_id" in state)
check("state carries stale_days", state["stale_days"] == focus.STALE_DAYS)
check("state carries week + calib fields",
      state["done_week"] and "est" in state["done_today"][0], state.get("done_week"))
check("state names home and root", state["home"] == TMP and state["root"])
status, body = api("/")
check("ui html carries new panels",
      "PR review" in body and "Triage" in body and "prPanel" in body)

# --- UI: add with estimate/notes, break with hint, draft polish/to
status, body = api("/api/add", {"title": "Estimated via UI",
                                "estimate_min": 15, "notes": "check the docs"})
t_add = json.loads(body)
check("ui add carries estimate+notes",
      t_add["estimate_min"] == 15 and t_add["notes"] == "check the docs", t_add)
status, body = api("/api/break", {"id": t_add["id"], "hint": "focus on tests"})
check("ui break with hint", len(json.loads(body)["subtasks"]) == 3)
status, body = api("/api/draft", {"text": "cover tuesday", "polish": True,
                                  "to": "Priya", "no_voice": True})
check("ui draft polish+to", status == 200 and json.loads(body)["message"])

# --- UI: triage decisions (stale_t was backdated above and is still untouched)
status, body = api("/api/triage", {"id": stale_t["id"], "decision": "keep"})
check("ui triage keep freshens", json.loads(body)["updated"][:10] != "2026-01-01")
check("ui triage keep logged",
      any(e["event"] == "triage" and e.get("decision") == "keep"
          and e.get("id") == stale_t["id"] for e in focus.load_history()))
store = focus.load_store()
focus.get_task(store, stale_t["id"])["updated"] = "2026-01-01T00:00:00+00:00"
focus.save_store(store)
status, body = api("/api/triage", {"id": stale_t["id"], "decision": "later"})
check("ui triage later moves", json.loads(body)["status"] == "later")
td = json.loads(api("/api/add", {"title": "Triage done target"})[1])["id"]
status, body = api("/api/triage", {"id": td, "decision": "done"})
check("ui triage done completes", json.loads(body).get("completed"))
tdr = json.loads(api("/api/add", {"title": "Triage drop target"})[1])["id"]
status, body = api("/api/triage", {"id": tdr, "decision": "drop"})
check("ui triage drop deletes", json.loads(body)["deleted"] == tdr
      and focus.get_task(focus.load_store(), tdr) is None)
check("ui triage rejects bad decision",
      api_status("/api/triage", {"id": stale_t["id"], "decision": "zap"}) == 400)

# --- UI: memory show/add (memory was left disabled by the CLI checks above)
m = json.loads(api("/api/memory", {"action": "show"})[1])
check("ui memory show reports disabled",
      m["disabled"] is True and any("tiny first steps" in f for f in m["facts"]), m)
m = json.loads(api("/api/memory", {"action": "add", "text": "works best before noon"})[1])
check("ui memory add re-enables",
      m["disabled"] is False and "works best before noon" in m["facts"], m)

# --- UI: pr review (memory is ON right now — it must still stay out of pr prompts)
status, body = api("/api/pr", {"action": "review", "diff": diff, "name": "ui-pr"})
s_pr = json.loads(body)
check("ui pr review summarises",
      "retry logic" in s_pr["summary"] and len(s_pr["checklist"]) == 2, s_pr)
check("ui pr session saved", os.path.exists(focus.pr_session_path("ui-pr")))
check("pr never carries user memory", "USER MEMORY" not in mock_llm.SEEN[-1])
status, body = api("/api/pr", {"action": "check", "n": 1, "name": "ui-pr"})
check("ui pr check ticks", json.loads(body)["checklist"][0]["done"] is True)
check("ui pr check out of range rejected",
      api_status("/api/pr", {"action": "check", "n": 99, "name": "ui-pr"}) == 400)
status, body = api("/api/pr", {"action": "resume"})
check("ui pr resume finds latest", json.loads(body)["name"] == "ui-pr")
check("git_working_diff returns a pair",
      isinstance(focus.git_working_diff(), tuple) and len(focus.git_working_diff()) == 2)

# --- UI: memory edit/off (back to disabled, the state the tail of the suite expects)
m = json.loads(api("/api/memory", {"action": "edit", "text": "fact one\n\nfact two\n"})[1])
check("ui memory edit replaces facts", m["facts"] == ["fact one", "fact two"], m)
m = json.loads(api("/api/memory", {"action": "off"})[1])
check("ui memory off keeps facts", m["disabled"] is True and len(m["facts"]) == 2)

# --- UI: voice show/setup/learn/edit/off (off is destructive — keep it last)
v = json.loads(api("/api/voice", {"action": "show"})[1])
check("ui voice show", v["profile"].startswith("Open Slack")
      and len(v["questions"]) == 8, v)
v = json.loads(api("/api/voice", {"action": "learn",
                                  "sample": "hey — shipping the fix today. Cheers, J"})[1])
check("ui voice learn grows samples", v["samples_count"] == 3, v)
v = json.loads(api("/api/voice", {"action": "setup",
                                  "answers": {"greeting": "hey"}, "samples": []})[1])
check("ui voice setup from answers", v["profile"].startswith("Open Slack"), v)
api("/api/voice", {"action": "edit", "profile": "Always sign off with Cheers, J."})
check("ui voice edit replaces",
      focus.load_voice()["profile"] == "Always sign off with Cheers, J.")
check("ui voice setup rejects empty",
      api_status("/api/voice", {"action": "setup", "answers": {}, "samples": []}) == 400)
api("/api/voice", {"action": "off"})
check("ui voice off clears everything", focus.load_voice() == {})

# --- UI: project show/setup/ls/edit/off (ends with no saved profile, as before)
p = json.loads(api("/api/project", {"action": "show"})[1])
check("ui project show falls back to repo docs",
      p.get("fallback") is True and p["source"] == "CLAUDE.md", p)
p = json.loads(api("/api/project", {"action": "setup"})[1])
check("ui project setup distils",
      p["source"] == "harvest" and "save_json" in p["profile"], p)
check("ui project setup persisted",
      focus.load_project(focus.project_key()).get("source") == "harvest")
status, body = api("/api/project", {"action": "ls"})
check("ui project ls marks here", any(x["current"] for x in json.loads(body)["projects"]))
api("/api/project", {"action": "edit", "profile": "## Stack\nedited"})
check("ui project edit saves",
      focus.load_project(focus.project_key()).get("source") == "edit")
p = json.loads(api("/api/project", {"action": "off"})[1])
check("ui project off clears",
      focus.load_project(focus.project_key()) == {} and p.get("fallback") is True, p)

req = urllib.request.Request(
    f"http://127.0.0.1:{ui_port}/api/add",
    data=json.dumps({"title": "evil"}).encode(),
    headers={"Content-Type": "application/json", "Origin": "http://evil.example"})
try:
    urllib.request.urlopen(req, timeout=10)
    forged = 200
except urllib.error.HTTPError as e:
    forged = e.code
check("cross-origin POST rejected", forged == 403, forged)
req = urllib.request.Request(
    f"http://127.0.0.1:{ui_port}/api/voice",
    data=json.dumps({"action": "off"}).encode(),
    headers={"Content-Type": "application/json", "Origin": "http://evil.example"})
try:
    urllib.request.urlopen(req, timeout=10)
    forged = 200
except urllib.error.HTTPError as e:
    forged = e.code
check("cross-origin POST to voice rejected", forged == 403, forged)

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
