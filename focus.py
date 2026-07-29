#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""
focus — a local-first ADHD companion for developers.

Everything runs on your Mac. AI features talk to a local model server
(LM Studio on :1234 or Ollama on :11434) over localhost. No accounts,
no cloud, no telemetry. Non-AI commands work with no model running.

Commands:
  focus dump [text]        Brain-dump -> structured tasks (AI, with fallback)
  focus add <title>        Add a single task (no AI)
  focus ls                 List tasks grouped by status
  focus break <id>         Split a task into <=25-minute subtasks (AI)
  focus next               Show exactly ONE next action
  focus start <id>         Move a task to Now
  focus done <id>[.n]      Complete a task or one subtask
  focus move <id> <status> Move between inbox/now/next/later
  focus pr [...]           Summarise a diff + build a review checklist (AI)
  focus draft [...]        Bullets -> polished message (AI)
  focus ui                 Open the local dashboard
  focus doctor             Check local model connectivity

Data lives in ~/.focus/ as plain JSON. Override with FOCUS_HOME.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------- paths

def focus_home():
    home = os.environ.get("FOCUS_HOME") or os.path.join(os.path.expanduser("~"), ".focus")
    os.makedirs(home, exist_ok=True)
    os.makedirs(os.path.join(home, "pr"), exist_ok=True)
    return home

def _path(name):
    return os.path.join(focus_home(), name)

def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default

def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# ---------------------------------------------------------------- store

STATUSES = ["inbox", "now", "next", "later", "done"]

def load_store():
    return load_json(_path("tasks.json"), {"next_id": 1, "tasks": []})

def save_store(store):
    save_json(_path("tasks.json"), store)

def get_task(store, tid):
    for t in store["tasks"]:
        if t["id"] == tid:
            return t
    return None

def new_task(store, title, status="inbox", priority=2, estimate_min=None, notes=""):
    t = {
        "id": store["next_id"],
        "title": title.strip(),
        "notes": notes,
        "status": status,
        "priority": priority,           # 1 high, 2 normal, 3 low
        "estimate_min": estimate_min,
        "subtasks": [],                  # {"text", "done", "estimate_min"}
        "created": now_iso(),
        "updated": now_iso(),
    }
    store["next_id"] += 1
    store["tasks"].append(t)
    return t

def touch(task):
    task["updated"] = now_iso()

# ---------------------------------------------------------------- config / llm

DEFAULT_ENDPOINTS = [
    ("LM Studio", "http://127.0.0.1:1234/v1"),
    ("Ollama", "http://127.0.0.1:11434/v1"),
]

def load_config():
    return load_json(_path("config.json"), {})

def http_json(url, payload=None, timeout=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout or 120) as resp:
        return json.loads(resp.read().decode())

class NoModelError(RuntimeError):
    pass

def detect_runtime():
    """Return (name, base_url, model) for the first reachable local server."""
    cfg = load_config()
    candidates = []
    if cfg.get("endpoint"):
        candidates.append(("configured", cfg["endpoint"].rstrip("/")))
    candidates += DEFAULT_ENDPOINTS
    for name, base in candidates:
        try:
            models = http_json(base + "/models", timeout=3)
            ids = [m.get("id") for m in models.get("data", []) if m.get("id")]
            model = cfg.get("model") or (ids[0] if ids else None)
            if model:
                return name, base, model
        except Exception:
            continue
    raise NoModelError(
        "No local model server found.\n"
        "  Start LM Studio (Developer tab -> Start Server) or `ollama serve`,\n"
        "  make sure a model is loaded, then try again.\n"
        "  Non-AI commands (add, ls, next, done, move) work without one."
    )

def chat(system, user, temperature=0.3):
    name, base, model = detect_runtime()
    resp = http_json(base + "/chat/completions", {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    })
    return resp["choices"][0]["message"]["content"]

def extract_json(text):
    """Pull the first JSON object out of a model reply, tolerating chatter/fences."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in model reply")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\" and in_str:
            esc = True
        elif c == '"':
            in_str = not in_str
        elif not in_str:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON object in model reply")

# ---------------------------------------------------------------- prompts

SYS_DUMP = """You help a developer with ADHD turn a messy brain-dump into a task list.
Extract every distinct actionable task. Keep titles short, verb-first, concrete.
Guess priority: 1 (urgent/blocking others), 2 (normal), 3 (someday).
Guess estimate_min only when obvious. Do not invent tasks that are not in the text.
Reply with ONLY this JSON:
{"tasks": [{"title": "...", "priority": 2, "estimate_min": null}]}"""

SYS_BREAK = """You help a developer with ADHD break a task into small, finishable steps.
Rules:
- Each subtask takes 25 minutes or less.
- Step 1 must be a tiny, concrete, physical first action (open a file, run a command,
  write one sentence) so starting is trivial.
- Steps are ordered and self-contained; no vague steps like "do research".
- 3 to 8 subtasks. Include estimate_min for each (5, 10, 15, 20 or 25).
Reply with ONLY this JSON:
{"subtasks": [{"text": "...", "estimate_min": 10}]}"""

SYS_PR = """You help a developer with ADHD review a pull request without drowning.
Given a unified diff, reply with ONLY this JSON:
{"summary": "3 short sentences max: what this change does and why",
 "risks": ["the 1-4 places a reviewer should look hardest, most important first"],
 "checklist": [{"file": "path", "item": "one specific thing to verify in this file"}]}
Checklist rules: one or two items per changed file, concrete and verifiable,
ordered so related files sit together. Do not pad. Do not invent files."""

SYS_DRAFT = """You help a developer with ADHD write work messages. They will give you
rough bullets or a half-written draft. Produce a ready-to-send message.
Rules:
- Sound like a normal, warm, direct colleague. UK English.
- No corporate filler, no "I hope this finds you well", no apologising for existing.
- Lead with the point. Keep it as short as the content allows.
- If the input asks for something, make the ask explicit and easy to say yes to.
Reply with ONLY the message text, no preamble, no quotes."""

SYS_VOICE = """You build a compact writing-voice profile from a person's interview
answers and real message samples, so a model can imitate how THEY write.
Focus on what is distinctive and reusable: greetings and sign-offs they actually
use, formality, sentence length and rhythm, warmth vs bluntness, emoji and
punctuation habits, pet phrases, and words they would never use.
Where the samples contradict the interview answers, trust the samples.
Keep it under 220 words. Write rules as imperatives ("Open Slack messages
with...", "Never say..."). Include one 1-2 line example message in their voice.
Reply with ONLY this JSON:
{"profile": "the profile text"}"""

VOICE_QUESTIONS = [
    ("greeting", "How do you usually open a Slack message? (e.g. 'hey', 'hi both', straight in with no greeting)"),
    ("signoff", "How do you sign off emails? (e.g. 'Cheers, J', 'Thanks!', nothing)"),
    ("emoji", "Emojis and exclamation marks — never, occasionally, or freely?"),
    ("formality", "Formality from 1 (mates) to 5 (lawyer) — and does it change for managers?"),
    ("directness", "Are you short-and-blunt, or warmer with a bit of context first?"),
    ("phrases", "Any pet phrases you actually use? (e.g. 'quick one', 'no worries', 'sanity check')"),
    ("never", "Words or phrases you'd NEVER use? (e.g. 'per my last email', 'circling back', 'utilise')"),
    ("spelling", "UK or US spelling?"),
]

def load_voice():
    return load_json(_path("voice.json"), {})

def save_voice(v):
    save_json(_path("voice.json"), v)

def voice_system_suffix(disabled=False):
    if disabled:
        return ""
    v = load_voice()
    if not v.get("profile"):
        return ""
    return ("\nMATCH THIS PERSON'S VOICE. Follow this profile over any generic "
            "style advice:\n" + v["profile"])

DRAFT_STYLES = {
    "slack": "Format: a Slack message. Casual-professional. No greeting needed unless addressing someone new.",
    "email": "Format: an email with a one-line subject on the first line as 'Subject: ...', then a blank line, then the body with a brief greeting and sign-off.",
    "pr-comment": "Format: a code review comment. Constructive, specific, assume good intent. Frame opinions as questions or suggestions where reasonable.",
    "standup": "Format: a standup update with three short labelled lines: Yesterday / Today / Blockers.",
}

# ---------------------------------------------------------------- helpers

PRI_LABEL = {1: "high", 2: "med", 3: "low"}

def remaining_estimate(task):
    subs = [s for s in task["subtasks"] if not s["done"]]
    if subs:
        known = [s["estimate_min"] for s in subs if s.get("estimate_min")]
        if known:
            return sum(known)
    return task.get("estimate_min")

def first_open_subtask(task):
    for i, s in enumerate(task["subtasks"]):
        if not s["done"]:
            return i, s
    return None, None

def pick_next(store, energy=None):
    """Deterministic: no AI needed to decide what to do — that IS the feature."""
    order = {"now": 0, "next": 1, "inbox": 2, "later": 3}
    cands = [t for t in store["tasks"] if t["status"] not in ("done",)]
    if not cands:
        return None
    def key(t):
        est = remaining_estimate(t)
        est_key = est if est is not None else 999
        if energy == "low":
            # tired: easiest wins over priority
            return (order[t["status"]], est_key, t["priority"], t["created"])
        return (order[t["status"]], t["priority"], est_key, t["created"])
    return sorted(cands, key=key)[0]

def parse_task_ref(ref):
    """'12' -> (12, None); '12.3' -> (12, 2) zero-based subtask index."""
    m = re.fullmatch(r"(\d+)(?:\.(\d+))?", ref.strip())
    if not m:
        raise SystemExit(f"Bad task id: {ref!r} (use e.g. 12 or 12.3)")
    tid = int(m.group(1))
    sub = int(m.group(2)) - 1 if m.group(2) else None
    return tid, sub

def fmt_task(t, verbose=False):
    est = remaining_estimate(t)
    bits = [f"#{t['id']:<3} {t['title']}"]
    meta = [PRI_LABEL[t["priority"]]]
    if est:
        meta.append(f"~{est}m")
    done_subs = sum(1 for s in t["subtasks"] if s["done"])
    if t["subtasks"]:
        meta.append(f"{done_subs}/{len(t['subtasks'])} steps")
    line = f"{bits[0]}  [{' · '.join(meta)}]"
    out = [line]
    if verbose:
        for i, s in enumerate(t["subtasks"], 1):
            tick = "x" if s["done"] else " "
            est_s = f" ({s['estimate_min']}m)" if s.get("estimate_min") else ""
            out.append(f"      [{tick}] {t['id']}.{i} {s['text']}{est_s}")
    return "\n".join(out)

def read_multiline(args_text, prompt):
    if args_text:
        return args_text
    if not sys.stdin.isatty():
        return sys.stdin.read()
    print(prompt + " (finish with Ctrl-D on its own line):")
    return sys.stdin.read()

# ---------------------------------------------------------------- commands

def cmd_dump(args):
    text = read_multiline(" ".join(args.text), "Brain-dump everything on your mind")
    if not text.strip():
        raise SystemExit("Nothing to dump.")
    store = load_store()
    created = []
    try:
        data = extract_json(chat(SYS_DUMP, text))
        for item in data.get("tasks", []):
            if not item.get("title"):
                continue
            pri = item.get("priority") if item.get("priority") in (1, 2, 3) else 2
            created.append(new_task(store, item["title"], priority=pri,
                                    estimate_min=item.get("estimate_min")))
    except NoModelError:
        print("(no local model — falling back to one task per line)\n")
        for line in text.splitlines():
            line = line.strip(" -*\t")
            if line:
                created.append(new_task(store, line))
    save_store(store)
    print(f"Captured {len(created)} task(s) into your inbox:\n")
    for t in created:
        print("  " + fmt_task(t))
    print("\nOut of your head, into the list. `focus next` when ready.")

def cmd_add(args):
    store = load_store()
    t = new_task(store, " ".join(args.title), status=args.status,
                 priority=args.priority)
    save_store(store)
    print("Added: " + fmt_task(t))

def cmd_ls(args):
    store = load_store()
    shown = 0
    for status in STATUSES:
        if status == "done" and not args.all:
            continue
        group = [t for t in store["tasks"] if t["status"] == status]
        if not group:
            continue
        print(f"\n{status.upper()}")
        for t in group:
            print("  " + fmt_task(t, verbose=args.verbose))
            shown += 1
    if shown == 0:
        print("Empty. `focus dump` or `focus add <title>` to get things out of your head.")
    print()

def cmd_break(args):
    store = load_store()
    tid, _ = parse_task_ref(args.id)
    t = get_task(store, tid)
    if not t:
        raise SystemExit(f"No task #{tid}")
    context = t["title"] + ("\nNotes: " + t["notes"] if t["notes"] else "")
    if args.hint:
        context += "\nExtra context: " + " ".join(args.hint)
    data = extract_json(chat(SYS_BREAK, context))
    subs = data.get("subtasks", [])
    if not subs:
        raise SystemExit("Model returned no subtasks — try `focus break` again with a --hint.")
    t["subtasks"] = [
        {"text": s["text"], "done": False, "estimate_min": s.get("estimate_min")}
        for s in subs if s.get("text")
    ]
    touch(t)
    save_store(store)
    print(f"Broke down #{t['id']} {t['title']}:\n")
    print("  " + fmt_task(t, verbose=True))
    print(f"\nFirst step is tiny on purpose. `focus start {t['id']}` to begin.")

def cmd_next(args):
    store = load_store()
    t = pick_next(store, energy=args.energy)
    if not t:
        print("Nothing on the list. Enjoy it, or `focus dump` what's on your mind.")
        return
    idx, sub = first_open_subtask(t)
    print("\n" + "=" * 56)
    if sub:
        est = f"  (~{sub['estimate_min']}m)" if sub.get("estimate_min") else ""
        print(f"  DO THIS ONE THING:{est}")
        print(f"  -> {sub['text']}")
        print(f"     from #{t['id']} {t['title']}  [{t['status']}]")
        done_ref = f"{t['id']}.{idx + 1}"
    else:
        est = remaining_estimate(t)
        est_s = f"  (~{est}m)" if est else ""
        print(f"  DO THIS ONE THING:{est_s}")
        print(f"  -> #{t['id']} {t['title']}  [{t['status']}]")
        done_ref = str(t["id"])
        if not t["subtasks"] and (est is None or est > 30):
            print(f"     Feels big? `focus break {t['id']}` first.")
    print("=" * 56)
    print(f"  When finished: focus done {done_ref}")
    if args.why:
        print(f"  (chosen because: status={t['status']}, priority={PRI_LABEL[t['priority']]}, "
              f"remaining ~{remaining_estimate(t) or '?'}m, oldest first)")
    print()

def cmd_start(args):
    store = load_store()
    tid, _ = parse_task_ref(args.id)
    t = get_task(store, tid)
    if not t:
        raise SystemExit(f"No task #{tid}")
    t["status"] = "now"
    touch(t)
    save_store(store)
    print(f"#{t['id']} -> NOW. One task in Now at a time works best.")
    others = [x for x in store["tasks"] if x["status"] == "now" and x["id"] != t["id"]]
    if others:
        print(f"  Heads-up: also in Now: " + ", ".join(f"#{x['id']}" for x in others))

def cmd_done(args):
    store = load_store()
    tid, sub_idx = parse_task_ref(args.id)
    t = get_task(store, tid)
    if not t:
        raise SystemExit(f"No task #{tid}")
    if sub_idx is not None:
        try:
            s = t["subtasks"][sub_idx]
        except IndexError:
            raise SystemExit(f"No subtask {tid}.{sub_idx + 1}")
        s["done"] = True
        touch(t)
        remaining = [x for x in t["subtasks"] if not x["done"]]
        save_store(store)
        if remaining:
            nxt = remaining[0]["text"]
            print(f"Done: {s['text']}")
            print(f"Next up on this task: {nxt}   ({len(remaining)} step(s) left)")
        else:
            t["status"] = "done"
            touch(t)
            save_store(store)
            print(f"That was the last step — task #{t['id']} '{t['title']}' COMPLETE. Nice.")
    else:
        t["status"] = "done"
        for s in t["subtasks"]:
            s["done"] = True
        touch(t)
        save_store(store)
        print(f"Task #{t['id']} '{t['title']}' COMPLETE. Nice.")

def cmd_move(args):
    store = load_store()
    tid, _ = parse_task_ref(args.id)
    t = get_task(store, tid)
    if not t:
        raise SystemExit(f"No task #{tid}")
    if args.status not in STATUSES:
        raise SystemExit(f"Status must be one of: {', '.join(STATUSES)}")
    t["status"] = args.status
    touch(t)
    save_store(store)
    print(f"#{t['id']} -> {args.status.upper()}")

# ------------------------------------------------ pr review

MAX_DIFF_CHARS = 60_000

def get_diff(args):
    if args.file:
        with open(args.file, encoding="utf-8", errors="replace") as f:
            return f.read(), os.path.basename(args.file)
    if not sys.stdin.isatty():
        return sys.stdin.read(), "stdin"
    # fall back to git
    for cmd in (["git", "diff", "--staged"], ["git", "diff", "HEAD"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout, " ".join(cmd)
        except (OSError, subprocess.TimeoutExpired):
            pass
    raise SystemExit(
        "No diff found. Pipe one in (`gh pr diff 123 | focus pr`), use -f file.diff,\n"
        "or run inside a repo with staged/unstaged changes."
    )

def pr_session_path(name):
    safe = re.sub(r"[^\w.-]", "_", name)[:60] or "session"
    return os.path.join(focus_home(), "pr", safe + ".json")

def cmd_pr(args):
    if args.resume or args.check is not None:
        return cmd_pr_resume(args)
    diff, source = get_diff(args)
    truncated = False
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS]
        truncated = True
    data = extract_json(chat(SYS_PR, diff))
    name = args.name or datetime.now().strftime("pr-%Y%m%d-%H%M")
    session = {
        "name": name,
        "source": source,
        "created": now_iso(),
        "summary": data.get("summary", ""),
        "risks": data.get("risks", []),
        "truncated": truncated,
        "checklist": [
            {"file": c.get("file", "?"), "item": c.get("item", ""), "done": False}
            for c in data.get("checklist", [])
        ],
    }
    save_json(pr_session_path(name), session)
    print_pr(session)
    print(f"Saved as '{name}'. After any interruption: focus pr --resume")

def latest_session():
    d = os.path.join(focus_home(), "pr")
    files = sorted(
        (os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json")),
        key=os.path.getmtime, reverse=True,
    )
    if not files:
        raise SystemExit("No PR sessions yet. Run `focus pr` on a diff first.")
    return files[0]

def print_pr(s):
    print(f"\nPR REVIEW: {s['name']}  (from {s['source']})")
    print("-" * 56)
    print("WHAT IT DOES:\n" + textwrap.fill(s["summary"], 72, initial_indent="  ",
                                            subsequent_indent="  "))
    if s.get("truncated"):
        print("\n  !! Diff was truncated — large PR. Review the tail manually.")
    if s["risks"]:
        print("\nLOOK HARDEST AT:")
        for r in s["risks"]:
            print("  ! " + r)
    print("\nCHECKLIST (tick with `focus pr --check N`):")
    undone_seen = False
    for i, c in enumerate(s["checklist"], 1):
        tick = "x" if c["done"] else " "
        marker = ""
        if not c["done"] and not undone_seen:
            marker = "   <- you are here"
            undone_seen = True
        print(f"  [{tick}] {i}. {c['file']}: {c['item']}{marker}")
    done = sum(1 for c in s["checklist"] if c["done"])
    print(f"\n  {done}/{len(s['checklist'])} done.")

def cmd_pr_resume(args):
    path = pr_session_path(args.name) if args.name else latest_session()
    if not os.path.exists(path):
        path = latest_session()
    s = load_json(path, None)
    if s is None:
        raise SystemExit("Session file unreadable.")
    if args.check is not None:
        idx = args.check - 1
        if not (0 <= idx < len(s["checklist"])):
            raise SystemExit(f"No checklist item {args.check}")
        s["checklist"][idx]["done"] = True
        save_json(path, s)
    print_pr(s)

# ------------------------------------------------ draft

def cmd_draft(args):
    text = read_multiline(" ".join(args.bullets),
                          "Rough bullets or half-draft of what you want to say")
    if not text.strip():
        raise SystemExit("Nothing to draft from.")
    style = DRAFT_STYLES[args.type]
    tone = {"friendly": "Tone: warm and friendly.",
            "neutral": "Tone: neutral and professional.",
            "firm": "Tone: polite but firm; do not soften the ask away."}[args.tone]
    to = f"Recipient: {args.to}." if args.to else ""
    system = "\n".join([SYS_DRAFT, style, tone, to]) + voice_system_suffix(args.no_voice)
    if args.polish:
        user = "Polish this draft, keep my meaning and roughly my voice:\n\n" + text
    else:
        user = "Write the message from these notes:\n\n" + text
    msg = chat(system, user, temperature=0.5).strip()
    print("\n" + "-" * 56)
    print(msg)
    print("-" * 56)
    print("(copy what you like, rerun with --tone/--type to adjust)")

# ------------------------------------------------ voice

def _derive_profile(answers, samples):
    parts = []
    if answers:
        parts.append("INTERVIEW ANSWERS:")
        for key, q in VOICE_QUESTIONS:
            if answers.get(key):
                parts.append(f"Q: {q}\nA: {answers[key]}")
    if samples:
        parts.append("\nREAL MESSAGES THEY HAVE SENT:")
        for i, s in enumerate(samples, 1):
            parts.append(f"--- sample {i} ---\n{s.strip()}")
    data = extract_json(chat(SYS_VOICE, "\n".join(parts)))
    profile = data.get("profile", "").strip()
    if not profile:
        raise SystemExit("Model returned an empty profile — try again.")
    return profile

def cmd_voice(args):
    v = load_voice()
    if args.action == "show":
        if not v.get("profile"):
            print("No voice profile yet. Run `focus voice setup`.")
        else:
            print("\nYOUR VOICE PROFILE (used by every draft):\n")
            print(textwrap.indent(v["profile"], "  "))
            print(f"\n  built from {len(v.get('samples', []))} sample(s), "
                  f"updated {v.get('updated', '?')}")
            print("  Tune it: focus voice edit · add samples: focus voice learn · disable: focus voice off")
        return
    if args.action == "off":
        if v:
            save_voice({})
            print("Voice profile cleared. Drafts go back to the default style.")
        else:
            print("No voice profile to clear.")
        return
    if args.action == "edit":
        if not v.get("profile"):
            raise SystemExit("Nothing to edit yet — run `focus voice setup` first.")
        editor = os.environ.get("EDITOR", "nano")
        path = _path("voice_profile.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(v["profile"])
        subprocess.run([editor, path])
        with open(path, encoding="utf-8") as f:
            v["profile"] = f.read().strip()
        v["updated"] = now_iso()
        save_voice(v)
        print("Profile updated.")
        return
    if args.action == "learn":
        samples = v.get("samples", [])
        if args.samples:
            for sp in args.samples:
                with open(sp, encoding="utf-8") as f:
                    samples.append(f.read())
        else:
            text = read_multiline("", "Paste a message you actually sent")
            if not text.strip():
                raise SystemExit("Nothing pasted.")
            samples.append(text)
        v["samples"] = samples[-10:]  # keep the freshest 10
        v["profile"] = _derive_profile(v.get("answers", {}), v["samples"])
        v["updated"] = now_iso()
        save_voice(v)
        print("Learned. Updated profile:\n")
        print(textwrap.indent(v["profile"], "  "))
        return
    # setup
    answers = {}
    samples = []
    if args.samples:
        for sp in args.samples:
            with open(sp, encoding="utf-8") as f:
                samples.append(f.read())
    if sys.stdin.isatty():
        print("\nQuick voice interview — 8 questions, a few words each is plenty.")
        print("Skip any with Enter. Your answers never leave this machine.\n")
        for key, q in VOICE_QUESTIONS:
            try:
                ans = input(f"  {q}\n  > ").strip()
            except EOFError:
                break
            if ans:
                answers[key] = ans
        if not args.samples:
            print("\nOptional but powerful: paste 1-3 real messages you've sent")
            print("(Slack or email, any length). Ctrl-D when done, or Ctrl-D now to skip.")
            pasted = sys.stdin.read().strip()
            if pasted:
                samples.append(pasted)
    elif not samples:
        raise SystemExit(
            "Non-interactive setup needs samples: focus voice setup --samples msg1.txt msg2.txt"
        )
    if not answers and not samples:
        raise SystemExit("Nothing to build a profile from.")
    profile = _derive_profile(answers, samples)
    save_voice({"profile": profile, "answers": answers, "samples": samples,
                "updated": now_iso()})
    print("\nDone — every `focus draft` now writes in your voice:\n")
    print(textwrap.indent(profile, "  "))
    print("\nIf a draft sounds off: `focus voice learn` with a real message,")
    print("or `focus voice edit` to tweak the profile directly.")

# ------------------------------------------------ doctor

def cmd_doctor(args):
    print("focus doctor\n")
    print(f"  data dir : {focus_home()}")
    store = load_store()
    open_n = sum(1 for t in store["tasks"] if t["status"] != "done")
    print(f"  tasks    : {open_n} open / {len(store['tasks'])} total")
    try:
        name, base, model = detect_runtime()
        print(f"  model    : OK — {name} at {base} (model: {model})")
    except NoModelError as e:
        print("  model    : NOT FOUND")
        print(textwrap.indent(str(e), "             "))

# ------------------------------------------------ ui

def api_state():
    store = load_store()
    nxt = pick_next(store)
    llm_ok = True
    try:
        detect_runtime()
    except NoModelError:
        llm_ok = False
    return {"tasks": store["tasks"], "next_id": nxt["id"] if nxt else None,
            "llm": llm_ok, "voice": bool(load_voice().get("profile"))}

class UIHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n).decode()) if n else {}

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, UI_HTML.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._send(200, api_state())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = self._body()
            if self.path == "/api/add":
                store = load_store()
                t = new_task(store, body["title"],
                             status=body.get("status", "inbox"),
                             priority=int(body.get("priority", 2)))
                save_store(store)
                self._send(200, t)
            elif self.path == "/api/update":
                store = load_store()
                t = get_task(store, int(body["id"]))
                if not t:
                    return self._send(404, {"error": "no such task"})
                if "status" in body and body["status"] in STATUSES:
                    t["status"] = body["status"]
                    if body["status"] == "done":
                        for s in t["subtasks"]:
                            s["done"] = True
                if "toggle_subtask" in body:
                    i = int(body["toggle_subtask"])
                    if 0 <= i < len(t["subtasks"]):
                        t["subtasks"][i]["done"] = not t["subtasks"][i]["done"]
                        if all(s["done"] for s in t["subtasks"]) and t["subtasks"]:
                            t["status"] = "done"
                touch(t)
                save_store(store)
                self._send(200, t)
            elif self.path == "/api/break":
                store = load_store()
                t = get_task(store, int(body["id"]))
                if not t:
                    return self._send(404, {"error": "no such task"})
                data = extract_json(chat(SYS_BREAK, t["title"]))
                t["subtasks"] = [
                    {"text": s["text"], "done": False,
                     "estimate_min": s.get("estimate_min")}
                    for s in data.get("subtasks", []) if s.get("text")
                ]
                touch(t)
                save_store(store)
                self._send(200, t)
            elif self.path == "/api/dump":
                store = load_store()
                created = []
                try:
                    data = extract_json(chat(SYS_DUMP, body["text"]))
                    for item in data.get("tasks", []):
                        if item.get("title"):
                            pri = item.get("priority") if item.get("priority") in (1, 2, 3) else 2
                            created.append(new_task(store, item["title"], priority=pri,
                                                    estimate_min=item.get("estimate_min")))
                except NoModelError:
                    for line in body["text"].splitlines():
                        line = line.strip(" -*\t")
                        if line:
                            created.append(new_task(store, line))
                save_store(store)
                self._send(200, {"created": created})
            elif self.path == "/api/draft":
                style = DRAFT_STYLES.get(body.get("type", "slack"), DRAFT_STYLES["slack"])
                tone = body.get("tone", "friendly")
                tone_line = {"friendly": "Tone: warm and friendly.",
                             "neutral": "Tone: neutral and professional.",
                             "firm": "Tone: polite but firm."}.get(tone, "")
                system = "\n".join([SYS_DRAFT, style, tone_line]) + \
                    voice_system_suffix(body.get("no_voice", False))
                msg = chat(system, "Write the message from these notes:\n\n" + body["text"],
                           temperature=0.5)
                self._send(200, {"message": msg.strip()})
            else:
                self._send(404, {"error": "not found"})
        except NoModelError as e:
            self._send(503, {"error": str(e)})
        except Exception as e:  # keep the local server alive
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

def cmd_ui(args):
    server = ThreadingHTTPServer(("127.0.0.1", args.port), UIHandler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"focus dashboard -> {url}   (Ctrl-C to stop)")
    if not args.no_browser:
        try:
            import webbrowser
            threading.Timer(0.4, webbrowser.open, [url]).start()
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")

# ---------------------------------------------------------------- UI html

UI_HTML = r"""<!DOCTYPE html>
<html lang="en-GB">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>focus — dashboard</title>
<style>
:root{--bg:#fafaf7;--card:#fff;--ink:#1a1a1a;--soft:#555;--accent:#0b5cad;
--accent-soft:#e7f0fa;--green:#1a7a3d;--green-soft:#e6f4ea;--amber:#8a5a00;
--amber-soft:#fdf3dd;--red:#a32020;--border:#d9d9d2;--purple:#5b2d86;--purple-soft:#f1e9f9;}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);
color:var(--ink);font-size:16px;line-height:1.5;padding:20px;}
.wrap{max-width:1100px;margin:0 auto;}
h1{font-size:1.3rem;margin-bottom:14px;}
h1 .llm{font-size:.75rem;font-weight:700;padding:3px 10px;border-radius:999px;vertical-align:middle;}
.llm.ok{background:var(--green-soft);color:var(--green);}
.llm.off{background:#fbeaea;color:var(--red);}
.one{background:var(--card);border:2px solid var(--green);border-radius:14px;
padding:18px 20px;margin-bottom:18px;}
.one .label{font-size:.75rem;font-weight:800;letter-spacing:.06em;color:var(--green);
text-transform:uppercase;}
.one .thing{font-size:1.25rem;font-weight:700;margin:6px 0;}
.one .from{color:var(--soft);font-size:.9rem;}
.one button{margin-top:10px;}
.timer{font-variant-numeric:tabular-nums;font-weight:800;font-size:1.2rem;color:var(--accent);
margin-left:12px;}
button{font:inherit;font-size:.88rem;font-weight:600;border:1px solid var(--border);
background:#fff;border-radius:8px;padding:6px 12px;cursor:pointer;color:var(--ink);}
button:hover{background:var(--accent-soft);border-color:var(--accent);}
button.primary{background:var(--green);border-color:var(--green);color:#fff;}
button.primary:hover{opacity:.9;}
.cols{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;}
@media(max-width:900px){.cols{grid-template-columns:repeat(2,1fr);}}
.col{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px;}
.col h2{font-size:.8rem;letter-spacing:.05em;text-transform:uppercase;color:var(--soft);
margin-bottom:8px;}
.task{border:1px solid var(--border);border-radius:10px;padding:8px 10px;margin-bottom:8px;
background:#fff;}
.task .t{font-weight:600;font-size:.92rem;cursor:pointer;}
.task .meta{font-size:.75rem;color:var(--soft);margin-top:2px;}
.task .pri1{color:var(--red);font-weight:700;}
.subs{margin-top:6px;border-top:1px dashed var(--border);padding-top:6px;}
.subs label{display:flex;gap:7px;font-size:.85rem;padding:2px 0;align-items:flex-start;cursor:pointer;}
.subs input{margin-top:3px;accent-color:var(--green);}
.subs .done{opacity:.5;text-decoration:line-through;}
.task .actions{margin-top:6px;display:flex;gap:6px;flex-wrap:wrap;}
.task .actions button{font-size:.75rem;padding:3px 8px;}
.panel{background:var(--card);border:1px solid var(--border);border-radius:12px;
padding:14px;margin-top:18px;}
.panel h2{font-size:1rem;margin-bottom:8px;}
textarea{width:100%;font:inherit;font-size:.92rem;border:1px solid var(--border);
border-radius:8px;padding:9px;min-height:76px;resize:vertical;background:#fff;color:var(--ink);}
.row{display:flex;gap:8px;margin-top:8px;flex-wrap:wrap;align-items:center;}
select{font:inherit;font-size:.88rem;border:1px solid var(--border);border-radius:8px;
padding:5px 8px;background:#fff;color:var(--ink);}
pre.out{white-space:pre-wrap;background:var(--accent-soft);border-radius:8px;padding:12px;
margin-top:10px;font:inherit;font-size:.92rem;color:#123a5c;}
.two{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
@media(max-width:900px){.two{grid-template-columns:1fr;}}
.muted{color:var(--soft);font-size:.85rem;}
</style>
</head>
<body>
<div class="wrap">
  <h1>focus <span id="llmBadge" class="llm off">checking model…</span></h1>

  <div class="one">
    <div class="label">Do this one thing</div>
    <div class="thing" id="oneThing">Loading…</div>
    <div class="from" id="oneFrom"></div>
    <button class="primary" id="doneBtn" style="display:none">Done ✓</button>
    <button id="timerBtn">Start 25-min timer</button><span class="timer" id="timerTxt"></span>
  </div>

  <div class="cols" id="board"></div>

  <div class="two">
    <div class="panel">
      <h2>Brain-dump</h2>
      <p class="muted">Everything on your mind, messy is fine. The model sorts it into tasks.</p>
      <textarea id="dumpText" placeholder="e.g. fix flaky auth test, reply to Priya about the rota, look at #4521, book desk for Thursday…"></textarea>
      <div class="row"><button class="primary" onclick="doDump()">Capture</button>
      <span class="muted" id="dumpStatus"></span></div>
    </div>
    <div class="panel">
      <h2>Draft a message</h2>
      <textarea id="draftText" placeholder="bullets of what you want to say…"></textarea>
      <div class="row">
        <select id="draftType"><option value="slack">Slack</option><option value="email">Email</option>
        <option value="pr-comment">PR comment</option><option value="standup">Standup</option></select>
        <select id="draftTone"><option value="friendly">Friendly</option>
        <option value="neutral">Neutral</option><option value="firm">Firm</option></select>
        <label class="muted" id="voiceWrap" style="display:none;cursor:pointer">
          <input type="checkbox" id="voiceUse" checked> in my voice</label>
        <button class="primary" onclick="doDraft()">Draft it</button>
        <span class="muted" id="draftStatus"></span>
      </div>
      <pre class="out" id="draftOut" style="display:none"></pre>
    </div>
  </div>
</div>
<script>
let state={tasks:[],next_id:null,llm:false};
const STATUSES=["inbox","now","next","later"];
const PRI={1:"high",2:"med",3:"low"};

async function api(path,body){
  const r=await fetch(path,body?{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body)}:{});
  if(!r.ok){const e=await r.json().catch(()=>({error:r.statusText}));throw new Error(e.error);}
  return r.json();
}
async function refresh(){
  state=await api("/api/state");
  document.getElementById("llmBadge").className="llm "+(state.llm?"ok":"off");
  document.getElementById("llmBadge").textContent=state.llm?"local model connected":"no model — AI features off";
  document.getElementById("voiceWrap").style.display=state.voice?"inline":"none";
  renderOne();renderBoard();
}
function nextAction(t){
  if(!t)return null;
  for(let i=0;i<t.subtasks.length;i++)if(!t.subtasks[i].done)
    return{text:t.subtasks[i].text,sub:i,est:t.subtasks[i].estimate_min};
  return{text:t.title,sub:null,est:t.estimate_min};
}
function renderOne(){
  const t=state.tasks.find(x=>x.id===state.next_id);
  const el=document.getElementById("oneThing"),from=document.getElementById("oneFrom"),
    btn=document.getElementById("doneBtn");
  if(!t){el.textContent="Nothing on the list — enjoy it.";from.textContent="";btn.style.display="none";return;}
  const a=nextAction(t);
  el.textContent=a.text+(a.est?"  (~"+a.est+"m)":"");
  from.textContent=a.sub!==null?("from #"+t.id+" "+t.title):("task #"+t.id);
  btn.style.display="inline-block";
  btn.onclick=async()=>{
    if(a.sub!==null)await api("/api/update",{id:t.id,toggle_subtask:a.sub});
    else await api("/api/update",{id:t.id,status:"done"});
    refresh();
  };
}
function renderBoard(){
  const board=document.getElementById("board");board.innerHTML="";
  for(const st of STATUSES){
    const col=document.createElement("div");col.className="col";
    col.innerHTML="<h2>"+st+"</h2>";
    for(const t of state.tasks.filter(x=>x.status===st)){
      const d=document.createElement("div");d.className="task";
      const doneSubs=t.subtasks.filter(s=>s.done).length;
      d.innerHTML='<div class="t">#'+t.id+" "+esc(t.title)+'</div>'+
        '<div class="meta"><span class="pri'+t.priority+'">'+PRI[t.priority]+"</span>"+
        (t.subtasks.length?" · "+doneSubs+"/"+t.subtasks.length+" steps":"")+"</div>";
      if(t.subtasks.length){
        const subs=document.createElement("div");subs.className="subs";
        t.subtasks.forEach((s,i)=>{
          const l=document.createElement("label");if(s.done)l.className="done";
          const c=document.createElement("input");c.type="checkbox";c.checked=s.done;
          c.onchange=async()=>{await api("/api/update",{id:t.id,toggle_subtask:i});refresh();};
          l.appendChild(c);l.appendChild(document.createTextNode(s.text+
            (s.estimate_min?" ("+s.estimate_min+"m)":"")));
          subs.appendChild(l);
        });
        d.appendChild(subs);
      }
      const act=document.createElement("div");act.className="actions";
      if(!t.subtasks.length&&state.llm)addBtn(act,"break down",async()=>{
        await api("/api/break",{id:t.id});refresh();});
      for(const to of STATUSES.filter(x=>x!==st))
        addBtn(act,"→ "+to,async()=>{await api("/api/update",{id:t.id,status:to});refresh();});
      addBtn(act,"done ✓",async()=>{await api("/api/update",{id:t.id,status:"done"});refresh();});
      d.appendChild(act);
      col.appendChild(d);
    }
    board.appendChild(col);
  }
}
function addBtn(parent,label,fn){const b=document.createElement("button");
  b.textContent=label;b.onclick=fn;parent.appendChild(b);}
function esc(s){return s.replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
async function doDump(){
  const t=document.getElementById("dumpText"),s=document.getElementById("dumpStatus");
  if(!t.value.trim())return;s.textContent="thinking…";
  try{const r=await api("/api/dump",{text:t.value});
    s.textContent="captured "+r.created.length+" task(s)";t.value="";refresh();}
  catch(e){s.textContent=e.message;}
}
async function doDraft(){
  const t=document.getElementById("draftText"),s=document.getElementById("draftStatus"),
    o=document.getElementById("draftOut");
  if(!t.value.trim())return;s.textContent="writing…";
  try{const r=await api("/api/draft",{text:t.value,
      type:document.getElementById("draftType").value,
      tone:document.getElementById("draftTone").value,
      no_voice:!document.getElementById("voiceUse").checked});
    o.style.display="block";o.textContent=r.message;s.textContent="";}
  catch(e){s.textContent=e.message;}
}
let timer=null,left=0;
document.getElementById("timerBtn").onclick=()=>{
  if(timer){clearInterval(timer);timer=null;left=0;
    document.getElementById("timerTxt").textContent="";
    document.getElementById("timerBtn").textContent="Start 25-min timer";return;}
  left=25*60;document.getElementById("timerBtn").textContent="Stop timer";
  timer=setInterval(()=>{left--;
    const m=String(Math.floor(left/60)).padStart(2,"0"),s=String(left%60).padStart(2,"0");
    document.getElementById("timerTxt").textContent=m+":"+s;
    if(left<=0){clearInterval(timer);timer=null;
      document.getElementById("timerTxt").textContent="break time!";
      document.getElementById("timerBtn").textContent="Start 25-min timer";}},1000);
};
refresh();setInterval(refresh,5000);
</script>
</body>
</html>"""

# ---------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(prog="focus",
                                description="local-first ADHD companion for developers")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("dump", help="brain-dump -> tasks")
    s.add_argument("text", nargs="*")
    s.set_defaults(fn=cmd_dump)

    s = sub.add_parser("add", help="add one task (no AI)")
    s.add_argument("title", nargs="+")
    s.add_argument("--status", default="inbox", choices=STATUSES)
    s.add_argument("-p", "--priority", type=int, default=2, choices=[1, 2, 3])
    s.set_defaults(fn=cmd_add)

    s = sub.add_parser("ls", help="list tasks")
    s.add_argument("-v", "--verbose", action="store_true", help="show subtasks")
    s.add_argument("-a", "--all", action="store_true", help="include done")
    s.set_defaults(fn=cmd_ls)

    s = sub.add_parser("break", help="split a task into small steps")
    s.add_argument("id")
    s.add_argument("--hint", nargs="*", help="extra context for the model")
    s.set_defaults(fn=cmd_break)

    s = sub.add_parser("next", help="show exactly one next action")
    s.add_argument("--energy", choices=["low"], help="low = easiest thing first")
    s.add_argument("--why", action="store_true")
    s.set_defaults(fn=cmd_next)

    s = sub.add_parser("start", help="move task to Now")
    s.add_argument("id")
    s.set_defaults(fn=cmd_start)

    s = sub.add_parser("done", help="finish task (12) or one subtask (12.3)")
    s.add_argument("id")
    s.set_defaults(fn=cmd_done)

    s = sub.add_parser("move", help="move task between lists")
    s.add_argument("id")
    s.add_argument("status", choices=STATUSES)
    s.set_defaults(fn=cmd_move)

    s = sub.add_parser("pr", help="summarise a diff + review checklist")
    s.add_argument("-f", "--file", help="read diff from file")
    s.add_argument("--name", help="session name")
    s.add_argument("--resume", action="store_true", help="show latest session")
    s.add_argument("--check", type=int, help="tick checklist item N (implies --resume)")
    s.set_defaults(fn=cmd_pr)

    s = sub.add_parser("draft", help="bullets -> polished message")
    s.add_argument("bullets", nargs="*")
    s.add_argument("--type", default="slack", choices=list(DRAFT_STYLES))
    s.add_argument("--tone", default="friendly", choices=["friendly", "neutral", "firm"])
    s.add_argument("--to", help="who it's for")
    s.add_argument("--polish", action="store_true",
                   help="treat input as a draft to clean up, not bullets")
    s.add_argument("--no-voice", action="store_true",
                   help="ignore your voice profile for this draft")
    s.set_defaults(fn=cmd_draft)

    s = sub.add_parser("voice", help="teach drafts to sound like you")
    s.add_argument("action", nargs="?", default="setup",
                   choices=["setup", "show", "edit", "learn", "off"])
    s.add_argument("--samples", nargs="*",
                   help="files containing real messages you've sent")
    s.set_defaults(fn=cmd_voice)

    s = sub.add_parser("ui", help="open the local dashboard")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(fn=cmd_ui)

    s = sub.add_parser("doctor", help="check model connectivity")
    s.set_defaults(fn=cmd_doctor)

    args = p.parse_args(argv)
    if not getattr(args, "fn", None):
        p.print_help()
        return 0
    try:
        args.fn(args)
    except NoModelError as e:
        print(str(e), file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    sys.exit(main())
