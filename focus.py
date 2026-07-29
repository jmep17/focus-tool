#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""
focus — a local-first ADHD companion for developers.

Everything you write runs on your Mac. AI features talk to a local model server
(LM Studio on :1234 or Ollama on :11434) over localhost. No accounts, no cloud,
no telemetry. Non-AI commands work with no model running.

Two features read from elsewhere, and only ever read: `focus pr` can pull the
current branch's pull request through your own `gh` CLI, and `focus shortcut`
fetches the ticket you're working on. Nothing is ever sent out.

Commands:
  focus dump [text]        Brain-dump -> structured tasks (AI, with fallback)
  focus add <title>        Add a single task (no AI)
  focus ls                 List tasks grouped by status
  focus break <id>         Split a task into <=25-minute subtasks (AI)
  focus next               Show exactly ONE next action
  focus start <id>         Move a task to Now
  focus done <id>[.n]      Complete a task or one subtask
  focus move <id> <status> Move between inbox/now/next/later
  focus today              What you finished today (--week for the week)
  focus note <id> [text]   Breadcrumb: jot where you left off, shown on resume
  focus triage             Work through stale tasks, one at a time
  focus timer [min]        Countdown with a desktop notification (default 25)
  focus pr [...]           Summarise a diff + build a review checklist (AI)
  focus draft [...]        Bullets -> polished message (AI)
  focus memory [...]       Facts the AI remembers about you (local only)
  focus project [...]      Teach the AI features about this codebase (AI)
  focus shortcut [...]     Pull the Shortcut ticket you're working on into prompts
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
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------- paths

def focus_home():
    home = os.environ.get("FOCUS_HOME") or os.path.join(os.path.expanduser("~"), ".focus")
    os.makedirs(home, exist_ok=True)
    os.makedirs(os.path.join(home, "pr"), exist_ok=True)
    os.makedirs(os.path.join(home, "projects"), exist_ok=True)
    os.makedirs(os.path.join(home, "memory"), exist_ok=True)
    os.makedirs(os.path.join(home, "shortcut"), exist_ok=True)
    return home

def _path(name):
    return os.path.join(focus_home(), name)

def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default

def save_json(path, data, mode=None):
    """Atomic write. `mode` (0o600 for the one file holding a token) is applied to the
    temp file before the rename — os.replace keeps the source's mode, so chmodding the
    final path instead would leave a window where the secret is world-readable."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    if mode is not None:
        os.chmod(tmp, mode)
    os.replace(tmp, path)

def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def safe_name(name, default="project"):
    """Anything -> a filename we're happy to write into ~/.focus/."""
    return re.sub(r"[^\w.-]", "_", name)[:60] or default

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
    log_event("created", id=t["id"], title=t["title"])
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

def http_json(url, payload=None, timeout=None, headers=None):
    data = json.dumps(payload).encode() if payload is not None else None
    head = {"Content-Type": "application/json"}
    head.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=head)
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

def chat(system, user, temperature=0.3, on_delta=None):
    name, base, model = detect_runtime()
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if on_delta is not None:
        # Streaming is a display nicety, never a new failure mode. A server that refuses
        # `stream`, or streams nothing usable, falls through to the plain request below —
        # the caller gets the same string it always got, just later.
        try:
            text = _chat_stream(base + "/chat/completions", payload, on_delta)
            if text.strip():
                return text
        except (OSError, ValueError, KeyError):
            pass
    resp = http_json(base + "/chat/completions", payload)
    return resp["choices"][0]["message"]["content"]

def _chat_stream(url, payload, on_delta):
    """Read an OpenAI-shaped SSE stream, calling on_delta(text so far) as each piece
    lands, and return the whole reply. Tolerates a server that ignores `stream` and
    answers with one plain JSON body: that arrives as a single line with a `message`
    rather than a `delta`, and is read the same way."""
    data = json.dumps(dict(payload, stream=True)).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json", "Accept": "text/event-stream"})
    parts = []
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:                      # one line at a time, as the socket gives it
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line or line == "[DONE]":
                continue
            try:
                obj = json.loads(line)
            except ValueError:                # keep-alive comments, framing noise
                continue
            for ch in obj.get("choices", []):
                piece = ((ch.get("delta") or {}).get("content")
                         or (ch.get("message") or {}).get("content") or "")
                if piece:
                    parts.append(piece)
                    on_delta("".join(parts))
    return "".join(parts)

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

_SUMMARY_RE = re.compile(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)')

def _partial_summary(text):
    """The `summary` field out of a *half-written* JSON reply, for live progress. The
    reply can't be parsed yet — extract_json needs a closed object — so this reads the
    one field worth watching appear, and returns "" until it starts."""
    m = _SUMMARY_RE.search(text)
    if not m:
        return ""
    try:
        return json.loads('"' + m.group(1) + '"')
    except ValueError:                        # cut mid-escape; show it raw this frame
        return m.group(1)

class ModelReplyError(RuntimeError):
    pass

def ask_model(system, user, temperature=0.3, raw=False, on_delta=None):
    """One model call with one retry. raw=True returns the reply text; otherwise the
    first JSON object in it. Garbage replies and HTTP failures become ModelReplyError
    so callers never traceback on a flaky local model. NoModelError passes through —
    it has its own fallbacks. on_delta(text so far) streams the reply as it is written;
    on the retry it simply starts again from the empty string."""
    last = None
    for _ in range(2):
        try:
            reply = chat(system, user, temperature, on_delta=on_delta)
            return reply if raw else extract_json(reply)
        except NoModelError:
            raise
        except (ValueError, KeyError, OSError) as e:
            last = e
    raise ModelReplyError(
        f"The model's reply couldn't be used ({type(last).__name__}: {last}).\n"
        "  Small local models are flaky — try once more, or load a different model.\n"
        "  `focus doctor` shows what you're connected to."
    )

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
You are given a unified diff. What a reviewer needs first is what is WRONG with it:
findings first, chores second.
How to read the diff, before anything else:
- A line starting with "-" has ALREADY BEEN DELETED. It is not in the code any more.
  Never review it, and never ask for it to be changed or removed.
- A line starting with "+" is the code as it stands now. Judge that.
- Before you write a finding, re-read the "+" lines: if they already do the thing you
  were about to ask for, there is no finding. Recommending a fix that is already in
  the diff is the single worst thing you can do here.
Rules:
- findings are concrete problems you can point at in this diff. Each one names the
  file, quotes the line or symbol it is about, and says in one sentence what breaks
  and when it bites. Worst first.
- Zero findings is a respectable answer. Say nothing rather than padding with
  "consider adding tests" or "make sure this is correct".
- severity is exactly one of: "bug" (wrong behaviour), "risk" (works today, will
  bite later), "nit" (naming, style, dead code).
- Never comment on code that is not in the diff. Never invent a file or a line.
- fix: for every finding, the concrete change you would make. Name the edit, don't
  describe the goal — "pass idempotency_key=key into the retry", not "handle keys
  correctly". A one-line code snippet in `backticks` is ideal. Say "not sure" only
  when you genuinely cannot suggest one.
- suggestions: improvements that are NOT defects — a simpler shape, a name that
  would carry better, a test worth having. Nothing that duplicates a finding. Zero
  is fine.
- checklist is only what a human has to check by running the code or reading AROUND
  the diff — the things you cannot verify from the diff alone. One or two per
  changed file, ordered so related files sit together. Do not pad.
- If the author's description contains a task list, a TODO list or unticked boxes,
  those are THEIR notes to themselves. Never copy them into the checklist, the
  findings or the suggestions. Review the code, not the author's to-do list.
- Inside "summary", "what", "fix", "item" and suggestions, write markdown:
  `backticks` around identifiers, paths and values. No headings, no bullets, no
  code fences.
- "file" and "where" are plain text — no backticks, no markdown. They get quoted
  for you.
Reply with ONLY this JSON:
{"summary": "3 short sentences max: what this change does and why",
 "findings": [{"severity": "bug", "file": "path", "where": "the line or symbol",
               "what": "what is wrong, and when it bites",
               "fix": "the change you would make"}],
 "suggestions": ["an improvement that is not a defect"],
 "checklist": [{"file": "path", "item": "one specific thing to verify in this file"}]}"""

SYS_PR_FILE = """You are reviewing ONE file out of a larger pull request, closely.
You get the whole diff for that one file, so there is no excuse for skimming: read
every changed line and say what is wrong with it.
How to read the diff, before anything else:
- A line starting with "-" has ALREADY BEEN DELETED. It is not in the code any more.
  Never review it, and never ask for it to be changed or removed.
- A line starting with "+" is the code as it stands now. Judge that.
- Before you write a finding, re-read the "+" lines: if they already do the thing you
  were about to ask for, there is no finding. Recommending a fix that is already in
  the diff is the single worst thing you can do here.
Rules:
- findings are concrete problems in THIS file's diff. Each quotes the line or symbol
  it is about, says in one sentence what breaks and when, and gives the fix you would
  make (name the edit, not the goal).
- Look for, at least: off-by-one and boundary errors, unhandled error paths and
  exceptions, None/null and empty-collection cases, resource leaks, ordering and
  concurrency, state mutated while iterated, silent exception swallowing, changed
  behaviour the callers of this function will not expect, and anything the
  surrounding code's conventions say should have been done differently.
- severity is exactly one of: "bug" (wrong behaviour), "risk" (works today, will
  bite later), "nit" (naming, style, dead code).
- Zero findings is a respectable answer for a clean file. Never pad.
- Never comment on code that is not in this diff. Never invent a line.
- If the author's description contains a task list or unticked boxes, ignore it
  entirely — it is their notes, not your review.
- checklist: at most 2 things a human must check by running the code or reading
  around this diff. Often zero.
- markdown `backticks` inside "what", "fix" and "item"; "where" stays plain.
Reply with ONLY this JSON:
{"findings": [{"severity": "bug", "file": "path", "where": "the line or symbol",
               "what": "what is wrong, and when it bites",
               "fix": "the change you would make"}],
 "checklist": [{"file": "path", "item": "one specific thing to verify"}]}"""

SYS_PR_SUMMARY = """Every file in this pull request has already been reviewed on its own.
You are given the change's description and the findings that came back from those
passes. Write the overview a reviewer reads first.
Rules:
- summary: 3 short sentences max — what this change does and why, across all files.
- suggestions: improvements that are NOT defects and do not repeat a finding —
  a simpler shape, a better name, a test worth having. Zero is fine. Never copy
  anything out of the author's own task list or TODOs.
- Do not restate the findings. Do not invent new ones.
- markdown `backticks` around identifiers and paths.
Reply with ONLY this JSON:
{"summary": "...", "suggestions": ["..."]}"""

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

# ---------------------------------------------------------------- project context

SYS_PROJECT = """You build a compact project profile from a repository brief, so that a
model working on this codebase knows how it actually works.
Keep only facts someone would act on: the stack, the conventions this repo follows,
invariants that must not break, areas that are fragile or legacy, and how tests are laid
out. Drop install instructions, badges, licence text and project history.
Structure it as markdown '## ' sections, in this order: '## Stack', '## Conventions',
'## Testing', then one section per significant area of the codebase, each named after its
directory (e.g. '## payments/'). Sections must be '## ' — nothing else is read.
Write facts, not prose about the project. Under 400 words total.
Reply with ONLY this JSON:
{"profile": "the markdown profile"}"""

# Read in order; the first one that exists wins. Not concatenated — a repo with both a
# CLAUDE.md and a README has the review-relevant material in the former.
PROJECT_DOCS = ["CLAUDE.md", "AGENTS.md", ".cursorrules", "CONTRIBUTING.md",
                "ARCHITECTURE.md", "docs/architecture.md", "README.md"]
PROJECT_MANIFESTS = ["pyproject.toml", "package.json", "go.mod", "Cargo.toml",
                     "Gemfile", "requirements.txt"]
# Sections whose heading contains one of these always survive selection. Coupled to the
# headings SYS_PROJECT dictates — change one, change the other.
PINNED_HEADINGS = ("stack", "conventions", "review", "invariant", "testing")

MAX_HARVEST_CHARS = 24_000       # ceiling on the brief we hand the model
MAX_CONTEXT_CHARS = 4_000        # ceiling on context injected into `focus pr`
BRIEF_CONTEXT_CHARS = 1_200      # ...and into dump/break/draft, which gain less

PROJECT_SKELETON = """## Stack


## Conventions


## Fragile areas


## What reviewers here flag

"""

def _git(argv, cwd=None):
    """Run a git subcommand, returning stdout, or "" on any failure."""
    try:
        out = subprocess.run(["git"] + argv, capture_output=True, text=True,
                             timeout=15, cwd=cwd)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout if out.returncode == 0 else ""

def _gh(argv, cwd=None, timeout=30):
    """Run a `gh` subcommand -> (stdout, reason). reason is "" on success, else a short
    sentence fit to print. Unlike _git this reports why it failed: `focus pr fetch` is an
    explicit request, and "nothing happened" is a bad answer to one. Never raises.

    This is the one place focus reaches GitHub, and it does it through the user's own
    already-authenticated gh — focus stores no token and speaks no GitHub API itself.
    """
    try:
        out = subprocess.run(["gh"] + argv, capture_output=True, text=True,
                             timeout=timeout, cwd=cwd)
    except FileNotFoundError:
        return "", "GitHub CLI not installed — `brew install gh`, then `gh auth login`."
    except (OSError, subprocess.TimeoutExpired):
        return "", "GitHub CLI didn't respond."
    if out.returncode == 0:
        return out.stdout, ""
    err = " ".join(out.stderr.split())
    low = err.lower()
    if "no pull requests found" in low or "no default remote" in low:
        return "", "No pull request found for this branch."
    if "auth" in low and "login" in low:
        return "", "GitHub CLI isn't logged in — run `gh auth login`."
    return "", err.split(". ")[0][:200] or "GitHub CLI failed."

def _remote_ident(root):
    """github.com/you/repo from either an https or an scp-style remote URL."""
    url = _git(["config", "--get", "remote.origin.url"], cwd=root).strip()
    if not url:
        return ""
    url = re.sub(r"^[a-zA-Z][\w+.-]*://", "", url)   # drop scheme
    url = re.sub(r"^[^@/]+@", "", url)               # drop user@
    url = url.replace(":", "/", 1)                   # host:path -> host/path
    return re.sub(r"\.git$", "", url).strip("/")

_ROOT_CACHE = {}

# Which repo everything root-derived resolves against. Only the UI's picker sets it —
# the CLI leaves it None and stays cwd-based, so no command changes behaviour. One
# value for the whole dashboard, not per-request: there is one active repo at a time.
_ACTIVE_ROOT = None

def active_root():
    return _ACTIVE_ROOT or os.getcwd()

def set_active_root(path):
    """Point every root-derived lookup at `path`, and return the repo root it resolved
    to. Raises ValueError if it isn't a readable directory."""
    global _ACTIVE_ROOT
    path = os.path.abspath(os.path.expanduser(path or ""))
    if not os.path.isdir(path):
        raise ValueError(f"Not a folder: {path}")
    # A subdirectory of a repo selects the repo, the same way the cwd path does.
    root = _git(["rev-parse", "--show-toplevel"], cwd=path).strip() or path
    _ROOT_CACHE.pop(root, None)   # re-selecting re-probes: a remote may have been added
    _ACTIVE_ROOT = root
    return root

def is_git_repo(path):
    """Cheap enough for a directory listing — a stat, not a shell-out. `exists` rather
    than `isdir` because a linked worktree's .git is a file."""
    return os.path.exists(os.path.join(path, ".git"))

def _root_and_key():
    """(repo root, storage key) for the active repo. Cached — this shells out to git
    twice, and api_state() is polled every few seconds by the dashboard."""
    cwd = active_root()
    if cwd not in _ROOT_CACHE:
        root = _git(["rev-parse", "--show-toplevel"], cwd=cwd).strip() or cwd
        ident = _remote_ident(root) or os.path.basename(root.rstrip(os.sep))
        _ROOT_CACHE[cwd] = (root, safe_name(ident))
    return _ROOT_CACHE[cwd]

def project_root():
    return _root_and_key()[0]

def project_key(name=None):
    return safe_name(name) if name else _root_and_key()[1]

def project_path(key):
    return os.path.join(focus_home(), "projects", key + ".json")

def load_project(key):
    return load_json(project_path(key), {})

def save_project(key, p):
    save_json(project_path(key), p)

def list_projects():
    d = os.path.join(focus_home(), "projects")
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json"))

# ---- the dashboard's repo selection, remembered between runs

MAX_RECENT_REPOS = 8

def load_ui_prefs():
    return load_json(_path("ui.json"), {})

def save_ui_prefs(p):
    save_json(_path("ui.json"), p)

def remember_repo(root):
    """Push `root` to the front of the recent list and persist it as the selection."""
    p = load_ui_prefs()
    recent = [r for r in p.get("recent", []) if r != root]
    p["repo"] = root
    p["recent"] = [root] + recent[:MAX_RECENT_REPOS - 1]
    save_ui_prefs(p)
    return p

# ---- turning a diff (or any text) into what to look for

def diff_paths(diff):
    """Changed file paths from a unified diff, in diff order."""
    paths = []
    for m in re.finditer(r"^\+\+\+ (?:b/)?(\S+)", diff, re.M):
        if m.group(1) != "/dev/null" and m.group(1) not in paths:
            paths.append(m.group(1))
    if not paths:  # pure renames and mode changes carry no +++ line
        for m in re.finditer(r"^diff --git a/(\S+)", diff, re.M):
            if m.group(1) not in paths:
                paths.append(m.group(1))
    return paths

def query_tokens(text):
    """Lowercase tokens from paths or free text. camelCase-aware, so
    payments/StripeClient_v2.py -> {payments, stripe, client}."""
    if not isinstance(text, str):
        text = " ".join(text)
    words = re.split(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])", text)
    return {w.lower() for w in words if len(w) > 2}

# ---- fitting a profile into a budget

def split_sections(md):
    """[(heading, text)]. Index 0 is whatever precedes the first '## '."""
    parts = re.split(r"^## ", md, flags=re.M)
    out = [("", parts[0])]
    for p in parts[1:]:
        out.append((p.split("\n", 1)[0], "## " + p))
    return out

def _clip(text, budget):
    """Hard-truncate at a line boundary. Without this a heading-less document, or one
    oversized section, would blow the budget and silently eat the diff's share."""
    if len(text) <= budget:
        return text
    cut = text[:budget]
    nl = cut.rfind("\n")
    return (cut[:nl] if nl > budget // 2 else cut).rstrip()

def select_sections(md, tokens, budget):
    """Trim a markdown profile to `budget` chars, keeping the sections that matter for
    `tokens`. Deterministic and model-free — see README for why this isn't embeddings."""
    md = md.strip()
    if len(md) <= budget:
        return md
    sections = split_sections(md)
    keep, used = {}, 0
    for i, (head, text) in enumerate(sections):
        if i == 0 or any(p in head.lower() for p in PINNED_HEADINGS):
            text = _clip(text, max(0, budget - used))
            if not text:
                break
            keep[i] = text
            used += len(text) + 1
    scored = sorted(
        (-(3 * len(tokens & query_tokens(head)) + len(tokens & query_tokens(text))), i, text)
        for i, (head, text) in enumerate(sections) if i not in keep
    )
    # If anything matched, spend what's left on matches only — padding the budget with
    # unrelated sections dilutes the signal for no gain.
    for score, i, text in ([s for s in scored if s[0] < 0] or scored):
        if used + len(text) + 1 <= budget:
            keep[i] = text
            used += len(text) + 1
    return "\n".join(keep[i] for i in sorted(keep))

# ---- turning a repo into LLM-friendly markdown

def _read_head(path, limit):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read(limit)
    except OSError:
        return ""

def _repo_files(root):
    out = _git(["ls-files"], cwd=root)
    if out.strip():
        return out.splitlines()
    files, skip = [], {"node_modules", "venv", "__pycache__", "dist", "build", "target"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith(".")]
        for fn in filenames:
            files.append(os.path.relpath(os.path.join(dirpath, fn), root))
        if len(files) > 5_000:
            break
    return files

def _tree_lines(files, limit=60):
    """Collapse a file list to two directory levels with counts — far more architecture
    per character than a flat listing."""
    counts = {}
    for f in files:
        parts = f.split("/")
        key = "/".join(parts[:2]) + "/" if len(parts) > 2 else \
              (parts[0] + "/" if len(parts) > 1 else parts[0])
        counts[key] = counts.get(key, 0) + 1
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return [f"- {k} ({v} files)" if k.endswith("/") else f"- {k}" for k, v in rows]

def _churn_lines(root, limit=10):
    counts = {}
    for line in _git(["log", "--name-only", "--pretty=format:", "-n", "200"],
                     cwd=root).splitlines():
        line = line.strip()
        if line:
            counts[line] = counts.get(line, 0) + 1
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return [f"- {p} ({n})" for p, n in rows]

def repo_brief(root):
    """A deterministic markdown brief of a repo. No model involved, so this doubles as
    the fallback profile when no model server is running."""
    parts, found = [], []
    manifests = []
    for name in PROJECT_MANIFESTS:
        body = _read_head(os.path.join(root, name), 2_000)
        if body:
            found.append(name)
            manifests.append(f"--- {name} ---\n{body}")
    if found:
        parts.append("## Stack\n" + "\n".join("- " + n for n in found))
    files = _repo_files(root)
    if files:
        parts.append("## Layout\n" + "\n".join(_tree_lines(files)))
    churn = _churn_lines(root)
    if churn:
        parts.append("## Churn (commits touching each file, last 200)\n" + "\n".join(churn))
    if manifests:
        parts.append("## Manifests\n" + "\n".join(manifests))
    docs = []
    for name in PROJECT_DOCS:
        body = _read_head(os.path.join(root, name), 6_000)
        if body:
            docs.append(f"--- {name} ---\n{body}")
            break
    docdir = os.path.join(root, "docs")
    if os.path.isdir(docdir):
        for name in sorted(n for n in os.listdir(docdir) if n.endswith(".md"))[:2]:
            body = _read_head(os.path.join(docdir, name), 3_000)
            if body:
                docs.append(f"--- docs/{name} ---\n{body}")
    if docs:
        parts.append("## Docs\n" + "\n".join(docs))
    return _clip("\n\n".join(parts), MAX_HARVEST_CHARS)

# ---- resolution + injection

def resolve_project_context(query="", budget=MAX_CONTEXT_CHARS, name=None, disabled=False):
    """(text, source) for this repo, first tier that yields something.

    Mirrors detect_runtime()'s try-in-order idiom, except no context is always a valid
    answer — this never raises, so it can sit in front of every AI command.
    """
    if disabled or name == "none":
        return "", ""
    tokens = query_tokens(query) if query else set()
    if name:
        key = project_key(name)
        tiers = [(load_project(key).get("profile", ""), name)]
    else:
        key = project_key()
        root = project_root()
        tiers = [
            (load_project(key).get("profile", ""), key),
            (_read_head(os.path.join(root, ".focus", "project.md"), MAX_HARVEST_CHARS),
             ".focus/project.md"),
        ]
        for doc in PROJECT_DOCS:
            body = _read_head(os.path.join(root, doc), MAX_HARVEST_CHARS)
            if body:
                tiers.append((body, doc))
                break
    for text, source in tiers:
        if (text or "").strip():
            return select_sections(text, tokens, budget), source
    return "", ""

PROJECT_HEADER = ("\nPROJECT CONTEXT — how THIS repo actually works. Prefer these "
                  "conventions, invariants and known-fragile areas over generic advice, "
                  "and name the convention when a risk follows from one. This is "
                  "reference material describing what is already true, not a list of work "
                  "to request: never turn a convention into a finding unless the code in "
                  "front of you actually breaks it:\n")

def project_block(text):
    """Format already-resolved context. Split out because cmd_pr needs the source label
    for the session record, so it resolves separately."""
    return PROJECT_HEADER + text if text else ""

def project_system_suffix(query="", budget=MAX_CONTEXT_CHARS, name=None, disabled=False):
    return project_block(resolve_project_context(query, budget, name, disabled)[0])

# ---------------------------------------------------------------- shortcut tickets

# The one remote host focus talks to itself. Overridable so the tests can point it at a
# local fake, the same way config.json's `endpoint` points the model client at mock_llm.
SHORTCUT_API = "https://api.app.shortcut.com/api/v3"

MAX_TICKET_CHARS = 2_000     # ceiling on the ticket injected into `focus pr`
BRIEF_TICKET_CHARS = 800     # ...and into dump/break/draft, which gain less
MAX_TICKET_COMMENTS = 3

class TicketError(RuntimeError):
    pass

def shortcut_config_path():
    return _path("shortcut.json")

def load_shortcut_config():
    return load_json(shortcut_config_path(), {})

def save_shortcut_config(c):
    # 0600: the only file in ~/.focus/ that holds a credential.
    save_json(shortcut_config_path(), c, mode=0o600)

def shortcut_token(cfg=None):
    """Env first, so a token can be scoped to one shell without ever hitting disk.
    Callers holding the config already pass it rather than re-reading the file."""
    cfg = load_shortcut_config() if cfg is None else cfg
    return (os.environ.get("SHORTCUT_API_TOKEN") or "").strip() \
        or (cfg.get("token") or "").strip()

def shortcut_endpoint(cfg=None):
    cfg = load_shortcut_config() if cfg is None else cfg
    return (cfg.get("endpoint") or SHORTCUT_API).rstrip("/")

def story_path(sid):
    return os.path.join(focus_home(), "shortcut", f"story-{safe_name(str(sid))}.json")

def load_story(sid):
    return load_json(story_path(sid), {})

def save_story(s):
    save_json(story_path(s["id"]), s)

def cached_story_ids():
    """Every cached story id, shortest first. The one place that knows the filename
    convention — clear paths and listings go through it rather than re-deriving it."""
    d = os.path.join(focus_home(), "shortcut")
    try:
        ids = [f[6:-5] for f in os.listdir(d)
               if f.startswith("story-") and f.endswith(".json")]
    except OSError:
        return []
    return sorted(ids, key=lambda x: (len(x), x))

def _any_cached_story():
    """Is anything cached at all? The gate in front of every AI command, so it stops at
    the first hit rather than listing and sorting the whole directory."""
    try:
        with os.scandir(os.path.join(focus_home(), "shortcut")) as it:
            return any(e.name.startswith("story-") and e.name.endswith(".json")
                       for e in it)
    except OSError:
        return False

def clear_story_cache():
    """Delete every cached story, returning the ids that went."""
    ids = cached_story_ids()
    for sid in ids:
        os.remove(story_path(sid))
    return ids

def ticket_repo_path(key):
    """This repo's branch->story pins. Same two-scope shape as memory: its own
    `disabled` flag, independent of the global one."""
    return os.path.join(focus_home(), "shortcut", "repo-" + key + ".json")

def load_ticket_repo(key=None):
    return load_json(ticket_repo_path(key or project_key()), {})

def save_ticket_repo(r, key=None):
    save_json(ticket_repo_path(key or project_key()), r)

def save_shortcut_token(token):
    cfg = load_shortcut_config()
    cfg["token"] = token
    cfg.pop("disabled", None)     # saving a token is opting in
    save_shortcut_config(cfg)

def pin_story(sid):
    """Pin a story to the branch you're on, and return that branch (or "" if git can't
    name one — a detached HEAD still gets the ticket, just not a durable pin)."""
    branch = current_branch(project_root())
    if branch:
        r = load_ticket_repo()
        r.setdefault("branches", {})[branch] = str(sid)
        r.pop("disabled", None)
        save_ticket_repo(r)
    return branch

def set_ticket_enabled(on, here=False):
    """Flip the `disabled` flag in one scope. Shared by the CLI verb and /api/shortcut so
    the two scopes' bookkeeping is written once, not once per caller per scope."""
    load, save = ((load_ticket_repo, save_ticket_repo) if here
                  else (load_shortcut_config, save_shortcut_config))
    cfg = load()
    if on:
        cfg.pop("disabled", None)
    else:
        cfg["disabled"] = True
    save(cfg)

def current_branch(root=None):
    return _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root).strip()

# Shortcut's own branch convention is `you/sc-12345/some-title`, and its UI links are
# .../story/12345/slug — so both spellings are worth recognising.
_SC_RE = re.compile(r"\bsc-(\d+)\b", re.I)
_SC_URL_RE = re.compile(r"app\.shortcut\.com/[\w.-]+/story/(\d+)")

def _story_id_in(text):
    if not text:
        return None
    m = _SC_URL_RE.search(text) or _SC_RE.search(text)
    return m.group(1) if m else None

def detect_story_id(story=None, pr_meta=None, root=None, repo=None, deep=True):
    """(id, how) — the story this work belongs to, first hit wins. Never raises: no
    ticket is always a valid answer, so this can sit in front of every AI command.

    `deep=False` drops the commit-log tier, which is a second `git` spawn to find the
    weakest signal. Callers on the everyday path (dump/break/draft) pass False; the ones
    the user asked for by name — pr, and the `shortcut` verbs — pay for it.
    """
    if story:
        return str(story).lstrip("#").replace("sc-", ""), "given"
    root = root or project_root()
    branch = current_branch(root)
    if branch:
        repo = load_ticket_repo() if repo is None else repo
        pinned = (repo.get("branches") or {}).get(branch)
        if pinned:
            return str(pinned), "pinned to " + branch
        sid = _story_id_in(branch)
        if sid:
            return sid, "branch name"
    if pr_meta:
        sid = _story_id_in((pr_meta.get("title") or "") + "\n" + (pr_meta.get("body") or ""))
        if sid:
            return sid, "PR description"
    if deep:
        sid = _story_id_in(_git(["log", "--pretty=%s", "-n", "20"], cwd=root))
        if sid:
            return sid, "commit message"
    return None, ""

def _story_state(d):
    if d.get("completed"):
        return "done"
    return "in progress" if d.get("started") else "not started"

def render_story(d):
    """A Shortcut story -> markdown, most important first. `_clip` cuts the tail, so the
    order here *is* the priority order: name and description survive, comments don't."""
    sid = d.get("id")
    head = [f"sc-{sid} — {(d.get('name') or '').strip()}"]
    facts = [f for f in (d.get("story_type"), _story_state(d)) if f]
    if d.get("estimate"):
        facts.append(f"estimate {d['estimate']}")
    if facts:
        head.append(" · ".join(facts))
    out = ["\n".join(head)]
    desc = (d.get("description") or "").strip()
    if desc:
        # Left whole: acceptance criteria usually live in here under the team's own
        # heading, and guessing at which one would drop as much as it kept.
        out.append("## Description\n" + desc)
    tasks = [t for t in (d.get("tasks") or []) if t.get("description")]
    if tasks:
        out.append("## Tasks\n" + "\n".join(
            f"- [{'x' if t.get('complete') else ' '}] {t['description'].strip()}"
            for t in tasks))
    comments = [c for c in (d.get("comments") or []) if (c.get("text") or "").strip()]
    if comments:
        out.append("## Latest comments\n" + "\n\n".join(
            f"({(c.get('created_at') or '')[:10]}) {c['text'].strip()}"
            for c in comments[-MAX_TICKET_COMMENTS:]))
    return "\n\n".join(out)

def fetch_story(sid, cfg=None):
    """Fetch one story from Shortcut and cache it. Raises TicketError on any failure —
    callers that must not break (a review, an injection) catch it and use the cache."""
    cfg = load_shortcut_config() if cfg is None else cfg
    token = shortcut_token(cfg)
    if not token:
        raise TicketError(
            "No Shortcut token. Run `focus shortcut token` (or set SHORTCUT_API_TOKEN).\n"
            "  Get one at app.shortcut.com -> Settings -> API Tokens.")
    url = f"{shortcut_endpoint(cfg)}/stories/{sid}"
    try:
        d = http_json(url, timeout=15, headers={"Shortcut-Token": token})
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise TicketError("Shortcut rejected the token — `focus shortcut token` "
                              "to replace it.")
        if e.code == 404:
            raise TicketError(f"Shortcut has no story sc-{sid}.")
        raise TicketError(f"Shortcut returned HTTP {e.code}.")
    except (urllib.error.URLError, OSError) as e:
        raise TicketError(f"Couldn't reach Shortcut ({e}). Cached tickets still work.")
    except (ValueError, KeyError) as e:
        raise TicketError(f"Shortcut sent something unreadable ({e}).")
    story = {
        # Type, state and estimate live inside `text` via render_story — kept there
        # rather than duplicated as fields nothing reads.
        "id": d.get("id", sid),
        "name": (d.get("name") or "").strip(),
        "url": d.get("app_url", ""),
        "text": render_story(d),
        "fetched": now_iso(),
    }
    save_story(story)
    return story

def resolve_ticket_context(budget=MAX_TICKET_CHARS, story=None, disabled=False,
                           pr_meta=None, refresh=False):
    """(text, info) for the story behind this work, or ("", None). info is
    {"id","name","url","how"}.

    Reads the cache. Only `refresh=True` — which is `run_pr_review` alone — goes to the
    network, and even then a failure falls back to the cache rather than breaking the
    review. That is what keeps `focus dump` instant and working offline.
    """
    if disabled or story == "none":
        return "", None
    # Cheapest gate first. With nothing cached and no way to fetch, no tier below can
    # succeed — so bail before the file reads and git shell-outs that detection costs.
    # A user who has never touched Shortcut pays one os.scandir and nothing else.
    cached = _any_cached_story()
    if not cached and not refresh:
        return "", None
    cfg = load_shortcut_config()
    if cfg.get("disabled") or not (cached or shortcut_token(cfg)):
        return "", None
    repo = load_ticket_repo()
    if repo.get("disabled"):
        return "", None
    sid, how = detect_story_id(story, pr_meta, repo=repo, deep=refresh)
    if not sid:
        return "", None
    s = {}
    if refresh:
        try:
            s = fetch_story(sid, cfg)
        except TicketError:
            pass                      # cache is the fallback; never break a review
    s = s or load_story(sid)
    if not s.get("text"):
        return "", None
    return _clip(s["text"], budget), {"id": str(sid), "name": s.get("name", ""),
                                      "url": s.get("url", ""), "how": how}

# The marker line the mock's dispatch splits on — keep in sync with tests/mock_llm.py.
TICKET_HEADER = ("\nTICKET CONTEXT — the story this work is meant to deliver. Judge the "
                 "work against what it asks for, and say so when the two disagree. Its "
                 "acceptance criteria are not a to-do list to repeat back: only mention "
                 "one when the code in front of you fails to meet it:\n")

def ticket_block(text):
    """Format already-resolved ticket text. Split out for the same reason as
    project_block: run_pr_review needs the info record for the session."""
    return TICKET_HEADER + text if text else ""

def ticket_system_suffix(budget=BRIEF_TICKET_CHARS, story=None, disabled=False):
    return ticket_block(resolve_ticket_context(budget, story, disabled)[0])

# ---------------------------------------------------------------- history + memory

def history_path():
    return _path("history.jsonl")

def log_event(event, **fields):
    """Append one event line to history.jsonl. Single-line appends, no rewrite — the
    log feeds `focus today` and the memory suffix, and must never block a command."""
    rec = {"ts": now_iso(), "event": event}
    rec.update(fields)
    try:
        with open(history_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass

def load_history():
    out = []
    try:
        with open(history_path(), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        pass
    return out

def memory_path(key=None):
    """Global memory, or one repo's. Same file shape either way — {facts[], disabled?} —
    so the two scopes share every reader and writer below."""
    return (os.path.join(focus_home(), "memory", key + ".json") if key
            else _path("memory.json"))

def load_memory(key=None):
    return load_json(memory_path(key), {})

def save_memory(m, key=None):
    save_json(memory_path(key), m)

def memory_facts(m):
    return [f["text"] for f in m.get("facts", []) if f.get("text")]

def _estimate_ratio(hist):
    """Median actual/estimate across finished tasks; None until 3 samples exist."""
    ratios = [e["actual_min"] / e["estimate_min"] for e in hist
              if e.get("event") == "done" and e.get("estimate_min") and e.get("actual_min")]
    if len(ratios) < 3:
        return None
    return sorted(ratios)[len(ratios) // 2]

def _done_dates(hist):
    return {e["ts"][:10] for e in hist if e.get("event") == "done" and e.get("ts")}

def _streak(dates):
    """Consecutive days (ending today or yesterday) with at least one task done."""
    d = datetime.now(timezone.utc).date()
    if d.isoformat() not in dates:
        d -= timedelta(days=1)
    n = 0
    while d.isoformat() in dates:
        n += 1
        d -= timedelta(days=1)
    return n

def memory_stats_lines(hist=None):
    """Deterministic facts about this user from history.jsonl — no model involved."""
    hist = load_history() if hist is None else hist
    lines = []
    ratio = _estimate_ratio(hist)
    if ratio and ratio >= 1.4:
        lines.append(f"Their time estimates run short — real tasks take ~{ratio:.1f}x "
                     "the guess. Size estimates up accordingly.")
    elif ratio and ratio <= 0.7:
        lines.append(f"They overestimate — tasks take ~{ratio:.1f}x the guess. "
                     "Size estimates down accordingly.")
    streak = _streak(_done_dates(hist))
    if streak >= 2:
        lines.append(f"On a {streak}-day streak of finishing at least one task.")
    return lines

# The marker line the mock's dispatch splits on — keep in sync with tests/mock_llm.py.
MEMORY_HEADER = ("\nUSER MEMORY — this user's real patterns and standing facts. Use "
                 "them to size estimates and shape advice:\n")
MAX_MEMORY_CHARS = 600   # small on purpose: local context is scarce and shared

def _memory_lines(m):
    return ["- " + x for x in memory_facts(m)] if not m.get("disabled") else []

def memory_system_suffix(disabled=False, no_project=False):
    """Stats derived from history, explicit global facts, and this repo's own facts, as
    one system-prompt suffix. Same contract as voice_system_suffix: "" when off or empty.

    Stats stay global: history.jsonl carries no repo, and how far a person's estimates
    run out is a fact about the person. `no_project` drops the repo block for the same
    reason --no-project drops the repo profile — one flag, one meaning.
    """
    if disabled:
        return ""
    glob = load_memory()
    lines = ([] if glob.get("disabled") else ["- " + x for x in memory_stats_lines()]) \
        + _memory_lines(glob)
    proj_lines = [] if no_project else _memory_lines(load_memory(project_key()))
    proj = ""
    if proj_lines:
        proj = "\n".join([f"In this project ({project_key()}):"] + proj_lines)
        # Repo facts are the specific ones — clip them first, and cap them at half, so a
        # long global list can neither crowd them out nor be crowded out by them.
        proj = _clip(proj, MAX_MEMORY_CHARS // 2)
    body = _clip("\n".join(lines), MAX_MEMORY_CHARS - len(proj) - bool(proj))
    body = "\n".join(x for x in (body, proj) if x)
    return MEMORY_HEADER + body if body else ""

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

def total_estimate(task):
    subs = [s.get("estimate_min") for s in task["subtasks"] if s.get("estimate_min")]
    return task.get("estimate_min") or (sum(subs) if subs else None)

def _minutes_between(a, b):
    try:
        delta = datetime.fromisoformat(b) - datetime.fromisoformat(a)
    except (TypeError, ValueError):
        return None
    return max(1, int(delta.total_seconds() // 60))

def actual_minutes(task):
    """Elapsed start->done in minutes, or None. Capped at 8h: beyond that the gap is
    days away from the desk, not time on the task, and it would poison calibration."""
    m = _minutes_between(task.get("started"), task.get("completed"))
    return m if m is not None and m <= 480 else None

def complete_task(task):
    """The one done-path: stamp `completed`, tick subtasks, log the event. Shared by
    cmd_done, cmd_triage and the UI so history never misses a finish."""
    task["status"] = "done"
    for s in task["subtasks"]:
        s["done"] = True
    task["completed"] = now_iso()
    touch(task)
    log_event("done", id=task["id"], title=task["title"],
              estimate_min=total_estimate(task), actual_min=actual_minutes(task))

def _calib(task):
    """'  (est ~15m · took ~40m)' when both sides are known, else ''. Factual, not
    judgemental — the point is calibration, not guilt."""
    est, act = total_estimate(task), actual_minutes(task)
    return f"  (est ~{est}m · took ~{act}m)" if est and act else ""

def last_note(task):
    notes = (task.get("notes") or "").strip()
    return notes.splitlines()[-1] if notes else ""

def add_note(task, text):
    stamp = datetime.now().strftime("%d %b %H:%M")
    line = f"[{stamp}] {text.strip()}"
    old = (task.get("notes") or "").rstrip()
    task["notes"] = old + "\n" + line if old else line
    touch(task)
    log_event("note", id=task["id"])

STALE_DAYS = 14

def task_age_days(task):
    try:
        upd = datetime.fromisoformat(task.get("updated") or task["created"])
    except (TypeError, ValueError):
        return 0
    return max(0, int((datetime.now(timezone.utc) - upd).total_seconds() // 86400))

def stale_tasks(store):
    out = [t for t in store["tasks"] if t["status"] in ("inbox", "later")
           and task_age_days(t) >= STALE_DAYS]
    return sorted(out, key=lambda t: t.get("updated") or t["created"])

def _notify(title, text):
    """Local desktop notification. macOS only (osascript), silently skipped elsewhere;
    nothing leaves the machine."""
    if sys.platform != "darwin":
        return
    try:
        subprocess.run(["osascript", "-e",
                        'display notification "{}" with title "{}"'.format(
                            text.replace('"', "'"), title.replace('"', "'"))],
                       capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass

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
    if sub is not None and sub < 0:
        raise SystemExit(f"Bad task id: {ref!r} (subtasks start at .1)")
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
    age = task_age_days(t)
    if t["status"] in ("inbox", "later") and age >= STALE_DAYS:
        meta.append(f"{age}d old")
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
        system = SYS_DUMP + project_system_suffix(text, BRIEF_CONTEXT_CHARS,
                                                  disabled=args.no_project) \
            + ticket_system_suffix(disabled=args.no_ticket) \
            + memory_system_suffix(args.no_memory, args.no_project)
        data = ask_model(system, text)
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
                 priority=args.priority, estimate_min=args.estimate,
                 notes=args.notes)
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
    system = SYS_BREAK + project_system_suffix(context, BRIEF_CONTEXT_CHARS,
                                               disabled=args.no_project) \
        + ticket_system_suffix(story=args.story, disabled=args.no_ticket) \
        + memory_system_suffix(args.no_memory, args.no_project)
    data = ask_model(system, context)
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
    ln = last_note(t)
    if ln:
        print(f"     last note: {ln}")
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
    if not t.get("started"):
        t["started"] = now_iso()
    touch(t)
    log_event("started", id=t["id"], title=t["title"])
    save_store(store)
    print(f"#{t['id']} -> NOW. One task in Now at a time works best.")
    ln = last_note(t)
    if ln:
        print(f"  last note: {ln}")
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
        log_event("subtask_done", id=t["id"], subtask=sub_idx + 1)
        remaining = [x for x in t["subtasks"] if not x["done"]]
        if remaining:
            touch(t)
            save_store(store)
            print(f"Done: {s['text']}")
            print(f"Next up on this task: {remaining[0]['text']}   "
                  f"({len(remaining)} step(s) left)")
        else:
            complete_task(t)
            save_store(store)
            print(f"That was the last step — task #{t['id']} '{t['title']}' "
                  f"COMPLETE. Nice.{_calib(t)}")
    else:
        complete_task(t)
        save_store(store)
        print(f"Task #{t['id']} '{t['title']}' COMPLETE. Nice.{_calib(t)}")

def cmd_move(args):
    store = load_store()
    tid, _ = parse_task_ref(args.id)
    t = get_task(store, tid)
    if not t:
        raise SystemExit(f"No task #{tid}")
    if args.status not in STATUSES:
        raise SystemExit(f"Status must be one of: {', '.join(STATUSES)}")
    if args.status == "done":
        complete_task(t)
    else:
        t["status"] = args.status
        touch(t)
        log_event("moved", id=t["id"], to=args.status)
    save_store(store)
    print(f"#{t['id']} -> {args.status.upper()}")

# ------------------------------------------------ today / note / triage / timer

def cmd_today(args):
    store = load_store()
    days = 7 if args.week else 1
    today = datetime.now(timezone.utc).date()
    window = {(today - timedelta(days=i)).isoformat() for i in range(days)}
    done = [t for t in store["tasks"] if t["status"] == "done"
            and (t.get("completed") or "")[:10] in window]
    label = "this week" if args.week else "today"
    if not done:
        print(f"Nothing ticked off {label} yet — that's fine.")
        print("`focus next` for one small thing.")
        return
    done.sort(key=lambda t: t.get("completed") or "")
    print(f"\nDONE {label.upper()} ({len(done)}):")
    for t in done:
        print(f"  ✓ #{t['id']} {t['title']}{_calib(t)}")
    hist = load_history()
    streak = _streak(_done_dates(hist))
    if streak > 1:
        print(f"\n  {streak}-day streak of finishing at least one thing.")
    ratio = _estimate_ratio(hist)
    if ratio:
        print(f"  Estimate check: things take ~{ratio:.1f}x your guess lately.")
    print()

def cmd_note(args):
    store = load_store()
    tid, _ = parse_task_ref(args.id)
    t = get_task(store, tid)
    if not t:
        raise SystemExit(f"No task #{tid}")
    text = " ".join(args.text).strip()
    if not text:
        if t.get("notes"):
            print(f"Notes on #{t['id']} {t['title']}:")
            print(textwrap.indent(t["notes"], "  "))
        else:
            print(f"No notes on #{t['id']} yet. "
                  f"`focus note {t['id']} where you left off` to leave a breadcrumb.")
        return
    add_note(t, text)
    save_store(store)
    print(f"Noted on #{t['id']}: {text}")
    print("It'll be shown when this task comes up again.")

def cmd_triage(args):
    store = load_store()
    stale = stale_tasks(store)
    if not stale:
        print(f"Nothing stale — no inbox/later task untouched for {STALE_DAYS}+ days.")
        return
    if not sys.stdin.isatty():
        print(f"{len(stale)} stale task(s), untouched {STALE_DAYS}+ days:")
        for t in stale:
            print("  " + fmt_task(t))
        print("Run `focus triage` in a terminal to work through them.")
        return
    print(f"{len(stale)} stale task(s). One at a time — "
          "[k]eep  [l]ater  [d]one  [x] drop  [q]uit")
    for t in stale:
        print(f"\n  {fmt_task(t)}")
        try:
            ans = input("  k/l/d/x/q > ").strip().lower()[:1]
        except EOFError:
            break
        if ans == "q":
            break
        if ans == "l":
            t["status"] = "later"
            touch(t)
            log_event("triage", id=t["id"], decision="later")
            print("  -> later")
        elif ans == "d":
            complete_task(t)
            log_event("triage", id=t["id"], decision="done")
            print("  -> done. Nice.")
        elif ans == "x":
            sure = input(f"  drop '{t['title']}' for good? [y/N] > ").strip().lower()
            if sure == "y":
                store["tasks"] = [x for x in store["tasks"] if x["id"] != t["id"]]
                log_event("triage", id=t["id"], decision="drop")
                print("  dropped — one less thing.")
            else:
                print("  kept")
        else:
            touch(t)   # keep: freshen so it stops nagging for another fortnight
            log_event("triage", id=t["id"], decision="keep")
            print("  kept (freshened)")
    save_store(store)

def cmd_timer(args):
    minutes = args.minutes
    store = load_store()
    now_tasks = [t for t in store["tasks"] if t["status"] == "now"]
    task_id = now_tasks[0]["id"] if now_tasks else None
    log_event("timer_start", minutes=minutes, task_id=task_id)
    if task_id:
        print(f"Timer: {minutes}m on #{task_id} {now_tasks[0]['title']}. Ctrl-C to stop.")
    else:
        print(f"Timer: {minutes}m. Ctrl-C to stop.")
    try:
        for left in range(minutes * 60, 0, -1):
            print(f"\r  {left // 60:02d}:{left % 60:02d}  ", end="", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        log_event("timer_cancel", minutes=minutes, task_id=task_id)
        print("\nTimer stopped. Stopping on purpose counts as a decision, not a failure.")
        return
    log_event("timer_done", minutes=minutes, task_id=task_id)
    print("\r  00:00  \nTime! Stand up, stretch, water. `focus next` when you're back.")
    _notify("focus", f"{minutes} minutes up — take a break.")

# ------------------------------------------------ memory

def _memory_scope(args):
    """Which memory the verb acts on: None for global, else a project key. `--here` is
    spelled as a flag rather than an optional-value `--project` because
    `memory add --project "a fact"` would silently eat the fact as the flag's value."""
    if getattr(args, "project", None):
        return project_key(args.project)
    return project_key() if getattr(args, "here", False) else None

def cmd_memory(args):
    key = _memory_scope(args)
    m = load_memory(key)
    if args.action == "show":
        stats = memory_stats_lines()
        facts = memory_facts(m)
        proj = load_memory(project_key()) if key is None else {}
        proj_facts = memory_facts(proj)
        if not stats and not facts and not proj_facts:
            print("Nothing remembered yet. `focus memory add <a fact about you>` —")
            print("estimate patterns appear on their own as you finish tasks.")
            return
        state = "OFF — not being injected" if m.get("disabled") else \
            "used by dump / break / draft"
        scope = f"project {key}" if key else "about you, everywhere"
        print(f"\nUSER MEMORY — {scope} ({state}):\n")
        if key is None:
            for line in stats:
                print(f"  - {line}   (derived from history)")
        for line in facts:
            print(f"  - {line}")
        if proj_facts:
            off = " — OFF" if proj.get("disabled") else ""
            print(f"\n  In this project ({project_key()}{off}):\n")
            for line in proj_facts:
                print(f"  - {line}")
        print("\n  add: focus memory add <fact> · this repo only: <fact> --here · "
              "edit: focus memory edit · off: focus memory off")
        return
    if args.action == "off":
        m["disabled"] = True
        save_memory(m, key)
        where = f"for {key} " if key else ""
        print(f"Memory injection {where}off (facts kept). `focus memory add` re-enables")
        print("it, or `--no-memory` skips it for a single command.")
        return
    if args.action == "edit":
        editor = os.environ.get("EDITOR", "nano")
        # Scoped filename so editing one scope can't clobber the other's buffer.
        path = _path(f"memory_facts_{key}.txt" if key else "memory_facts.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(memory_facts(m)))
        subprocess.run([editor, path])
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
        m["facts"] = [{"text": ln, "ts": now_iso()} for ln in lines]
        m["updated"] = now_iso()
        save_memory(m, key)
        print(f"Memory updated — {len(lines)} fact(s).")
        return
    # add
    text = " ".join(args.text).strip() or read_multiline("", "A fact worth remembering")
    if not text.strip():
        raise SystemExit("Nothing to remember.")
    m.setdefault("facts", []).append({"text": text.strip(), "ts": now_iso()})
    m.pop("disabled", None)
    m["updated"] = now_iso()
    save_memory(m, key)
    print(f"Remembered{' for ' + key if key else ''}: {text.strip()}")
    print("Every dump/break/draft now knows it. `focus memory show` to review.")

# ------------------------------------------------ pr review

# Project context, the ticket and the PR description are all charged against this, not
# added on top of it — see run_pr_review. An 8k model is already over budget at 60k chars
# of diff; nothing injected may make that worse.
MAX_DIFF_CHARS = 60_000
MAX_PR_BODY_CHARS = 1_500    # ceiling on the PR description put ahead of the diff

def git_working_diff(root=None):
    """(diff, source) for everything since the last commit, or (None, None). The UI calls
    this directly — get_diff's stdin tier would block a server thread forever — and
    passes the picked repo as `root`, so the diff comes from the repo on screen.

    `git diff HEAD` first, and it wins in every normal case: it is staged + unstaged, a
    strict superset of `git diff --staged`. Asking for the staged snapshot first — which
    this used to do — means that the moment you stage something and keep working, the
    review is of an older version of the file, and it comes back recommending fixes you
    already made. `--staged` survives only for a repo with no commits yet, where there is
    no HEAD to diff against."""
    for cmd in (["git", "diff", "HEAD"], ["git", "diff", "--staged"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=15,
                                 cwd=root)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout, " ".join(cmd)
        except (OSError, subprocess.TimeoutExpired):
            pass
    return None, None

def unpushed_commits(root=None):
    """How many commits the branch has that its remote doesn't, or 0 when there is no
    upstream / no git / anything odd. `gh pr diff` returns what is *pushed*, so with
    commits sitting locally the review is of an older version of the branch — the other
    way a review comes back recommending a fix you already made."""
    out = _git(["rev-list", "--count", "@{upstream}..HEAD"], cwd=root)
    try:
        return int(out.strip())
    except (ValueError, AttributeError):
        return 0

PR_FIELDS = "number,title,body,url,headRefName,baseRefName,isDraft"

# `focus pr fetch` is a deliberate act and can afford to wait; the fallback tier fires
# when the user typed nothing but `focus pr`, and must not sit there for a minute.
GH_TIMEOUT = 30
GH_FALLBACK_TIMEOUT = 10

def gh_pr_meta(root=None, number=None, timeout=GH_TIMEOUT):
    """(meta dict, reason) for a PR — the one named, or the current branch's."""
    argv = ["pr", "view"] + ([str(number)] if number else []) + ["--json", PR_FIELDS]
    out, reason = _gh(argv, cwd=root, timeout=timeout)
    if reason:
        return None, reason
    try:
        return json.loads(out), ""
    except json.JSONDecodeError:
        return None, "Couldn't read the PR details GitHub CLI returned."

def gh_pr_diff(root=None, number=None, timeout=GH_TIMEOUT):
    """(diff, meta, reason). Metadata is fetched first: it names the session and gives
    the review the description the author already wrote. `pr_source()` turns the meta
    into the session's source label, so that string lives in one place."""
    meta, reason = gh_pr_meta(root, number, timeout)
    if not meta:
        return None, None, reason
    diff, reason = _gh(["pr", "diff", str(meta["number"])], cwd=root, timeout=timeout)
    if not diff.strip():
        return None, None, reason or f"PR #{meta['number']} has an empty diff."
    return diff, meta, ""

def pr_source(meta):
    return f"PR #{meta['number']}"

def get_diff(args):
    """(diff, source, pr_meta). Tiers, first hit wins — the PR sits *below* the working
    tree on purpose: uncommitted changes are what you're in the middle of, and pulling
    the PR instead would silently review something else. A clean tree is the "I pushed,
    now review it" moment, and that's where the auto-fetch lands."""
    if args.file:
        with open(args.file, encoding="utf-8", errors="replace") as f:
            return f.read(), os.path.basename(args.file), None
    if not sys.stdin.isatty():
        return sys.stdin.read(), "stdin", None
    diff, source = git_working_diff()
    if diff:
        return diff, source, None
    diff, meta, reason = gh_pr_diff(project_root(), timeout=GH_FALLBACK_TIMEOUT)
    if diff:
        return diff, pr_source(meta), meta
    raise SystemExit(
        "No diff found — nothing staged or unstaged here, and no PR to fall back on\n"
        f"  ({reason})\n"
        "  Pipe one in (`git diff main | focus pr`), use -f file.diff, or name a PR\n"
        "  with `focus pr fetch 123`."
    )

def pr_session_path(name):
    return os.path.join(focus_home(), "pr", safe_name(name, "session") + ".json")

def _pr_action(args):
    """(action, item) from either the verb form or the older --resume/--check flags."""
    if args.action == "check":
        return "check", args.n
    if args.check is not None:
        return "check", args.check
    if args.action == "resume" or args.resume:
        return "resume", None
    if args.action == "fetch":
        return "fetch", args.n
    return "review", None

def _pr_preamble(meta):
    """The PR's own title and description, ahead of the diff. Cheap, high-signal context
    the author already wrote — and the thing the diff is supposed to deliver."""
    if not meta:
        return ""
    head = f"PULL REQUEST #{meta.get('number')}: {meta.get('title', '')}".strip()
    body = _clip(_strip_todos((meta.get("body") or "").strip()), MAX_PR_BODY_CHARS)
    branch, base = meta.get("headRefName", ""), meta.get("baseRefName", "")
    line = f"{branch} -> {base}" if branch and base else ""
    if body:
        body = ("AUTHOR'S DESCRIPTION (their words, context only — never review it, "
                "never copy from it):\n" + body)
    return "\n".join(x for x in (head, line, body) if x) + "\n\nDIFF:\n"

_TODO_RE = re.compile(r"^\s*[-*+]\s*\[( |x|X)\]\s*")

def _strip_todos(body):
    """Take the author's task list out of the PR description. An unticked box is work
    they haven't done yet — feed it to a model and it comes back as a review item, which
    is how a review ends up handing someone their own to-do list. Ticked boxes describe
    what the PR *did*, so those stay, minus the checkbox that makes them look tickable."""
    out = []
    for raw in body.splitlines():
        m = _TODO_RE.match(raw)
        if not m:
            out.append(raw)
        elif m.group(1) in ("x", "X"):
            out.append("- " + raw[m.end():])
    return "\n".join(out).strip()

MAX_DEEP_FILES = 12          # a per-file pass each; past this it stops being one sitting
MAX_DEEP_FILE_CHARS = 24_000  # one file's diff, before its share of context is charged

def split_diff_by_file(diff):
    """[(path, that file's diff)] in diff order. Splits on `diff --git`, which is the
    only boundary git guarantees — hunk headers appear inside a file's diff too."""
    starts = [m.start() for m in re.finditer(r"^diff --git ", diff, re.M)]
    if not starts:
        paths = diff_paths(diff)
        return [(paths[0] if paths else "?", diff)] if diff.strip() else []
    starts.append(len(diff))
    out = []
    for i in range(len(starts) - 1):
        chunk = diff[starts[i]:starts[i + 1]]
        paths = diff_paths(chunk)
        out.append((paths[0] if paths else "?", chunk))
    return out

def _dedupe_key(*parts):
    return tuple(" ".join(str(p).lower().split()) for p in parts)

def _dedupe_findings(findings):
    """Per-file passes never see each other, so a change threaded through five files
    comes back as the same finding five times. Same file and same sentence is the same
    finding; the first one wins, since findings arrive worst-first within a file."""
    seen, out = set(), []
    for f in findings:
        k = _dedupe_key(f["file"], f["what"])
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out

def _dedupe_checklist(items):
    seen, out = set(), []
    for c in items:
        k = _dedupe_key(c.get("file", ""), c.get("item", ""))
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out

def run_deep_review(diff, ctx, ticket, preamble, say):
    """One model pass per changed file, then one pass over the findings for the summary.

    This is what depth costs on a local model: a 60k-char diff in one request gets
    skimmed, the same diff in seven requests gets read. Returns the same dict shape the
    single-pass reply has, so run_pr_review stores it without caring which way it came."""
    files = split_diff_by_file(diff)
    skipped = files[MAX_DEEP_FILES:]
    files = files[:MAX_DEEP_FILES]
    sys_file = SYS_PR_FILE + project_block(ctx) + ticket_block(ticket)
    findings, checklist = [], []
    for i, (path, chunk) in enumerate(files, 1):
        say("file", f"{i}/{len(files)} {path}")
        say("model", "")
        part = ask_model(sys_file, preamble + _clip(chunk, MAX_DEEP_FILE_CHARS),
                         on_delta=lambda text: say("model", text))
        # The pass saw one file, so that file is where its findings are — a model naming
        # another one is guessing, and a wrong path is worse than no path.
        findings += [dict(f, file=path) for f in session_findings(part)]
        checklist += [dict(c, file=path) for c in part.get("checklist", [])
                      if isinstance(c, dict)]
    findings, checklist = _dedupe_findings(findings), _dedupe_checklist(checklist)
    # The per-file passes never saw each other, so the overview is its own pass. It reads
    # the findings, not the diff — that is the whole reason it fits in one request.
    say("summarising", f"{len(findings)} finding(s) across {len(files)} file(s)")
    say("model", "")
    digest = "\n".join(
        f"- {f['severity']} {f['file']}: {f['what']}" for f in findings) or "(none)"
    over = ask_model(SYS_PR_SUMMARY + project_block(ctx) + ticket_block(ticket),
                     f"{preamble}FILES: {', '.join(p for p, _ in files)}\n\nFINDINGS:\n"
                     f"{digest}", on_delta=lambda text: say("model", text))
    return {"summary": over.get("summary", ""),
            "suggestions": over.get("suggestions", []),
            "findings": findings, "checklist": checklist,
            "deep": {"files": [p for p, _ in files],
                     "skipped": [p for p, _ in skipped]}}

def run_pr_review(diff, source, name=None, project=None, no_project=False,
                  pr_meta=None, story=None, no_ticket=False, progress=None, deep=False):
    """Resolve context, charge it against the diff budget, ask the model, save the
    session. Shared by cmd_pr and the UI's /api/pr — never prints, never SystemExits.

    progress(stage, text) is called as the review happens: "context"/"ticket"/"diff"
    with a human-readable line, then "model" repeatedly with the raw reply so far (the
    caller renders what it likes from it), then "saved". A review is the one command
    that can sit silent for a minute — this is what says it is still going."""
    say = progress or (lambda stage, text="": None)
    # Resolve context first, then charge it to the diff's budget so the request never
    # grows. Keyed on the paths the diff touches, so only relevant sections survive.
    ctx, ctx_source = resolve_project_context(
        diff_paths(diff), budget=MAX_CONTEXT_CHARS, name=project, disabled=no_project)
    if ctx:
        say("context", f"{ctx_source} ({len(ctx)} chars)")
    # The ticket is the one thing a diff can't tell you: what it was *asked* to do. This
    # is the only injection site that refreshes over the network — a review is already a
    # deliberate, slow act, and a stale acceptance criterion is worse here than anywhere.
    ticket, ticket_info = resolve_ticket_context(
        budget=MAX_TICKET_CHARS, story=story, disabled=no_ticket, pr_meta=pr_meta,
        refresh=True)
    if ticket_info:
        say("ticket", f"sc-{ticket_info['id']} {ticket_info.get('name', '')}".strip())
    # Only on the PR path: this is a review of what GitHub has, and what GitHub has is
    # what was pushed. One `git` call, and only where two `gh` ones were already spent.
    ahead = unpushed_commits(active_root()) if pr_meta else 0
    if ahead:
        say("stale", f"{ahead} local commit(s) not pushed — reviewing what's on GitHub")
    preamble = _pr_preamble(pr_meta)
    truncated = False
    budget = MAX_DIFF_CHARS - len(ctx) - len(ticket) - len(preamble)
    if len(diff) > budget:
        diff = diff[:budget]
        truncated = True
    say("diff", f"{len(diff)} chars from {source}" + (" (truncated)" if truncated else ""))
    if deep:
        data = run_deep_review(diff, ctx, ticket, preamble, say)
    else:
        say("model", "")        # the wait starts here, before the first token lands
        data = ask_model(SYS_PR + project_block(ctx) + ticket_block(ticket),
                         preamble + diff, on_delta=lambda text: say("model", text))
    name = name or (f"pr-{pr_meta['number']}" if pr_meta
                    else datetime.now().strftime("pr-%Y%m%d-%H%M"))
    session = {
        "name": name,
        "source": source,
        "created": now_iso(),
        "summary": data.get("summary", ""),
        "findings": session_findings(data),
        "suggestions": [s.strip() for s in data.get("suggestions") or []
                        if isinstance(s, str) and s.strip()],
        "deep": data.get("deep"),
        "truncated": truncated,
        "unpushed": ahead,
        "project": {"source": ctx_source, "chars": len(ctx)} if ctx else None,
        "pr": {"number": pr_meta["number"], "title": pr_meta.get("title", ""),
               "url": pr_meta.get("url", ""), "branch": pr_meta.get("headRefName", ""),
               "draft": pr_meta.get("isDraft", False)} if pr_meta else None,
        "ticket": dict(ticket_info, chars=len(ticket)) if ticket_info else None,
        "checklist": [
            {"file": c.get("file", "?"), "item": c.get("item", ""), "done": False}
            for c in data.get("checklist", [])
        ],
    }
    save_json(pr_session_path(name), session)
    say("saved", name)
    return session

_SPIN = "|/-\\"
_STAGE_LABEL = {"context": "context", "ticket": "ticket", "diff": "diff",
                "file": "reviewing", "summarising": "summarising", "stale": "!!"}

def _term_width(stream):
    """Columns to draw the live line in. A pty with no window size set — CI, some ssh
    sessions — answers 0, which would clip every frame by a character and pad the clear
    with nothing, so anything implausible falls back to 80."""
    try:
        cols = os.get_terminal_size(stream.fileno()).columns
    except Exception:
        cols = 0
    return cols if cols >= 20 else 80

def pr_progress_printer(stream=None):
    """A run_pr_review progress callback that shows the review happening on stderr: one
    line per resolved input, then a live line the summary writes itself into as the model
    produces it. On a tty a ticker thread repaints that line, so the silence *before* the
    first token still moves; piped, each stage prints once and nothing rewrites."""
    stream = stream or sys.stderr
    try:
        tty = stream.isatty()
    except Exception:
        tty = False
    lock = threading.Lock()
    st = {"text": "", "start": time.time(), "stop": None, "thread": None, "said": False}

    def paint():
        secs = int(time.time() - st["start"])
        tail = _partial_summary(st["text"]) or (
            f"{len(st['text'])} chars in" if st["text"] else "thinking")
        frame = _SPIN[int(time.time() * 5) % len(_SPIN)]
        width = _term_width(stream) - 1
        with lock:
            if st["thread"] is None:
                return
            stream.write("\r" + f"  {frame} reading the diff… {secs}s · {tail}"[:width]
                         .ljust(width))
            stream.flush()

    def tick():
        while not st["stop"].wait(0.2):
            paint()

    def stop_ticker():
        if st["thread"] is None:
            return
        st["stop"].set()
        st["thread"].join(timeout=1)
        with lock:
            st["thread"] = None
            stream.write("\r" + " " * (_term_width(stream) - 1) + "\r")
            stream.flush()

    def progress(stage, text=""):
        if stage == "model":
            st["text"] = text
            if not tty:
                if not st["said"]:
                    st["said"] = True
                    stream.write("  · asking the model (this is the slow bit)…\n")
                    stream.flush()
                return
            if st["thread"] is None:
                st["start"], st["stop"] = time.time(), threading.Event()
                st["thread"] = threading.Thread(target=tick, daemon=True)
                st["thread"].start()
            paint()
            return
        stop_ticker()
        if stage in _STAGE_LABEL:
            stream.write(f"  · {_STAGE_LABEL[stage]}: {text}\n")
            stream.flush()

    return progress

def cmd_pr(args):
    action, item = _pr_action(args)
    if action in ("resume", "check"):
        if action == "check" and item is None:
            raise SystemExit("Which item? `focus pr check 3`")
        return cmd_pr_resume(args, item)
    if action == "fetch":
        # Asked for by name, so say why it failed rather than falling through quietly.
        diff, pr_meta, reason = gh_pr_diff(project_root(), item)
        if not diff:
            raise SystemExit(reason or "Couldn't fetch that pull request.")
        source = pr_source(pr_meta)
    else:
        diff, source, pr_meta = get_diff(args)
    session = run_pr_review(diff, source, name=args.name, project=args.project,
                            no_project=args.no_project, pr_meta=pr_meta,
                            story=args.story, no_ticket=args.no_ticket,
                            progress=pr_progress_printer(), deep=args.deep)
    print_pr(session)
    print(f"\nSaved as '{session['name']}'. After any interruption: focus pr resume")

def _latest_session_path():
    """Newest session file in pr/, or None (dir missing or empty)."""
    d = os.path.join(focus_home(), "pr")
    try:
        files = sorted(
            (os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json")),
            key=os.path.getmtime, reverse=True,
        )
    except OSError:
        return None
    return files[0] if files else None

def latest_session():
    path = _latest_session_path()
    if not path:
        raise SystemExit("No PR sessions yet. Run `focus pr` on a diff first.")
    return path

SEVERITIES = ("bug", "risk", "nit")
MAX_WHERE_CHARS = 90

def session_findings(d):
    """Findings out of a model reply or a saved session. Sessions written before
    findings existed carry `risks` — a list of bare strings — and there is no migration
    layer, so the reader converts rather than the file changing under someone."""
    out = []
    for f in d.get("findings") or []:
        if isinstance(f, str) and f.strip():
            out.append({"severity": "risk", "file": "", "where": "", "what": f.strip(),
                        "fix": ""})
        elif isinstance(f, dict) and (f.get("what") or f.get("where")):
            sev = str(f.get("severity") or "").strip().lower()
            # file/where are asked for bare and quoted by the renderer. Models quote them
            # anyway — telling one not to use markdown in two fields out of five never
            # sticks — so strip it here rather than printing ``sleep(1)``.
            out.append({"severity": sev if sev in SEVERITIES else "risk",
                        "file": str(f.get("file") or "").strip().strip("`"),
                        "where": _clip(str(f.get("where") or "").strip().strip("`"),
                                       MAX_WHERE_CHARS),
                        "what": str(f.get("what") or ""),
                        "fix": str(f.get("fix") or "").strip()})
    if out:
        return out
    return [{"severity": "risk", "file": "", "where": "", "what": r.strip(), "fix": ""}
            for r in d.get("risks") or [] if isinstance(r, str) and r.strip()]

def pr_markdown(s):
    """The review as markdown — the one renderer there is. `focus pr` prints it and the
    dashboard's copy button asks for it, so what you read in the terminal is exactly what
    lands in the PR comment. Terminal-first: no tables, no reference links, nothing that
    only makes sense once something else renders it."""
    out = [f"## PR review — {s['name']}"]
    meta = [f"from {s['source']}"]
    if s.get("project"):
        meta.append(f"project context: {s['project']['source']} "
                    f"({s['project']['chars']} chars)")
    if s.get("ticket"):
        t = s["ticket"]
        meta.append(f"ticket: sc-{t['id']} {t.get('name', '')} "
                    f"(via {t.get('how', '?')}, {t['chars']} chars)".replace("  ", " "))
    if s.get("deep"):
        n = len(s["deep"].get("files") or [])
        meta.append(f"deep review: {n} file{'s' if n != 1 else ''}, one pass each")
    out.append("*" + " · ".join(meta) + "*")
    if s.get("pr"):
        pr = s["pr"]
        title = f"#{pr['number']} {pr['title']}".strip()
        out.append(("**[draft]** " if pr.get("draft") else "")
                   + (f"[{title}]({pr['url']})" if pr.get("url") else title))
    out += ["", "### What it does", textwrap.fill(s.get("summary", ""), 78)]
    if s.get("truncated"):
        out += ["", "> **Diff was truncated** — large PR. Review the tail manually."]
    if s.get("unpushed"):
        out += ["", f"> **{s['unpushed']} commit(s) not pushed.** This reviews what is on "
                "GitHub, so anything you fixed locally since is not in it."]
    findings = session_findings(s)
    out += ["", "### Findings"]
    if not findings:
        out.append("Nothing flagged. Read it yourself anyway — a small local model is "
                   "not a reviewer, it is a first pass.")
    for f in findings:
        head = f"**{f['severity']}**"
        if f["file"]:
            head += f" `{f['file']}`"
        if f["where"]:
            head += f" — `{f['where']}`"
        out.append(f"- {head}")
        if f["what"]:
            out.append(textwrap.fill(f["what"], 78, initial_indent="  ",
                                     subsequent_indent="  "))
        if f.get("fix"):
            out.append(textwrap.fill("**Fix:** " + f["fix"], 78, initial_indent="  ",
                                     subsequent_indent="  "))
    if s.get("suggestions"):
        out += ["", "### Suggestions"]
        for sug in s["suggestions"]:
            out.append(textwrap.fill(sug, 78, initial_indent="- ",
                                     subsequent_indent="  "))
    if (s.get("deep") or {}).get("skipped"):
        out += ["", "> Not reviewed (past the "
                f"{MAX_DEEP_FILES}-file deep-review cap): "
                + ", ".join(f"`{p}`" for p in s["deep"]["skipped"])]
    done = sum(1 for c in s["checklist"] if c["done"])
    out += ["", f"### Checklist — {done}/{len(s['checklist'])} done",
            "Tick with `focus pr check N`.", ""]
    undone_seen = False
    for i, c in enumerate(s["checklist"], 1):
        marker = ""
        if not c["done"] and not undone_seen:
            marker = "   <- you are here"
            undone_seen = True
        out.append(f"- [{'x' if c['done'] else ' '}] {i}. `{c['file']}` — "
                   f"{c['item']}{marker}")
    return "\n".join(out)

def print_pr(s):
    print("\n" + pr_markdown(s))

def cmd_pr_resume(args, item=None):
    path = pr_session_path(args.name) if args.name else latest_session()
    if not os.path.exists(path):
        path = latest_session()
    s = load_json(path, None)
    if s is None:
        raise SystemExit("Session file unreadable.")
    if item is not None:
        idx = item - 1
        if not (0 <= idx < len(s["checklist"])):
            raise SystemExit(f"No checklist item {item}")
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
    system = ("\n".join([SYS_DRAFT, style, tone, to])
              + voice_system_suffix(args.no_voice)
              + project_system_suffix(text, BRIEF_CONTEXT_CHARS, disabled=args.no_project)
              + ticket_system_suffix(disabled=args.no_ticket)
              + memory_system_suffix(args.no_memory, args.no_project))
    if args.polish:
        user = "Polish this draft, keep my meaning and roughly my voice:\n\n" + text
    else:
        user = "Write the message from these notes:\n\n" + text
    msg = ask_model(system, user, temperature=0.5, raw=True).strip()
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
    data = ask_model(SYS_VOICE, "\n".join(parts))
    profile = data.get("profile", "").strip()
    if not profile:
        raise ModelReplyError("Model returned an empty profile — try again.")
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

# ------------------------------------------------ project

def _derive_project(root):
    """Repo -> markdown profile. Falls back to the raw brief with no model running."""
    brief = repo_brief(root)
    if not brief.strip():
        raise SystemExit(f"Nothing to harvest in {root} — no docs, manifests or files.")
    try:
        data = ask_model(SYS_PROJECT, brief)
    except NoModelError:
        return _clip(brief, MAX_CONTEXT_CHARS * 2), "brief"
    profile = data.get("profile", "").strip()
    if not profile:
        raise ModelReplyError("Model returned an empty profile — try again.")
    return profile, "harvest"

def cmd_project(args):
    key = project_key(args.project)
    p = load_project(key)
    if args.action == "show":
        if not p.get("profile"):
            ctx, source = resolve_project_context()
            if ctx:
                print(f"\nNo saved profile for '{key}'. Falling back to {source}:\n")
                print(textwrap.indent(ctx, "  "))
                print("\n  Distil it into a proper profile: focus project setup")
            else:
                print(f"No project context for '{key}'. Run `focus project setup`.")
            return
        print(f"\nPROJECT PROFILE: {key}  (used by every AI command)\n")
        print(textwrap.indent(p["profile"], "  "))
        print(f"\n  from {p.get('root', '?')} via {p.get('source', '?')}, "
              f"updated {p.get('updated', '?')}")
        print("  Tune it: focus project edit · rebuild: focus project setup · "
              "disable: focus project off")
        return
    if args.action == "ls":
        keys = list_projects()
        if not keys:
            print("No project profiles yet. Run `focus project setup` inside a repo.")
            return
        print("\nPROJECT PROFILES:")
        for k in keys:
            here = "  <- here" if k == key else ""
            print(f"  {k}  ({len(load_project(k).get('profile', ''))} chars){here}")
        return
    if args.action == "off":
        if p:
            os.remove(project_path(key))
            print(f"Cleared the profile for '{key}'.")
        else:
            print(f"No profile to clear for '{key}'.")
        print("Note: focus still reads this repo's own CLAUDE.md/README as a fallback.\n"
              "      Use --no-project on a command to suppress context entirely.")
        return
    if args.action == "edit":
        seed = p.get("profile")
        if not seed:
            seed, _ = resolve_project_context(budget=MAX_CONTEXT_CHARS)
            seed = seed or PROJECT_SKELETON
        editor = os.environ.get("EDITOR", "nano")
        path = _path("project_profile.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(seed)
        subprocess.run([editor, path])
        with open(path, encoding="utf-8") as f:
            profile = f.read().strip()
        if not profile:
            raise SystemExit("Empty profile — nothing saved.")
        save_project(key, {"profile": profile, "root": p.get("root", project_root()),
                           "source": "edit", "updated": now_iso()})
        print(f"Profile for '{key}' updated.")
        return
    # setup
    root = project_root()
    print(f"Reading {root} ...")
    profile, source = _derive_project(root)
    if source == "brief":
        print("(no local model — saving the raw repo brief instead of a distilled one)\n")
    save_project(key, {"profile": profile, "root": root,
                       "source": source, "updated": now_iso()})
    print(f"\nPROJECT PROFILE: {key}\n")
    print(textwrap.indent(profile, "  "))
    print(f"\nSaved. Every AI command now knows this repo — `focus project edit` to "
          f"correct it, `--no-project` to skip it.")

# ------------------------------------------------ shortcut

def _print_story(s, how=""):
    print(f"\nTICKET: sc-{s['id']} — {s.get('name', '')}")
    if how:
        print(f"  found via {how}")
    if s.get("url"):
        print("  " + s["url"])
    print(f"  fetched {s.get('fetched', '?')}\n")
    print(textwrap.indent(_clip(s.get("text", ""), MAX_TICKET_CHARS), "  "))

def cmd_shortcut(args):
    sid = args.n
    if args.action == "token":
        # Read from stdin, never argv — a token in a command line is a token in your
        # shell history and in `ps`.
        token = read_multiline("", "Paste your Shortcut API token").strip()
        if not token:
            raise SystemExit("No token given — nothing saved.")
        save_shortcut_token(token)
        print(f"Saved to {shortcut_config_path()} (mode 600).")
        print("Now: `focus shortcut use 12345`, or just name branches `sc-12345/...`.")
        return
    if args.action in ("on", "off"):
        on = args.action == "on"
        set_ticket_enabled(on, args.here)
        where = f"for '{project_key()}'" if args.here else "everywhere"
        print(f"Ticket context {'back on' if on else 'off'} {where}.")
        if not on:
            print("Your token and cached tickets are kept. Back on: `focus shortcut on`"
                  + (" --here" if args.here else ""))
        return
    if args.action == "ls":
        ids = cached_story_ids()
        if not ids:
            print("No tickets cached yet. `focus shortcut use 12345`.")
            return
        pins = load_ticket_repo().get("branches") or {}
        here, _ = detect_story_id()
        print("\nCACHED TICKETS:")
        for i in ids:
            s = load_story(i)
            mark = "  <- here" if here == i else ""
            print(f"  sc-{i}  {s.get('name', '?')[:60]}{mark}")
        if pins:
            print("\nPINNED IN THIS REPO:")
            for branch, pin in sorted(pins.items()):
                print(f"  {branch} -> sc-{pin}")
        return
    if args.action == "clear":
        ids = clear_story_cache()
        had_token = bool(load_shortcut_config().get("token"))
        save_shortcut_config({})
        print(f"Cleared {len(ids)} cached ticket(s)"
              + (" and the saved token." if had_token else "."))
        return
    if args.action == "use":
        if not sid:
            raise SystemExit("Which story? `focus shortcut use 12345`")
        story = fetch_story(str(sid))
        branch = pin_story(sid)
        _print_story(story, f"pinned to {branch}" if branch else "given")
        print("\nEvery dump/break/draft/pr in this repo now knows it.")
        return
    if args.action == "fetch":
        detected, how = detect_story_id(sid)
        if not detected:
            raise SystemExit(
                "No story to refresh. Name one (`focus shortcut fetch 12345`), pin one\n"
                "  (`focus shortcut use 12345`), or work on an `sc-12345` branch.")
        _print_story(fetch_story(detected), how)
        return
    # show
    detected, how = detect_story_id(sid)
    if not detected:
        print("No ticket for this branch.")
        print("  focus shortcut use 12345   pin one to this branch")
        print("  ...or name branches `sc-12345/thing`, and focus finds them itself.")
        return
    s = load_story(detected)
    if not s.get("text"):
        print(f"Found sc-{detected} (via {how}) but it isn't cached yet.")
        print(f"  focus shortcut fetch {detected}")
        return
    _print_story(s, how)
    print("\n  Refresh: focus shortcut fetch · silence it: focus shortcut off --here")

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

MAX_BROWSE_DIRS = 200

def repo_state(recent=False):
    """The active repo, and — only when the picker asks — where it has been. The recent
    list stays off the 5-second poll; entries that no longer exist are dropped from the
    view, not from the file, so an unmounted disk comes back."""
    root, key = _root_and_key()
    out = {"root": root, "key": key, "git": is_git_repo(root)}
    if recent:
        out["recent"] = [{"path": r, "name": os.path.basename(r.rstrip(os.sep)) or r,
                          "git": is_git_repo(r)}
                         for r in load_ui_prefs().get("recent", []) if os.path.isdir(r)]
    return out

def shortcut_state():
    """Everything the Shortcut panel renders. Off the 5-second poll on purpose: resolving
    a ticket needs the current branch, which changes without the repo root changing and so
    can't ride _ROOT_CACHE. The panel asks when it opens instead."""
    cfg = load_shortcut_config()
    repo = load_ticket_repo()
    root = project_root()
    branch = current_branch(root)          # resolved once, then handed to detection
    sid, how = detect_story_id(root=root, repo=repo)
    s = load_story(sid) if sid else {}
    return {
        "branch": branch,
        "has_token": bool(shortcut_token(cfg)),
        "env_token": bool(os.environ.get("SHORTCUT_API_TOKEN")),
        "disabled": bool(cfg.get("disabled")),
        "disabled_here": bool(repo.get("disabled")),
        "id": sid, "how": how, "url": s.get("url", ""),
        "text": _clip(s.get("text", ""), MAX_TICKET_CHARS),
        "fetched": s.get("fetched", ""),
        "cached": cached_story_ids(),
        "pins": repo.get("branches") or {},
    }

# One slot, not a registry: the dashboard is one person in one browser, and a second
# concurrent review would only be overwriting the line the first one is drawing. The
# review POST blocks for a minute, so progress rides a separate poll on another thread
# rather than the response body — no streaming socket to keep alive, nothing added to
# /api/state, and a reloaded page can still find the review it left running.
_PR_PROGRESS = {"active": False, "stage": "", "text": "", "chars": 0, "summary": "",
                "file": "", "started": 0.0}
_PR_PROGRESS_LOCK = threading.Lock()

def ui_pr_progress(stage, text=""):
    """run_pr_review's progress callback for the dashboard: keep the latest state, let
    the page ask for it. The model stage carries the raw half-written reply, so the
    summary is pulled out here and the page renders words rather than JSON."""
    with _PR_PROGRESS_LOCK:
        _PR_PROGRESS.update(active=True, stage=stage)
        if stage == "model":
            # `file` survives a model stage on purpose: in a deep review it is which of
            # the seven files this minute is being spent on, which is the whole question.
            _PR_PROGRESS.update(chars=len(text), summary=_partial_summary(text), text="")
        else:
            _PR_PROGRESS["text"] = text
            if stage == "file":
                _PR_PROGRESS["file"] = text
            elif stage in ("diff", "fetch"):
                _PR_PROGRESS["file"] = ""

def pr_progress_begin(stage, text=""):
    with _PR_PROGRESS_LOCK:
        _PR_PROGRESS.update(active=True, stage=stage, text=text, chars=0, summary="",
                            file="", started=time.time())

def pr_progress_end():
    with _PR_PROGRESS_LOCK:
        _PR_PROGRESS["active"] = False

def pr_progress_state():
    with _PR_PROGRESS_LOCK:
        s = dict(_PR_PROGRESS)
    s["elapsed"] = int(time.time() - s["started"]) if s["started"] else 0
    return s

def browse_dirs(path):
    """Subdirectories of `path`, for the repo picker. Raises OSError if unreadable.
    `git` is a stat rather than a shell-out — this runs once per listed folder."""
    path = os.path.abspath(os.path.expanduser(path))
    dirs = []
    for name in sorted((n for n in os.listdir(path) if not n.startswith(".")),
                       key=str.lower):
        p = os.path.join(path, name)
        if os.path.isdir(p):
            dirs.append({"name": name, "path": p, "git": is_git_repo(p)})
        if len(dirs) >= MAX_BROWSE_DIRS:
            break
    parent = os.path.dirname(path.rstrip(os.sep))
    return {"path": path, "git": is_git_repo(path), "dirs": dirs,
            "parent": parent if parent != path else ""}

def api_state():
    store = load_store()
    nxt = pick_next(store)
    nxt_low = pick_next(store, energy="low")
    llm_ok = True
    try:
        name, base, model_id = detect_runtime()
        model = {"runtime": name, "endpoint": base, "model": model_id}
    except NoModelError as e:
        llm_ok = False
        model = {"error": str(e)}
    ctx, ctx_source = resolve_project_context(budget=BRIEF_CONTEXT_CHARS)
    today = datetime.now(timezone.utc).date()
    week_ago = (today - timedelta(days=7)).isoformat()

    def _done_entry(t):
        return {"id": t["id"], "title": t["title"], "completed": t.get("completed"),
                "est": total_estimate(t), "actual": actual_minutes(t)}

    done_week = sorted(
        (_done_entry(t) for t in store["tasks"] if t["status"] == "done"
         and (t.get("completed") or "")[:10] >= week_ago),
        key=lambda d: d["completed"] or "")
    done_today = [d for d in done_week
                  if (d["completed"] or "")[:10] == today.isoformat()]
    hist = load_history()
    return {"tasks": store["tasks"], "next_id": nxt["id"] if nxt else None,
            "next_low_id": nxt_low["id"] if nxt_low else None,
            "llm": llm_ok, "model": model,
            "voice": bool(load_voice().get("profile")),
            "project": ctx_source if ctx else "",
            "memory": bool(memory_system_suffix()),
            "done_today": done_today, "done_week": done_week,
            "streak": _streak(_done_dates(hist)), "ratio": _estimate_ratio(hist),
            "stale_days": STALE_DAYS, "home": focus_home(), "root": project_root(),
            "repo": repo_state()}

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

    _LOCAL_HOSTS = ("127.0.0.1", "localhost")

    def _origin_ok(self):
        """Only the local dashboard may POST. Without this, any webpage could CSRF the
        API with a text/plain form post — mutating tasks and burning model calls."""
        host = (self.headers.get("Host") or "").split(":")[0].lower()
        if host not in self._LOCAL_HOSTS:
            return False
        origin = self.headers.get("Origin")
        if origin:
            m = re.match(r"https?://([^/:]+)", origin)
            if not m or m.group(1).lower() not in self._LOCAL_HOSTS:
                return False
        return True

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, UI_HTML.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._send(200, api_state())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._origin_ok():
            return self._send(403, {"error": "forbidden origin"})
        try:
            body = self._body()
            if self.path == "/api/add":
                store = load_store()
                est = body.get("estimate_min")
                t = new_task(store, body["title"],
                             status=body.get("status", "inbox"),
                             priority=int(body.get("priority", 2)),
                             estimate_min=int(est) if est else None,
                             notes=body.get("notes", ""))
                save_store(store)
                self._send(200, t)
            elif self.path == "/api/update":
                store = load_store()
                t = get_task(store, int(body["id"]))
                if not t:
                    return self._send(404, {"error": "no such task"})
                if body.get("delete"):
                    store["tasks"] = [x for x in store["tasks"] if x["id"] != t["id"]]
                    save_store(store)
                    log_event("triage", id=t["id"], decision="drop")
                    return self._send(200, {"deleted": t["id"]})
                if "status" in body and body["status"] in STATUSES:
                    if body["status"] == "done":
                        complete_task(t)
                    else:
                        t["status"] = body["status"]
                        if body["status"] == "now" and not t.get("started"):
                            t["started"] = now_iso()
                        log_event("moved" if body["status"] != "now" else "started",
                                  id=t["id"], to=body["status"])
                if "toggle_subtask" in body:
                    i = int(body["toggle_subtask"])
                    if 0 <= i < len(t["subtasks"]):
                        t["subtasks"][i]["done"] = not t["subtasks"][i]["done"]
                        if t["subtasks"][i]["done"]:
                            log_event("subtask_done", id=t["id"], subtask=i + 1)
                        if all(s["done"] for s in t["subtasks"]) and t["subtasks"]:
                            complete_task(t)
                touch(t)
                save_store(store)
                self._send(200, t)
            elif self.path == "/api/break":
                store = load_store()
                t = get_task(store, int(body["id"]))
                if not t:
                    return self._send(404, {"error": "no such task"})
                context = t["title"] + ("\nNotes: " + t["notes"] if t["notes"] else "")
                if body.get("hint"):
                    context += "\nExtra context: " + str(body["hint"])
                system = SYS_BREAK + project_system_suffix(
                    context, BRIEF_CONTEXT_CHARS,
                    disabled=body.get("no_project", False)) \
                    + ticket_system_suffix(story=body.get("story"),
                                           disabled=body.get("no_ticket", False)) \
                    + memory_system_suffix(body.get("no_memory", False),
                                           body.get("no_project", False))
                data = ask_model(system, context)
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
                    system = SYS_DUMP + project_system_suffix(
                        body["text"], BRIEF_CONTEXT_CHARS,
                        disabled=body.get("no_project", False)) \
                        + ticket_system_suffix(
                            disabled=body.get("no_ticket", False)) \
                        + memory_system_suffix(body.get("no_memory", False),
                                               body.get("no_project", False))
                    data = ask_model(system, body["text"])
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
                             "firm": "Tone: polite but firm; do not soften the ask "
                                     "away."}.get(tone, "")
                to = f"Recipient: {body['to']}." if body.get("to") else ""
                system = "\n".join([SYS_DRAFT, style, tone_line, to]) + \
                    voice_system_suffix(body.get("no_voice", False)) + \
                    project_system_suffix(body["text"], BRIEF_CONTEXT_CHARS,
                                          disabled=body.get("no_project", False)) + \
                    ticket_system_suffix(disabled=body.get("no_ticket", False)) + \
                    memory_system_suffix(body.get("no_memory", False),
                                         body.get("no_project", False))
                prefix = ("Polish this draft, keep my meaning and roughly my voice:\n\n"
                          if body.get("polish")
                          else "Write the message from these notes:\n\n")
                msg = ask_model(system, prefix + body["text"],
                                temperature=0.5, raw=True)
                self._send(200, {"message": msg.strip()})
            elif self.path == "/api/note":
                store = load_store()
                t = get_task(store, int(body["id"]))
                if not t:
                    return self._send(404, {"error": "no such task"})
                text = (body.get("text") or "").strip()
                if text:
                    add_note(t, text)
                    save_store(store)
                self._send(200, t)
            elif self.path == "/api/timer":
                ev = body.get("event")
                task_id = None
                if ev in ("timer_start", "timer_done", "timer_cancel"):
                    now_tasks = [t for t in load_store()["tasks"]
                                 if t["status"] == "now"]
                    task_id = now_tasks[0]["id"] if now_tasks else None
                    log_event(ev, minutes=int(body.get("minutes", 25)),
                              task_id=task_id, source="ui")
                self._send(200, {"ok": True, "task_id": task_id})
            elif self.path == "/api/triage":
                store = load_store()
                t = get_task(store, int(body["id"]))
                if not t:
                    return self._send(404, {"error": "no such task"})
                decision = body.get("decision")
                if decision not in ("keep", "later", "done", "drop"):
                    return self._send(400, {"error": "bad decision"})
                if decision == "drop":
                    store["tasks"] = [x for x in store["tasks"] if x["id"] != t["id"]]
                elif decision == "done":
                    complete_task(t)
                elif decision == "later":
                    t["status"] = "later"
                    touch(t)
                else:  # keep: freshen so it stops nagging for another fortnight
                    touch(t)
                save_store(store)
                log_event("triage", id=t["id"], decision=decision)
                self._send(200, {"deleted": t["id"]} if decision == "drop" else t)
            elif self.path == "/api/memory":
                # `scope` picks which file a verb writes; the reply always carries both,
                # so one round-trip renders the whole panel.
                pkey = project_key()
                key = pkey if body.get("scope") == "project" else None
                m = load_memory(key)
                action = body.get("action", "show")
                if action == "add":
                    text = (body.get("text") or "").strip()
                    if not text:
                        return self._send(400, {"error": "Nothing to remember."})
                    m.setdefault("facts", []).append({"text": text, "ts": now_iso()})
                    m.pop("disabled", None)
                    m["updated"] = now_iso()
                    save_memory(m, key)
                elif action == "edit":
                    lines = [ln.strip() for ln in (body.get("text") or "").splitlines()
                             if ln.strip()]
                    m["facts"] = [{"text": ln, "ts": now_iso()} for ln in lines]
                    m["updated"] = now_iso()
                    save_memory(m, key)
                elif action == "off":
                    m["disabled"] = True
                    save_memory(m, key)
                elif action != "show":
                    return self._send(400, {"error": "bad action"})
                glob = load_memory() if key else m
                proj = m if key else load_memory(pkey)
                self._send(200, {
                    "stats": memory_stats_lines(),
                    "facts": memory_facts(glob),
                    "disabled": bool(glob.get("disabled")),
                    "project": {"key": pkey, "facts": memory_facts(proj),
                                "disabled": bool(proj.get("disabled"))}})
            elif self.path == "/api/voice":
                v = load_voice()
                action = body.get("action", "show")
                if action == "setup":
                    answers = {k: str(a).strip() for k, a in
                               (body.get("answers") or {}).items() if str(a).strip()}
                    samples = [s for s in (body.get("samples") or []) if s.strip()]
                    if not answers and not samples:
                        return self._send(
                            400, {"error": "Nothing to build a profile from."})
                    profile = _derive_profile(answers, samples)
                    v = {"profile": profile, "answers": answers, "samples": samples,
                         "updated": now_iso()}
                    save_voice(v)
                elif action == "learn":
                    sample = (body.get("sample") or "").strip()
                    if not sample:
                        return self._send(400, {"error": "Nothing pasted."})
                    samples = v.get("samples", [])
                    samples.append(sample)
                    v["samples"] = samples[-10:]  # keep the freshest 10
                    v["profile"] = _derive_profile(v.get("answers", {}), v["samples"])
                    v["updated"] = now_iso()
                    save_voice(v)
                elif action == "edit":
                    profile = (body.get("profile") or "").strip()
                    if not profile:
                        return self._send(400, {"error": "Empty profile — not saved."})
                    v["profile"] = profile
                    v["updated"] = now_iso()
                    save_voice(v)
                elif action == "off":
                    v = {}
                    save_voice(v)
                elif action != "show":
                    return self._send(400, {"error": "bad action"})
                self._send(200, {
                    "profile": v.get("profile", ""),
                    "samples_count": len(v.get("samples", [])),
                    "updated": v.get("updated", ""),
                    "questions": VOICE_QUESTIONS})
            elif self.path == "/api/project":
                key = project_key(body.get("name"))
                action = body.get("action", "show")
                p = load_project(key)
                if action == "setup":
                    root = project_root()
                    profile, source = _derive_project(root)
                    p = {"profile": profile, "root": root, "source": source,
                         "updated": now_iso()}
                    save_project(key, p)
                elif action == "edit":
                    profile = (body.get("profile") or "").strip()
                    if not profile:
                        return self._send(400, {"error": "Empty profile — not saved."})
                    p = {"profile": profile, "root": p.get("root", project_root()),
                         "source": "edit", "updated": now_iso()}
                    save_project(key, p)
                elif action == "ls":
                    return self._send(200, {"projects": [
                        {"key": k, "chars": len(load_project(k).get("profile", "")),
                         "current": k == key}
                        for k in list_projects()]})
                elif action == "off":
                    if p:
                        os.remove(project_path(key))
                    p = {}
                elif action != "show":
                    return self._send(400, {"error": "bad action"})
                out = {"key": key, "profile": p.get("profile", ""),
                       "source": p.get("source", ""), "root": p.get("root", ""),
                       "updated": p.get("updated", "")}
                if not p.get("profile"):
                    ctx, ctx_source = resolve_project_context()
                    if ctx:
                        out.update(profile=ctx, source=ctx_source, fallback=True)
                seed = p.get("profile")
                if not seed:
                    seed, _ = resolve_project_context(budget=MAX_CONTEXT_CHARS)
                out["edit_seed"] = seed or PROJECT_SKELETON
                self._send(200, out)
            elif self.path == "/api/repo":
                action = body.get("action", "show")
                if action == "browse":
                    start = (body.get("path")
                             or os.path.dirname(project_root().rstrip(os.sep))
                             or project_root())
                    try:
                        return self._send(200, browse_dirs(start))
                    except OSError as e:
                        return self._send(400, {"error":
                            f"Can't read {start}: {e.strerror or e}"})
                if action == "use":
                    try:
                        remember_repo(set_active_root(body.get("path")))
                    except ValueError as e:
                        return self._send(400, {"error": str(e)})
                elif action != "show":
                    return self._send(400, {"error": "bad action"})
                self._send(200, repo_state(recent=True))
            elif self.path == "/api/shortcut":
                # POST, like every other route here, so _origin_ok covers it — this one
                # accepts an API token, and must never be reachable cross-origin.
                action = body.get("action", "show")
                if action == "token":
                    token = (body.get("token") or "").strip()
                    if not token:
                        return self._send(400, {"error": "No token given."})
                    save_shortcut_token(token)
                elif action == "use":
                    sid = str(body.get("n") or "").strip()
                    if not sid:
                        return self._send(400, {"error": "Which story?"})
                    fetch_story(sid)
                    pin_story(sid)
                elif action == "fetch":
                    sid, _ = detect_story_id(body.get("n") or None)
                    if not sid:
                        return self._send(400, {"error": "No ticket to refresh."})
                    fetch_story(sid)
                elif action in ("on", "off"):
                    set_ticket_enabled(action == "on", body.get("here", False))
                elif action == "clear":
                    clear_story_cache()
                    save_shortcut_config({})
                elif action != "show":
                    return self._send(400, {"error": "bad action"})
                self._send(200, shortcut_state())
            elif self.path == "/api/pr":
                action = body.get("action", "review")
                if action == "progress":
                    # Polled from the page *while* a review POST is still blocking on
                    # another thread. Cheap by construction: one dict copy, no I/O.
                    return self._send(200, pr_progress_state())
                if action in ("review", "fetch"):
                    pr_meta = None
                    pr_progress_begin("fetch" if action == "fetch" else "diff")
                    if action == "fetch":
                        diff, pr_meta, reason = gh_pr_diff(project_root(),
                                                           body.get("n") or None)
                        if not diff:
                            pr_progress_end()
                            return self._send(400, {"error": reason})
                        source = pr_source(pr_meta)
                    else:
                        # No PR tier here, unlike the CLI: `gh` is two subprocesses on a
                        # server thread, and the panel has an explicit button for it.
                        diff = body.get("diff") or ""
                        source = "ui paste"
                        if not diff.strip():
                            diff, source = git_working_diff(project_root())
                            if not diff:
                                pr_progress_end()
                                return self._send(400, {"error":
                                    "No diff. Paste one, stage changes in "
                                    + project_root()
                                    + ", or use Review this branch's PR."})
                    try:
                        session = run_pr_review(
                            diff, source, name=body.get("name") or None,
                            project=body.get("project") or None,
                            no_project=body.get("no_project", False), pr_meta=pr_meta,
                            story=body.get("story") or None,
                            no_ticket=body.get("no_ticket", False),
                            progress=ui_pr_progress, deep=body.get("deep", False))
                    finally:
                        # However this ends — reply, dead model, raised error — the poll
                        # has to stop saying a review is running.
                        pr_progress_end()
                    return self._send(200, session)
                if action not in ("resume", "check", "markdown"):
                    return self._send(400, {"error": "bad action"})
                path = (pr_session_path(body["name"]) if body.get("name")
                        else _latest_session_path())
                if path and not os.path.exists(path):
                    path = _latest_session_path()
                s = load_json(path, None) if path else None
                if s is None:
                    return self._send(404, {"error":
                        "No PR sessions yet. Run a review first."})
                if action == "markdown":
                    # The same text `focus pr` prints, so what the copy button puts on
                    # the clipboard is the review, not a second rendering of it.
                    return self._send(200, {"name": s["name"],
                                            "markdown": pr_markdown(s)})
                if action == "check":
                    idx = int(body.get("n", 0)) - 1
                    if not (0 <= idx < len(s["checklist"])):
                        return self._send(400, {"error": f"No checklist item {idx + 1}"})
                    s["checklist"][idx]["done"] = True
                    save_json(path, s)
                self._send(200, s)
            else:
                self._send(404, {"error": "not found"})
        except NoModelError as e:
            self._send(503, {"error": str(e)})
        except (ModelReplyError, TicketError) as e:
            self._send(502, {"error": str(e)})
        except SystemExit as e:  # a CLI-shaped helper leaked; don't kill the thread
            self._send(500, {"error": f"unexpected exit: {e}"})
        except Exception as e:  # keep the local server alive
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

def cmd_ui(args):
    # Reopen on the repo the picker was last pointed at. A deleted or unmounted one
    # must not stop the dashboard from starting — fall back to the cwd, silently.
    saved = load_ui_prefs().get("repo")
    if saved:
        try:
            set_active_root(saved)
        except ValueError:
            pass
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), UIHandler)
    except OSError as e:
        raise SystemExit(
            f"Can't open 127.0.0.1:{args.port} ({e.strerror or e}).\n"
            f"  Another `focus ui` already running? Try --port {args.port + 1}.")
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
.task .stale{color:var(--amber);font-weight:700;}
.task .lastnote{font-size:.78rem;color:var(--soft);font-style:italic;margin-top:2px;}
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
input[type=text],input[type=number]{font:inherit;font-size:.88rem;border:1px solid var(--border);
border-radius:8px;padding:5px 8px;background:#fff;color:var(--ink);}
details.panel summary{cursor:pointer;font-size:1rem;font-weight:700;}
details.panel summary .muted{font-weight:400;}
details.sub{margin-top:10px;}
details.sub summary{cursor:pointer;font-size:.88rem;font-weight:600;color:var(--accent);}
details.sub>*{margin-top:8px;}
.facts li,.risks li,.findings li{margin:2px 0 2px 18px;font-size:.9rem;}
.findings li{margin-bottom:8px;}
.findings code,.check code{font-size:.82rem;opacity:.85;}
.fix{margin-top:2px;font-size:.88rem;color:var(--soft);}
.prose{white-space:pre-wrap;background:var(--accent-soft);border-radius:8px;
  padding:8px 10px;margin-top:8px;font-size:.92rem;}
.prose code,.fix code{background:rgba(0,0,0,.06);border-radius:4px;padding:0 3px;}
.sev{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em;
  border-radius:4px;padding:1px 5px;margin-right:6px;}
.sev-bug{background:var(--red);color:#fff;}
.sev-risk{background:var(--amber);color:#fff;}
.sev-nit{background:#6f6f6f;color:#fff;}
.check label{display:flex;gap:7px;font-size:.9rem;padding:2px 0;align-items:flex-start;cursor:pointer;}
.check .done{opacity:.55;text-decoration:line-through;}
.here{color:var(--green);font-weight:700;}
.warn{color:var(--amber);font-weight:600;font-size:.9rem;}
.repobar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:14px;
font-size:.9rem;color:var(--soft);}
.repobar .path{font-weight:700;color:var(--ink);}
.repobar button{font-size:.78rem;padding:3px 9px;}
.picker{background:var(--card);border:1px solid var(--border);border-radius:12px;
padding:12px;margin-bottom:14px;}
.picker .at{font-weight:700;font-size:.9rem;word-break:break-all;}
.picker ul{list-style:none;max-height:230px;overflow-y:auto;margin:8px 0;
border:1px solid var(--border);border-radius:8px;}
.picker li{display:flex;gap:8px;align-items:center;padding:4px 8px;font-size:.9rem;
border-bottom:1px solid var(--border);}
.picker li:last-child{border-bottom:none;}
.picker li:hover{background:var(--accent-soft);}
.picker li .name{flex:1;cursor:pointer;text-align:left;font:inherit;border:none;
background:none;padding:0;color:var(--accent);font-weight:600;}
.picker li .tag{font-size:.72rem;font-weight:700;color:var(--green);
background:var(--green-soft);border-radius:999px;padding:1px 8px;}
.memscope{font-weight:700;font-size:.92rem;margin-top:14px;}
.memscope:first-child{margin-top:0;}
</style>
</head>
<body>
<div class="wrap">
  <h1>focus <span id="llmBadge" class="llm off">checking model…</span>
    <label class="muted" id="projectWrap" style="display:none;cursor:pointer">
      <input type="checkbox" id="projectUse" checked> using <span id="projectSrc"></span></label>
    <label class="muted" id="memoryWrap" style="display:none;cursor:pointer">
      <input type="checkbox" id="memoryUse" checked> memory</label></h1>

  <div class="repobar">
    <span>repo</span><span class="path" id="repoPath">…</span>
    <span class="warn" id="repoWarn"></span>
    <button id="repoBtn" onclick="togglePicker()">change</button>
  </div>
  <div class="picker" id="repoPicker" style="display:none"></div>

  <div class="panel" id="donePanel" style="display:none;margin:0 0 18px;">
    <h2>🎉 Done today (<span id="doneCount"></span>)</h2>
    <div class="muted" id="doneList"></div>
  </div>

  <div class="one">
    <div class="label">Do this one thing
      <label class="muted" style="cursor:pointer;float:right;text-transform:none;letter-spacing:0">
        <input type="checkbox" id="lowEnergy" onchange="renderOne()"> low energy — smallest first</label></div>
    <div class="thing" id="oneThing">Loading…</div>
    <div class="from" id="oneFrom"></div>
    <div class="muted" id="oneWhy"></div>
    <button class="primary" id="doneBtn" style="display:none">Done ✓</button>
    <button id="timerBtn">Start timer</button>
    <input type="number" id="timerMin" value="25" min="1" max="180" style="width:64px" title="minutes"><span class="muted"> min</span>
    <span class="timer" id="timerTxt"></span>
  </div>

  <div class="row" style="margin:0 0 12px">
    <input type="text" id="addTitle" placeholder="add a task…" style="flex:1;min-width:180px">
    <select id="addPri"><option value="1">high</option><option value="2" selected>med</option>
      <option value="3">low</option></select>
    <input type="number" id="addEst" placeholder="est. min" style="width:88px" min="1">
    <select id="addStatus"><option>inbox</option><option>now</option><option>next</option>
      <option>later</option></select>
    <input type="text" id="addNotes" placeholder="notes (optional)" style="width:170px">
    <button class="primary" onclick="doAdd()">Add</button>
    <label class="muted" style="cursor:pointer;margin-left:auto">
      <input type="checkbox" id="showDone" onchange="renderBoard()"> show done</label>
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
        <input type="text" id="draftTo" placeholder="to (optional)" style="width:110px">
        <label class="muted" style="cursor:pointer" title="treat the input as a draft to clean up, not bullets">
          <input type="checkbox" id="draftPolish"> polish</label>
        <label class="muted" id="voiceWrap" style="display:none;cursor:pointer">
          <input type="checkbox" id="voiceUse" checked> in my voice</label>
        <button class="primary" onclick="doDraft()">Draft it</button>
        <span class="muted" id="draftStatus"></span>
      </div>
      <pre class="out" id="draftOut" style="display:none"></pre>
    </div>
  </div>

  <details class="panel" id="triagePanel">
    <summary>Triage <span class="muted" id="triageCount">no stale tasks</span></summary>
    <p class="muted" style="margin-top:8px">Inbox/later tasks untouched for a fortnight.
      Keep freshens them, drop lets them go — deciding either way counts.</p>
    <div id="triageList"></div>
  </details>

  <details class="panel" id="prPanel">
    <summary>PR review <span class="muted">what's wrong with this diff, and what to do about it</span></summary>
    <textarea id="prDiff" placeholder="paste a unified diff… (or use the repo button below)" style="margin-top:10px"></textarea>
    <div class="row">
      <input type="text" id="prName" placeholder="session name (optional)" style="width:180px">
      <button class="primary" id="prPasteBtn" onclick="doPr('paste')">Review pasted diff</button>
      <button id="prGitBtn" onclick="doPr('git')">Review changes in repo</button>
      <button id="prFetchBtn" onclick="doPr('fetch')">Review this branch's PR</button>
      <button onclick="doPr('resume')">Resume last session</button>
      <label class="muted" style="cursor:pointer" title="One model pass per changed file, then a summary pass. Much deeper, and minutes rather than seconds."><input type="checkbox" id="prDeep"> deep (a pass per file)</label>
      <label class="muted" style="cursor:pointer"><input type="checkbox" id="prNoProject"> skip project context</label>
      <label class="muted" style="cursor:pointer"><input type="checkbox" id="prNoTicket"> skip ticket</label>
      <span class="muted" id="prStatus"></span>
    </div>
    <div id="prOut"></div>
  </details>

  <details class="panel" id="shortcutPanel">
    <summary>Ticket <span class="muted" id="shortcutSummary">the Shortcut story behind this work</span></summary>
    <div id="shortcutBody" class="muted">open to load…</div>
  </details>

  <details class="panel" id="voicePanel">
    <summary>Voice <span class="muted" id="voiceSummary">how drafts sound like you</span></summary>
    <div id="voiceBody" class="muted">open to load…</div>
  </details>

  <details class="panel" id="memoryPanel">
    <summary>Memory <span class="muted" id="memorySummary">facts the AI knows about you</span></summary>
    <div id="memoryBody" class="muted">open to load…</div>
  </details>

  <details class="panel" id="projectPanel">
    <summary>Project <span class="muted" id="projectSummary">what the AI knows about this repo</span></summary>
    <div id="projectBody" class="muted">open to load…</div>
  </details>

  <details class="panel" id="settingsPanel">
    <summary>Doctor <span class="muted">data &amp; model</span></summary>
    <div id="settingsBody" class="muted" style="margin-top:8px"></div>
  </details>
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
  const badge=document.getElementById("llmBadge");
  badge.className="llm "+(state.llm?"ok":"off");
  badge.textContent=state.llm?"local model — "+((state.model||{}).model||"connected")
    :"no model — AI features off";
  badge.title=state.llm?((state.model||{}).runtime+" at "+(state.model||{}).endpoint)
    :((state.model||{}).error||"");
  document.getElementById("voiceWrap").style.display=state.voice?"inline":"none";
  document.getElementById("projectWrap").style.display=state.project?"inline":"none";
  document.getElementById("projectSrc").textContent=state.project;
  document.getElementById("memoryWrap").style.display=state.memory?"inline":"none";
  const repo=state.repo||{};
  document.getElementById("repoPath").textContent=repo.root||state.root||"?";
  document.getElementById("repoWarn").textContent=
    repo.root&&!repo.git?"not a git repo — paste diffs instead":"";
  document.getElementById("prPasteBtn").disabled=!state.llm;
  document.getElementById("prGitBtn").disabled=!state.llm||!repo.git;
  document.getElementById("prGitBtn").textContent="Review changes in "+(state.root||"repo");
  document.getElementById("prFetchBtn").disabled=!state.llm||!repo.git;
  renderOne();renderBoard();renderDone();renderTriage();renderSettings();
}
// one toggle for every AI panel — project context is a property of the repo, not the panel
function noProject(){const b=document.getElementById("projectUse");return !!b&&!b.checked;}
function noMemory(){const b=document.getElementById("memoryUse");return !!b&&!b.checked;}
function renderDone(){
  const p=document.getElementById("donePanel"),d=state.done_today||[];
  if(!d.length){p.style.display="none";return;}
  p.style.display="block";
  document.getElementById("doneCount").textContent=d.length;
  document.getElementById("doneList").textContent=
    d.map(x=>{
      const c=[x.est?"est ~"+x.est+"m":"",x.actual?"took ~"+x.actual+"m":""]
        .filter(Boolean).join(" · ");
      return "#"+x.id+" "+x.title+(c?" ("+c+")":"");
    }).join("  ·  ")+
    (state.streak>1?"   —  "+state.streak+"-day streak":"")+
    (state.ratio?"   —  tasks take ~"+state.ratio.toFixed(1)+"x your estimates":"");
}
function nextAction(t){
  if(!t)return null;
  for(let i=0;i<t.subtasks.length;i++)if(!t.subtasks[i].done)
    return{text:t.subtasks[i].text,sub:i,est:t.subtasks[i].estimate_min};
  return{text:t.title,sub:null,est:t.estimate_min};
}
function remainingEst(t){
  const subs=t.subtasks.filter(s=>!s.done&&s.estimate_min);
  if(subs.length)return subs.reduce((a,s)=>a+s.estimate_min,0);
  return t.estimate_min;
}
function renderOne(){
  const low=document.getElementById("lowEnergy").checked;
  const t=state.tasks.find(x=>x.id===(low?state.next_low_id:state.next_id));
  const el=document.getElementById("oneThing"),from=document.getElementById("oneFrom"),
    btn=document.getElementById("doneBtn"),why=document.getElementById("oneWhy");
  if(!t){el.textContent="Nothing on the list — enjoy it.";from.textContent="";
    why.textContent="";btn.style.display="none";return;}
  why.textContent="chosen because: status="+t.status+", priority="+PRI[t.priority]+
    ", remaining ~"+(remainingEst(t)||"?")+"m, "+(low?"smallest":"oldest")+" first";
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
function calib(t){
  const est=t.estimate_min||t.subtasks.reduce((a,s)=>a+(s.estimate_min||0),0)||null;
  let took=null;
  if(t.started&&t.completed){
    const m=Math.max(1,Math.round((Date.parse(t.completed)-Date.parse(t.started))/60000));
    if(m<=480)took=m;
  }
  if(!est&&!took)return"";
  return(est?"est ~"+est+"m":"")+(est&&took?" · ":"")+(took?"took ~"+took+"m":"");
}
function staleList(){
  const days=state.stale_days||14;
  return state.tasks
    .filter(t=>(t.status==="inbox"||t.status==="later")&&
      (Date.now()-Date.parse(t.updated||t.created))/86400000>=days)
    .sort((a,b)=>(a.updated||a.created)<(b.updated||b.created)?-1:1);
}
function renderBoard(){
  const board=document.getElementById("board");board.innerHTML="";
  const cols=document.getElementById("showDone").checked?STATUSES.concat("done"):STATUSES;
  board.style.gridTemplateColumns="repeat("+cols.length+",1fr)";
  for(const st of cols){
    const col=document.createElement("div");col.className="col";
    col.innerHTML="<h2>"+st+"</h2>";
    for(const t of state.tasks.filter(x=>x.status===st)){
      const d=document.createElement("div");d.className="task";
      const doneSubs=t.subtasks.filter(s=>s.done).length;
      const age=Math.floor((Date.now()-Date.parse(t.updated||t.created))/86400000);
      const stale=(st==="inbox"||st==="later")&&age>=(state.stale_days||14);
      const note=lastNote(t);
      const cal=st==="done"?calib(t):"";
      d.innerHTML='<div class="t">#'+t.id+" "+esc(t.title)+'</div>'+
        '<div class="meta"><span class="pri'+t.priority+'">'+PRI[t.priority]+"</span>"+
        (t.subtasks.length?" · "+doneSubs+"/"+t.subtasks.length+" steps":"")+
        (stale?' · <span class="stale">'+age+'d old</span>':"")+
        (cal?" · "+esc(cal):"")+"</div>"+
        (note?'<div class="lastnote">'+esc(note)+'</div>':"");
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
        const h=prompt("Optional hint for the breakdown (leave empty to skip):");
        if(h===null)return;
        await api("/api/break",{id:t.id,hint:h.trim(),
          no_project:noProject(),no_memory:noMemory()});refresh();});
      for(const to of STATUSES.filter(x=>x!==st))
        addBtn(act,"→ "+to,async()=>{await api("/api/update",{id:t.id,status:to});refresh();});
      addBtn(act,"note",async()=>{
        const v=prompt("Where did you leave off on #"+t.id+"?");
        if(v&&v.trim()){await api("/api/note",{id:t.id,text:v});refresh();}});
      addBtn(act,"done ✓",async()=>{await api("/api/update",{id:t.id,status:"done"});refresh();});
      addBtn(act,"drop ✕",async()=>{
        if(confirm('Drop "#'+t.id+' '+t.title+'" for good?')){
          await api("/api/update",{id:t.id,delete:true});refresh();}});
      d.appendChild(act);
      col.appendChild(d);
    }
    board.appendChild(col);
  }
}
// Inline markdown — `code`, **bold**, *italic* — the three things the prompts ask the
// model to write. Built as DOM nodes and never innerHTML: this is model output quoting
// the user's own diff, and a stray < or & has to stay a character, not become a tag.
function mdInline(text,parent){
  const re=/(`[^`]+`|\*\*[^*]+\*\*|\*[^*\s][^*]*\*)/g;
  let last=0,m;
  while((m=re.exec(text))!==null){
    if(m.index>last)parent.appendChild(document.createTextNode(text.slice(last,m.index)));
    const t=m[0];
    let el;
    if(t[0]==="`"){el=document.createElement("code");el.textContent=t.slice(1,-1);}
    else if(t.startsWith("**")){el=document.createElement("strong");el.textContent=t.slice(2,-2);}
    else{el=document.createElement("em");el.textContent=t.slice(1,-1);}
    parent.appendChild(el);
    last=m.index+t.length;
  }
  if(last<text.length)parent.appendChild(document.createTextNode(text.slice(last)));
  return parent;
}
function mdDiv(text,cls){const d=document.createElement("div");
  if(cls)d.className=cls;return mdInline(text||"",d);}
function addBtn(parent,label,fn){const b=document.createElement("button");
  b.textContent=label;b.onclick=fn;parent.appendChild(b);}
function addLink(parent,url){if(!url)return;const a=document.createElement("a");
  a.href=url;a.target="_blank";a.rel="noopener";a.textContent=" ↗";
  a.style.marginLeft="4px";parent.appendChild(a);}
function lastNote(t){const n=(t.notes||"").trim();if(!n)return"";
  const ls=n.split("\n");return ls[ls.length-1];}
function esc(s){return s.replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
async function doDump(){
  const t=document.getElementById("dumpText"),s=document.getElementById("dumpStatus");
  if(!t.value.trim())return;s.textContent="thinking…";
  try{const r=await api("/api/dump",{text:t.value,no_project:noProject(),no_memory:noMemory()});
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
      to:document.getElementById("draftTo").value.trim(),
      polish:document.getElementById("draftPolish").checked,
      no_voice:!document.getElementById("voiceUse").checked,
      no_project:noProject(),no_memory:noMemory()});
    o.style.display="block";o.textContent=r.message;s.textContent="";}
  catch(e){s.textContent=e.message;}
}
async function doAdd(){
  const title=document.getElementById("addTitle"),est=document.getElementById("addEst"),
    notes=document.getElementById("addNotes");
  if(!title.value.trim())return;
  await api("/api/add",{title:title.value,
    priority:Number(document.getElementById("addPri").value),
    status:document.getElementById("addStatus").value,
    estimate_min:est.value?Number(est.value):null,
    notes:notes.value.trim()});
  title.value="";est.value="";notes.value="";refresh();
}
// --- triage: the CLI's k/l/d/x walkthrough as a list, non-blocking
async function doTriage(id,decision){
  if(decision==="drop"&&!confirm("Drop #"+id+" for good?"))return;
  await api("/api/triage",{id:id,decision:decision});refresh();
}
function renderTriage(){
  const list=staleList(),el=document.getElementById("triageList"),
    c=document.getElementById("triageCount");
  c.textContent=list.length?list.length+" stale":"no stale tasks";
  el.innerHTML="";
  for(const t of list){
    const d=document.createElement("div");d.className="task";
    const age=Math.floor((Date.now()-Date.parse(t.updated||t.created))/86400000);
    d.innerHTML='<div class="t">#'+t.id+" "+esc(t.title)+'</div>'+
      '<div class="meta">'+t.status+' · <span class="stale">'+age+'d untouched</span></div>';
    const act=document.createElement("div");act.className="actions";
    addBtn(act,"keep",()=>doTriage(t.id,"keep"));
    addBtn(act,"→ later",()=>doTriage(t.id,"later"));
    addBtn(act,"done ✓",()=>doTriage(t.id,"done"));
    addBtn(act,"drop ✕",()=>doTriage(t.id,"drop"));
    d.appendChild(act);el.appendChild(d);
  }
}
// --- pr review
// The review POST blocks for as long as the model takes, so progress comes from a second
// request on another server thread rather than the response body — and the summary shows
// up word by word as the model writes it, which is the bit that says it's alive.
let prName=null,prPoll=null;
function prProgressText(p){
  if(!p||!p.active)return "";
  const t=" · "+p.elapsed+"s";
  if(p.stage==="fetch")return "fetching the pull request…"+t;
  if(p.stage==="context")return "project context: "+p.text+t;
  if(p.stage==="ticket")return "ticket "+p.text+t;
  if(p.stage==="file")return "reading "+p.text+t;
  if(p.stage==="stale")return "⚠ "+p.text;
  if(p.stage==="summarising")return "writing the overview — "+p.text+t;
  if(p.stage==="model"){
    const what=p.file?"reading "+p.file:"reading the diff";
    return what+(p.summary?": "+p.summary:" — "+p.chars+" chars in")+t;
  }
  if(p.stage==="saved")return "";
  return p.text+t;
}
function stopPrPoll(){if(prPoll){clearInterval(prPoll);prPoll=null;}}
function startPrPoll(){
  stopPrPoll();
  prPoll=setInterval(async()=>{
    try{
      const txt=prProgressText(await api("/api/pr",{action:"progress"}));
      if(txt)document.getElementById("prStatus").textContent=txt;
    }catch(e){}
  },700);
}
async function doPr(mode){
  const s=document.getElementById("prStatus");
  const body={action:mode==="resume"?"resume":(mode==="fetch"?"fetch":"review"),
    name:document.getElementById("prName").value.trim()};
  if(mode==="paste"){
    const d=document.getElementById("prDiff").value;
    if(!d.trim()){s.textContent="paste a diff first";return;}
    body.diff=d;
  }
  if(mode!=="resume"){
    body.no_project=document.getElementById("prNoProject").checked;
    body.no_ticket=document.getElementById("prNoTicket").checked;
    body.deep=document.getElementById("prDeep").checked;
    s.textContent=mode==="fetch"?"fetching the PR, then reviewing…"
      :(body.deep?"deep review — one pass per changed file, this takes a while…"
        :"reviewing… (local models take a minute)");
    startPrPoll();
  }
  try{const r=await api("/api/pr",body);s.textContent="";renderPr(r);
    document.getElementById("prOut").scrollIntoView({behavior:"smooth",block:"nearest"});}
  catch(e){s.textContent=e.message;}
  finally{stopPrPoll();}
}
async function prCheck(n){
  try{renderPr(await api("/api/pr",{action:"check",n:n,name:prName}));}
  catch(e){document.getElementById("prStatus").textContent=e.message;}
}
function renderPr(sn){
  prName=sn.name;
  const o=document.getElementById("prOut");o.innerHTML="";
  const head=document.createElement("div");head.className="muted";
  head.style.marginTop="10px";
  head.textContent=sn.name+" · from "+sn.source+
    (sn.project?" · context: "+sn.project.source+" ("+sn.project.chars+" chars)":"")+
    (sn.ticket?" · ticket sc-"+sn.ticket.id+" (via "+sn.ticket.how+")":"");
  o.appendChild(head);
  if(sn.pr){
    const p=document.createElement("div");p.className="muted";
    p.textContent="#"+sn.pr.number+" "+sn.pr.title+(sn.pr.draft?" [draft]":"");
    addLink(p,sn.pr.url);
    o.appendChild(p);
  }
  if(sn.deep){
    const d=document.createElement("div");d.className="muted";
    const n=(sn.deep.files||[]).length;
    d.textContent="deep review · "+n+" file"+(n===1?"":"s")+", one pass each";
    o.appendChild(d);
  }
  o.appendChild(mdDiv(sn.summary,"prose"));
  if(sn.truncated){const w=document.createElement("div");w.className="warn";
    w.textContent="!! Diff was truncated — large PR. Review the tail manually.";
    o.appendChild(w);}
  if(sn.unpushed){const w=document.createElement("div");w.className="warn";
    w.textContent="!! "+sn.unpushed+" commit(s) not pushed — this reviews what is on "+
      "GitHub, so anything you fixed locally since is not in it.";
    o.appendChild(w);}
  // sessions written before findings existed carry `risks`, a list of bare strings
  const fs=(sn.findings||[]).map(f=>typeof f==="string"
    ?{severity:"risk",file:"",where:"",what:f}:f)
    .concat((sn.findings||[]).length?[]:(sn.risks||[])
      .map(r=>({severity:"risk",file:"",where:"",what:r})));
  const h=document.createElement("div");h.className="muted";
  h.style.marginTop="6px";h.textContent=fs.length?"Findings:":"Nothing flagged.";
  o.appendChild(h);
  if(fs.length){
    const ul=document.createElement("ul");ul.className="findings";
    for(const f of fs){
      const li=document.createElement("li");
      const s2=document.createElement("span");s2.className="sev sev-"+(f.severity||"risk");
      s2.textContent=f.severity||"risk";li.appendChild(s2);
      const where=[f.file,f.where].filter(Boolean).join(" · ");
      if(where){const c=document.createElement("code");c.textContent=where;
        li.appendChild(c);li.appendChild(document.createElement("br"));}
      mdInline(f.what||"",li);
      if(f.fix){
        const fx=document.createElement("div");fx.className="fix";
        const lab=document.createElement("strong");lab.textContent="Fix: ";
        fx.appendChild(lab);mdInline(f.fix,fx);li.appendChild(fx);
      }
      ul.appendChild(li);
    }
    o.appendChild(ul);
  }
  if((sn.suggestions||[]).length){
    const h2=document.createElement("div");h2.className="muted";
    h2.style.marginTop="6px";h2.textContent="Suggestions:";o.appendChild(h2);
    const ul2=document.createElement("ul");ul2.className="findings";
    for(const g of sn.suggestions){const li=document.createElement("li");
      mdInline(g,li);ul2.appendChild(li);}
    o.appendChild(ul2);
  }
  if((sn.deep||{}).skipped&&sn.deep.skipped.length){
    const w=document.createElement("div");w.className="warn";
    w.textContent="Not reviewed (past the file cap): "+sn.deep.skipped.join(", ");
    o.appendChild(w);
  }
  const chk=document.createElement("div");chk.className="check";
  chk.style.marginTop="8px";
  let here=false;
  sn.checklist.forEach((c,i)=>{
    const l=document.createElement("label");if(c.done)l.className="done";
    const b=document.createElement("input");b.type="checkbox";
    b.checked=c.done;b.disabled=c.done;
    b.onchange=()=>prCheck(i+1);
    l.appendChild(b);
    const txt=document.createElement("span");
    const cf=document.createElement("code");cf.textContent=c.file;
    txt.appendChild(cf);txt.appendChild(document.createTextNode(" — "));
    mdInline(c.item,txt);l.appendChild(txt);
    if(!c.done&&!here){here=true;
      const m=document.createElement("span");m.className="here";
      m.textContent=" ← you are here";l.appendChild(m);}
    chk.appendChild(l);
  });
  o.appendChild(chk);
  const done=sn.checklist.filter(c=>c.done).length;
  const prog=document.createElement("div");prog.className="muted";
  prog.textContent=done+"/"+sn.checklist.length+" done";
  addBtn(prog,"copy as markdown",copyPrMarkdown);
  o.appendChild(prog);
}
// The server renders it, so the clipboard gets exactly what `focus pr` prints. If the
// browser refuses the clipboard, show the text instead of swallowing it.
async function copyPrMarkdown(){
  const s=document.getElementById("prStatus");
  try{
    const r=await api("/api/pr",{action:"markdown",name:prName});
    try{await navigator.clipboard.writeText(r.markdown);s.textContent="copied";}
    catch(e){
      const pre=document.createElement("pre");pre.className="out";
      pre.textContent=r.markdown;document.getElementById("prOut").appendChild(pre);
      s.textContent="couldn't reach the clipboard — select it below";
    }
  }catch(e){s.textContent=e.message;}
}
// --- voice
async function loadVoice(){
  try{renderVoice(await api("/api/voice",{action:"show"}));}
  catch(e){document.getElementById("voiceSummary").textContent=e.message;}
}
async function doVoice(body,confirmMsg){
  if(confirmMsg&&!confirm(confirmMsg))return;
  const s=document.getElementById("voiceSummary");
  if(body.action==="setup"||body.action==="learn")s.textContent="deriving profile…";
  try{renderVoice(await api("/api/voice",body));refresh();}
  catch(e){s.textContent=e.message;}
}
function renderVoice(v){
  const el=document.getElementById("voiceBody"),
    s=document.getElementById("voiceSummary");
  s.textContent=v.profile
    ?"built from "+v.samples_count+" sample(s), updated "+(v.updated||"?").slice(0,10)
    :"no profile yet";
  el.innerHTML="";el.className="";
  if(v.profile){
    const pre=document.createElement("pre");pre.className="out";
    pre.textContent=v.profile;el.appendChild(pre);
    const learn=document.createElement("details");learn.className="sub";
    learn.innerHTML="<summary>Teach it a real message</summary>";
    const lta=document.createElement("textarea");
    lta.placeholder="paste a message you actually sent…";learn.appendChild(lta);
    const lrow=document.createElement("div");lrow.className="row";
    addBtn(lrow,"Learn from this",()=>{
      if(lta.value.trim())doVoice({action:"learn",sample:lta.value});});
    learn.appendChild(lrow);el.appendChild(learn);
    const ed=document.createElement("details");ed.className="sub";
    ed.innerHTML="<summary>Edit the profile</summary>";
    const eta=document.createElement("textarea");eta.value=v.profile;
    eta.style.minHeight="140px";ed.appendChild(eta);
    const erow=document.createElement("div");erow.className="row";
    addBtn(erow,"Save profile",()=>doVoice({action:"edit",profile:eta.value}));
    ed.appendChild(erow);el.appendChild(ed);
  }
  const su=document.createElement("details");su.className="sub";
  su.innerHTML="<summary>"+(v.profile?"Redo the interview"
    :"Set up your voice — 8 quick questions, skip any")+"</summary>";
  const inputs={};
  for(const qa of v.questions){
    const l=document.createElement("label");l.style.display="block";
    l.style.marginTop="6px";l.className="muted";l.textContent=qa[1];
    const inp=document.createElement("input");inp.type="text";inp.style.width="100%";
    inputs[qa[0]]=inp;su.appendChild(l);su.appendChild(inp);
  }
  const sta=document.createElement("textarea");
  sta.placeholder="optional but powerful: paste 1-3 real messages you have sent…";
  su.appendChild(sta);
  const srow=document.createElement("div");srow.className="row";
  addBtn(srow,"Build profile",()=>{
    const answers={};
    for(const k in inputs)if(inputs[k].value.trim())answers[k]=inputs[k].value.trim();
    doVoice({action:"setup",answers:answers,
      samples:sta.value.trim()?[sta.value.trim()]:[]});
  });
  su.appendChild(srow);el.appendChild(su);
  if(v.profile){
    const row=document.createElement("div");row.className="row";
    addBtn(row,"Clear voice profile",()=>doVoice({action:"off"},
      "This deletes the profile AND its samples and answers. Sure?"));
    el.appendChild(row);
  }
}
// --- repo picker: the dashboard follows whichever repo you point it at
let browsePath=null;
function togglePicker(){
  const p=document.getElementById("repoPicker");
  if(p.style.display==="none"){p.style.display="block";loadRepo();}
  else p.style.display="none";
}
async function loadRepo(){
  try{
    const r=await api("/api/repo",{action:"show"});
    const b=await api("/api/repo",{action:"browse",path:browsePath});
    renderPicker(r,b);
  }catch(e){document.getElementById("repoPicker").textContent=e.message;}
}
async function browseRepo(path){browsePath=path;await loadRepo();}
async function useRepo(path){
  try{
    await api("/api/repo",{action:"use",path:path});
    browsePath=null;
    document.getElementById("repoPicker").style.display="none";
    await refresh();
    // both panels describe the repo — reload whichever are open, or they go stale
    if(document.getElementById("projectPanel").open)loadProject();
    if(document.getElementById("memoryPanel").open)loadMemory();
  }catch(e){document.getElementById("repoPicker").textContent=e.message;}
}
function renderPicker(r,b){
  const el=document.getElementById("repoPicker");
  el.innerHTML="";
  const at=document.createElement("div");at.className="at";at.textContent=b.path;
  el.appendChild(at);
  const ul=document.createElement("ul");
  if(b.parent){
    const li=document.createElement("li");
    const up=document.createElement("button");up.className="name";
    up.textContent="↑ ..";up.onclick=()=>browseRepo(b.parent);
    li.appendChild(up);ul.appendChild(li);
  }
  for(const d of b.dirs){
    const li=document.createElement("li");
    const nm=document.createElement("button");nm.className="name";
    nm.textContent=d.name;nm.onclick=()=>browseRepo(d.path);
    li.appendChild(nm);
    if(d.git){const t=document.createElement("span");t.className="tag";
      t.textContent="git";li.appendChild(t);}
    addBtn(li,"Use",()=>useRepo(d.path));
    ul.appendChild(li);
  }
  if(!b.dirs.length){
    const li=document.createElement("li");li.className="muted";
    li.textContent="no folders in here";ul.appendChild(li);
  }
  el.appendChild(ul);
  const row=document.createElement("div");row.className="row";
  const inp=document.createElement("input");inp.type="text";inp.style.flex="1";
  inp.placeholder="/path/to/a/repo";inp.value=b.path;
  inp.addEventListener("keydown",e=>{if(e.key==="Enter")useRepo(inp.value);});
  row.appendChild(inp);
  addBtn(row,"Use this folder",()=>useRepo(inp.value));
  addBtn(row,"Open",()=>browseRepo(inp.value));
  el.appendChild(row);
  if((r.recent||[]).length){
    const rec=document.createElement("div");rec.className="row";
    const lbl=document.createElement("span");lbl.className="muted";
    lbl.textContent="recent:";rec.appendChild(lbl);
    for(const x of r.recent){
      const btn=document.createElement("button");
      btn.textContent=x.name+(x.path===r.root?" ← here":"");
      btn.title=x.path;btn.onclick=()=>useRepo(x.path);rec.appendChild(btn);
    }
    el.appendChild(rec);
  }
}
// --- memory
async function loadMemory(){
  try{renderMemory(await api("/api/memory",{action:"show"}));}
  catch(e){document.getElementById("memorySummary").textContent=e.message;}
}
async function doMemory(body,confirmMsg){
  if(confirmMsg&&!confirm(confirmMsg))return;
  try{renderMemory(await api("/api/memory",body));refresh();}
  catch(e){document.getElementById("memorySummary").textContent=e.message;}
}
// One block per scope, same controls either side — global facts follow you everywhere,
// project facts follow the repo bar. Every verb carries its scope back to the server.
function memorySection(el,title,scope,facts,stats,disabled){
  const h=document.createElement("div");h.className="memscope";
  h.textContent=title+(disabled?"  — off, facts kept":"");
  el.appendChild(h);
  if(stats.length||facts.length){
    const ul=document.createElement("ul");ul.className="facts";
    for(const x of stats){const li=document.createElement("li");
      li.className="muted";li.textContent=x+"  (derived from history)";
      ul.appendChild(li);}
    for(const x of facts){const li=document.createElement("li");
      li.textContent=x;ul.appendChild(li);}
    el.appendChild(ul);
  }else{
    const p=document.createElement("div");p.className="muted";
    p.textContent=scope==="project"
      ?"Nothing for this repo yet — conventions, invariants, whatever you keep re-explaining."
      :"Nothing remembered yet — estimate patterns appear on their own as you finish tasks.";
    el.appendChild(p);
  }
  const row=document.createElement("div");row.className="row";
  const inp=document.createElement("input");inp.type="text";inp.style.flex="1";
  inp.placeholder=scope==="project"?"a fact about this repo…":"a fact worth remembering…";
  row.appendChild(inp);
  addBtn(row,"Remember",()=>{if(inp.value.trim())
    doMemory({action:"add",scope:scope,text:inp.value});});
  if(disabled){
    const note=document.createElement("span");note.className="muted";
    note.textContent="adding a fact switches it back on";row.appendChild(note);
  }else addBtn(row,"Turn off (facts kept)",()=>doMemory({action:"off",scope:scope}));
  el.appendChild(row);
  const ed=document.createElement("details");ed.className="sub";
  ed.innerHTML="<summary>Edit all facts</summary>";
  const ta=document.createElement("textarea");ta.value=facts.join("\n");
  ta.placeholder="one fact per line";ed.appendChild(ta);
  const erow=document.createElement("div");erow.className="row";
  addBtn(erow,"Save facts",()=>doMemory({action:"edit",scope:scope,text:ta.value},
    "Replace ALL facts in this section with this list?"));
  ed.appendChild(erow);el.appendChild(ed);
}
function renderMemory(m){
  const el=document.getElementById("memoryBody"),
    s=document.getElementById("memorySummary"),
    p=m.project||{key:"?",facts:[],disabled:false};
  const live=(m.disabled?0:m.facts.length+m.stats.length)
    +(p.disabled?0:p.facts.length);
  s.textContent=live?live+" reaching dump / break / draft":"nothing remembered yet";
  el.innerHTML="";el.className="";
  memorySection(el,"About you — everywhere","global",m.facts,m.stats,m.disabled);
  memorySection(el,"In this project ("+p.key+")","project",p.facts,[],p.disabled);
}
// --- project
async function loadProject(){
  try{
    const p=await api("/api/project",{action:"show"});
    const l=await api("/api/project",{action:"ls"});
    renderProject(p,l.projects||[]);
  }catch(e){document.getElementById("projectSummary").textContent=e.message;}
}
async function doProject(body,confirmMsg){
  if(confirmMsg&&!confirm(confirmMsg))return;
  const s=document.getElementById("projectSummary");
  if(body.action==="setup")s.textContent="reading the repo + distilling… can take a minute";
  try{await api("/api/project",body);await loadProject();refresh();}
  catch(e){s.textContent=e.message;}
}
function renderProject(p,projects){
  const el=document.getElementById("projectBody"),
    s=document.getElementById("projectSummary");
  s.textContent=p.profile?(p.key+" via "+p.source+(p.fallback?" (fallback)":""))
    :"no context for "+p.key;
  el.innerHTML="";el.className="";
  if(p.profile){
    const pre=document.createElement("pre");pre.className="out";
    pre.textContent=p.profile;el.appendChild(pre);
    const meta=document.createElement("div");meta.className="muted";
    meta.textContent=p.fallback
      ?"no saved profile — falling back to "+p.source+". Build one to distil it."
      :"from "+(p.root||"?")+" via "+p.source+", updated "+(p.updated||"?").slice(0,10);
    el.appendChild(meta);
  }
  const row=document.createElement("div");row.className="row";
  addBtn(row,"Build profile from this repo",()=>doProject({action:"setup"}));
  if(p.profile&&!p.fallback)
    addBtn(row,"Clear saved profile",()=>doProject({action:"off"},
      "Delete the saved profile for "+p.key+"? Repo docs (CLAUDE.md/README) still apply as fallback."));
  el.appendChild(row);
  const ed=document.createElement("details");ed.className="sub";
  ed.innerHTML="<summary>Edit the profile</summary>";
  const ta=document.createElement("textarea");ta.value=p.edit_seed||"";
  ta.style.minHeight="140px";ed.appendChild(ta);
  const erow=document.createElement("div");erow.className="row";
  addBtn(erow,"Save profile",()=>doProject({action:"edit",profile:ta.value}));
  ed.appendChild(erow);el.appendChild(ed);
  if(projects.length){
    const h=document.createElement("div");h.className="muted";
    h.style.marginTop="8px";h.textContent="Saved profiles:";el.appendChild(h);
    const ul=document.createElement("ul");ul.className="facts";
    for(const pr of projects){const li=document.createElement("li");
      li.textContent=pr.key+" ("+pr.chars+" chars)"+(pr.current?" ← here":"");
      ul.appendChild(li);}
    el.appendChild(ul);
  }
}
// --- shortcut ticket
async function loadShortcut(){
  try{renderShortcut(await api("/api/shortcut",{action:"show"}));}
  catch(e){document.getElementById("shortcutSummary").textContent=e.message;}
}
async function doShortcut(body,confirmMsg){
  if(confirmMsg&&!confirm(confirmMsg))return;
  const s=document.getElementById("shortcutSummary");
  s.textContent="talking to Shortcut…";
  try{renderShortcut(await api("/api/shortcut",body));}
  catch(e){s.textContent=e.message;}
}
function renderShortcut(t){
  const el=document.getElementById("shortcutBody"),
    s=document.getElementById("shortcutSummary");
  const off=t.disabled||t.disabled_here;
  s.textContent=off?"off"+(t.disabled_here&&!t.disabled?" for this repo":"")
    :(t.id?"sc-"+t.id+" via "+t.how:"no ticket for "+(t.branch||"this branch"));
  el.innerHTML="";el.className="";
  if(!t.has_token){
    const w=document.createElement("div");w.className="warn";
    w.textContent="No Shortcut token yet — paste one below to fetch tickets. "+
      "It is stored in ~/.focus/shortcut.json, mode 600, and never leaves this machine.";
    el.appendChild(w);
  }
  if(t.text){
    const pre=document.createElement("pre");pre.className="out";
    pre.textContent=t.text;el.appendChild(pre);
    const meta=document.createElement("div");meta.className="muted";
    meta.textContent="sc-"+t.id+" · found via "+t.how+" · fetched "+
      (t.fetched||"?").slice(0,10);
    addLink(meta,t.url);
    el.appendChild(meta);
  }else if(t.id){
    const d=document.createElement("div");d.className="muted";
    d.textContent="sc-"+t.id+" found via "+t.how+", but not fetched yet.";
    el.appendChild(d);
  }
  const row=document.createElement("div");row.className="row";
  const idIn=document.createElement("input");idIn.type="text";
  idIn.placeholder="story id, e.g. 12345";idIn.style.width="150px";
  row.appendChild(idIn);
  addBtn(row,"Pin to this branch",()=>{
    if(!idIn.value.trim()){s.textContent="which story?";return;}
    doShortcut({action:"use",n:idIn.value.trim()});});
  if(t.id)addBtn(row,"Refresh",()=>doShortcut({action:"fetch"}));
  addBtn(row,off?"Turn ticket context on":"Turn it off",
    ()=>doShortcut({action:off?"on":"off"}));
  el.appendChild(row);
  const ed=document.createElement("details");ed.className="sub";
  ed.innerHTML="<summary>API token</summary>";
  const ti=document.createElement("input");ti.type="password";
  ti.placeholder=t.env_token?"set from SHORTCUT_API_TOKEN":
    (t.has_token?"saved — paste a new one to replace":"paste your Shortcut API token");
  ti.style.width="320px";ed.appendChild(ti);
  const trow=document.createElement("div");trow.className="row";
  addBtn(trow,"Save token",()=>{
    if(!ti.value.trim())return;
    doShortcut({action:"token",token:ti.value.trim()});ti.value="";});
  addBtn(trow,"Forget everything",()=>doShortcut({action:"clear"},
    "Delete the saved token and every cached ticket?"));
  ed.appendChild(trow);el.appendChild(ed);
  if((t.cached||[]).length){
    const h=document.createElement("div");h.className="muted";
    h.style.marginTop="8px";h.textContent="Cached tickets:";el.appendChild(h);
    const ul=document.createElement("ul");ul.className="facts";
    const pins=t.pins||{};
    const byId={};for(const b in pins){(byId[pins[b]]=byId[pins[b]]||[]).push(b);}
    for(const c of t.cached){const li=document.createElement("li");
      li.textContent="sc-"+c+(byId[c]?" — "+byId[c].join(", "):"")+
        (c===t.id?" ← here":"");
      ul.appendChild(li);}
    el.appendChild(ul);
  }
}
// --- doctor
function renderSettings(){
  const el=document.getElementById("settingsBody");
  const open=state.tasks.filter(t=>t.status!=="done").length;
  const m=state.model||{};
  el.style.whiteSpace="pre-wrap";
  el.textContent=["data dir : "+(state.home||"?"),
    "tasks    : "+open+" open / "+state.tasks.length+" total",
    state.llm?"model    : OK — "+m.runtime+" at "+m.endpoint+" (model: "+m.model+")"
      :"model    : NOT FOUND — "+(m.error||"")].join("\n");
}
document.getElementById("voicePanel").addEventListener("toggle",e=>{if(e.target.open)loadVoice();});
document.getElementById("memoryPanel").addEventListener("toggle",e=>{if(e.target.open)loadMemory();});
document.getElementById("projectPanel").addEventListener("toggle",e=>{if(e.target.open)loadProject();});
document.getElementById("shortcutPanel").addEventListener("toggle",e=>{if(e.target.open)loadShortcut();});
// Timer state lives in localStorage as an end-timestamp, so a reload (or closing the
// tab mid-pomodoro) never loses a running timer.
let timer=null;
function timerEnd(){return Number(localStorage.getItem("focusTimerEnd")||0);}
function setTimerEnd(ts){ts?localStorage.setItem("focusTimerEnd",ts)
  :localStorage.removeItem("focusTimerEnd");}
function timerMins(){return Number(localStorage.getItem("focusTimerMin")||25);}
function tickTimer(){
  const btn=document.getElementById("timerBtn"),txt=document.getElementById("timerTxt");
  const end=timerEnd();
  if(!end){if(timer){clearInterval(timer);timer=null;}
    btn.textContent="Start timer";return;}
  const left=Math.round((end-Date.now())/1000);
  if(left<=0){
    const mins=timerMins();
    setTimerEnd(0);clearInterval(timer);timer=null;
    txt.textContent="break time!";btn.textContent="Start timer";
    if("Notification" in window&&Notification.permission==="granted")
      new Notification("focus",{body:mins+" minutes up — take a break."});
    api("/api/timer",{event:"timer_done",minutes:mins}).catch(()=>{});
    return;
  }
  txt.textContent=String(Math.floor(left/60)).padStart(2,"0")+":"+
    String(left%60).padStart(2,"0");
  btn.textContent="Stop timer";
}
document.getElementById("timerBtn").onclick=()=>{
  if(timerEnd()){setTimerEnd(0);document.getElementById("timerTxt").textContent="";
    api("/api/timer",{event:"timer_cancel",minutes:timerMins()}).catch(()=>{});
    tickTimer();return;}
  if("Notification" in window&&Notification.permission==="default")
    Notification.requestPermission();
  const mins=Math.max(1,Number(document.getElementById("timerMin").value)||25);
  localStorage.setItem("focusTimerMin",mins);
  setTimerEnd(Date.now()+mins*60*1000);
  api("/api/timer",{event:"timer_start",minutes:mins}).catch(()=>{});
  timer=setInterval(tickTimer,1000);tickTimer();
};
if(timerEnd()){timer=setInterval(tickTimer,1000);tickTimer();}
refresh();setInterval(refresh,5000);
</script>
</body>
</html>"""

# ---------------------------------------------------------------- main

NO_PROJECT_HELP = "ignore this repo's project context"
NO_MEMORY_HELP = "ignore your memory facts for this command"
NO_TICKET_HELP = "ignore the Shortcut ticket for this command"
STORY_HELP = "Shortcut story id to use, or 'none' to skip the ticket"

def main(argv=None):
    p = argparse.ArgumentParser(prog="focus",
                                description="local-first ADHD companion for developers")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("dump", help="brain-dump -> tasks")
    s.add_argument("text", nargs="*")
    s.add_argument("--no-project", action="store_true", help=NO_PROJECT_HELP)
    s.add_argument("--no-memory", action="store_true", help=NO_MEMORY_HELP)
    s.add_argument("--no-ticket", action="store_true", help=NO_TICKET_HELP)
    s.set_defaults(fn=cmd_dump)

    s = sub.add_parser("add", help="add one task (no AI)")
    s.add_argument("title", nargs="+")
    s.add_argument("--status", default="inbox", choices=STATUSES)
    s.add_argument("-p", "--priority", type=int, default=2, choices=[1, 2, 3])
    s.add_argument("-e", "--estimate", type=int, metavar="MIN",
                   help="estimate in minutes")
    s.add_argument("--notes", default="", help="initial note on the task")
    s.set_defaults(fn=cmd_add)

    s = sub.add_parser("ls", help="list tasks")
    s.add_argument("-v", "--verbose", action="store_true", help="show subtasks")
    s.add_argument("-a", "--all", action="store_true", help="include done")
    s.set_defaults(fn=cmd_ls)

    s = sub.add_parser("break", help="split a task into small steps")
    s.add_argument("id")
    s.add_argument("--hint", nargs="*", help="extra context for the model")
    s.add_argument("--no-project", action="store_true", help=NO_PROJECT_HELP)
    s.add_argument("--no-memory", action="store_true", help=NO_MEMORY_HELP)
    s.add_argument("--no-ticket", action="store_true", help=NO_TICKET_HELP)
    s.add_argument("--story", help=STORY_HELP)
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

    s = sub.add_parser("today", help="what you finished today")
    s.add_argument("-w", "--week", action="store_true", help="last 7 days instead")
    s.set_defaults(fn=cmd_today)

    s = sub.add_parser("note", help="jot where you left off on a task")
    s.add_argument("id")
    s.add_argument("text", nargs="*")
    s.set_defaults(fn=cmd_note)

    s = sub.add_parser("triage", help="work through stale tasks, one at a time")
    s.set_defaults(fn=cmd_triage)

    s = sub.add_parser("timer", help="countdown with a desktop notification")
    s.add_argument("minutes", nargs="?", type=int, default=25)
    s.set_defaults(fn=cmd_timer)

    s = sub.add_parser("pr", help="summarise a diff + review checklist")
    s.add_argument("action", nargs="?", default="review",
                   choices=["review", "resume", "check", "fetch"],
                   help="review a diff (default), fetch this branch's PR, "
                        "resume the last one, or check N")
    s.add_argument("n", nargs="?", type=int,
                   help="item number for `check N`, PR number for `fetch N`")
    s.add_argument("-f", "--file", help="read diff from file")
    s.add_argument("--name", help="session name")
    s.add_argument("--project", help="project profile to use, or 'none' to skip context")
    s.add_argument("--no-ticket", action="store_true", help=NO_TICKET_HELP)
    s.add_argument("--story", help=STORY_HELP)
    s.add_argument("--deep", action="store_true",
                   help="one model pass per changed file, then a summary pass — much "
                        "deeper, and minutes rather than seconds")
    # pre-verb spellings, still honoured so nothing anyone has typed before breaks
    s.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    s.add_argument("--check", type=int, help=argparse.SUPPRESS)
    s.add_argument("--no-project", action="store_true", help=argparse.SUPPRESS)
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
    s.add_argument("--no-project", action="store_true", help=NO_PROJECT_HELP)
    s.add_argument("--no-memory", action="store_true", help=NO_MEMORY_HELP)
    s.add_argument("--no-ticket", action="store_true", help=NO_TICKET_HELP)
    s.set_defaults(fn=cmd_draft)

    s = sub.add_parser("voice", help="teach drafts to sound like you")
    s.add_argument("action", nargs="?", default="setup",
                   choices=["setup", "show", "edit", "learn", "off"])
    s.add_argument("--samples", nargs="*",
                   help="files containing real messages you've sent")
    s.set_defaults(fn=cmd_voice)

    s = sub.add_parser("memory", help="facts the AI remembers about you (local only)")
    s.add_argument("action", nargs="?", default="show",
                   choices=["show", "add", "edit", "off"])
    s.add_argument("text", nargs="*", help="the fact, for `memory add`")
    s.add_argument("--here", action="store_true",
                   help="act on this repo's memory, not your global facts")
    s.add_argument("--project", help="act on this project's memory, by profile name")
    s.set_defaults(fn=cmd_memory)

    s = sub.add_parser("project", help="teach AI features about this codebase")
    s.add_argument("action", nargs="?", default="setup",
                   choices=["setup", "show", "edit", "ls", "off"])
    s.add_argument("--project", help="act on this profile name, not the detected one")
    s.set_defaults(fn=cmd_project)

    s = sub.add_parser("shortcut", help="pull Shortcut tickets into the AI features")
    s.add_argument("action", nargs="?", default="show",
                   choices=["show", "use", "fetch", "token", "ls", "on", "off", "clear"],
                   help="show this branch's ticket (default), pin one with `use N`, "
                        "refresh it, save a token, list, or turn injection on/off")
    s.add_argument("n", nargs="?", type=int, help="story id, for `use` and `fetch`")
    s.add_argument("--here", action="store_true",
                   help="act on this repo only, not everywhere")
    s.set_defaults(fn=cmd_shortcut)

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
    except (NoModelError, ModelReplyError, TicketError) as e:
        print(str(e), file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    sys.exit(main())
