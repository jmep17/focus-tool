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
  focus today              What you finished today (--week for the week)
  focus note <id> [text]   Breadcrumb: jot where you left off, shown on resume
  focus triage             Work through stale tasks, one at a time
  focus timer [min]        Countdown with a desktop notification (default 25)
  focus pr [...]           Summarise a diff + build a review checklist (AI)
  focus draft [...]        Bullets -> polished message (AI)
  focus memory [...]       Facts the AI remembers about you (local only)
  focus project [...]      Teach the AI features about this codebase (AI)
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

class ModelReplyError(RuntimeError):
    pass

def ask_model(system, user, temperature=0.3, raw=False):
    """One model call with one retry. raw=True returns the reply text; otherwise the
    first JSON object in it. Garbage replies and HTTP failures become ModelReplyError
    so callers never traceback on a flaky local model. NoModelError passes through —
    it has its own fallbacks."""
    last = None
    for _ in range(2):
        try:
            reply = chat(system, user, temperature)
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

def _root_and_key():
    """(repo root, storage key) for the cwd. Cached — this shells out to git twice,
    and api_state() is polled every few seconds by the dashboard."""
    cwd = os.getcwd()
    if cwd not in _ROOT_CACHE:
        root = _git(["rev-parse", "--show-toplevel"]).strip() or cwd
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
                  "and name the convention when a risk follows from one:\n")

def project_block(text):
    """Format already-resolved context. Split out because cmd_pr needs the source label
    for the session record, so it resolves separately."""
    return PROJECT_HEADER + text if text else ""

def project_system_suffix(query="", budget=MAX_CONTEXT_CHARS, name=None, disabled=False):
    return project_block(resolve_project_context(query, budget, name, disabled)[0])

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

def load_memory():
    return load_json(_path("memory.json"), {})

def save_memory(m):
    save_json(_path("memory.json"), m)

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

def memory_system_suffix(disabled=False):
    """Stats derived from history plus explicit `focus memory` facts, as a system-prompt
    suffix. Same contract as voice_system_suffix: "" when off or empty."""
    if disabled:
        return ""
    m = load_memory()
    if m.get("disabled"):
        return ""
    facts = [f["text"] for f in m.get("facts", []) if f.get("text")]
    body = "\n".join("- " + x for x in memory_stats_lines() + facts)
    if not body:
        return ""
    return MEMORY_HEADER + _clip(body, MAX_MEMORY_CHARS)

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
            + memory_system_suffix(args.no_memory)
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
        + memory_system_suffix(args.no_memory)
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

def cmd_memory(args):
    m = load_memory()
    if args.action == "show":
        stats = memory_stats_lines()
        facts = [f["text"] for f in m.get("facts", []) if f.get("text")]
        if not stats and not facts:
            print("Nothing remembered yet. `focus memory add <a fact about you>` —")
            print("estimate patterns appear on their own as you finish tasks.")
            return
        state = "OFF — not being injected" if m.get("disabled") else \
            "used by dump / break / draft"
        print(f"\nUSER MEMORY ({state}):\n")
        for line in stats:
            print(f"  - {line}   (derived from history)")
        for line in facts:
            print(f"  - {line}")
        print("\n  add: focus memory add <fact> · edit: focus memory edit · "
              "off: focus memory off")
        return
    if args.action == "off":
        m["disabled"] = True
        save_memory(m)
        print("Memory injection off (facts kept). `focus memory add` re-enables it,")
        print("or `--no-memory` skips it for a single command.")
        return
    if args.action == "edit":
        facts = [f["text"] for f in m.get("facts", []) if f.get("text")]
        editor = os.environ.get("EDITOR", "nano")
        path = _path("memory_facts.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(facts))
        subprocess.run([editor, path])
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
        m["facts"] = [{"text": ln, "ts": now_iso()} for ln in lines]
        m["updated"] = now_iso()
        save_memory(m)
        print(f"Memory updated — {len(lines)} fact(s).")
        return
    # add
    text = " ".join(args.text).strip() or read_multiline("", "A fact worth remembering")
    if not text.strip():
        raise SystemExit("Nothing to remember.")
    m.setdefault("facts", []).append({"text": text.strip(), "ts": now_iso()})
    m.pop("disabled", None)
    m["updated"] = now_iso()
    save_memory(m)
    print(f"Remembered: {text.strip()}")
    print("Every dump/break/draft now knows it. `focus memory show` to review.")

# ------------------------------------------------ pr review

# Project context is charged against this, not added on top of it — see cmd_pr. An 8k
# model is already over budget at 60k chars of diff; context must not make that worse.
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
    return os.path.join(focus_home(), "pr", safe_name(name, "session") + ".json")

def _pr_action(args):
    """(action, item) from either the verb form or the older --resume/--check flags."""
    if args.action == "check":
        return "check", args.n
    if args.check is not None:
        return "check", args.check
    if args.action == "resume" or args.resume:
        return "resume", None
    return "review", None

def cmd_pr(args):
    action, item = _pr_action(args)
    if action != "review":
        if action == "check" and item is None:
            raise SystemExit("Which item? `focus pr check 3`")
        return cmd_pr_resume(args, item)
    diff, source = get_diff(args)
    # Resolve context first, then charge it to the diff's budget so the request never
    # grows. Keyed on the paths the diff touches, so only relevant sections survive.
    ctx, ctx_source = resolve_project_context(
        diff_paths(diff), MAX_CONTEXT_CHARS, args.project, args.no_project)
    truncated = False
    if len(diff) > MAX_DIFF_CHARS - len(ctx):
        diff = diff[:MAX_DIFF_CHARS - len(ctx)]
        truncated = True
    data = ask_model(SYS_PR + project_block(ctx), diff)
    name = args.name or datetime.now().strftime("pr-%Y%m%d-%H%M")
    session = {
        "name": name,
        "source": source,
        "created": now_iso(),
        "summary": data.get("summary", ""),
        "risks": data.get("risks", []),
        "truncated": truncated,
        "project": {"source": ctx_source, "chars": len(ctx)} if ctx else None,
        "checklist": [
            {"file": c.get("file", "?"), "item": c.get("item", ""), "done": False}
            for c in data.get("checklist", [])
        ],
    }
    save_json(pr_session_path(name), session)
    print_pr(session)
    print(f"Saved as '{name}'. After any interruption: focus pr resume")

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
    if s.get("project"):
        print(f"  project context: {s['project']['source']} "
              f"({s['project']['chars']} chars)")
    print("-" * 56)
    print("WHAT IT DOES:\n" + textwrap.fill(s["summary"], 72, initial_indent="  ",
                                            subsequent_indent="  "))
    if s.get("truncated"):
        print("\n  !! Diff was truncated — large PR. Review the tail manually.")
    if s["risks"]:
        print("\nLOOK HARDEST AT:")
        for r in s["risks"]:
            print("  ! " + r)
    print("\nCHECKLIST (tick with `focus pr check N`):")
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
              + memory_system_suffix(args.no_memory))
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

# ------------------------------------------------ project

def _derive_project(root):
    """Repo -> markdown profile. Falls back to the raw brief with no model running."""
    brief = repo_brief(root)
    if not brief.strip():
        raise SystemExit(f"Nothing to harvest in {root} — no docs, manifests or files.")
    try:
        data = ask_model(SYS_PROJECT, brief)
    except NoModelError:
        print("(no local model — saving the raw repo brief instead of a distilled one)\n")
        return _clip(brief, MAX_CONTEXT_CHARS * 2), "brief"
    profile = data.get("profile", "").strip()
    if not profile:
        raise SystemExit("Model returned an empty profile — try again.")
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
    save_project(key, {"profile": profile, "root": root,
                       "source": source, "updated": now_iso()})
    print(f"\nPROJECT PROFILE: {key}\n")
    print(textwrap.indent(profile, "  "))
    print(f"\nSaved. Every AI command now knows this repo — `focus project edit` to "
          f"correct it, `--no-project` to skip it.")

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
    ctx, ctx_source = resolve_project_context(budget=BRIEF_CONTEXT_CHARS)
    today = datetime.now(timezone.utc).date().isoformat()
    done_today = [{"id": t["id"], "title": t["title"]}
                  for t in store["tasks"] if t["status"] == "done"
                  and (t.get("completed") or "")[:10] == today]
    return {"tasks": store["tasks"], "next_id": nxt["id"] if nxt else None,
            "llm": llm_ok, "voice": bool(load_voice().get("profile")),
            "project": ctx_source if ctx else "",
            "memory": bool(memory_system_suffix()),
            "done_today": done_today, "streak": _streak(_done_dates(load_history()))}

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
                system = SYS_BREAK + project_system_suffix(
                    t["title"], BRIEF_CONTEXT_CHARS,
                    disabled=body.get("no_project", False)) \
                    + memory_system_suffix(body.get("no_memory", False))
                data = ask_model(system, t["title"])
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
                        + memory_system_suffix(body.get("no_memory", False))
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
                             "firm": "Tone: polite but firm."}.get(tone, "")
                system = "\n".join([SYS_DRAFT, style, tone_line]) + \
                    voice_system_suffix(body.get("no_voice", False)) + \
                    project_system_suffix(body["text"], BRIEF_CONTEXT_CHARS,
                                          disabled=body.get("no_project", False)) + \
                    memory_system_suffix(body.get("no_memory", False))
                msg = ask_model(system,
                                "Write the message from these notes:\n\n" + body["text"],
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
                if ev in ("timer_start", "timer_done", "timer_cancel"):
                    log_event(ev, minutes=int(body.get("minutes", 25)), source="ui")
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "not found"})
        except NoModelError as e:
            self._send(503, {"error": str(e)})
        except ModelReplyError as e:
            self._send(502, {"error": str(e)})
        except Exception as e:  # keep the local server alive
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

def cmd_ui(args):
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
</style>
</head>
<body>
<div class="wrap">
  <h1>focus <span id="llmBadge" class="llm off">checking model…</span>
    <label class="muted" id="projectWrap" style="display:none;cursor:pointer">
      <input type="checkbox" id="projectUse" checked> using <span id="projectSrc"></span></label>
    <label class="muted" id="memoryWrap" style="display:none;cursor:pointer">
      <input type="checkbox" id="memoryUse" checked> memory</label></h1>

  <div class="panel" id="donePanel" style="display:none;margin:0 0 18px;">
    <h2>🎉 Done today (<span id="doneCount"></span>)</h2>
    <div class="muted" id="doneList"></div>
  </div>

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
  document.getElementById("projectWrap").style.display=state.project?"inline":"none";
  document.getElementById("projectSrc").textContent=state.project;
  document.getElementById("memoryWrap").style.display=state.memory?"inline":"none";
  renderOne();renderBoard();renderDone();
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
    d.map(x=>"#"+x.id+" "+x.title).join("  ·  ")+
    (state.streak>1?"   —  "+state.streak+"-day streak":"");
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
      const age=Math.floor((Date.now()-Date.parse(t.updated||t.created))/86400000);
      const stale=(st==="inbox"||st==="later")&&age>=14;
      const note=lastNote(t);
      d.innerHTML='<div class="t">#'+t.id+" "+esc(t.title)+'</div>'+
        '<div class="meta"><span class="pri'+t.priority+'">'+PRI[t.priority]+"</span>"+
        (t.subtasks.length?" · "+doneSubs+"/"+t.subtasks.length+" steps":"")+
        (stale?' · <span class="stale">'+age+'d old</span>':"")+"</div>"+
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
        await api("/api/break",{id:t.id,no_project:noProject(),no_memory:noMemory()});refresh();});
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
function addBtn(parent,label,fn){const b=document.createElement("button");
  b.textContent=label;b.onclick=fn;parent.appendChild(b);}
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
      no_voice:!document.getElementById("voiceUse").checked,
      no_project:noProject(),no_memory:noMemory()});
    o.style.display="block";o.textContent=r.message;s.textContent="";}
  catch(e){s.textContent=e.message;}
}
// Timer state lives in localStorage as an end-timestamp, so a reload (or closing the
// tab mid-pomodoro) never loses a running timer.
let timer=null;
function timerEnd(){return Number(localStorage.getItem("focusTimerEnd")||0);}
function setTimerEnd(ts){ts?localStorage.setItem("focusTimerEnd",ts)
  :localStorage.removeItem("focusTimerEnd");}
function tickTimer(){
  const btn=document.getElementById("timerBtn"),txt=document.getElementById("timerTxt");
  const end=timerEnd();
  if(!end){if(timer){clearInterval(timer);timer=null;}
    btn.textContent="Start 25-min timer";return;}
  const left=Math.round((end-Date.now())/1000);
  if(left<=0){
    setTimerEnd(0);clearInterval(timer);timer=null;
    txt.textContent="break time!";btn.textContent="Start 25-min timer";
    if("Notification" in window&&Notification.permission==="granted")
      new Notification("focus",{body:"25 minutes up — take a break."});
    api("/api/timer",{event:"timer_done",minutes:25}).catch(()=>{});
    return;
  }
  txt.textContent=String(Math.floor(left/60)).padStart(2,"0")+":"+
    String(left%60).padStart(2,"0");
  btn.textContent="Stop timer";
}
document.getElementById("timerBtn").onclick=()=>{
  if(timerEnd()){setTimerEnd(0);document.getElementById("timerTxt").textContent="";
    api("/api/timer",{event:"timer_cancel",minutes:25}).catch(()=>{});
    tickTimer();return;}
  if("Notification" in window&&Notification.permission==="default")
    Notification.requestPermission();
  setTimerEnd(Date.now()+25*60*1000);
  api("/api/timer",{event:"timer_start",minutes:25}).catch(()=>{});
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

def main(argv=None):
    p = argparse.ArgumentParser(prog="focus",
                                description="local-first ADHD companion for developers")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("dump", help="brain-dump -> tasks")
    s.add_argument("text", nargs="*")
    s.add_argument("--no-project", action="store_true", help=NO_PROJECT_HELP)
    s.add_argument("--no-memory", action="store_true", help=NO_MEMORY_HELP)
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
                   choices=["review", "resume", "check"],
                   help="review a diff (default), resume the last one, or check N")
    s.add_argument("n", nargs="?", type=int, help="item number, for `focus pr check N`")
    s.add_argument("-f", "--file", help="read diff from file")
    s.add_argument("--name", help="session name")
    s.add_argument("--project", help="project profile to use, or 'none' to skip context")
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
    s.set_defaults(fn=cmd_memory)

    s = sub.add_parser("project", help="teach AI features about this codebase")
    s.add_argument("action", nargs="?", default="setup",
                   choices=["setup", "show", "edit", "ls", "off"])
    s.add_argument("--project", help="act on this profile name, not the detected one")
    s.set_defaults(fn=cmd_project)

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
    except (NoModelError, ModelReplyError) as e:
        print(str(e), file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    sys.exit(main())
