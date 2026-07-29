# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python3 tests/test_focus.py           # the whole test suite (~303 checks, no pytest)
uv tool install --editable .          # install `focus` on PATH, running this folder live
uv run --script focus.py <subcommand> # run without installing (PEP 723 header, zero deps)
focus doctor                          # check which local model server is reachable
focus ui --port 8765 --no-browser     # dashboard on 127.0.0.1 only
```

There is no linter, formatter, or CI config in the repo.

**Running one test:** the suite is a linear script, not a framework — `check()` asserts and
dies on first failure, and later checks depend on earlier state (task ids 1/2/3, a voice
profile built mid-file). There is no way to select a single test; comment out or truncate
the tail of `tests/test_focus.py` instead. `FOCUS_HOME` is pointed at a temp dir and
`mock_llm` is started *before* `import focus`, so never reorder those first ~20 lines. The
`os.chdir` in that block belongs there too — project context resolves against the cwd, so
without it the checks depend on where you invoked the suite from.

The suite never touches the network, GitHub or Shortcut: `tests/mock_shortcut.py` is a
loopback fake wired in through `shortcut.json`'s `endpoint` (the same config-injection
trick that points the model client at `mock_llm`), and `tests/fake_gh.py` is put on `PATH`
as `gh`, with `FOCUS_TEST_GH=nopr` selecting the no-PR case. Two things in the gh block are
load-bearing: `sys.stdin` is swapped for a tty-shaped object, because `get_diff` reads
piped stdin before anything else and the suite runs piped; and `git_working_diff` is
monkeypatched rather than the checkout dirtied, so the tier-order checks don't depend on
the state of the working tree.

## Architecture

Everything is `focus.py`: CLI, storage, LLM client, HTTP server, and the dashboard HTML.
`pyproject.toml` exposes it as `focus = "focus:main"`. Keep it stdlib-only — the PEP 723
header declares `dependencies = []`, which is what makes `uv run --script` and the bare
`chmod +x focus.py` install path work.

**Storage** — plain JSON under `~/.focus/`, overridable with `FOCUS_HOME` (tests rely on
this): `tasks.json` (`{next_id, tasks[]}`), `config.json` (`{endpoint, model}`),
`voice.json`, `memory.json` (`{facts[], disabled?}`), `memory/<key>.json` in the same shape
for one repo's facts, `projects/<key>.json` per repo, `ui.json` (`{repo, recent[]}` — the
dashboard's repo selection), `pr/<name>.json` per review session, `shortcut.json`
(`{token, endpoint?, disabled?}`) and `shortcut/story-<id>.json` +
`shortcut/repo-<key>.json` for tickets. All writes go through `save_json`,
which writes `.tmp` then `os.replace` for atomicity. `load_json` swallows missing/corrupt
files and returns the default — there is no migration layer, so any new task field must be
read with `.get()` (`started` and `completed` already follow this rule).

`history.jsonl` is the exception to the save_json rule: an append-only event log
(`log_event()` writes one JSON line per created/started/done/subtask_done/moved/triage/
timer event and never raises). It is the source of truth for `focus today`, streaks, and
the derived estimate-accuracy stats — don't rewrite it in place, only append.

**LLM layer** — `detect_runtime()` probes `/models` on the configured endpoint, then LM
Studio (`:1234`), then Ollama (`:11434`), and picks the first model id it sees unless
`config.json` names one. `chat()` posts an OpenAI-shaped `/chat/completions`.
`extract_json()` is a brace-depth scanner that survives markdown fences, preamble chatter,
and braces inside strings — small local models produce all three, so use it rather than
`json.loads` on any model reply. Failure to find a server raises `NoModelError`, which
`main()` turns into exit code 2; only `dump` degrades gracefully (one line = one task).

`chat(on_delta=...)` streams: `_chat_stream` posts `"stream": true` and reads the SSE
lines as the socket gives them, calling `on_delta(text so far)`. Streaming is a **display
nicety, never a new failure mode** — a server that refuses `stream`, or one that answers
it with a plain JSON body, falls back to the ordinary request and the caller gets the
same string it always got. Don't turn a streaming failure into an error; the two mock
servers behind `FOCUS_TEST_NOSTREAM` (`plain` / `error`) exist to keep that true.

Every AI call site goes through `ask_model()` — chat + extract_json with one retry, and
any garbage reply or HTTP failure surfaced as `ModelReplyError` (exit 2 in the CLI, 502
in the UI; `NoModelError` stays 503, and `TicketError` joins `ModelReplyError` at 2/502). Don't call `chat()`/`extract_json()` directly from
a command: the traceback-on-flaky-model failure mode is exactly what `ask_model` exists
to prevent. `raw=True` returns the reply text for non-JSON prompts (`draft`).

**Prompts are the product.** `SYS_DUMP`, `SYS_BREAK`, `SYS_PR`, `SYS_PR_FILE`,
`SYS_PR_SUMMARY`, `SYS_VOICE`, `SYS_PROJECT` each end with a literal JSON schema, and the
parsing code downstream assumes exactly those keys. Editing a prompt is a three-place
change:

1. the `SYS_*` constant,
2. the consumer (e.g. `cmd_break` reads `data["subtasks"][].text/estimate_min`),
3. `tests/mock_llm.py` — `canned_reply()` dispatches on **substrings of the system prompt**
   (`"brain-dump"`, `"break a task"`, `"unified diff"`, `"ONE file out of a larger pull
   request"`, `"already been reviewed on its own"`, `"writing-voice profile"`, `"write work
   messages"`, `"compact project profile"`). Rewording a prompt past one of those substrings
   makes the mock fall through to `"ok"` and tests fail far from the real cause.

Two traps in that dispatch. A substring must sit on **one line** of the prompt — these are
triple-quoted strings, and a phrase that wraps across a newline silently never matches.
And the specific prompts are tested **before** the general one (`SYS_PR_FILE` and
`SYS_PR_SUMMARY` above `SYS_PR`), or a deep review's per-file pass gets answered with a
single-pass reply.

The mock dispatches on the prompt **up to the first suffix marker** (`MATCH THIS PERSON'S
VOICE`, `PROJECT CONTEXT`, `TICKET CONTEXT`, `USER MEMORY`), not the whole string. It has
to: everything after a marker is arbitrary repo or ticket text, and this repo's own
`CLAUDE.md` contains the literal string `"brain-dump"` — dispatching on the full prompt
answers a PR request with a task list. If you add another suffix, add its marker to that
split. `mock_llm.SEEN` records system prompts and `SEEN_USER` the user message, which is
where the diff and the PR preamble ride.

**`focus pr` takes verbs, not flags** — `pr` / `pr fetch [N]` / `pr resume` / `pr check N`,
matching `voice` and `project`. The pre-verb `--resume` / `--check N` / `--no-project`
spellings still work but are `argparse.SUPPRESS`ed out of `--help`; `_pr_action()`
normalises both forms in one place so `cmd_pr` doesn't branch on flags. The one `n`
positional serves both `check N` and `fetch N`. `--project none` is the documented way to
skip context for a review, `--story none` / `--no-ticket` the way to skip the ticket. Tests
cover both spellings — don't delete the legacy ones without deleting their checks.

**`pick_next` is deliberately AI-free.** Sorting by (status order, priority, remaining
estimate, created) — with `--energy low` swapping priority and size — is the product
decision, not a shortcut. Don't "improve" it by asking the model.

**Voice profile** — `voice_system_suffix()` appends `voice.json`'s profile to the draft
system prompt, and returns `""` when disabled or absent. It is invoked from both
`cmd_draft` and the UI's `/api/draft`; `focus voice learn` keeps the freshest 10 samples and
re-derives the whole profile from scratch each time.

**User memory** — `memory_system_suffix()` follows the same contract (`""` when off or
empty) and injects three things under the one `USER MEMORY` marker: explicit `focus memory
add` facts, stats `memory_stats_lines()` derives *deterministically* from `history.jsonl`
(median estimate-vs-actual ratio after 3 samples, finishing streak), and — under an
`In this project (<key>):` line — the active repo's own facts from `memory/<key>.json`.
No model round-trip in the derivation — that's the point; it works with no server running.
Injected into dump/break/draft and their UI twins only, never `pr`: the pr budget belongs
to the diff. `actual_min` is capped at 8h in `actual_minutes()` so a task started Monday
and finished Friday doesn't poison calibration.

The two memory scopes are independent — each file has its own `disabled` flag, and
`memory_path(key=None)` is the only thing that knows which is which. The stats stay global
on purpose: `history.jsonl` carries no repo, and how far someone's estimates run out is a
fact about the person. Project facts are `_clip`ped first and capped at half of
`MAX_MEMORY_CHARS` so a long global list can't crowd out the specific ones. `--no-project`
(and the UI's `no_project`) drops the project block along with the project profile — one
flag, one meaning. CLI scope flags trail the text, as everywhere else here:
`focus memory add "<fact>" --here`, since argparse can't take a flag between two
positionals.

**Project context** — `project_system_suffix()` appends a repo profile to *every* AI system
prompt (dump, break, pr, draft, and their UI twins), the same shape as the voice suffix.
`resolve_project_context()` tries, first hit wins, never raising: `--project NAME`, then
`projects/<key>.json`, then a repo-committed `.focus/project.md`, then the first present of
`PROJECT_DOCS` read live. That last tier is why the feature works with no setup — don't
"fix" it by requiring `focus project setup`. focus reads `.focus/project.md` but never
writes into a user's repo.

`select_sections()` trims a profile to a char budget: the preamble and any heading matching
`PINNED_HEADINGS` always survive, the rest are scored by lexical overlap with tokens from
the diff's changed paths (or the user's text). `PINNED_HEADINGS` is coupled to the headings
`SYS_PROJECT` dictates — change one, change the other. It is deliberately not embeddings:
`/v1/embeddings` isn't reliably served, `detect_runtime()` has no notion of model *type*,
and ranking ten sections of a 4kB document doesn't need vectors.

Context is charged **against** `MAX_DIFF_CHARS`, not added to it (`run_pr_review` shrinks
the diff budget by `len(ctx) + len(ticket) + len(preamble)`). An 8k-context model is
already over budget on diff alone; anything that grows the request is a regression.
`_clip()` exists so a heading-less doc or one oversized section can't blow the budget and
silently eat the diff's share.

**Shortcut tickets** — `ticket_system_suffix()` is a third suffix with the same contract as
the voice and project ones (`""` when off or empty), under a `TICKET CONTEXT` marker. It
goes into dump, break, draft **and** pr — unlike memory, which stays out of pr. The ticket
is the one thing a diff genuinely cannot tell you, so it earns its share of that budget.

`resolve_ticket_context()` **never fetches** unless `refresh=True`, which only
`run_pr_review` passes; the `focus shortcut use/fetch` verbs call `fetch_story()` directly.
Everything else reads `shortcut/story-<id>.json`. That is what keeps `focus dump` instant
and working on a train — don't "improve" it by fetching on every command. Even with
`refresh=True` a `TicketError` falls back to the cache rather than breaking the review.

**The gate order in `resolve_ticket_context` is load-bearing.** `_any_cached_story()` (one
`os.scandir`, early-exits on the first hit) runs *before* `load_shortcut_config()` and
`load_ticket_repo()`, because the latter needs `project_key()` → two `git` shell-outs.
Measured: a user who has never touched Shortcut pays **zero** subprocesses; putting the
gate below those loads costs them two. Everything after the gate takes what it loaded as a
parameter — `repo` into `detect_story_id`, `cfg` into `fetch_story` — so each file is read
once per command, not two or three times.

`detect_story_id()` is the tiered lookup, first hit wins, never raising: explicit id, then
this repo's `branches[<current branch>]` pin, then `sc-12345` in the branch name (Shortcut's
own convention), then the PR title/body, then recent commit subjects. `_SC_RE` requires the
`sc-` prefix on purpose — a bare number in a commit message is not a story id. The last
tier is a second `git` spawn for the weakest signal, so it is gated behind `deep=` — pr and
the `shortcut` verbs pay for it, dump/break/draft don't.

Two scopes with independent `disabled` flags, exactly like memory: `shortcut.json` (global,
also holds the token) and `shortcut/repo-<key>.json` (this repo's pins). The token is read
from `SHORTCUT_API_TOKEN` first so it can be scoped to one shell. `shortcut.json` is the
only file in `~/.focus/` holding a credential, and the only caller of `save_json(...,
mode=0o600)` — the chmod lands on the temp file *before* `os.replace`, since replace
preserves the source's mode. `focus shortcut token` reads from stdin, never argv, so a
token never enters shell history or `ps`.

**A review must be of the code as it is now.** Two ways that quietly stops being true, and
both come back as "the model is recommending fixes I already made":

1. `git_working_diff` asks `git diff HEAD` **first** — staged + unstaged. It used to ask
   `git diff --staged` first, which means the moment you stage something and keep working,
   the review is of the older file. `--staged` is a strict subset; it survives only as the
   fallback for a repo with no commits, where there is no HEAD to diff against. Don't put
   it back on top.
2. `gh pr diff` returns what is **pushed**. `unpushed_commits()` (one `git rev-list`, only
   on the PR path, 0 on any failure) puts the count in the session so `pr_markdown` and the
   dashboard can say the review is of an older branch than the one on disk.

The prompts also state diff semantics outright — `-` lines are already deleted, `+` lines
are the current code, re-read them before writing a finding — because a small model
reviewing a `-` line as if it were live produces exactly the same complaint. For the same
reason `PROJECT_HEADER` and `TICKET_HEADER` say their contents are reference material, not
work to request: a conventions list and a ticket's acceptance criteria both read as to-do
lists, and come back as findings against code that already satisfies them.

**`focus pr` can pull its own diff.** `get_diff()` tiers are `-f` → stdin → working tree →
the branch's PR. The PR sits **below** the working tree deliberately: uncommitted changes
are what you're in the middle of, and silently reviewing something else is worse than not
auto-detecting at all. A clean tree is the "I pushed, now review it" moment.

`focus pr fetch [N]` does *not* go through `get_diff` — it calls `gh_pr_diff()` directly,
exactly as the UI does, because forcing the PR means skipping every tier including `args`.
That is also why it can surface `_gh`'s reason as a `SystemExit` while the fallback tier
folds it into the "no diff found" message.

GitHub is reached only by shelling out to the user's own `gh` — focus stores no GitHub
token and speaks no GitHub API. `_gh()` is `_git()`'s sibling but returns `(stdout, reason)`
rather than collapsing failures to `""`: an explicit `focus pr fetch` has to be able to say
*why* nothing happened. Two timeouts, not one: `GH_TIMEOUT` (30s) for the asked-for fetch,
`GH_FALLBACK_TIMEOUT` (10s) for the implicit tier, which fires when the user typed nothing
but `focus pr` and must not sit there for a minute × two `gh` invocations. When a PR is
found its title and body ride ahead of the diff in the user message (`_pr_preamble`) —
cheap, high-signal context the author already wrote.

**A review is findings first, chores second.** `SYS_PR` asks for `findings`
(`severity` ∈ `SEVERITIES`, `file`, `where`, `what`) — concrete problems quoted out of
the diff — and demotes `checklist` to what a human can only check by reading *around* it.
"Zero findings is a respectable answer" is in the prompt on purpose: a model padding a
clean diff with "consider adding tests" is what made the old output feel like nothing but
a checklist. `session_findings()` normalises the reply *and* reads sessions written before
findings existed (a `risks` list of bare strings) — there is no migration layer, so the
reader converts. It also strips backticks off `file`/`where`: the prompt asks for those two
bare, and models ignore that, so the renderer would print ` ``sleep(1)`` `.

Every finding carries a `fix` (the edit to make, not the goal), and non-defects go in a
separate top-level `suggestions` list so they can't dilute the findings.

**Silence is a valid review.** All three PR prompts say outright that an empty findings
list, empty checklist and no suggestions is a complete answer, and the checklist rule
carries "there is no quota" — it used to say *one or two per changed file*, which is a
quota, and a quota gets filled. Don't reintroduce a per-file minimum anywhere. The
renderers hold up their end: `pr_markdown` omits the whole `### Checklist` section when
there is nothing to check (an empty one reads as a form the model failed to fill in), the
dashboard says `nothing to chase — no checklist` instead of `0/0 done`, and
`run_pr_review` drops checklist items whose `item` is blank.

**The author's to-do list is not review material.** `_strip_todos()` drops unticked
checkbox lines from the PR description before `_pr_preamble` puts it in front of the diff,
and keeps ticked ones minus the box — an unticked box is work not done yet, and feeding it
to a model hands the author back their own TODOs as review items. The body is also labelled
`AUTHOR'S DESCRIPTION (their words, context only — never review it, never copy from it)`,
because it is untrusted text sitting in a prompt.

**`--deep` is a pass per file.** `run_deep_review()` splits on `diff --git`, reviews each
file with `SYS_PR_FILE`, then runs one `SYS_PR_SUMMARY` pass over *the findings only* (not
the diff — that is what keeps the overview inside one request). It returns the same dict
shape as the single-pass reply, so `run_pr_review` stores it without knowing which way it
came. Depth on a local model comes from the number of passes, not from prompt wording: 60k
chars in one request gets skimmed, the same diff in seven gets read. A file's findings are
force-attributed to that file's path — the pass only saw one file, so a model naming
another is guessing — and `_dedupe_findings`/`_dedupe_checklist` merge what repeats across
passes. `MAX_DEEP_FILES` is capped at 12 and the skipped paths are *printed*, never
silently dropped.

**`pr_markdown()` is the only renderer.** `print_pr` prints it and `/api/pr`
`action: "markdown"` serves it to the dashboard's copy button, so what you read in the
terminal is byte-identical to what lands in a PR comment. Terminal-first markdown: no
tables, no reference links, nothing that only makes sense once something else renders it.
The checklist is a GitHub task list (`- [x] 3. …`) because that is also the clearest thing
to look at raw. Keep `<- you are here`, `N/M done` and the `project context:` / `ticket:`
provenance wording — the suite matches those substrings, and they are what `pr resume`
exists to show.

**A review says it is working.** `run_pr_review(progress=...)` emits `context` / `ticket` /
`diff` with a human-readable line, then `model` repeatedly with the **raw half-written
reply** — each consumer renders what it wants from it, which is why `_partial_summary()`
(a regex over an unparseable prefix, not `extract_json`) is a shared helper rather than
living in either renderer. The `model` stage fires once with `""` *before* `ask_model`, so
both renderers know the wait has started before the first token lands.

`pr_progress_printer()` is the CLI's: static lines, then a live line repainted by a ticker
thread so the silence before the first token still moves. It checks `isatty()` — piped, it
prints `asking the model…` once and never rewrites, which is also what the test suite sees
(it captures stderr into a `StringIO`). `_term_width` floors an implausible width at 80: a
pty with no winsize reports 0, which silently clipped every frame by a character.

The dashboard's is a **poll, not a stream**: `_PR_PROGRESS` is one slot (one person, one
browser) that `ui_pr_progress` writes and `/api/pr` `action: "progress"` reads, so the page
can ask on another thread while the review POST blocks. That keeps the response shape
unchanged, adds nothing to `/api/state`, and needs no long-lived socket. `pr_progress_end()`
belongs in a `finally` — a dead model must not leave the page saying a review is running.

**The dashboard has no implicit PR tier.** `/api/pr` `review` is paste → working tree →
400; pulling a PR is the separate `fetch` action behind its own button. `gh` is two
subprocesses on a server thread, and the reason CLAUDE.md already forbids `get_diff()` in
the UI is thread-blocking — a silent 20-second version of it from a button labelled
"Review changes in <repo>" would be the same mistake.

**Which repo is "here" is `active_root()`.** `_root_and_key()` resolves against it, so
`project_root()`, `project_key()`, `resolve_project_context()` and everything downstream
follow one variable. `_ACTIVE_ROOT` is `None` for the CLI — that's the whole reason CLI
behaviour is unchanged — and is set only by the dashboard's repo picker via
`set_active_root()`, which resolves a subdirectory to its toplevel, rejects non-directories
with `ValueError`, and drops the `_ROOT_CACHE` entry so re-selecting re-probes. Don't
reintroduce a bare `os.getcwd()` in that path. `git_working_diff(root)` takes the root
explicitly because it is a `subprocess` cwd, not a lookup.

`_ROOT_CACHE` memoises the two `git` shell-outs per root — `api_state()` resolves context on
every dashboard poll. Keep that poll cheap: `repo_state()` only builds the recent-repos list
when the picker asks (`recent=True`), never for `/api/state`.

**Task lifecycle goes through `complete_task()`** — it stamps `completed`, ticks
subtasks, and logs the `done` event, and it is shared by `cmd_done`, `cmd_move`,
`cmd_triage` and both UI done-paths. Bypassing it (setting `status = "done"` by hand)
silently breaks `focus today`, streaks and calibration. `cmd_start` and the UI's
move-to-now stamp `started` on first start only.

**UI duplicates CLI logic.** `UIHandler` re-implements every feature against the same
prompts and shared helpers rather than calling the `cmd_*` functions (which print and
`SystemExit`). A change to any AI feature usually needs the same edit in both places — the
tests exercise both. The dashboard is one raw string, `UI_HTML`, with inline CSS/JS and no
build step (so no `"""` inside it). `NoModelError` maps to 503, `ModelReplyError` to 502,
and everything else — including a leaked `SystemExit`, which the handler traps separately
because it is a `BaseException` that would otherwise kill the thread — to 500 so the local
server stays up. `do_POST` rejects requests whose `Host`/`Origin` isn't local
(`_origin_ok`) — a text/plain form post from any webpage reaches the handler otherwise, so
don't remove the check when adding endpoints.

Feature areas each get one multiplexed POST route dispatching on an `action` field
(`/api/pr`, `/api/triage`, `/api/memory`, `/api/voice`, `/api/project`, `/api/repo`,
`/api/shortcut`), mirroring the CLI's verbs; panel bodies lazy-load via `action: "show"` on
open, and nothing heavy (pr sessions, profile text, `repo_brief`, the repo picker's
directory listing, `shortcut_state()`) is ever added to the 5-second `/api/state` poll.
The ticket badge in particular stays off the poll: resolving one needs the current branch,
which changes without the repo root changing and so can't ride `_ROOT_CACHE`.

`/api/shortcut` accepts an API token, which is the strongest reason `_origin_ok` covers
every POST — and `shortcut_state()` reports `has_token` but never returns the token itself.

`/api/repo` (`show` / `browse` / `use`) is the picker, and the only writer of `_ACTIVE_ROOT`.
It is a POST route so `_origin_ok` covers it — a filesystem listing must not be reachable by
a cross-origin form post. `browse` is a `os.listdir` plus one `.git` stat per entry, never a
shell-out. After `use`, the page reloads the project and memory panels if they're open;
both describe the repo and go stale otherwise.
The UI must never call `get_diff()` — its stdin tier blocks a server thread forever;
`git_working_diff()` and `gh_pr_diff()` are the UI-safe extractions, just as
`run_pr_review()`, `pin_story()`, `set_ticket_enabled()`, `clear_story_cache()` and
`_latest_session_path()` are the print-free, SystemExit-free cores shared with the `cmd_*`
functions. When a UI branch and its CLI verb would otherwise write the same state two ways,
add the core rather than the second copy — the duplication CLAUDE.md sanctions is command
*flow*, not bookkeeping.

## Constraints

**Nothing you write ever leaves the machine.** Diffs, drafts, voice samples, memory facts
and harvested repo context go to the local model and nowhere else, the UI binds
`127.0.0.1`, and there is no telemetry and no third-party dependency. That is the premise,
and it is the thing to protect.

Two features read from elsewhere, and the rule is precise: **outbound reads only, to
exactly two places.**

1. `gh` (a binary the user already installed and authenticated) for the current branch's
   PR. focus stores no GitHub credential and constructs no GitHub URL — everything goes
   through `_gh()`.
2. `api.app.shortcut.com` for the ticket you're on, via `http_json` with a
   `Shortcut-Token` header, from `fetch_story()` only.

Both are `GET`-shaped: focus never posts, comments, updates or deletes anything on either
service. Nothing from `~/.focus/` — no diff, no draft, no memory fact — is ever included in
an outbound request; the Shortcut request carries a story id and the token, and that is all.
A new remote host, a write to a remote service, or any outbound call carrying user content
is a change to the product's promise, not a feature — it needs the README's Privacy section
rewritten alongside it.

`repo_brief()` still only shells out to offline git subcommands (`rev-parse`,
`config --get`, `ls-files`, `log`). Note `tomllib` is 3.11+ and the PEP 723 header says
`>=3.9`, so manifests are read as text, never parsed with it.
