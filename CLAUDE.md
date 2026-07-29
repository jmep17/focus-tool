# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python3 tests/test_focus.py           # the whole test suite (~114 checks, no pytest)
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

## Architecture

Everything is `focus.py`: CLI, storage, LLM client, HTTP server, and the dashboard HTML.
`pyproject.toml` exposes it as `focus = "focus:main"`. Keep it stdlib-only — the PEP 723
header declares `dependencies = []`, which is what makes `uv run --script` and the bare
`chmod +x focus.py` install path work.

**Storage** — plain JSON under `~/.focus/`, overridable with `FOCUS_HOME` (tests rely on
this): `tasks.json` (`{next_id, tasks[]}`), `config.json` (`{endpoint, model}`),
`voice.json`, `memory.json` (`{facts[], disabled?}`), `projects/<key>.json` per repo, and
`pr/<name>.json` per review session. All writes go through `save_json`,
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

Every AI call site goes through `ask_model()` — chat + extract_json with one retry, and
any garbage reply or HTTP failure surfaced as `ModelReplyError` (exit 2 in the CLI, 502
in the UI; `NoModelError` stays 503). Don't call `chat()`/`extract_json()` directly from
a command: the traceback-on-flaky-model failure mode is exactly what `ask_model` exists
to prevent. `raw=True` returns the reply text for non-JSON prompts (`draft`).

**Prompts are the product.** `SYS_DUMP`, `SYS_BREAK`, `SYS_PR`, `SYS_VOICE`, `SYS_PROJECT`
each end with a literal JSON schema, and the parsing code downstream assumes exactly those
keys. Editing a prompt is a three-place change:

1. the `SYS_*` constant,
2. the consumer (e.g. `cmd_break` reads `data["subtasks"][].text/estimate_min`),
3. `tests/mock_llm.py` — `canned_reply()` dispatches on **substrings of the system prompt**
   (`"brain-dump"`, `"break a task"`, `"unified diff"`, `"writing-voice profile"`,
   `"write work messages"`, `"compact project profile"`). Rewording a prompt past one of
   those substrings makes the mock fall through to `"ok"` and tests fail far from the real
   cause.

The mock dispatches on the prompt **up to the first suffix marker** (`MATCH THIS PERSON'S
VOICE`, `PROJECT CONTEXT`, `USER MEMORY`), not the whole string. It has to: everything
after a marker is arbitrary repo text, and this repo's own `CLAUDE.md` contains the
literal string `"brain-dump"` — dispatching on the full prompt answers a PR request with
a task list. If you add another suffix, add its marker to that split.

**`focus pr` takes verbs, not flags** — `pr` / `pr resume` / `pr check N`, matching `voice`
and `project`. The pre-verb `--resume` / `--check N` / `--no-project` spellings still work
but are `argparse.SUPPRESS`ed out of `--help`; `_pr_action()` normalises both forms in one
place so `cmd_pr` doesn't branch on flags. `--project none` is the documented way to skip
context for a review. Tests cover both spellings — don't delete the legacy ones without
deleting their checks.

**`pick_next` is deliberately AI-free.** Sorting by (status order, priority, remaining
estimate, created) — with `--energy low` swapping priority and size — is the product
decision, not a shortcut. Don't "improve" it by asking the model.

**Voice profile** — `voice_system_suffix()` appends `voice.json`'s profile to the draft
system prompt, and returns `""` when disabled or absent. It is invoked from both
`cmd_draft` and the UI's `/api/draft`; `focus voice learn` keeps the freshest 10 samples and
re-derives the whole profile from scratch each time.

**User memory** — `memory_system_suffix()` follows the same contract (`""` when off or
empty) and injects two things under the `USER MEMORY` marker: explicit `focus memory add`
facts, and stats `memory_stats_lines()` derives *deterministically* from `history.jsonl`
(median estimate-vs-actual ratio after 3 samples, finishing streak). No model round-trip
in the derivation — that's the point; it works with no server running. Injected into
dump/break/draft and their UI twins only, never `pr`: the pr budget belongs to the diff.
`actual_min` is capped at 8h in `actual_minutes()` so a task started Monday and finished
Friday doesn't poison calibration.

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

Context is charged **against** `MAX_DIFF_CHARS`, not added to it (`cmd_pr` shrinks the diff
budget by `len(ctx)`). An 8k-context model is already over budget on diff alone; anything
that grows the request is a regression. `_clip()` exists so a heading-less doc or one
oversized section can't blow the budget and silently eat the diff's share.

`_ROOT_CACHE` memoises the two `git` shell-outs per cwd — `api_state()` resolves context on
every dashboard poll.

**Task lifecycle goes through `complete_task()`** — it stamps `completed`, ticks
subtasks, and logs the `done` event, and it is shared by `cmd_done`, `cmd_move`,
`cmd_triage` and both UI done-paths. Bypassing it (setting `status = "done"` by hand)
silently breaks `focus today`, streaks and calibration. `cmd_start` and the UI's
move-to-now stamp `started` on first start only.

**UI duplicates CLI logic.** `UIHandler` re-implements break/dump/draft against the same
prompts rather than calling the `cmd_*` functions (which print and `SystemExit`). A change to
any AI feature usually needs the same edit in both places — the tests exercise both. The
dashboard is one raw string, `UI_HTML`, with inline CSS/JS and no build step. `NoModelError`
maps to 503, `ModelReplyError` to 502, and everything else to 500 so the local server
stays up. `do_POST` rejects requests whose `Host`/`Origin` isn't local (`_origin_ok`) —
a text/plain form post from any webpage reaches the handler otherwise, so don't remove
the check when adding endpoints.

## Constraints

Network calls only ever go to `127.0.0.1`, and the UI binds `127.0.0.1`. The whole premise
is that diffs, drafts, voice samples and harvested repo context never leave the machine —
don't add a remote endpoint, telemetry, or a third-party dependency. `repo_brief()` only
shells out to offline git subcommands (`rev-parse`, `config --get`, `ls-files`, `log`).
Note `tomllib` is 3.11+ and the PEP 723 header says `>=3.9`, so manifests are read as text,
never parsed with it.
