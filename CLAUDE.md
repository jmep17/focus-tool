# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python3 tests/test_focus.py           # the whole test suite (~44 checks, no pytest)
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
`mock_llm` is started *before* `import focus`, so never reorder those first ~20 lines.

## Architecture

Everything is `focus.py`: CLI, storage, LLM client, HTTP server, and the dashboard HTML.
`pyproject.toml` exposes it as `focus = "focus:main"`. Keep it stdlib-only — the PEP 723
header declares `dependencies = []`, which is what makes `uv run --script` and the bare
`chmod +x focus.py` install path work.

**Storage** — plain JSON under `~/.focus/`, overridable with `FOCUS_HOME` (tests rely on
this): `tasks.json` (`{next_id, tasks[]}`), `config.json` (`{endpoint, model}`),
`voice.json`, and `pr/<name>.json` per review session. All writes go through `save_json`,
which writes `.tmp` then `os.replace` for atomicity. `load_json` swallows missing/corrupt
files and returns the default — there is no migration layer, so any new task field must be
read with `.get()`.

**LLM layer** — `detect_runtime()` probes `/models` on the configured endpoint, then LM
Studio (`:1234`), then Ollama (`:11434`), and picks the first model id it sees unless
`config.json` names one. `chat()` posts an OpenAI-shaped `/chat/completions`.
`extract_json()` is a brace-depth scanner that survives markdown fences, preamble chatter,
and braces inside strings — small local models produce all three, so use it rather than
`json.loads` on any model reply. Failure to find a server raises `NoModelError`, which
`main()` turns into exit code 2; only `dump` degrades gracefully (one line = one task).

**Prompts are the product.** `SYS_DUMP`, `SYS_BREAK`, `SYS_PR`, `SYS_VOICE` each end with a
literal JSON schema, and the parsing code downstream assumes exactly those keys. Editing a
prompt is a three-place change:

1. the `SYS_*` constant,
2. the consumer (e.g. `cmd_break` reads `data["subtasks"][].text/estimate_min`),
3. `tests/mock_llm.py` — `canned_reply()` dispatches on **substrings of the system prompt**
   (`"brain-dump"`, `"break a task"`, `"unified diff"`, `"writing-voice profile"`,
   `"write work messages"`). Rewording a prompt past one of those substrings makes the mock
   fall through to `"ok"` and tests fail far from the real cause.

**`pick_next` is deliberately AI-free.** Sorting by (status order, priority, remaining
estimate, created) — with `--energy low` swapping priority and size — is the product
decision, not a shortcut. Don't "improve" it by asking the model.

**Voice profile** — `voice_system_suffix()` appends `voice.json`'s profile to the draft
system prompt, and returns `""` when disabled or absent. It is invoked from both
`cmd_draft` and the UI's `/api/draft`; `focus voice learn` keeps the freshest 10 samples and
re-derives the whole profile from scratch each time.

**UI duplicates CLI logic.** `UIHandler` re-implements break/dump/draft against the same
prompts rather than calling the `cmd_*` functions (which print and `SystemExit`). A change to
any AI feature usually needs the same edit in both places — the tests exercise both. The
dashboard is one raw string, `UI_HTML`, with inline CSS/JS and no build step. `NoModelError`
maps to 503 and everything else to 500 so the local server stays up.

## Constraints

Network calls only ever go to `127.0.0.1`, and the UI binds `127.0.0.1`. The whole premise
is that diffs, drafts and voice samples never leave the machine — don't add a remote
endpoint, telemetry, or a third-party dependency.
