# focus — a local-first ADHD companion for developers

Everything runs on your Mac. AI features use your **local model** (LM Studio or
Ollama) over localhost. No accounts, no cloud, no telemetry. Plain-JSON storage
in `~/.focus/` that you can read, grep and back up.

---

## Install with uv (1 minute)

```bash
# 1. From the unzipped folder — installs a `focus` command onto your PATH:
uv tool install --editable .

# If `focus` isn't found afterwards, let uv fix your PATH and reopen the terminal:
uv tool update-shell

# 2. Start your local model (either):
#    LM Studio  ->  load a model, Developer tab -> Start Server (port 1234)
#    Ollama     ->  ollama serve  (port 11434), with a model pulled

# 3. From ANY directory:
focus doctor
```

`--editable` means the installed command runs this folder's `focus.py` directly —
edit the file (or its prompts) and the change is live immediately, no reinstall.
Keep the folder somewhere permanent (e.g. `~/tools/focus-tool`).
Uninstall any time: `uv tool uninstall focus-cli`.

No uv? `brew install uv` first — or skip uv entirely: the script is
stdlib-only, so `chmod +x focus.py && ln -s "$(pwd)/focus.py" /usr/local/bin/focus`
still works, as does `uv run --script focus.py` for a one-off.
Prefer a specific endpoint or model? Create `~/.focus/config.json`:

```json
{ "endpoint": "http://127.0.0.1:1234/v1", "model": "llama-3.1-8b-instruct" }
```

---

## The 60-second tour — mapped to why each exists

### "Everything is in my head and it's too loud" → `focus dump`

```bash
focus dump   # then type/paste everything, messy is fine, Ctrl-D to finish
```

The model turns your ramble into clean, prioritised tasks in an inbox.
Capture is the whole point — get it out of your head *fast*.
(No model running? It still works: one line = one task.)

### "This task is too big to start" → `focus break`

```bash
focus break 4
```

Splits task #4 into 3–8 steps of **25 minutes or less**, and the first step is
deliberately tiny (open the file, run the command) so starting costs nothing.

### "I don't know what to do first" → `focus next`

```bash
focus next            # exactly ONE action, nothing else
focus next --energy low   # tired? easiest thing instead
focus next --why      # show the reasoning
```

One thing. Not a list. The choice is made deterministically (status, priority,
size, age) so you never burn decision energy on *choosing* — the ADHD tax is
the choosing, not the doing.

### "I did stuff all day and feel like I did nothing" → `focus today`

```bash
focus today            # what you actually finished today
focus today --week     # the last 7 days
```

Every completed task, with `est ~15m · took ~40m` when focus knows both, plus your
finishing streak. Visible progress is fuel — and the estimate feedback quietly teaches
you what your "15 minutes" really means. `focus done` shows the same calibration line
the moment you finish something.

### "What was I even doing?" → `focus note`

```bash
focus note 4 "left off mid-refactor in parse_config, tests failing"
focus note 4            # read the trail back
```

A timestamped breadcrumb on any task. `focus next` and `focus start` show the latest
one back to you, so an interruption costs a sentence, not your whole mental state —
the same trick `focus pr resume` plays, for every task.

### "My list is full of guilt" → `focus triage`

```bash
focus triage
```

Anything in inbox/later untouched for 14+ days comes up **one at a time**:
keep / later / done / drop. Dropping a task you were never going to do is a
decision, not a failure. Stale tasks also show their age in `focus ls`
(`[med · 21d old]`).

### "I sat down and it's somehow 6pm" → `focus timer`

```bash
focus timer          # 25 minutes, desktop notification when it's up
focus timer 10       # any length
```

A countdown in your terminal, a macOS notification at zero (local `osascript` —
nothing leaves the machine), and the session is logged so `focus today` knows you
showed up. The dashboard timer survives a page reload and can fire a browser
notification too.

### "I lose my place reviewing PRs every time I'm interrupted" → `focus pr`

```bash
gh pr diff 4521 | focus pr --name pr-4521    # or: git diff HEAD | focus pr
focus pr resume          # after ANY interruption: shows "<- you are here"
focus pr check 3         # tick item 3 as reviewed
```

The model writes a 3-sentence summary, flags where to look hardest, and builds
a file-by-file checklist. Your position is saved on disk, so a meeting or a
Slack ping can't wipe your mental state — the checklist *is* the state.

### "I stare at empty message boxes" → `focus draft`

```bash
focus draft "can't make thursday, can do tue" --type slack --tone friendly
focus draft --type email --tone firm      # reads bullets from stdin
focus draft --polish < my-rough-draft.txt # cleans up something you wrote
```

Bullets in, ready-to-send message out. Types: `slack`, `email`, `pr-comment`,
`standup`. Nothing you write ever leaves the machine.

### "The drafts don't sound like me" → `focus voice`

```bash
focus voice            # 8-question interview + paste a few real messages
focus voice show       # see the profile every draft now uses
focus voice learn      # paste another real message; profile re-learns
focus voice edit       # tweak the profile text directly in $EDITOR
focus voice off        # back to default style
focus draft ... --no-voice   # skip your voice for one draft
```

The interview takes two minutes (greetings, sign-offs, emoji habits, formality,
pet phrases, banned corporate speak). Pasting 1–3 real messages matters more
than the interview — where they disagree, the samples win. Your local model
distils it all into a compact profile in `~/.focus/voice.json`, which is
appended to every draft prompt. Interview answers and samples never leave
your machine.

Best loop: when a draft sounds off, fix it by hand, send it, then
`focus voice learn` and paste what you actually sent. It converges fast.

### "The AI doesn't know *me*" → `focus memory`

```bash
focus memory add "I underestimate anything involving other people's code"
focus memory show      # everything the AI is told about you
focus memory edit      # rewrite the facts in $EDITOR
focus memory off       # stop injecting (facts kept); --no-memory skips one command
```

Two kinds of memory reach the model. **Facts you add** — standing truths about how
you work. **Patterns focus derives on its own** from `~/.focus/history.jsonl`, the
local event log every command appends to: once three finished tasks have both an
estimate and an actual, it computes how far off your guesses run and tells the model
to size estimates accordingly ("real tasks take ~2.1x the guess"). Derivation is
deterministic — no model round-trip, so it works even with no server running, and
it's injected as a short `USER MEMORY` suffix into `dump`, `break` and `draft`
(not `pr`, whose context budget belongs to the diff). Like everything else, the log
and the facts are plain JSON on your machine.

### "The advice doesn't know my codebase" → `focus project`

```bash
focus project          # read this repo, build a profile of how it works
focus project show     # see what every AI command is being told
focus project edit     # correct it by hand in $EDITOR
focus project ls       # every repo you've profiled
focus project off      # drop the profile for this repo
focus pr --project none       # ignore context for one review
focus break 4 --no-project    # same, for dump / break / draft
```

Generic review advice ("check error handling") is worthless. Advice that knows
*this* repo — that writes go through `save_json`, that `payments/` must carry an
idempotency key, that the auth module is legacy — is the reason to review at all.

`focus project` reads what your repo already has: the first of `CLAUDE.md`,
`AGENTS.md`, `.cursorrules`, `CONTRIBUTING.md`, `ARCHITECTURE.md` or `README.md`,
plus your manifests (`pyproject.toml`, `package.json`, `go.mod`, …), the directory
tree, and which files churn most in the last 200 commits. Your local model distils
that into a ≤400-word markdown profile in `~/.focus/projects/<repo>.json`, which is
appended to the prompt of every AI command.

**It works before you set anything up.** With no profile stored, focus reads your
repo's own `CLAUDE.md` (or whichever doc it finds) live. `focus project` just
distils that into something smaller and sharper. Resolution order:

1. `--project NAME` — an explicit profile, for when the diff arrives on stdin and
   you're sitting in a different directory
2. `~/.focus/projects/<repo>.json` — what `focus project` built
3. `.focus/project.md` **committed in your repo** — the way to share one reviewed
   context file with your team; focus reads it and never writes there
4. the repo's own `CLAUDE.md` / `AGENTS.md` / `README.md`, live

**Only the relevant parts are sent.** A local model's context window is small, and
a diff already eats most of it. So focus parses the changed paths out of the diff,
scores each `## ` section of the profile against them, and sends the sections that
match plus the always-on ones (`## Stack`, `## Conventions`, `## Testing`) up to a
4,000-character budget — charged *against* the diff's budget, never added on top.
Change `payments/client.py` and you get the payments section, not the frontend one.

This is deliberately lexical, not embeddings: `/v1/embeddings` isn't reliably
available on a local server, an index goes stale, and ranking ten sections of a
4kB document doesn't need vectors. `focus pr` tells you what it used, so the advice
is never coming from somewhere you can't see:

```
PR REVIEW: pr-4521  (from stdin)
  project context: github.com_you_repo (2143 chars)
```

Everything harvested — your docs, your file tree, your commit history — stays on
the machine, same as diffs and drafts.

### "I want to see it" → `focus ui`

```bash
focus ui
```

Opens a local dashboard (127.0.0.1 only): a big **DO THIS ONE THING** banner,
a 25-minute timer that survives page reloads, a **Done today** strip with your
streak, an Inbox/Now/Next/Later board with tick-able subtasks, breadcrumb notes
and age badges on stale cards, a brain-dump box and a message drafter. The API
rejects requests from other origins, so a random webpage can't poke your tasks.

### Everyday plumbing

```bash
focus add "book desk for thursday" -p 3   # quick add, no AI
focus add "write rota doc" -e 20          # ...with an estimate (feeds calibration)
focus ls -v          # list everything with subtasks
focus start 4        # move to Now (stamps the start time, shows your last note)
focus done 4.2       # tick subtask 2  (ticking the last one completes the task)
focus done 4         # complete the whole task
focus move 4 later   # inbox / now / next / later
```

---

## Suggested daily shape

1. Morning: `focus dump` the noise, `focus next`, `focus timer`.
2. Before a PR: `focus pr`, read the summary, work the checklist.
3. After every interruption: `focus pr resume` or `focus next` — the tool
   remembers so you don't have to. Leaving your desk? `focus note` first.
4. Any message that's been sitting unsent for 10+ minutes: `focus draft`.
5. End of day: `focus today` — look at what you actually did.
6. Once a week or so: `focus triage`.

## Testing / hacking

```bash
python3 tests/test_focus.py   # 114 end-to-end checks against a mock model
```

Single file, MIT-style — change anything. Prompts live near the top of `focus.py`
(`SYS_DUMP`, `SYS_BREAK`, `SYS_PR`, `SYS_DRAFT`, `SYS_VOICE`, `SYS_PROJECT`) — tune
them to your taste; that's half the fun.

## Privacy

- Talks only to `127.0.0.1` (your model server). The UI binds to `127.0.0.1`
  and refuses cross-origin requests. Timer notifications use local `osascript`.
- Data: plain JSON under `~/.focus/` — tasks, the `history.jsonl` event log,
  memory facts, voice and project profiles. Delete the folder, it's gone.
- Diffs, drafts, memory facts, and anything `focus project` harvests from your
  repo (docs, manifests, file tree, commit history) are sent to your local model
  only. On a work machine, that's the whole point.
