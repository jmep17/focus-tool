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
focus pr                 # your staged/unstaged changes — or, if the tree is
                         # clean, this branch's own PR, pulled for you
focus pr fetch           # always this branch's PR
focus pr fetch 4600      # ...or that one
focus pr resume          # after ANY interruption: shows "<- you are here"
focus pr check 3         # tick item 3 as reviewed
```

The model writes a 3-sentence summary, flags where to look hardest, and builds
a file-by-file checklist. Your position is saved on disk, so a meeting or a
Slack ping can't wipe your mental state — the checklist *is* the state.

You don't have to find the diff yourself. With nothing staged, `focus pr` asks
your own `gh` for the pull request on the branch you're standing on, names the
session after it (`pr-4521`), and puts the PR's title and description in front of
the diff. Uncommitted changes still win — what you're in the middle of is what
you meant. `-f file.diff` and piping still work exactly as before, and if `gh`
isn't installed or the branch has no PR, focus says which.

Nothing is stored on your behalf: focus never sees a GitHub token, it just runs
the CLI you already logged into, and only ever reads.

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
focus memory add "never add a third-party dependency" --here   # this repo only
focus memory show      # everything the AI is told about you, both scopes
focus memory edit      # rewrite the facts in $EDITOR
focus memory off       # stop injecting (facts kept); --no-memory skips one command
focus memory off --here                    # …just this repo's facts
focus memory add "…" --project acme-api    # another repo's, by profile name
```

Three kinds of memory reach the model. **Facts you add** — standing truths about how
you work. **Patterns focus derives on its own** from `~/.focus/history.jsonl`, the
local event log every command appends to: once three finished tasks have both an
estimate and an actual, it computes how far off your guesses run and tells the model
to size estimates accordingly ("real tasks take ~2.1x the guess"). Derivation is
deterministic — no model round-trip, so it works even with no server running. And
**facts about one repo**, added with `--here`: the conventions you keep re-explaining,
which reach the model only while you're in that repo, under an `In this project (…)`
line. The two scopes are separate files with separate off switches — turning global
memory off doesn't silence a repo's own facts, and `--no-project` drops the repo's
facts along with its profile.

It all goes in as one short `USER MEMORY` suffix on `dump`, `break` and `draft`
(not `pr`, whose context budget belongs to the diff), capped so repo facts can't be
crowded out by a long global list. Like everything else, the log and the facts are
plain JSON on your machine.

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
   you're sitting in a different directory (in the dashboard, point the repo bar at
   the folder instead)
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
PR REVIEW: pr-4521  (from PR #4521)
  #4521 Retry payments on transient errors
  https://github.com/you/repo/pull/4521
  project context: github.com_you_repo (2143 chars)
  ticket: sc-12345 Retry payments on transient errors (via branch name, 986 chars)
```

Everything harvested — your docs, your file tree, your commit history — stays on
the machine, same as diffs and drafts.

### "The AI doesn't know what I was asked to build" → `focus shortcut`

```bash
focus shortcut token          # paste your Shortcut API token (stdin, not argv)
focus shortcut use 12345      # pin that story to the branch you're on
focus shortcut                # what the AI is being told, and how it found it
focus shortcut fetch          # refresh it
focus shortcut ls             # what's cached
focus shortcut off [--here]   # stop injecting, everywhere or just this repo
focus pr --no-ticket          # skip it for one command (dump/break/draft too)
```

Your repo profile tells the model how the code works. Your memory tells it how
*you* work. Neither tells it **what you were asked to do** — so a review can only
ask "is this code correct", never "does this actually do what the ticket said".

Point focus at a Shortcut story and the answer changes. The story's description,
its acceptance criteria and its task list go into `focus pr` — which now flags the
criterion your diff quietly doesn't meet — and into `dump`, `break` and `draft`,
so a breakdown is the acceptance criteria rather than your best guess at them.

**Usually you don't tell it anything.** focus finds the story itself, first hit
wins: a story you pinned to this branch, then `sc-12345` in the branch name
(Shortcut's own convention, so this is free if you use its branch button), then
the PR title or description, then your recent commit messages.

**It reads the cache, not the network.** `focus dump` never makes a request — it
uses `~/.focus/shortcut/story-<id>.json`, so it stays instant and works on a
train. Only `focus pr` and the explicit `use`/`fetch` verbs refresh, and if
Shortcut is unreachable they fall back to the cached copy rather than failing.

Your token lives in `~/.focus/shortcut.json`, mode 600, and is sent to Shortcut
and nowhere else — never to the model, never into a session file. `SHORTCUT_API_TOKEN`
works too if you'd rather not put it on disk. `focus shortcut clear` forgets the
lot. The request going out contains a story id and your token; nothing from your
machine rides along.

### "I want to see it" → `focus ui`

```bash
focus ui
```

Opens a local dashboard (127.0.0.1 only): a big **DO THIS ONE THING** banner,
a 25-minute timer that survives page reloads, a **Done today** strip with your
streak, an Inbox/Now/Next/Later board with tick-able subtasks, breadcrumb notes
and age badges on stale cards, a brain-dump box and a message drafter. The API
rejects requests from other origins, so a random webpage can't poke your tasks.

**It follows whichever repo you're in.** The bar at the top shows the current one;
**change** opens a folder picker — browse, paste a path, or click one you've used
before, with git repos flagged. Everything repo-shaped follows the choice: which
diff **Review changes in …** picks up, which project profile is appended to every
prompt, which ticket the **Ticket** panel resolves, and which project memory the
Memory panel edits. The choice is remembered,
so the dashboard reopens where you left it — the CLI is unaffected and stays
wherever you're `cd`'d to. Browsing never leaves the machine; it's the same local
server that was already reading your repo.

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
2. Starting a ticket: `focus shortcut use 12345`, then `focus break` it.
3. Before a PR: `focus pr`, read the summary, work the checklist.
4. After every interruption: `focus pr resume` or `focus next` — the tool
   remembers so you don't have to. Leaving your desk? `focus note` first.
5. Any message that's been sitting unsent for 10+ minutes: `focus draft`.
6. End of day: `focus today` — look at what you actually did.
7. Once a week or so: `focus triage`.

## Testing / hacking

```bash
python3 tests/test_focus.py   # 233 end-to-end checks against local fakes
```

No network, no real model, no GitHub and no Shortcut account: `tests/mock_llm.py`
and `tests/mock_shortcut.py` are loopback servers, and `tests/fake_gh.py` stands in
for the `gh` binary on `PATH`.

Single file, MIT-style — change anything. Prompts live near the top of `focus.py`
(`SYS_DUMP`, `SYS_BREAK`, `SYS_PR`, `SYS_DRAFT`, `SYS_VOICE`, `SYS_PROJECT`) — tune
them to your taste; that's half the fun.

## Privacy

**Nothing you write ever leaves the machine.** Diffs, drafts, notes, memory facts,
voice samples, and everything `focus project` harvests from your repo (docs,
manifests, file tree, commit history) go to your local model server on
`127.0.0.1` and nowhere else. On a work machine, that's the whole point.

Two features *read* from elsewhere, and it's worth being exact about them:

- **`focus pr`** runs your own `gh` CLI to fetch the branch's pull request. focus
  never sees a GitHub token — it shells out to the tool you already logged into,
  and only reads. Skip it entirely by piping a diff or using `-f`.
- **`focus shortcut`** fetches the story you're working on from
  `api.app.shortcut.com`. The request carries a story id and your token; nothing
  from your machine goes with it. The token lives in `~/.focus/shortcut.json` at
  mode 600, is sent to Shortcut only, and is never given to the model. Don't
  configure a token and this never runs.

Neither ever writes: focus does not comment, update, close or push anything. And
neither is on the path of the everyday commands — `dump`, `break` and `draft` read
a local cache, so they work with the network off.

- The UI binds to `127.0.0.1` and refuses cross-origin requests. Timer
  notifications use local `osascript`.
- Data: plain JSON under `~/.focus/` — tasks, the `history.jsonl` event log,
  memory facts (global and per-repo), voice and project profiles, cached tickets,
  and which repo the dashboard was last pointed at. Delete the folder, it's gone.
- The dashboard's folder picker lists directories on your machine to a page served
  from `127.0.0.1`, and nowhere else. Nothing is uploaded, indexed or phoned home.
