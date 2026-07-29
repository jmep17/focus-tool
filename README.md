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

### "I lose my place reviewing PRs every time I'm interrupted" → `focus pr`

```bash
gh pr diff 4521 | focus pr --name pr-4521    # or: git diff HEAD | focus pr
focus pr --resume        # after ANY interruption: shows "<- you are here"
focus pr --check 3       # tick item 3 as reviewed
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

### "I want to see it" → `focus ui`

```bash
focus ui
```

Opens a local dashboard (127.0.0.1 only): a big **DO THIS ONE THING** banner,
a 25-minute timer, an Inbox/Now/Next/Later board with tick-able subtasks, a
brain-dump box and a message drafter.

### Everyday plumbing

```bash
focus add "book desk for thursday" -p 3   # quick add, no AI
focus ls -v          # list everything with subtasks
focus start 4        # move to Now
focus done 4.2       # tick subtask 2  (ticking the last one completes the task)
focus done 4         # complete the whole task
focus move 4 later   # inbox / now / next / later
```

---

## Suggested daily shape

1. Morning: `focus dump` the noise, `focus next`, start the timer.
2. Before a PR: `focus pr`, read the summary, work the checklist.
3. After every interruption: `focus pr --resume` or `focus next` — the tool
   remembers so you don't have to.
4. Any message that's been sitting unsent for 10+ minutes: `focus draft`.

## Testing / hacking

```bash
python3 tests/test_focus.py   # 44 end-to-end checks against a mock model
```

Single file, ~900 lines, MIT-style — change anything. Prompts live near the
top of `focus.py` (`SYS_DUMP`, `SYS_BREAK`, `SYS_PR`, `SYS_DRAFT`) — tune them
to your taste; that's half the fun.

## Privacy

- Talks only to `127.0.0.1` (your model server). The UI binds to `127.0.0.1`.
- Data: plain JSON under `~/.focus/`. Delete the folder, it's gone.
- Diffs and drafts are sent to your local model only. On a work machine,
  that's the whole point.
