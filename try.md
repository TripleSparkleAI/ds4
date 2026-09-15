# try.sh - the DwarfStar arm runner

`try.sh` runs on the M5 and drives a `ds4-server` on the Spark over ssh.
It loads the model, runs TEN fixed tests (five short prompts, five long
prompts - the same ten every run), prints tokens/second for each with a
min/median/max summary, writes a markdown-table result file, and unloads
the model on every way out: trap, 30 s idle, or a watchdog ON the Spark
itself if this machine dies mid-run.

POSIX sh.  No python, no jq - ssh, curl and the ordinary text tools.

## run it

    ./try.sh                       # menu; defaults to triple-all-fastest
    ./try.sh --list                # branches and what is built, no run
    ./try.sh --branch triple-pool  # a different branch from the fork
    ./try.sh --arm stack           # the branch as it ships (default)
    ./try.sh --arm all-off         # the control: every off switch, off
    ./try.sh --arm all-on          # every lever forced on, opt-ins included
    ./try.sh --arm stack --off hits-first
    ./try.sh --dry-run             # plan only: branch, build, env, filename

## branches

The menu is discovered live with `git ls-remote --heads sparkle`: every
branch starting with `triple-` appears, so a branch pushed tomorrow needs
no change here.  Everything is keyed on the branch slug.

Which build directory runs a branch, in order: `--dir`, then `try-arms.txt`
(`<branch-slug>  <build-dir>` lines), then a worktree named after the
branch, then a worktree checked out on it, then a labelled best guess.
A guess says it is a guess and can be pinned with `--dir`.

## the levers and their switches (verified in the source)

| lever | switch | default | off |
| --- | --- | --- | --- |
| pool | `DS4_CUDA_STREAMING_EXPERT_PREAD_POOL` | on | `=0` |
| hotlist | `DS4_CUDA_EXPERT_HOTLIST_WRITE` | on | `=0` |
| hits-first | `DS4_CUDA_HITS_FIRST` | OFF | (default) |
| pagecache | `DS4_CUDA_KEEP_MODEL_PAGES` | new order | `=1` restores old |
| engram-lead | `DS4_V41_ENGRAM_LEAD_OFF` | on | set anything |
| draincut | `DS4_CUDA_SELECTED_DRAIN_SYNC` | ON | `=1` restores the blocking read |
| margin | `DS4_CUDA_EXPERT_CACHE_MARGIN_GB` | 8 GiB | knob, no Boolean |
| prefetch-pool | `DS4_CUDA_SSD_PREFETCH_CHUNK_MB` | 8 MB | **no off, `0` is ignored** (code not in this tree) |
| readahead-order | `DS4_CUDA_SSD_PREFETCH_STATS` | on | **stats only - the reorder has no off** |
| engram-read-threads | `DS4_ENGRAM_READ_THREADS` | request count | **no off** |

Six of the ten have a real off switch, which is what lets one binary serve
several arms.  `all-off` warns loudly that margin, prefetch-pool,
readahead-order and engram-read-threads cannot be switched off by
environment, so that control is NOT clean for them - those arms still need
a build without the code.

## results

`try-results/<date>-<branch>-<arm>-<N>on[-<lever>-off|-<lever>-on].txt`,
for example `try-results/2026-09-15-triple-all-fastest-stack-8on-hits-off.txt`.
The file is plain text with markdown tables, so a row can be pasted
straight into a README table.  Every run also appends one line to
`try-results/INDEX.md`: date, branch, what was ON, and the filename.

## unload safety

- a trap on EXIT/INT/TERM/HUP kills the server,
- a local watchdog unloads after `IDLE_SECS` (default 30) with no test in
  flight,
- a watchdog ON the Spark unloads if this machine dies or the shell is
  killed hard: a heartbeat keeps a beat file fresh for as long as the
  harness lives, and when the heart stops the Spark cleans up after itself.
