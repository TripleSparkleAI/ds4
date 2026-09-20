# ARTIFACTS - the local working artifacts, one named folder per kind

> **Every large file in here is GITIGNORED. Every small file in here is TRACKED.** The rule is
> `../.gitignore`, and it ignores **by extension**, never by wildcarding a directory, so a README,
> an INDEX, a manifest, a checksum file or a small log inside an artifact directory stays visible
> to git.

The navigator's rule, 2026-09-20: *"our trained weights and all things checkpoints all things we'll
keep in good named folders and structure in this root folder"* and *"the final weights we make too
all gitignored, the large files only."*

## ⛔ THE SKELETON IS TRACKED ON PURPOSE

A gitignored tree **clones as nothing**, and then the next person rebuilds a different structure and
two layouts exist. So every directory below carries a **tracked README.md** and a **tracked
INDEX.md**, and those files are what survive a fresh clone. ★ **If you add a directory here, give it
both files in the same commit.**

## ⛔⛔ AND THE INDEX IS THE ONLY RECORD THERE EVER WAS

The artifacts are invisible to git. **So an artifact with no INDEX row did not happen** - six weeks
from now nobody knows what was ever in this tree. Every row carries:

| field | why |
|---|---|
| **full name** | descriptive, per the naming law below. `run3.gguf` tells a reader nothing. |
| **exact bytes** | the only size that is a fact. Not GB, not "about". |
| **sha256** | so a re-download is verifiable and bit rot is detectable |
| **came from** | the source artifact or URL, and the commit that produced it |
| **date** | when it landed |
| **box** | which machine made it, with its clock: spark is UTC+10, rented 5090 is UTC+0000 |
| **re-obtainable** | yes with a TESTED and DATED command, or no. ⛔ An untested re-obtain path is a wish. |

## ★ THE NAMING LAW: folders for the KIND, files for the CHOICE

**Name the folder for the kind of thing. Name the file for the choice that distinguishes it from its
siblings.** A long true name beats a short cryptic one, so the tree reads as an index of where the
work has gone.

    ✅ ternary-gguf/v41flash-ptq1_0-layers-0-2-8-9-40-42-imatrix-wiki103.gguf
    ⛔ ternary-gguf/run3.gguf

## The directories

| directory | what kind of thing | state |
|---|---|---|
| `fp16-source/` | the FP16 safetensors shards we quantise **FROM** - the P0 precondition | ⛔ **empty. we do not hold them.** |
| `imatrix/` | calibration matrices, per corpus and per layer set | ⚠ empty, nothing calibrated |
| `checkpoints/` | intermediate states of a quantisation run, resumable | ⚠ empty, nothing run |
| `ternary-gguf/` | the PTQ1_0 output, per layer set and per method variant | ⚠ empty, nothing quantised |
| `final/` | the versions we would stand behind and host | ⛔ empty. `../HF_PUBLICATION_PLAN.md` |
| `scratch/` | runs that failed or were superseded - **kept, not deleted** | ⚠ empty |

★ **`scratch/` is kept rather than deleted on purpose.** A superseded run is the evidence for why
the current one is shaped the way it is, and this estate's storage law distinguishes
RE-DOWNLOADABLE, DERIVED and IRREPLACEABLE rather than treating everything as disposable.

## ⚠ Where the large ones actually belong

**Not the spark's nvme.** The FP16 source is a Keep-class artifact: **THE KEEP** is
`/media/hologram/M200/dwarfstar`, an external 3.8 TB drive on the spark. ⇒ this tree may hold a
**symlink or a STONE** rather than the bytes, and the INDEX row says which. A file that vanishes is
a mystery; a file that leaves a marker is a pointer.

## ⛔ Secrets

Nothing tracked in here carries a credential or a Hugging Face token. **Environment variable NAME
only**, never a value - not in a README, not in an INDEX row, not in a committed log.
