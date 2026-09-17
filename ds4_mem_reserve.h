#ifndef DS4_MEM_RESERVE_H
#define DS4_MEM_RESERVE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/*
 * The live OS memory reserve.
 *
 * Adopted from the eight-Spark ceiling rig: there the reserve is a NAMED
 * policy number (minimum_available_gib), it is read against live kernel
 * memory (MemAvailable from /proc/meminfo), it is surfaced with a
 * meets_headroom answer, and the floor is recorded next to its
 * authorization and the observed minimum.  References:
 *   research/v41-flash-landscape/06-octuple-spark-ceiling/CODE.md   (4.1, 5.2)
 *   research/v41-flash-landscape/06-octuple-spark-ceiling/HOW-WE-USE-IT.md (2.1, 2.5, 4)
 * Their rig's numbers do not transfer (a node holding a fully resident model
 * plus resident Engram has a different budget shape than our streaming box),
 * so the default here stays at the per-device reserve we already subtract
 * statically, as a policy value a human names rather than one we compute.
 *
 * WHAT THIS REPLACES: ds4_gpu_args.c:17 held the reserve as a compile-time
 * constant that ds4.c subtracted once at packing time
 * (transmits to ds4.c engine_classify_multi_tier, the
 * `safety_margin_bytes + cublas_workspace_overhead` line) and never
 * re-checked.  A static subtraction and a live alarm diverge exactly when
 * something outside the engine changes the equation.
 *
 * ENFORCEMENT IS OPT-IN (2026-09-16).  THE DEFAULT IS OBSERVE.  An earlier
 * revision of this module defaulted `enforce` to true, which made an unset
 * configuration abort ds4_session_eval on a reading below the floor - and
 * because the sampler is throttled, the very next call within
 * DS4_MEM_RESERVE_SAMPLE_INTERVAL_MS passed, so the failure was intermittent
 * with no switch to turn it off.  A lever whose OFF state is "abort the run"
 * is not a lever.  The floor is now recorded and logged by default and
 * DS4_MEM_RESERVE_ENFORCE=1 turns the stop threshold on, which is also what
 * ds4_help.c has said all along.
 *
 * ENFORCEMENT DECISION (breach aborts, it does not clamp cache growth):
 * our expert cache is sized once in ds4_ssd_auto_cache_plan()
 * (ds4_ssd.c) from the recommended working set and there is no runtime
 * growth path to clamp; the only moment a clamp could act is before
 * allocation, which is exactly what the packing-time reserve already does.
 * A late clamp would also silently change the cache size under a running
 * measurement, making the throughput unattributable.  So a breach is a STOP
 * THRESHOLD, which is what the reference calls it too
 * (sglang/docs/eight-million-token-results.md:83).
 *
 * WHAT THIS MEASURES, AND WHAT IT DOES NOT (name both, because two memory
 * effects that look alike are not the same instrument):
 *   IT MEASURES: whole-node MemAvailable, sampled at load, cache warm, the
 *   first long prefill and (throttled) steady state, against a floor a human
 *   authorized via DS4_MEM_RESERVE_MIB.  It answers "how close to the edge
 *   did this run get, and where", for pressure that arrives from OUTSIDE the
 *   engine as well as inside it.
 *   IT DOES NOT MEASURE: any in-process residency effect.  The 2.2x resident
 *   warm-up (cold 15.87 to warm 35.14 tokens/s) and the --warm-weights flag
 *   (+4.4% same-power) are two DIFFERENT effects that a memory reading
 *   cannot separate (WIKI/theory/48-decode-speedup-levers.md section 4,
 *   lines 165-205).  Nor does it see the page-cache trap: an A/B whose "cold"
 *   leg is still page-cache-warm measures the cache, not the lever
 *   (same file, lines 227-234).  A clean floor reading is not evidence that
 *   an experiment was controlled.
 */

#define DS4_MEM_RESERVE_DEFAULT_MIB 512u
#define DS4_MEM_RESERVE_MIN_MIB 1u
#define DS4_MEM_RESERVE_MAX_MIB (1024u * 1024u) /* 1 TiB */
#define DS4_MEM_RESERVE_MAX_HISTORY 16u

/*
 * A prompt this long or longer is the "first long prefill" phase.  The
 * reference's observed minimum lived in load, cache warm and the first long
 * prefill, not in steady state (HOW-WE-USE-IT.md:131-144).
 */
#define DS4_MEM_RESERVE_LONG_PREFILL_TOKENS 8192u

/* Steady-state samples are throttled to this interval. */
#define DS4_MEM_RESERVE_SAMPLE_INTERVAL_MS 2000u

/*
 * The per-device VRAM safety margin is a DIFFERENT QUANTITY from the host
 * memory floor above: one is per-device VRAM subtracted at placement time,
 * the other is whole-node MemAvailable checked at runtime.  They had one
 * name between them (DS4_MEM_RESERVE_MIB rewrote both), so an operator who
 * named a 16 GiB OS floor also took 16 GiB off every device's placement
 * budget.  Two quantities, two names.
 */
#define DS4_GPU_SAFETY_MARGIN_DEFAULT_MIB 512u

typedef enum {
    DS4_MEM_PHASE_STARTUP = 0,
    DS4_MEM_PHASE_LOAD,
    DS4_MEM_PHASE_CACHE_WARM,
    DS4_MEM_PHASE_FIRST_PREFILL,
    DS4_MEM_PHASE_STEADY,
    DS4_MEM_PHASE_COUNT
} ds4_mem_phase;

typedef enum {
    DS4_MEM_HEADROOM_UNKNOWN = 0,
    DS4_MEM_HEADROOM_MET,
    DS4_MEM_HEADROOM_BREACHED
} ds4_mem_headroom;

const char *ds4_mem_phase_name(ds4_mem_phase phase);

typedef struct {
    uint64_t floor_bytes;
    bool     floor_from_env;
    char     authorization[96];
    bool     enforce;      /* DS4_MEM_RESERVE_ENFORCE=1: a breach stops the run. */
    bool     sample_steady; /* sample from the per-token decode path at all. */

    uint32_t floor_history_mib[DS4_MEM_RESERVE_MAX_HISTORY];
    uint32_t floor_history_count;

    bool     available_known;
    bool     breached;
    bool     aborted; /* enforce && breach: the caller must stop */
    uint64_t available_bytes;
    uint64_t observed_min_bytes;
    ds4_mem_phase observed_min_phase;
    uint64_t phase_min_bytes[DS4_MEM_PHASE_COUNT];
    bool     phase_sampled[DS4_MEM_PHASE_COUNT];

    uint64_t samples;
    uint64_t breaches;
    ds4_mem_phase last_breach_phase;

    uint64_t sample_interval_ms;
    uint64_t last_sample_ms;
} ds4_mem_reserve;

/*
 * Test seam: when set, this replaces the /proc/meminfo read.  It returns
 * false when the host cannot report available memory (macOS, a kernel
 * without /proc), in which case nothing is claimed and nothing aborts.
 */
extern bool (*ds4_mem_reserve_read_hook)(uint64_t *bytes);

void ds4_mem_reserve_init(ds4_mem_reserve *r);
uint64_t ds4_mem_reserve_floor_bytes(const ds4_mem_reserve *r);
ds4_mem_headroom ds4_mem_reserve_headroom(const ds4_mem_reserve *r);

/* Record one observation. Returns false when a breach must abort. */
bool ds4_mem_reserve_observe(ds4_mem_reserve *r,
                             uint64_t available_bytes,
                             ds4_mem_phase phase);
/* Read live kernel memory (throttled) and record it. */
bool ds4_mem_reserve_sample(ds4_mem_reserve *r, ds4_mem_phase phase);

/*
 * The per-token decode path asks this BEFORE calling the sampler, so that
 * with nothing set the decode loop pays neither the clock_gettime nor the
 * /proc/meminfo parse.  Off unless DS4_MEM_RESERVE_ENFORCE=1 (a stop
 * threshold nobody samples for cannot stop anything) or
 * DS4_MEM_RESERVE_SAMPLE_STEADY=1 (observe steady state without stopping).
 */
static inline bool ds4_mem_reserve_steady_enabled(const ds4_mem_reserve *r) {
    return r != NULL && r->sample_steady;
}

/* Reads the per-device VRAM safety margin in bytes: its own name, its own
 * unit, its own default.  Never the host floor. */
uint64_t ds4_gpu_safety_margin_bytes_from_env(void);

/* FLOOR + AUTHORIZATION + OBSERVED MINIMUM + meets-headroom, one line. */
void ds4_mem_reserve_status_line(const ds4_mem_reserve *r, char *buf, size_t buflen);
void ds4_mem_reserve_log_status(const ds4_mem_reserve *r, const char *tag);

/* The end-of-run record a result file can carry. */
bool ds4_mem_reserve_record_json(const ds4_mem_reserve *r, char *buf, size_t buflen);
/* Appends the record to $DS4_MEM_RESERVE_RECORD when that is set. */
bool ds4_mem_reserve_write_record(const ds4_mem_reserve *r);

#endif /* DS4_MEM_RESERVE_H */
