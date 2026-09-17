#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_mem_reserve.h"

#define MIB (1024ull * 1024ull)
#define GIB (1024ull * 1024ull * 1024ull)

/*
 * A non-aborting check, so that every case below reports independently:
 * assert() stops at the first failure, which makes a red run show one line
 * and hide the rest.  One CHECK per independent claim.
 */
static int g_failures = 0;
static void check(bool ok, const char *name) {
    if (!ok) {
        g_failures++;
        fprintf(stderr, "FAIL: %s\n", name);
    }
}

static void clear_env(void) {
    unsetenv("DS4_MEM_RESERVE_MIB");
    unsetenv("DS4_MEM_RESERVE_ENFORCE");
    unsetenv("DS4_MEM_RESERVE_SAMPLE_STEADY");
    unsetenv("DS4_GPU_SAFETY_MARGIN_MIB");
    unsetenv("DS4_MEM_RESERVE_FLOOR_HISTORY_MIB");
    unsetenv("DS4_MEM_RESERVE_RECORD");
    ds4_mem_reserve_read_hook = NULL;
}

static bool contains(const char *hay, const char *needle) {
    return hay && needle && strstr(hay, needle) != NULL;
}

static bool hook_known(uint64_t *bytes) {
    *bytes = 3ull * GIB;
    return true;
}

static bool hook_unknown(uint64_t *bytes) {
    (void)bytes;
    return false;
}

int main(void) {
    ds4_mem_reserve r;
    char line[1024];
    char json[2048];

    /* Defaults: same number as the static per-device reserve this replaces. */
    clear_env();
    ds4_mem_reserve_init(&r);
    assert(r.floor_bytes == 512ull * MIB);
    assert(!r.floor_from_env);
    /* DEFAULT CHANGED 2026-09-16: OBSERVE, not ENFORCE. This assert pinned
     * the old direction, so leaving it would have refused the repair. */
    assert(!r.enforce);
    assert(contains(r.authorization, "default 512 MiB"));
    assert(r.floor_history_count == 1 && r.floor_history_mib[0] == 512);
    assert(ds4_mem_reserve_floor_bytes(&r) == 512ull * MIB);
    assert(ds4_mem_reserve_headroom(&r) == DS4_MEM_HEADROOM_UNKNOWN);

    /* The named value. */
    clear_env();
    setenv("DS4_MEM_RESERVE_MIB", "2048", 1);
    ds4_mem_reserve_init(&r);
    assert(r.floor_bytes == 2ull * GIB);
    assert(r.floor_from_env);
    assert(contains(r.authorization, "DS4_MEM_RESERVE_MIB=2048"));
    assert(r.floor_history_count == 1 && r.floor_history_mib[0] == 2048);

    /* Bad values fall back to the default rather than a silent zero. */
    clear_env();
    setenv("DS4_MEM_RESERVE_MIB", "abc", 1);
    ds4_mem_reserve_init(&r);
    assert(!r.floor_from_env && r.floor_bytes == 512ull * MIB);
    clear_env();
    setenv("DS4_MEM_RESERVE_MIB", "0", 1);
    ds4_mem_reserve_init(&r);
    assert(!r.floor_from_env && r.floor_bytes == 512ull * MIB);

    /* Enforcement switch. */
    clear_env();
    setenv("DS4_MEM_RESERVE_ENFORCE", "0", 1);
    ds4_mem_reserve_init(&r);
    assert(!r.enforce);
    clear_env();
    setenv("DS4_MEM_RESERVE_ENFORCE", "1", 1);
    ds4_mem_reserve_init(&r);
    assert(r.enforce);
    clear_env();
    setenv("DS4_MEM_RESERVE_ENFORCE", "maybe", 1);
    ds4_mem_reserve_init(&r);
    assert(!r.enforce); /* junk follows the default, which is now observe */

    /* The ladder is recorded next to the observed minimum. */
    clear_env();
    setenv("DS4_MEM_RESERVE_MIB", "2048", 1);
    setenv("DS4_MEM_RESERVE_FLOOR_HISTORY_MIB", "8192,4096,2048", 1);
    ds4_mem_reserve_init(&r);
    assert(r.floor_history_count == 3);
    assert(r.floor_history_mib[0] == 8192);
    assert(r.floor_history_mib[1] == 4096);
    assert(r.floor_history_mib[2] == 2048);

    /* A breach is a stop threshold when enforcing. */
    clear_env();
    setenv("DS4_MEM_RESERVE_MIB", "2048", 1);
    setenv("DS4_MEM_RESERVE_ENFORCE", "1", 1);
    ds4_mem_reserve_init(&r);
    assert(ds4_mem_reserve_observe(&r, 3ull * GIB, DS4_MEM_PHASE_LOAD));
    assert(!r.breached);
    assert(ds4_mem_reserve_headroom(&r) == DS4_MEM_HEADROOM_MET);
    assert(!ds4_mem_reserve_observe(&r, 1ull * GIB, DS4_MEM_PHASE_FIRST_PREFILL));
    assert(r.breached && r.aborted);
    assert(r.breaches == 1);
    assert(r.last_breach_phase == DS4_MEM_PHASE_FIRST_PREFILL);
    assert(r.observed_min_bytes == 1ull * GIB);
    assert(r.observed_min_phase == DS4_MEM_PHASE_FIRST_PREFILL);
    assert(ds4_mem_reserve_headroom(&r) == DS4_MEM_HEADROOM_BREACHED);

    /* Observe-only mode records the breach and keeps going. */
    clear_env();
    setenv("DS4_MEM_RESERVE_MIB", "2048", 1);
    setenv("DS4_MEM_RESERVE_ENFORCE", "0", 1);
    ds4_mem_reserve_init(&r);
    assert(ds4_mem_reserve_observe(&r, 1ull * GIB, DS4_MEM_PHASE_CACHE_WARM));
    assert(r.breached && !r.aborted);

    /* FLOOR, AUTHORIZATION and OBSERVED MINIMUM in one status line. */
    ds4_mem_reserve_status_line(&r, line, sizeof(line));
    assert(contains(line, "FLOOR=2.00 GiB"));
    assert(contains(line, "auth=DS4_MEM_RESERVE_MIB=2048"));
    assert(contains(line, "OBSERVED_MIN=1.00 GiB@cache-warm"));
    assert(contains(line, "meets_headroom=no"));
    assert(contains(line, "enforce=0"));

    /* The record carries the floor, its authorization, the ladder and the
     * observed minimum, so a result file can hold all four. */
    assert(ds4_mem_reserve_record_json(&r, json, sizeof(json)));
    assert(contains(json, "\"floor_mib\":2048.000"));
    assert(contains(json, "\"authorization\":\"DS4_MEM_RESERVE_MIB=2048\""));
    assert(contains(json, "\"floor_history_mib\":[2048]"));
    assert(contains(json, "\"observed_min_mib\":1024.000"));
    assert(contains(json, "\"observed_min_phase\":\"cache-warm\""));
    assert(contains(json, "\"meets_headroom\":false"));
    assert(contains(json, "\"phase_min_gib\":{\"cache-warm\":1.000}"));

    /* No samples yet: nothing is claimed. */
    clear_env();
    ds4_mem_reserve_init(&r);
    assert(ds4_mem_reserve_record_json(&r, json, sizeof(json)));
    assert(contains(json, "\"observed_min_mib\":null"));
    assert(contains(json, "\"meets_headroom\":null"));
    ds4_mem_reserve_status_line(&r, line, sizeof(line));
    assert(contains(line, "OBSERVED_MIN=not-sampled"));
    assert(contains(line, "meets_headroom=unknown"));

    /* The live read path, with an injected sampler. */
    clear_env();
    ds4_mem_reserve_read_hook = hook_known;
    ds4_mem_reserve_init(&r);
    assert(ds4_mem_reserve_sample(&r, DS4_MEM_PHASE_STARTUP));
    assert(r.samples == 1);
    assert(r.available_known);
    assert(ds4_mem_reserve_headroom(&r) == DS4_MEM_HEADROOM_MET);
    /* A repeat inside the throttle window does not re-read. */
    assert(ds4_mem_reserve_sample(&r, DS4_MEM_PHASE_STEADY));
    assert(ds4_mem_reserve_sample(&r, DS4_MEM_PHASE_STEADY));
    assert(r.samples == 2);
    /* Every phase gets its first observation regardless of the throttle. */
    assert(ds4_mem_reserve_sample(&r, DS4_MEM_PHASE_LOAD));
    assert(ds4_mem_reserve_sample(&r, DS4_MEM_PHASE_CACHE_WARM));
    assert(ds4_mem_reserve_sample(&r, DS4_MEM_PHASE_FIRST_PREFILL));
    assert(r.samples == 5);
    assert(ds4_mem_reserve_record_json(&r, json, sizeof(json)));
    assert(contains(json, "\"startup\":3.000"));
    assert(contains(json, "\"load\":3.000"));
    assert(contains(json, "\"cache-warm\":3.000"));
    assert(contains(json, "\"first-long-prefill\":3.000"));
    assert(contains(json, "\"steady\":3.000"));

    /* A host that cannot report available memory claims nothing. */
    clear_env();
    ds4_mem_reserve_read_hook = hook_unknown;
    ds4_mem_reserve_init(&r);
    assert(ds4_mem_reserve_sample(&r, DS4_MEM_PHASE_STARTUP));
    assert(!r.available_known);
    assert(r.samples == 0);
    assert(ds4_mem_reserve_headroom(&r) == DS4_MEM_HEADROOM_UNKNOWN);
    ds4_mem_reserve_status_line(&r, line, sizeof(line));
    assert(contains(line, "AVAILABLE=unavailable"));
    assert(contains(line, "meets_headroom=unknown"));

    /* The record file is appended to, one line per run. */
    clear_env();
    setenv("DS4_MEM_RESERVE_MIB", "640", 1);
    setenv("DS4_MEM_RESERVE_RECORD", "tests/_mem_reserve_record.jsonl", 1);
    ds4_mem_reserve_init(&r);
    assert(ds4_mem_reserve_observe(&r, 2ull * GIB, DS4_MEM_PHASE_LOAD));
    assert(ds4_mem_reserve_write_record(&r));
    assert(ds4_mem_reserve_write_record(&r));
    remove("tests/_mem_reserve_record.jsonl");

    /* ------------------------------------------------------------------
     * THE DEFAULT IS OBSERVE, NOT ENFORCE (2026-09-16).
     * An unset configuration must not be able to fail a healthy session.
     * ------------------------------------------------------------------ */

    /* Case: with nothing set, a breach is recorded and the caller continues.
     * This is the regression guard for the class "the off state aborts the
     * run": ds4_session_eval returns 1 exactly when a sample returns false. */
    clear_env();
    ds4_mem_reserve_init(&r);
    check(!r.enforce, "default: enforce is off");
    check(ds4_mem_reserve_observe(&r, 1ull * MIB, DS4_MEM_PHASE_STEADY),
          "default: a breach does NOT abort the caller");
    check(r.breached && !r.aborted,
          "default: the breach is still recorded, it just does not stop");

    /* Case: with nothing set, the live sampler is never reached from the
     * per-token decode path, so there is no clock_gettime and no
     * /proc/meminfo parse per token. */
    clear_env();
    ds4_mem_reserve_init(&r);
    check(!ds4_mem_reserve_steady_enabled(&r),
          "default: the per-token decode path does not sample");

    /* Case: enforcement fires when it is explicitly asked for. */
    clear_env();
    setenv("DS4_MEM_RESERVE_ENFORCE", "1", 1);
    ds4_mem_reserve_init(&r);
    check(r.enforce, "ENFORCE=1: enforcement is on");
    check(!ds4_mem_reserve_observe(&r, 1ull * MIB, DS4_MEM_PHASE_STEADY),
          "ENFORCE=1: a breach aborts the caller");
    check(r.aborted, "ENFORCE=1: the abort is recorded");

    /* Case: enforcing implies sampling - a stop threshold nobody reads
     * cannot stop anything. */
    clear_env();
    setenv("DS4_MEM_RESERVE_ENFORCE", "1", 1);
    ds4_mem_reserve_init(&r);
    check(ds4_mem_reserve_steady_enabled(&r),
          "ENFORCE=1: the per-token path samples");

    /* Case: observe the decode path without stopping it. */
    clear_env();
    setenv("DS4_MEM_RESERVE_SAMPLE_STEADY", "1", 1);
    ds4_mem_reserve_init(&r);
    check(ds4_mem_reserve_steady_enabled(&r) && !r.enforce,
          "SAMPLE_STEADY=1: samples steady state and still does not stop");

    /* Case: and the sampling can be turned off under enforcement too. */
    clear_env();
    setenv("DS4_MEM_RESERVE_ENFORCE", "1", 1);
    setenv("DS4_MEM_RESERVE_SAMPLE_STEADY", "0", 1);
    ds4_mem_reserve_init(&r);
    check(r.enforce && !ds4_mem_reserve_steady_enabled(&r),
          "SAMPLE_STEADY=0 beats the ENFORCE implication");

    /* ------------------------------------------------------------------
     * TWO QUANTITIES, TWO NAMES.
     * The host memory floor and the per-device VRAM placement margin used
     * to share DS4_MEM_RESERVE_MIB.  Each must now move only its own.
     * ------------------------------------------------------------------ */

    /* Case: the host floor does not touch the VRAM margin. */
    clear_env();
    setenv("DS4_MEM_RESERVE_MIB", "16384", 1);
    ds4_mem_reserve_init(&r);
    check(r.floor_bytes == 16384ull * MIB,
          "MEM_RESERVE_MIB moves the host floor");
    check(ds4_gpu_safety_margin_bytes_from_env() == 512ull * MIB,
          "MEM_RESERVE_MIB leaves the VRAM margin at its own default");

    /* Case: the VRAM margin does not touch the host floor. */
    clear_env();
    setenv("DS4_GPU_SAFETY_MARGIN_MIB", "2048", 1);
    ds4_mem_reserve_init(&r);
    check(ds4_gpu_safety_margin_bytes_from_env() == 2048ull * MIB,
          "GPU_SAFETY_MARGIN_MIB moves the VRAM margin");
    check(r.floor_bytes == 512ull * MIB && !r.floor_from_env,
          "GPU_SAFETY_MARGIN_MIB leaves the host floor at its own default");

    /* Case: both named at once, each lands on its own quantity. */
    clear_env();
    setenv("DS4_MEM_RESERVE_MIB", "16384", 1);
    setenv("DS4_GPU_SAFETY_MARGIN_MIB", "2048", 1);
    ds4_mem_reserve_init(&r);
    check(r.floor_bytes == 16384ull * MIB &&
          ds4_gpu_safety_margin_bytes_from_env() == 2048ull * MIB,
          "the two margins are independent");

    /* Case: with neither named, the VRAM margin is the 512 MiB the removed
     * DS4_GPU_ARGS_DEFAULT_SAFETY_MARGIN constant carried - byte-identical
     * placement for every invocation that sets nothing. */
    clear_env();
    check(ds4_gpu_safety_margin_bytes_from_env() == 512ull * MIB,
          "unset: the VRAM margin is the old constant");

    clear_env();
    if (g_failures != 0) {
        fprintf(stderr, "Live memory reserve: %d FAILED\n", g_failures);
        return 1;
    }
    puts("Live memory reserve: PASS");
    return 0;
}
