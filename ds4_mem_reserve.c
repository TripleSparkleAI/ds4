#include "ds4_mem_reserve.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "ds4_linux_memory.h"

#define DS4_MIB (1024ull * 1024ull)
#define DS4_GIB (1024ull * 1024ull * 1024ull)

bool (*ds4_mem_reserve_read_hook)(uint64_t *bytes) = NULL;

const char *ds4_mem_phase_name(ds4_mem_phase phase) {
    switch (phase) {
        case DS4_MEM_PHASE_STARTUP:      return "startup";
        case DS4_MEM_PHASE_LOAD:         return "load";
        case DS4_MEM_PHASE_CACHE_WARM:   return "cache-warm";
        case DS4_MEM_PHASE_FIRST_PREFILL: return "first-long-prefill";
        case DS4_MEM_PHASE_STEADY:       return "steady";
        default:                         return "unknown";
    }
}

static uint64_t ds4_mem_now_ms(void) {
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (uint64_t)ts.tv_sec * 1000ull + (uint64_t)(ts.tv_nsec / 1000000l);
    }
#endif
    struct timeval tv;
    if (gettimeofday(&tv, NULL) == 0) {
        return (uint64_t)tv.tv_sec * 1000ull + (uint64_t)(tv.tv_usec / 1000l);
    }
    return 0;
}

static bool ds4_mem_parse_ulong(const char *s, unsigned long *out) {
    if (!s || !s[0] || !out) return false;
    errno = 0;
    char *end = NULL;
    const unsigned long v = strtoul(s, &end, 10);
    if (end == s || *end != '\0' || errno != 0) return false;
    *out = v;
    return true;
}

/* Comma-separated MiB ladder, recorded next to the observed minimum the way
 * the reference records emergency_floor_history_gib next to it
 * (sglang/results/eight-million/summary.json:922-927). */
static void ds4_mem_reserve_parse_history(ds4_mem_reserve *r, const char *spec) {
    const char *p = spec;
    while (p && *p && r->floor_history_count < DS4_MEM_RESERVE_MAX_HISTORY) {
        char *end = NULL;
        errno = 0;
        const unsigned long v = strtoul(p, &end, 10);
        if (end == p || errno != 0 ||
            v < DS4_MEM_RESERVE_MIN_MIB || v > DS4_MEM_RESERVE_MAX_MIB) {
            fprintf(stderr,
                    "ds4: invalid DS4_MEM_RESERVE_FLOOR_HISTORY_MIB entry near '%s' "
                    "(want MiB values %u..%u); ignoring the rest of the ladder\n",
                    p, DS4_MEM_RESERVE_MIN_MIB, DS4_MEM_RESERVE_MAX_MIB);
            break;
        }
        r->floor_history_mib[r->floor_history_count++] = (uint32_t)v;
        if (*end == ',') {
            p = end + 1;
            continue;
        }
        if (*end != '\0') {
            fprintf(stderr,
                    "ds4: junk after DS4_MEM_RESERVE_FLOOR_HISTORY_MIB entry: '%s'\n",
                    end);
        }
        break;
    }
}

void ds4_mem_reserve_init(ds4_mem_reserve *r) {
    if (!r) return;
    memset(r, 0, sizeof(*r));

    /* Default = the static per-device reserve this module supersedes, so an
     * invocation that sets nothing behaves exactly as before. */
    r->floor_bytes = (uint64_t)DS4_MEM_RESERVE_DEFAULT_MIB * DS4_MIB;
    r->floor_from_env = false;
    snprintf(r->authorization, sizeof(r->authorization),
             "default %u MiB (ds4_gpu_args.c DS4_GPU_ARGS_DEFAULT_SAFETY_MARGIN)",
             DS4_MEM_RESERVE_DEFAULT_MIB);

    const char *mib = getenv("DS4_MEM_RESERVE_MIB");
    if (mib && mib[0]) {
        unsigned long v = 0;
        if (ds4_mem_parse_ulong(mib, &v) &&
            v >= DS4_MEM_RESERVE_MIN_MIB && v <= DS4_MEM_RESERVE_MAX_MIB) {
            r->floor_bytes = (uint64_t)v * DS4_MIB;
            r->floor_from_env = true;
            snprintf(r->authorization, sizeof(r->authorization),
                     "DS4_MEM_RESERVE_MIB=%lu", v);
        } else {
            fprintf(stderr,
                    "ds4: invalid DS4_MEM_RESERVE_MIB=%s (want %u..%u MiB); "
                    "using the %u MiB default\n",
                    mib, DS4_MEM_RESERVE_MIN_MIB, DS4_MEM_RESERVE_MAX_MIB,
                    DS4_MEM_RESERVE_DEFAULT_MIB);
        }
    }

    /* OBSERVE BY DEFAULT. A breach is a stop threshold only when the
     * operator asks for one: DS4_MEM_RESERVE_ENFORCE=1. Defaulting this to
     * true made an unset configuration abort a healthy decode on one low
     * reading, with no switch to turn the sampler off and - because the
     * sampler is throttled - a next call that passed, so the failure was
     * intermittent. The floor is still named, logged and recorded either
     * way; only the stop is opt-in. */
    r->enforce = false;
    const char *enforce = getenv("DS4_MEM_RESERVE_ENFORCE");
    if (enforce && enforce[0]) {
        if (!strcmp(enforce, "1")) {
            r->enforce = true;
        } else if (!strcmp(enforce, "0")) {
            r->enforce = false;
        } else {
            fprintf(stderr,
                    "ds4: invalid DS4_MEM_RESERVE_ENFORCE=%s (want 1 or 0); "
                    "using the default 0 (observe, do not stop)\n",
                    enforce);
        }
    }

    /* The per-token decode path only samples when someone asked for it.
     * Enforcing implies sampling (a stop threshold nobody reads cannot
     * stop anything); DS4_MEM_RESERVE_SAMPLE_STEADY=1 buys the steady-state
     * observation without the stop. With neither set the decode loop pays
     * no clock_gettime and no /proc/meminfo parse at all. */
    r->sample_steady = r->enforce;
    const char *steady = getenv("DS4_MEM_RESERVE_SAMPLE_STEADY");
    if (steady && steady[0]) {
        if (!strcmp(steady, "1")) {
            r->sample_steady = true;
        } else if (!strcmp(steady, "0")) {
            r->sample_steady = false;
        } else {
            fprintf(stderr,
                    "ds4: invalid DS4_MEM_RESERVE_SAMPLE_STEADY=%s (want 1 or 0); "
                    "following DS4_MEM_RESERVE_ENFORCE\n",
                    steady);
        }
    }

    r->observed_min_bytes = UINT64_MAX;
    r->observed_min_phase = DS4_MEM_PHASE_STARTUP;
    r->sample_interval_ms = DS4_MEM_RESERVE_SAMPLE_INTERVAL_MS;

    const char *history = getenv("DS4_MEM_RESERVE_FLOOR_HISTORY_MIB");
    if (history && history[0]) {
        ds4_mem_reserve_parse_history(r, history);
    }
    if (r->floor_history_count == 0) {
        r->floor_history_mib[0] = (uint32_t)(r->floor_bytes / DS4_MIB);
        r->floor_history_count = 1;
    }
}

uint64_t ds4_mem_reserve_floor_bytes(const ds4_mem_reserve *r) {
    if (!r) return (uint64_t)DS4_MEM_RESERVE_DEFAULT_MIB * DS4_MIB;
    return r->floor_bytes;
}

/*
 * The per-device VRAM safety margin. A SEPARATE QUANTITY from the host
 * memory floor: DS4_MEM_RESERVE_MIB used to set both, so naming a whole-node
 * OS floor silently subtracted the same number from every device's placement
 * budget - one knob, two units. This one is per-device VRAM, in MiB, and it
 * is read nowhere else.
 */
uint64_t ds4_gpu_safety_margin_bytes_from_env(void) {
    const uint64_t fallback =
        (uint64_t)DS4_GPU_SAFETY_MARGIN_DEFAULT_MIB * DS4_MIB;
    const char *mib = getenv("DS4_GPU_SAFETY_MARGIN_MIB");
    if (!mib || !mib[0]) return fallback;
    unsigned long v = 0;
    if (ds4_mem_parse_ulong(mib, &v) &&
        v >= DS4_MEM_RESERVE_MIN_MIB && v <= DS4_MEM_RESERVE_MAX_MIB) {
        return (uint64_t)v * DS4_MIB;
    }
    fprintf(stderr,
            "ds4: invalid DS4_GPU_SAFETY_MARGIN_MIB=%s (want %u..%u MiB); "
            "using the %u MiB default\n",
            mib, DS4_MEM_RESERVE_MIN_MIB, DS4_MEM_RESERVE_MAX_MIB,
            DS4_GPU_SAFETY_MARGIN_DEFAULT_MIB);
    return fallback;
}

ds4_mem_headroom ds4_mem_reserve_headroom(const ds4_mem_reserve *r) {
    if (!r || !r->available_known) return DS4_MEM_HEADROOM_UNKNOWN;
    return r->breached ? DS4_MEM_HEADROOM_BREACHED : DS4_MEM_HEADROOM_MET;
}

static const char *ds4_mem_headroom_name(ds4_mem_headroom h) {
    switch (h) {
        case DS4_MEM_HEADROOM_MET:       return "yes";
        case DS4_MEM_HEADROOM_BREACHED:  return "no";
        default:                         return "unknown";
    }
}

bool ds4_mem_reserve_observe(ds4_mem_reserve *r,
                             uint64_t available_bytes,
                             ds4_mem_phase phase) {
    if (!r) return true;
    if ((unsigned)phase >= (unsigned)DS4_MEM_PHASE_COUNT) {
        phase = DS4_MEM_PHASE_STEADY;
    }

    r->available_known = true;
    r->available_bytes = available_bytes;
    r->samples++;
    if (!r->phase_sampled[phase]) r->phase_sampled[phase] = true;
    if (available_bytes < r->phase_min_bytes[phase] ||
        r->phase_min_bytes[phase] == 0) {
        r->phase_min_bytes[phase] = available_bytes;
    }
    if (available_bytes < r->observed_min_bytes) {
        r->observed_min_bytes = available_bytes;
        r->observed_min_phase = phase;
    }

    if (available_bytes < r->floor_bytes) {
        r->breached = true;
        r->breaches++;
        r->last_breach_phase = phase;
        fprintf(stderr,
                "ds4: mem reserve BREACH floor=%.2f GiB (authorization: %s) "
                "available=%.2f GiB phase=%s\n",
                (double)r->floor_bytes / (double)DS4_GIB,
                r->authorization,
                (double)available_bytes / (double)DS4_GIB,
                ds4_mem_phase_name(phase));
        if (r->enforce) {
            r->aborted = true;
            return false;
        }
    }
    return true;
}

bool ds4_mem_reserve_sample(ds4_mem_reserve *r, ds4_mem_phase phase) {
    if (!r) return true;
    if ((unsigned)phase >= (unsigned)DS4_MEM_PHASE_COUNT) {
        phase = DS4_MEM_PHASE_STEADY;
    }

    /* First observation of a phase always lands; repeats are throttled so a
     * decode loop cannot turn this into a meminfo poll per token. */
    if (r->phase_sampled[phase] && r->sample_interval_ms != 0) {
        const uint64_t now = ds4_mem_now_ms();
        if (now != 0 && now - r->last_sample_ms < r->sample_interval_ms) {
            return true;
        }
    }
    r->last_sample_ms = ds4_mem_now_ms();

    uint64_t available = 0;
    bool known = false;
    if (ds4_mem_reserve_read_hook) {
        known = ds4_mem_reserve_read_hook(&available);
    } else {
        known = ds4_linux_nonmovable_memory(&available);
    }
    if (!known) {
        /* macOS host, or a kernel without /proc: claim nothing, stop
         * nothing. The status line reports meets_headroom=unknown. */
        r->available_known = false;
        return true;
    }
    return ds4_mem_reserve_observe(r, available, phase);
}

void ds4_mem_reserve_status_line(const ds4_mem_reserve *r, char *buf, size_t buflen) {
    if (!buf || buflen == 0) return;
    if (!r) {
        snprintf(buf, buflen, "ds4: mem reserve unavailable\n");
        return;
    }
    const ds4_mem_headroom headroom = ds4_mem_reserve_headroom(r);
    char observed[64];
    char available[64];
    if (r->samples == 0) {
        snprintf(observed, sizeof(observed), "not-sampled");
    } else {
        snprintf(observed, sizeof(observed), "%.2f GiB@%s",
                 (double)r->observed_min_bytes / (double)DS4_GIB,
                 ds4_mem_phase_name(r->observed_min_phase));
    }
    if (r->available_known) {
        snprintf(available, sizeof(available), "%.2f GiB",
                 (double)r->available_bytes / (double)DS4_GIB);
    } else {
        snprintf(available, sizeof(available), "unavailable");
    }
    snprintf(buf, buflen,
             "ds4: mem reserve FLOOR=%.2f GiB auth=%s enforce=%d "
             "OBSERVED_MIN=%s AVAILABLE=%s meets_headroom=%s samples=%llu breaches=%llu",
             (double)r->floor_bytes / (double)DS4_GIB,
             r->authorization,
             r->enforce ? 1 : 0,
             observed,
             available,
             ds4_mem_headroom_name(headroom),
             (unsigned long long)r->samples,
             (unsigned long long)r->breaches);
}

void ds4_mem_reserve_log_status(const ds4_mem_reserve *r, const char *tag) {
    char line[512];
    ds4_mem_reserve_status_line(r, line, sizeof(line));
    fprintf(stderr, "%s%s\n", tag && tag[0] ? tag : "", line);
}

bool ds4_mem_reserve_record_json(const ds4_mem_reserve *r, char *buf, size_t buflen) {
    if (!r || !buf || buflen == 0) return false;

    const ds4_mem_headroom headroom = ds4_mem_reserve_headroom(r);
    const char *headroom_json = headroom == DS4_MEM_HEADROOM_MET ? "true" :
                                headroom == DS4_MEM_HEADROOM_BREACHED ? "false" :
                                "null";

    size_t off = 0;
    int n = snprintf(buf + off, buflen - off,
                     "{\"floor_mib\":%.3f,\"floor_bytes\":%llu,"
                     "\"authorization\":\"%s\",\"enforce\":%s,"
                     "\"floor_history_mib\":[",
                     (double)r->floor_bytes / (double)DS4_MIB,
                     (unsigned long long)r->floor_bytes,
                     r->authorization,
                     r->enforce ? "true" : "false");
    if (n < 0 || (size_t)n >= buflen - off) return false;
    off += (size_t)n;
    for (uint32_t i = 0; i < r->floor_history_count; i++) {
        n = snprintf(buf + off, buflen - off, "%s%u",
                     i ? "," : "", r->floor_history_mib[i]);
        if (n < 0 || (size_t)n >= buflen - off) return false;
        off += (size_t)n;
    }
    n = snprintf(buf + off, buflen - off, "],");
    if (n < 0 || (size_t)n >= buflen - off) return false;
    off += (size_t)n;

    if (r->samples == 0) {
        n = snprintf(buf + off, buflen - off,
                     "\"observed_min_mib\":null,\"observed_min_bytes\":null,"
                     "\"observed_min_phase\":null,");
    } else {
        n = snprintf(buf + off, buflen - off,
                     "\"observed_min_mib\":%.3f,\"observed_min_bytes\":%llu,"
                     "\"observed_min_phase\":\"%s\",",
                     (double)r->observed_min_bytes / (double)DS4_MIB,
                     (unsigned long long)r->observed_min_bytes,
                     ds4_mem_phase_name(r->observed_min_phase));
    }
    if (n < 0 || (size_t)n >= buflen - off) return false;
    off += (size_t)n;

    n = snprintf(buf + off, buflen - off,
                 "\"meets_headroom\":%s,\"breached\":%s,\"aborted\":%s,"
                 "\"breaches\":%llu,\"samples\":%llu,\"available_known\":%s,"
                 "\"phase_min_gib\":{",
                 headroom_json,
                 r->breached ? "true" : "false",
                 r->aborted ? "true" : "false",
                 (unsigned long long)r->breaches,
                 (unsigned long long)r->samples,
                 r->available_known ? "true" : "false");
    if (n < 0 || (size_t)n >= buflen - off) return false;
    off += (size_t)n;

    bool first = true;
    for (int p = 0; p < DS4_MEM_PHASE_COUNT; p++) {
        if (!r->phase_sampled[p]) continue;
        n = snprintf(buf + off, buflen - off,
                     "%s\"%s\":%.3f",
                     first ? "" : ",",
                     ds4_mem_phase_name((ds4_mem_phase)p),
                     (double)r->phase_min_bytes[p] / (double)DS4_GIB);
        if (n < 0 || (size_t)n >= buflen - off) return false;
        off += (size_t)n;
        first = false;
    }
    n = snprintf(buf + off, buflen - off, "}}");
    if (n < 0 || (size_t)n >= buflen - off) return false;
    return true;
}

bool ds4_mem_reserve_write_record(const ds4_mem_reserve *r) {
    const char *path = getenv("DS4_MEM_RESERVE_RECORD");
    if (!path || !path[0]) return false;

    char json[1024];
    if (!ds4_mem_reserve_record_json(r, json, sizeof(json))) {
        fprintf(stderr, "ds4: could not render the memory reserve record\n");
        return false;
    }

    /* Append: one result file may carry several runs, and the reference keeps
     * the floor history next to the observed minimum rather than replacing
     * it (research/v41-flash-landscape/06-octuple-spark-ceiling/CODE.md:325-341). */
    FILE *fp = fopen(path, "a");
    if (!fp) {
        fprintf(stderr, "ds4: cannot append memory reserve record to %s: %s\n",
                path, strerror(errno));
        return false;
    }
    const int rc = fprintf(fp, "%s\n", json);
    fclose(fp);
    if (rc < 0) {
        fprintf(stderr, "ds4: failed writing memory reserve record to %s\n", path);
        return false;
    }
    return true;
}
