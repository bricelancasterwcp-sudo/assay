#!/bin/bash
# Tier re-profile campaign, 2026-08 — 15 models against assay 0.9.0 (schema v8),
# tier enthusiast-16gb, real hardware, full mode. One profile + one transcript
# per model into docs/superpowers/evidence/tier-enthusiast-2026-08/.
#
# LAUNCH (detached, never in the foreground of a session that may end):
#
#   setsid nohup scripts/campaign-2026-08.sh \
#     > /home/brice/workspace/assay-campaign-2026-08.log 2>&1 &
#
# Watch:  tail -f /home/brice/workspace/assay-campaign-2026-08.log
# Timing: docs/superpowers/evidence/tier-enthusiast-2026-08/campaign-run.log
# Done:   docs/superpowers/evidence/tier-enthusiast-2026-08/.DONE exists
# Stop:   kill "$(cat .../campaign.pid)"  — by hand, by the operator, never here.
#
# Deliberately absent, and why:
#   * No `timeout` anywhere. A probe that runs long is data; a probe killed
#     mid-call writes a truncated transcript and a profile that lies about what
#     it measured. The operator kills, the script does not.
#   * The script never kills a process it did not start. `ollama stop` unloads a
#     model from VRAM through the daemon's own API; it does not signal the
#     daemon, and no pkill/kill of anything appears below.
#   * No `set -e`. One model failing is one row missing, not fifteen.
#   * No wait for an idle daemon. The brief said "run on an idle daemon" and
#     this script implements MAKE idle, not WAIT for idle: `unload_all` asks the
#     daemon to stop every resident model before each probe rather than blocking
#     until someone else's work finishes. The difference matters to whoever
#     launches it — this needs an exclusive window, and it will evict another
#     session's warm model to get one. The VRAM floor (exit 91) and the
#     still-resident check (exit 90) are what stop it from profiling into
#     contention it could not clear; they refuse, they do not wait.
#
# Run log format (whitespace-separated, four fields then a trailing comment):
#   start <model> -     0        # <iso8601>
#   done  <model> <rc>  <secs>   # <iso8601>
#   skip  <model> <rc>  <secs>   # <iso8601>
# `awk '{print $1,$2,$3,$4}'` reads it. The run opens with full-line '#'
# comments pinning assay_bin / assay_version / assay_commit — the instrument is
# an EDITABLE install, so the durable timing record must name the commit that
# produced it, not just a version string. Those lines never begin with
# start/done/skip, so a parser selecting on field 1 ignores them.
# Non-probe exit codes are ours:
#   90 = models still resident after the unload wait (someone else holds the GPU)
#   91 = VRAM below the floor at preflight
#   92 = probe exited 0 but the written profile names a different model
# Any other <rc> is `assay probe`'s own exit status.

set -u

REPO=/home/brice/workspace/assay/.worktrees/v17
# The repo's own venv, not a session scratch dir: /tmp is wiped on reboot and
# belongs to whichever session created it, so a campaign launched days from now
# must not depend on one. This is an EDITABLE install pointing at $REPO/src, so
# the bytes that run are the worktree's — which is exactly why the run log
# pins both the version and the commit below.
ASSAY="$REPO/.venv/bin/assay"
VENV_PY="${ASSAY%/assay}/python"
OLLAMA_URL=http://127.0.0.1:11434
EVDIR="$REPO/docs/superpowers/evidence/tier-enthusiast-2026-08"
RUNLOG="$EVDIR/campaign-run.log"
PIDFILE="$EVDIR/campaign.pid"
TIER=enthusiast-16gb

# VRAM floor for the preflight, in MiB.
#
# The box is a 16 GiB RTX 5080 (16303 MiB total) whose desktop holds ~660 MiB,
# so an otherwise-idle card reports ~15.2 GiB free. The largest model here that
# fits the card outright is deepseek-coder-v2:16b-lite at 11.9 GiB of weights;
# 13 GiB free leaves it ~1.2 GiB for KV and compute. Below that, something else
# is resident and this run would either fight it for VRAM or silently profile a
# model that spilled to CPU — so the model is skipped by name and the campaign
# moves on. This is a courtesy/validity check, not a fit check: qwen3.8:27b
# (17.7 GiB) exceeds the card at any reading and splits to CPU by design.
FLOOR_MIB=13312

# How long to wait for /api/ps to drain after stopping every resident model.
PS_WAIT_TRIES=60
PS_WAIT_SLEEP=2

# The 15 models, ordered by committed model.weights_bytes ASCENDING (source:
# docs/superpowers/evidence/tier-enthusiast/*.json). Small first so a broken
# instrument costs minutes, not hours, before anyone notices.
MODELS=(
  qwen2.5-coder:1.5b-instruct-q8_0                          #  1,646,586,631
  hermes3:latest                                            #  4,661,227,243
  llama3.1:8b                                               #  4,920,753,328
  qwen3:8b                                                  #  5,225,388,164
  gemma2:9b                                                 #  5,443,152,417
  gemma-4-12b-it-qat-q4_0:latest                            #  6,975,879,982
  mistral-nemo:latest                                       #  7,071,713,227
  qwen2.5-coder:7b-instruct-q8_0                            #  8,098,539,207
  deepseek-r1:14b                                           #  8,988,112,209
  qwen2.5-coder:14b-instruct-q4_K_M                         #  8,988,124,298
  hf.co/bartowski/NousResearch_Hermes-4-14B-GGUF:Q4_K_M     #  9,001,755,690
  phi4:14b                                                  #  9,053,116,391
  qwen3:14b                                                 #  9,276,198,565
  deepseek-coder-v2:16b-lite-instruct-q5_K_M                # 11,851,329,771
  qwen3.8:27b                                               # 17,741,872,154
)

say() { printf '%s  %s\n' "$(date '+%F %T')" "$*"; }

# Four fields then a trailing comment, per the header.
runlog() {
  printf '%s %s %s %s  # %s\n' "$1" "$2" "$3" "$4" "$(date -Is)" >> "$RUNLOG"
}

# A full-line '#' comment. Never starts with start/done/skip, so a parser
# selecting on field 1 skips it without needing to know it exists.
runlog_note() { printf '# %s\n' "$*" >> "$RUNLOG"; }

# The E1-sweep slug rule: ':' and '/' become '-', everything else is kept.
slug_of() { printf '%s' "$1" | tr ':/' '--'; }

loaded_models() {
  curl -sf "$OLLAMA_URL/api/ps" 2>/dev/null | python3 -c '
import json, sys
try:
    doc = json.load(sys.stdin)
except Exception:
    sys.exit(1)
for entry in doc.get("models") or []:
    name = entry.get("name") or entry.get("model")
    if name:
        print(name)
'
}

# Ask the daemon to unload everything, then wait for /api/ps to report empty.
# Returns 1 if anything is still resident when the wait runs out.
unload_all() {
  local name resident
  while IFS= read -r name; do
    [ -n "$name" ] || continue
    say "unload $name"
    ollama stop "$name" >/dev/null 2>&1 || say "WARN 'ollama stop $name' failed"
  done < <(loaded_models)

  local i
  for ((i = 0; i < PS_WAIT_TRIES; i++)); do
    # A failed read is NOT an empty GPU. loaded_models exits non-zero when the
    # daemon is unreachable; treating that as "nothing resident" would march on
    # and probe a daemon that is not there.
    if ! resident=$(loaded_models); then
      say "WARN /api/ps unreadable"
      sleep "$PS_WAIT_SLEEP"
      continue
    fi
    [ -z "$resident" ] && return 0
    sleep "$PS_WAIT_SLEEP"
  done
  resident=$(echo "$resident" | tr '\n' ' ')
  say "WARN /api/ps still resident after $((PS_WAIT_TRIES * PS_WAIT_SLEEP))s: $resident"
  return 1
}

vram_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | head -1 | tr -cd '0-9'
}

# The profile must name the model we asked for. Cheap guard against a tag that
# resolved elsewhere; the amp2 pattern's served-blob identity check, in the
# shape this instrument allows.
profile_names() {
  python3 -c '
import json, sys
try:
    print((json.load(open(sys.argv[1])).get("model") or {}).get("name") or "")
except Exception:
    print("")
' "$1"
}

run_model() {
  local model=$1
  local slug started elapsed rc free named
  slug=$(slug_of "$model")
  started=$(date +%s)

  say "=== $model (slug $slug) ==="
  runlog start "$model" - 0

  if ! unload_all; then
    elapsed=$(( $(date +%s) - started ))
    say "SKIP $model — models still resident, refusing to share the GPU"
    runlog skip "$model" 90 "$elapsed"
    return 0
  fi

  free=$(vram_free_mib)
  if [ -z "$free" ]; then
    elapsed=$(( $(date +%s) - started ))
    say "SKIP $model — nvidia-smi gave no free-memory reading"
    runlog skip "$model" 91 "$elapsed"
    return 0
  fi
  if [ "$free" -lt "$FLOOR_MIB" ]; then
    elapsed=$(( $(date +%s) - started ))
    say "SKIP $model — VRAM free ${free} MiB < floor ${FLOOR_MIB} MiB"
    runlog skip "$model" 91 "$elapsed"
    return 0
  fi
  say "preflight ok — ${free} MiB free (floor ${FLOOR_MIB})"

  "$ASSAY" probe "$OLLAMA_URL" --model "$model" --full \
    --tier "$TIER" --real-hardware \
    --record "$EVDIR/$slug-transcript.jsonl" \
    --json "$EVDIR/$slug.json"
  rc=$?
  elapsed=$(( $(date +%s) - started ))

  if [ "$rc" -eq 0 ]; then
    named=$(profile_names "$EVDIR/$slug.json")
    if [ "$named" != "$model" ]; then
      say "IDENTITY MISMATCH for $model — profile names '$named'"
      runlog done "$model" 92 "$elapsed"
      return 0
    fi
  fi

  say "probe exit $rc after ${elapsed}s"
  runlog done "$model" "$rc" "$elapsed"
  return 0
}

cd "$REPO" || { echo "no such repo: $REPO"; exit 1; }

# Fail before the first unload rather than turning a bad path into 15 skips.
[ -x "$ASSAY" ] || { echo "assay not executable: $ASSAY"; exit 1; }
curl -sf "$OLLAMA_URL/api/tags" >/dev/null || {
  echo "ollama unreachable at $OLLAMA_URL"; exit 1; }

mkdir -p "$EVDIR"
rm -f "$EVDIR/.DONE"
echo $$ > "$PIDFILE"

ASSAY_VERSION=$("$VENV_PY" -c 'import assay; print(assay.__version__)' 2>&1)
ASSAY_COMMIT=$(git -C "$REPO" rev-parse HEAD 2>&1)

say "=== tier re-profile campaign 2026-08 START (pid $$, ${#MODELS[@]} models) ==="
say "assay: $ASSAY (version $ASSAY_VERSION, commit $ASSAY_COMMIT)"

# The same provenance into the RUN LOG, not just stdout. The run log is the
# durable timing record the estimate and any later re-analysis are read from,
# and an editable install means a version string alone does not identify the
# code that ran — the commit does.
runlog_note "campaign 2026-08 start $(date -Is)"
runlog_note "assay_bin=$ASSAY"
runlog_note "assay_version=$ASSAY_VERSION"
runlog_note "assay_commit=$ASSAY_COMMIT"
runlog_note "tier=$TIER mode=full vram_floor_mib=$FLOOR_MIB models=${#MODELS[@]}"

for m in "${MODELS[@]}"; do
  run_model "$m"
done

say "unloading the last model"
unload_all || say "WARN GPU not left empty"

say "=== tier re-profile campaign 2026-08 END ==="
touch "$EVDIR/.DONE"
