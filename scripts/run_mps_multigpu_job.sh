#!/usr/bin/env bash
# Author: Cunxi Gong <ansatzMe@outlook.com>

set -euo pipefail

MODE=${1:?usage: run_mps_multigpu_job.sh MODE OUTPUT_DIR [job options...]}
OUTPUT_DIR=${2:?usage: run_mps_multigpu_job.sh MODE OUTPUT_DIR [job options...]}
shift 2

NPROC=${RENO_MULTIGPU_NPROC:?RENO_MULTIGPU_NPROC must be set to 1, 2, 4, or 8}
case "$NPROC" in 1|2|4|8) ;; *) echo "invalid RENO_MULTIGPU_NPROC=$NPROC" >&2; exit 64 ;; esac
VISIBLE=${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must explicitly reserve physical GPU indices or UUIDs}

probe_output=$(VISIBLE="$VISIBLE" NPROC="$NPROC" python - <<'PY'
import os
import subprocess
import time

def query(args):
    result = subprocess.run(["nvidia-smi", *args], text=True, capture_output=True)
    if result.returncode:
        raise SystemExit(result.stderr.strip() or "nvidia-smi query failed")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]

tokens = [token.strip() for token in os.environ["VISIBLE"].split(",") if token.strip()]
expected = int(os.environ["NPROC"])
if len(tokens) != expected or len(set(tokens)) != expected:
    raise SystemExit("CUDA_VISIBLE_DEVICES must contain exactly N unique entries")

samples = []
selections = []
for sample in range(2):
    inventory = {}
    for line in query(["--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"]):
        index, uuid, name = [item.strip() for item in line.split(",", 2)]
        inventory[index] = (index, uuid, name)
        inventory[uuid] = (index, uuid, name)
    try:
        selected = [inventory[token] for token in tokens]
    except KeyError as error:
        raise SystemExit(f"unknown physical GPU index or UUID: {error.args[0]}")
    if len({item[1] for item in selected}) != expected:
        raise SystemExit("CUDA_VISIBLE_DEVICES resolves to duplicate GPU UUIDs")
    if any("H100" not in item[2] for item in selected):
        raise SystemExit("formal Stage 4 jobs require H100 devices")
    selected_uuids = {item[1] for item in selected}
    applications = query(
        ["--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"],
    )
    for line in applications:
        fields = [item.strip() for item in line.split(",")]
        if fields and fields[0] in selected_uuids:
            raise SystemExit(f"selected GPU {fields[0]} has compute process {fields[1]}")
    current = {}
    for line in query(["--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]):
        uuid, memory, utilization = [item.strip() for item in line.split(",")]
        current[uuid] = (int(memory), int(utilization))
    samples.append(current)
    selections.append(selected)
    if sample == 0:
        time.sleep(2.0)
if selections[0] != selections[1]:
    raise SystemExit("selected GPU inventory changed between occupancy probes")
selected = selections[-1]
selected_uuids = {item[1] for item in selected}
for uuid in selected_uuids:
    if any(sample[uuid][0] > 64 for sample in samples):
        raise SystemExit(f"selected GPU {uuid} has nontrivial allocated memory")
    if all(sample[uuid][1] > 5 for sample in samples):
        raise SystemExit(f"selected GPU {uuid} has sustained utilization")

print(",".join(item[0] for item in selected) + "|" + ",".join(item[1] for item in selected))
PY
)
IFS='|' read -r PHYSICAL_INDICES GPU_UUIDS <<<"$probe_output"

lock_fds=()
IFS=',' read -ra uuid_list <<<"$GPU_UUIDS"
for uuid in "${uuid_list[@]}"; do
    exec {lock_fd}>"/tmp/renormalizer-${uuid}.lock"
    flock -n "$lock_fd" || { echo "GPU lock busy: $uuid" >&2; exit 75; }
    lock_fds+=("$lock_fd")
done

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

dmon_path="${OUTPUT_DIR}.nvidia-smi-dmon.txt"
nvidia-smi dmon -s pucvmte -d 1 -o T -i "$PHYSICAL_INDICES" >"$dmon_path" 2>&1 &
dmon_pid=$!
torchrun_pid=""
owned_pids=""
declare -A owned_start_identities=()
owned_pid_discovery_succeeded=true
owned_pid_discovery_error=""
owned_identity_revalidation_succeeded=true
owned_identity_errors=""
owned_listener_query_succeeded=false
owned_listener_query_error=""
owned_signal_errors=""
owned_torchrun_term_attempted=false
owned_torchrun_term_succeeded=false
owned_dmon_term_attempted=false
owned_dmon_term_succeeded=false
owned_torchrun_waited=false
owned_torchrun_wait_status=""
owned_dmon_waited=false
owned_dmon_wait_status=""
owned_dmon_stopped=false

process_start_identity() {
    local pid=$1
    local stat_line remainder
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    IFS= read -r stat_line <"/proc/${pid}/stat" || return 1
    remainder=${stat_line##*) }
    set -- $remainder
    [[ $# -ge 20 ]] || return 1
    printf '%s\n' "${20}"
}

remember_owned_pid() {
    local pid=$1
    local label=$2
    local identity
    if ! identity=$(process_start_identity "$pid"); then
        owned_pid_discovery_succeeded=false
        [[ -n "$owned_pid_discovery_error" ]] || owned_pid_discovery_error="${label} ${pid}: process start identity unavailable"
        return 1
    fi
    if [[ -n "${owned_start_identities[$pid]:-}" && "${owned_start_identities[$pid]}" != "$identity" ]]; then
        owned_identity_revalidation_succeeded=false
        owned_identity_errors+="${label} ${pid}: process start identity changed during discovery"$'\n'
        return 1
    fi
    owned_start_identities[$pid]=$identity
}

owned_process_is_current() {
    local pid=$1
    local expected=${owned_start_identities[$pid]:-}
    local current
    [[ -n "$expected" ]] || return 1
    current=$(process_start_identity "$pid") || return 1
    [[ "$current" == "$expected" ]]
}

signal_owned_process() {
    local pid=$1
    local label=$2
    local expected=${owned_start_identities[$pid]:-}
    local current
    if [[ -z "$expected" ]]; then
        printf '%s\n' "${label} ${pid}: recorded process start identity missing" >&2
        return 125
    fi
    if ! current=$(process_start_identity "$pid"); then
        printf '%s\n' "${label} ${pid}: process start identity unavailable before signal" >&2
        return 125
    fi
    if [[ "$current" != "$expected" ]]; then
        printf '%s\n' "${label} ${pid}: process start identity changed before signal" >&2
        return 125
    fi
    kill -TERM "$pid"
}

if ! remember_owned_pid "$dmon_pid" "dmon"; then
    owned_identity_revalidation_succeeded=false
fi

refresh_owned_pids() {
    [[ -n "$torchrun_pid" ]] || return 0
    if [[ -z "$owned_pids" ]]; then
        if remember_owned_pid "$torchrun_pid" "torchrun"; then
            owned_pids=$torchrun_pid
        else
            return 0
        fi
    fi
    while true; do
        traversal_parents=""
        for parent in $owned_pids; do
            if kill -0 "$parent" 2>/dev/null; then
                if ! owned_process_is_current "$parent"; then
                    owned_pid_discovery_succeeded=false
                    owned_identity_revalidation_succeeded=false
                    parent_error="parent ${parent}: process start identity changed or unavailable before descendant traversal"
                    [[ -n "$owned_pid_discovery_error" ]] || owned_pid_discovery_error=$parent_error
                    owned_identity_errors+="$parent_error"$'\n'
                    return 0
                fi
                traversal_parents="${traversal_parents:+$traversal_parents }$parent"
            fi
        done
        [[ -n "$traversal_parents" ]] || return 0
        set +e
        process_snapshot=$(ps -eo pid=,ppid= 2>&1)
        process_status=$?
        set -e
        if [[ $process_status -ne 0 ]]; then
            owned_pid_discovery_succeeded=false
            [[ -n "$owned_pid_discovery_error" ]] || owned_pid_discovery_error=${process_snapshot:-"ps process discovery failed"}
            return 0
        fi
        set +e
        children=$(printf '%s\n' "$process_snapshot" | awk -v parents="$traversal_parents" '
            BEGIN {count=split(parents, values, " "); for (i=1; i<=count; i++) parent[values[i]]=1}
            ($2 in parent) {print $1}
        ' | xargs 2>&1)
        parse_status=$?
        set -e
        if [[ $parse_status -ne 0 ]]; then
            owned_pid_discovery_succeeded=false
            [[ -n "$owned_pid_discovery_error" ]] || owned_pid_discovery_error=${children:-"process discovery parse failed"}
            return 0
        fi
        next="$owned_pids"
        for child in $children; do
            case " $next " in
                *" $child "*) ;;
                *)
                    if remember_owned_pid "$child" "child"; then
                        next="$next $child"
                    fi
                    ;;
            esac
        done
        [[ "$next" == "$owned_pids" ]] && break
        owned_pids=$next
    done
}

cleanup() {
    status=$?
    cleanup_failed=false
    trap - EXIT INT TERM
    refresh_owned_pids
    if [[ "$owned_pid_discovery_succeeded" != true ]]; then cleanup_failed=true; fi
    if [[ -n "$torchrun_pid" ]] && kill -0 "$torchrun_pid" 2>/dev/null; then
        set +e
        signal_error=$(signal_owned_process "$torchrun_pid" "torchrun" 2>&1)
        signal_status=$?
        set -e
        if [[ $signal_status -ne 125 ]]; then owned_torchrun_term_attempted=true; fi
        if [[ $signal_status -eq 0 ]]; then
            owned_torchrun_term_succeeded=true
        else
            cleanup_failed=true
            owned_signal_errors+="torchrun ${torchrun_pid}: ${signal_error:-signal failed}"$'\n'
            if [[ $signal_status -eq 125 ]]; then
                owned_identity_revalidation_succeeded=false
                owned_identity_errors+="${signal_error:-torchrun identity revalidation failed}"$'\n'
            fi
        fi
    fi
    if [[ -n "$torchrun_pid" && "$owned_torchrun_waited" != true ]]; then
        set +e
        wait "$torchrun_pid"
        owned_torchrun_wait_status=$?
        set -e
        owned_torchrun_waited=true
        case "$owned_torchrun_wait_status" in 0|143) ;; *) cleanup_failed=true ;; esac
    fi
    for pid in $owned_pids; do
        if [[ "$pid" != "$torchrun_pid" ]] && kill -0 "$pid" 2>/dev/null; then
            set +e
            signal_error=$(signal_owned_process "$pid" "child" 2>&1)
            signal_status=$?
            set -e
            if [[ $signal_status -ne 0 ]]; then
                cleanup_failed=true
                owned_signal_errors+="child ${pid}: ${signal_error:-signal failed}"$'\n'
                if [[ $signal_status -eq 125 ]]; then
                    owned_identity_revalidation_succeeded=false
                    owned_identity_errors+="${signal_error:-child identity revalidation failed}"$'\n'
                fi
            fi
        fi
    done
    for _ in $(seq 1 20); do
        owned_alive=false
        for pid in $owned_pids; do
            if kill -0 "$pid" 2>/dev/null && owned_process_is_current "$pid"; then
                owned_alive=true
            fi
        done
        [[ "$owned_alive" == false ]] && break
        sleep 0.1
    done
    if kill -0 "$dmon_pid" 2>/dev/null; then
        set +e
        signal_error=$(signal_owned_process "$dmon_pid" "dmon" 2>&1)
        signal_status=$?
        set -e
        if [[ $signal_status -ne 125 ]]; then owned_dmon_term_attempted=true; fi
        if [[ $signal_status -eq 0 ]]; then
            owned_dmon_term_succeeded=true
        else
            cleanup_failed=true
            owned_signal_errors+="dmon ${dmon_pid}: ${signal_error:-signal failed}"$'\n'
            if [[ $signal_status -eq 125 ]]; then
                owned_identity_revalidation_succeeded=false
                owned_identity_errors+="${signal_error:-dmon identity revalidation failed}"$'\n'
            fi
        fi
    fi
    set +e
    wait "$dmon_pid"
    owned_dmon_wait_status=$?
    set -e
    owned_dmon_waited=true
    case "$owned_dmon_wait_status" in 0|143) ;; *) cleanup_failed=true ;; esac
    if kill -0 "$dmon_pid" 2>/dev/null && owned_process_is_current "$dmon_pid"; then
        cleanup_failed=true
        owned_dmon_stopped=false
    else
        owned_dmon_stopped=true
    fi
    owned_process_survivors=""
    verified_owned_pids=""
    for pid in $owned_pids; do
        if kill -0 "$pid" 2>/dev/null && owned_process_is_current "$pid"; then
            owned_process_survivors+="$pid"$'\n'
            verified_owned_pids+="$pid "
        fi
    done
    set +e
    listener_snapshot=$(ss -ltnp 2>&1)
    listener_query_status=$?
    set -e
    if [[ $listener_query_status -eq 0 ]]; then
        owned_listener_query_succeeded=true
    else
        cleanup_failed=true
        owned_listener_query_succeeded=false
        owned_listener_query_error=${listener_snapshot:-"ss listener query failed"}
        listener_snapshot=""
    fi
    owned_listener_survivors=$(OWNED_PIDS="$verified_owned_pids" LISTENERS="$listener_snapshot" python - <<'PY'
import os, re
owned = {int(value) for value in os.environ.get("OWNED_PIDS", "").split()}
for line in os.environ.get("LISTENERS", "").splitlines():
    if owned.intersection(int(value) for value in re.findall(r"pid=(\d+)", line)):
        print(line)
PY
)
    set +e
    compute_snapshot=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits 2>&1)
    compute_query_status=$?
    set -e
    compute_query_error=""
    survivors=""
    if [[ $compute_query_status -ne 0 ]]; then
        compute_query_error="$compute_snapshot"
        cleanup_failed=true
    else
        survivors=$(printf '%s\n' "$compute_snapshot" | awk -F, -v uuids=",$GPU_UUIDS," '{uuid=$1; gsub(/^[ \t]+|[ \t]+$/, "", uuid); if (index(uuids, "," uuid ",") > 0) print $0}')
    fi
    if [[ -n "$survivors" || -n "$owned_process_survivors" || -n "$owned_listener_survivors" ]]; then cleanup_failed=true; fi
    if [[ "$owned_identity_revalidation_succeeded" != true ]]; then cleanup_failed=true; fi
    if [[ "$cleanup_failed" == true && $status -eq 0 ]]; then status=1; fi
    owned_identity_records=""
    for pid in "${!owned_start_identities[@]}"; do
        owned_identity_records+="${pid}=${owned_start_identities[$pid]}"$'\n'
    done
    mkdir -p "$OUTPUT_DIR"
    STATUS="$status" SURVIVORS="$survivors" COMPUTE_QUERY_ERROR="$compute_query_error" OWNED_PIDS="$owned_process_survivors" OWNED_LISTENERS="$owned_listener_survivors" TORCHRUN_PID="$torchrun_pid" DMON_PID="$dmon_pid" GPU_UUIDS="$GPU_UUIDS" DMON_PATH="$dmon_path" OUTPUT_DIR="$OUTPUT_DIR" PID_DISCOVERY_SUCCEEDED="$owned_pid_discovery_succeeded" PID_DISCOVERY_ERROR="$owned_pid_discovery_error" IDENTITY_REVALIDATION_SUCCEEDED="$owned_identity_revalidation_succeeded" IDENTITY_ERRORS="$owned_identity_errors" PROCESS_START_IDENTITIES="$owned_identity_records" LISTENER_QUERY_SUCCEEDED="$owned_listener_query_succeeded" LISTENER_QUERY_ERROR="$owned_listener_query_error" SIGNAL_ERRORS="$owned_signal_errors" TORCHRUN_TERM_ATTEMPTED="$owned_torchrun_term_attempted" TORCHRUN_TERM_SUCCEEDED="$owned_torchrun_term_succeeded" DMON_TERM_ATTEMPTED="$owned_dmon_term_attempted" DMON_TERM_SUCCEEDED="$owned_dmon_term_succeeded" TORCHRUN_WAITED="$owned_torchrun_waited" TORCHRUN_WAIT_STATUS="$owned_torchrun_wait_status" DMON_WAITED="$owned_dmon_waited" DMON_WAIT_STATUS="$owned_dmon_wait_status" DMON_STOPPED="$owned_dmon_stopped" python - <<'PY'
import json, os, pathlib, tempfile
path = pathlib.Path(os.environ["OUTPUT_DIR"]) / "launcher-summary.json"
def boolean(name):
    return os.environ[name] == "true"
def optional_int(name):
    value = os.environ[name]
    return int(value) if value else None
def identities(name):
    result = {}
    for line in os.environ[name].splitlines():
        pid, identity = line.split("=", 1)
        result[pid] = identity
    return dict(sorted(result.items(), key=lambda item: int(item[0])))
payload = {
    "exit_code": int(os.environ["STATUS"]),
    "gpu_uuids": os.environ["GPU_UUIDS"].split(","),
    "dmon_path": os.environ["DMON_PATH"],
    "owned_dmon_stopped": boolean("DMON_STOPPED"),
    "owned_dmon_pid": int(os.environ["DMON_PID"]),
    "owned_dmon_term_attempted": boolean("DMON_TERM_ATTEMPTED"),
    "owned_dmon_term_succeeded": boolean("DMON_TERM_SUCCEEDED"),
    "owned_dmon_waited": boolean("DMON_WAITED"),
    "owned_dmon_wait_status": optional_int("DMON_WAIT_STATUS"),
    "owned_torchrun_pid": int(os.environ["TORCHRUN_PID"]) if os.environ["TORCHRUN_PID"] else None,
    "owned_torchrun_term_attempted": boolean("TORCHRUN_TERM_ATTEMPTED"),
    "owned_torchrun_term_succeeded": boolean("TORCHRUN_TERM_SUCCEEDED"),
    "owned_torchrun_waited": boolean("TORCHRUN_WAITED"),
    "owned_torchrun_wait_status": optional_int("TORCHRUN_WAIT_STATUS"),
    "owned_pid_discovery_succeeded": boolean("PID_DISCOVERY_SUCCEEDED"),
    "owned_pid_discovery_error": os.environ["PID_DISCOVERY_ERROR"] or None,
    "owned_identity_revalidation_succeeded": boolean("IDENTITY_REVALIDATION_SUCCEEDED"),
    "owned_identity_errors": os.environ["IDENTITY_ERRORS"].splitlines(),
    "owned_process_start_identities": identities("PROCESS_START_IDENTITIES"),
    "owned_listener_query_succeeded": boolean("LISTENER_QUERY_SUCCEEDED"),
    "owned_listener_query_error": os.environ["LISTENER_QUERY_ERROR"] or None,
    "owned_signal_errors": os.environ["SIGNAL_ERRORS"].splitlines(),
    "surviving_owned_pids": os.environ["OWNED_PIDS"].splitlines(),
    "surviving_owned_listeners": os.environ["OWNED_LISTENERS"].splitlines(),
    "surviving_selected_gpu_contexts": os.environ["SURVIVORS"].splitlines(),
    "selected_gpu_context_query_error": os.environ["COMPUTE_QUERY_ERROR"] or None,
}
with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
    json.dump(payload, stream, sort_keys=True, indent=2); stream.write("\n"); temporary = stream.name
os.replace(temporary, path)
PY
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    -m renormalizer.mps.distributed_job \
    --mode "$MODE" --output-dir "$OUTPUT_DIR" --expected-world-size "$NPROC" "$@" &
torchrun_pid=$!
if remember_owned_pid "$torchrun_pid" "torchrun"; then
    owned_pids=$torchrun_pid
else
    owned_identity_revalidation_succeeded=false
fi
for _ in $(seq 1 20); do
    refresh_owned_pids
    [[ $(wc -w <<<"$owned_pids") -ge $((NPROC + 1)) ]] && break
    kill -0 "$torchrun_pid" 2>/dev/null || break
    sleep 0.1
done
set +e
wait "$torchrun_pid"
torchrun_status=$?
set -e
owned_torchrun_waited=true
owned_torchrun_wait_status=$torchrun_status
exit "$torchrun_status"
