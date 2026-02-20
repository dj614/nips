#!/bin/bash
set -euo pipefail

# -------------------------------
# Required env (must be provided)
# -------------------------------
if [ -z "${MASTER_ADDR:-}" ]; then
  echo "ERROR: MASTER_ADDR is not set" >&2
  exit 1
fi

if [ -z "${RAY_PORT:-}" ]; then
  echo "ERROR: RAY_PORT is not set" >&2
  exit 1
fi

if ! [[ "${RAY_PORT}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: RAY_PORT must be numeric, got '${RAY_PORT}'" >&2
  exit 1
fi

# -------------------------------
# Optional env
# -------------------------------
: "${NNODES:=1}"
: "${RANK:=0}"
: "${RAY_CLEANUP:=1}"
: "${RAY_NODE_IP:=}"            # optional
: "${RAY_TMPDIR:=/tmp/ray}"      # user sets this in init.sh
: "${RAY_STOP_RETRIES:=5}"      # how many times to run ray stop --force
: "${RAY_STOP_INTERVAL:=2}"     # seconds between ray stop attempts
: "${RAY_PORT_FALLBACK_START:=20000}"  # fallback port range start (single-node only)
: "${RAY_PORT_FALLBACK_END:=40000}"    # fallback port range end (single-node only)
: "${RAY_PORT_FALLBACK_TRIES:=5}"     # number of random attempts

# Ensure Ray uses the caller-provided tmpdir (Ray 2.49.x supports env-based temp dir).
export RAY_TMPDIR="${RAY_TMPDIR}"

# Avoid accidentally connecting to another cluster via environment variables.
# These vars are sometimes set by launchers/frameworks.
unset RAY_ADDRESS || true
unset RAY_REDIS_ADDRESS || true
unset RAY_GCS_ADDRESS || true

# Safety: never delete anything under /sls-log
is_under_sls_log() {
  local p="$1"
  case "$p" in
    /sls-log|/sls-log/*) return 0 ;;
  esac
  if command -v realpath >/dev/null 2>&1; then
    local rp
    rp=$(realpath -m "$p" 2>/dev/null || true)
    case "$rp" in
      /sls-log|/sls-log/*) return 0 ;;
    esac
  fi
  return 1
}

ray_start_supports() {
  local flag="$1"
  ray start --help 2>/dev/null | grep -q -- "$flag"
}

# Best-effort: show what is listening on a port
show_port_owner() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -lntp 2>/dev/null | grep -E ":${port}\b" || true
    return 0
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP -sTCP:LISTEN -n -P 2>/dev/null | grep ":${port} " || true
    return 0
  fi

  # Fallback: python + /proc (works in minimal containers)
  if command -v python >/dev/null 2>&1; then
    PORT_TO_CHECK="${port}" python - <<'PY'
import os
import re
import sys

port = int(os.environ.get("PORT_TO_CHECK", "0"))
if port <= 0:
    print("(invalid port)")
    sys.exit(0)

# Convert port to hex (uppercase, no 0x), width 4.
port_hex = format(port, '04X')

# Parse /proc/net/tcp and /proc/net/tcp6 for LISTEN sockets on this port.
# Fields: sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode
# st == 0A means LISTEN.

def find_inodes(path):
    inodes = set()
    try:
        with open(path, 'r') as f:
            next(f, None)  # skip header
            for line in f:
                parts = line.split()
                if len(parts) < 10:
                    continue
                local_addr = parts[1]
                st = parts[3]
                inode = parts[9]
                # local_addr is like "0100007F:1F90"
                if st != '0A':
                    continue
                if local_addr.endswith(':' + port_hex):
                    inodes.add(inode)
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return inodes

inodes = set()
inodes |= find_inodes('/proc/net/tcp')
inodes |= find_inodes('/proc/net/tcp6')

if not inodes:
    print(f"(no LISTEN socket found for port {port})")
    sys.exit(0)

# Map socket inode -> PIDs by scanning /proc/<pid>/fd symlinks.
results = []
proc_dir = '/proc'

for pid in os.listdir(proc_dir):
    if not pid.isdigit():
        continue
    fd_dir = os.path.join(proc_dir, pid, 'fd')
    try:
        fds = os.listdir(fd_dir)
    except Exception:
        continue

    for fd in fds:
        p = os.path.join(fd_dir, fd)
        try:
            target = os.readlink(p)
        except Exception:
            continue
        m = re.match(r'socket:\[(\d+)\]$', target)
        if not m:
            continue
        inode = m.group(1)
        if inode in inodes:
            # Read cmdline
            cmdline = ''
            try:
                with open(os.path.join(proc_dir, pid, 'cmdline'), 'rb') as f:
                    raw = f.read().replace(b'\x00', b' ').strip()
                    cmdline = raw.decode('utf-8', errors='replace')
            except Exception:
                cmdline = '(cmdline unavailable)'

            # Read comm
            comm = ''
            try:
                with open(os.path.join(proc_dir, pid, 'comm'), 'r') as f:
                    comm = f.read().strip()
            except Exception:
                comm = '(comm unavailable)'

            results.append((int(pid), comm, cmdline, inode))
            break

# Deduplicate by pid
seen = set()
uniq = []
for r in sorted(results, key=lambda x: x[0]):
    if r[0] in seen:
        continue
    seen.add(r[0])
    uniq.append(r)

print(f"Listening processes for port {port}:")
for pid, comm, cmdline, inode in uniq:
    print(f"  pid={pid} comm={comm} inode={inode} cmdline={cmdline}")

if not uniq:
    print("  (could not map inode to a PID; permission may be restricted)")
PY
    return 0
  fi

  echo "(no ss/lsof/python available to inspect ports)"
  return 0
}

is_port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -lnt 2>/dev/null | awk '{print $4}' | grep -qE ":${port}$"
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP -sTCP:LISTEN -n -P 2>/dev/null | grep -q ":${port} "
  else
    # If we cannot check, assume not in use
    return 1
  fi
}

is_port_listening() {
  local host="$1"
  local port="$2"

  # Prefer ss/lsof when available (local-only)
  if [ "${host}" = "127.0.0.1" ] || [ "${host}" = "localhost" ]; then
    if is_port_in_use "${port}"; then
      return 0
    fi
  fi

  # Reliable fallback: python TCP connect test (works in minimal images)
  if command -v python >/dev/null 2>&1; then
    CHK_HOST="${host}" CHK_PORT="${port}" python - <<'PY'
import socket, os, sys
host = os.environ.get('CHK_HOST', '127.0.0.1')
port = int(os.environ.get('CHK_PORT', '0'))
try:
    s = socket.create_connection((host, port), timeout=0.5)
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
    return $?
  fi

  return 1
}

pick_free_port() {
  local start="$RAY_PORT_FALLBACK_START"
  local end="$RAY_PORT_FALLBACK_END"
  local tries="$RAY_PORT_FALLBACK_TRIES"

  if command -v python >/dev/null 2>&1; then
    START="${start}" END="${end}" TRIES="${tries}" python - <<'PY'
import os, random, socket, sys
start = int(os.environ.get('START','20000'))
end = int(os.environ.get('END','40000'))
tries = int(os.environ.get('TRIES','50'))
for _ in range(tries):
    p = random.randint(start, end)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(('127.0.0.1', p))
        s.close()
        print(p)
        sys.exit(0)
    except Exception:
        try:
            s.close()
        except Exception:
            pass
        continue
print(0)
sys.exit(0)
PY
    return 0
  fi

  # If python is not available, no safe way to pick; return 0
  echo 0
  return 0
}

ray_is_healthy() {
  # Return 0 if we can query ray status successfully.
  ray status >/dev/null 2>&1
}

ray_stop_until_clean() {
  local retries="$RAY_STOP_RETRIES"
  local interval="$RAY_STOP_INTERVAL"

  echo "cleanup: ray stop --force (up to ${retries} attempts)"

  for i in $(seq 1 "${retries}"); do
    echo "cleanup: ray stop attempt ${i}/${retries}"
    # Capture output to help debugging but do not fail the script on non-zero.
    local out
    out=$(ray stop --force 2>&1 || true)
    if [ -n "${out}" ]; then
      echo "cleanup: ray stop output: ${out}"
    fi

    # If ray reports no active processes, we still might have a lingering redis on the port.
    # Prefer checking port availability when possible.
    if ! is_port_listening "127.0.0.1" "${RAY_PORT}"; then
      echo "cleanup: port ${RAY_PORT} is free after ray stop"
      return 0
    fi

    echo "cleanup: port ${RAY_PORT} still in use; sleeping ${interval}s"
    sleep "${interval}"
  done

  echo "cleanup: ray stop retries exhausted; port ${RAY_PORT} still in use"
  return 1
}

cleanup_ray() {
  if [ "${RAY_CLEANUP}" != "1" ]; then
    echo "cleanup: skipped (RAY_CLEANUP=${RAY_CLEANUP})"
    return 0
  fi

  ray_stop_until_clean || true

  echo "cleanup: kill leftover ray processes"
  pkill -9 -f raylet >/dev/null 2>&1 || true
  pkill -9 -f gcs_server >/dev/null 2>&1 || true
  pkill -9 -f "ray::" >/dev/null 2>&1 || true
  # Try to only kill Ray-managed redis (avoid killing system redis)
  pkill -9 -f "redis-server.*ray" >/dev/null 2>&1 || true

  echo "cleanup: remove local ray state"
  for p in /tmp/ray /tmp/ray/* /tmp/ray_session* /tmp/ray/session_* /dev/shm/ray "${HOME}/.ray" "${RAY_TMPDIR}"; do
    if is_under_sls_log "$p"; then
      echo "cleanup: SKIP deleting $p (under /sls-log)"
      continue
    fi
    if [ -e "$p" ] || [ -L "$p" ]; then
      echo "cleanup: rm -rf $p"
      rm -rf "$p" 2>/dev/null || true
    fi
  done

  echo "cleanup: done"
}

get_node_count() {
  # Prefer `ray list nodes` when available (newer Ray).
  if ray list nodes >/dev/null 2>&1; then
    ray list nodes 2>/dev/null | grep -E "\bALIVE\b" | wc -l | tr -d ' '
    return 0
  fi

  # Fallback for older Ray: `ray status` prints active nodes as lines like:
  #   1 node_xxxxx...
  # Count those lines.
  # Avoid pipefail-induced double-output by not using `|| echo 0` in a pipeline.
  local n
  n=$(ray status 2>/dev/null | grep -E "^\s*[0-9]+\s+node_" | wc -l | tr -d ' ' || true)
  if [ -z "${n}" ]; then
    echo 0
  else
    echo "${n}"
  fi
  return 0
}

verify_nodes() {
  local expected="$NNODES"
  local retries=30
  local interval=2

  echo "verify: expecting ${expected} node(s)"
  for i in $(seq 1 "$retries"); do
    local alive
    alive=$(get_node_count)
    # Sanitize in case of unexpected whitespace
    alive=$(echo "${alive}" | tr -cd '0-9')
    if [ -z "${alive}" ]; then
      alive=0
    fi
    if [ "${alive}" -ge "${expected}" ]; then
      echo "verify: OK (${alive}/${expected})"
      return 0
    fi
    echo "verify: waiting... (${alive}/${expected}) attempt ${i}/${retries}"
    sleep "$interval"
  done

  echo "verify: FAILED"
  ray status 2>/dev/null || true
  return 1
}

init_ray() {
  echo "[Ray ENV] MASTER_ADDR=${MASTER_ADDR} RAY_PORT=${RAY_PORT} NNODES=${NNODES} RANK=${RANK} RAY_TMPDIR=${RAY_TMPDIR}"
  echo "[Ray ENV] effective RAY_TMPDIR=${RAY_TMPDIR}"
  env | grep -E '^RAY_' | sort || true

  if [ "${RANK}" -eq 0 ]; then
    echo "head: starting on ${MASTER_ADDR}:${RAY_PORT}"

    echo "head: preflight check port ${RAY_PORT}"
    if is_port_listening "127.0.0.1" "${RAY_PORT}"; then
      if ray_is_healthy; then
        echo "head: Ray already running on port ${RAY_PORT}; skipping start"
        verify_nodes
        return 0
      fi

      echo "head: port ${RAY_PORT} is in use but Ray is not healthy; attempting cleanup"
      echo "head: listener(s) on port ${RAY_PORT}:"
      show_port_owner "${RAY_PORT}" || true

      cleanup_ray

      echo "head: re-check port ${RAY_PORT} after cleanup"
      if is_port_listening "127.0.0.1" "${RAY_PORT}" && ! ray_is_healthy; then
        echo "head: port ${RAY_PORT} still in use and Ray still not healthy"
        echo "head: listener(s) on port ${RAY_PORT}:"
        show_port_owner "${RAY_PORT}" || true

        if [ "${NNODES}" != "1" ]; then
          echo "ERROR: Cannot auto-change RAY_PORT for multi-node runs (NNODES=${NNODES}). Please set a unique RAY_PORT externally." >&2
          exit 1
        fi

        local new_port
        new_port=$(pick_free_port | tr -cd '0-9')
        if [ -z "${new_port}" ] || [ "${new_port}" -eq 0 ]; then
          echo "ERROR: Failed to find a free fallback port in range ${RAY_PORT_FALLBACK_START}-${RAY_PORT_FALLBACK_END}." >&2
          exit 1
        fi

        echo "head: switching RAY_PORT from ${RAY_PORT} to ${new_port} (single-node fallback)"
        export RAY_PORT="${new_port}"
      fi
    else
      # Port not listening: still do cleanup to avoid stale on-disk session state causing mismatches.
      cleanup_ray
    fi

    # Build the most portable ray start command.
    local -a cmd
    cmd=(ray start --head --disable-usage-stats --port="${RAY_PORT}")

    if [ -n "${RAY_NODE_IP}" ] && ray_start_supports "--node-ip-address"; then
      echo "head: using node ip ${RAY_NODE_IP}"
      cmd+=(--node-ip-address="${RAY_NODE_IP}")
    fi

    echo "head: launching: ${cmd[*]}"
    "${cmd[@]}"

    echo "head: started"
    verify_nodes

  else
    echo "worker: connecting to ${MASTER_ADDR}:${RAY_PORT}"
    local -a wcmd
    wcmd=(ray start --address="${MASTER_ADDR}:${RAY_PORT}")

    if [ -n "${RAY_NODE_IP}" ] && ray_start_supports "--node-ip-address"; then
      wcmd+=(--node-ip-address="${RAY_NODE_IP}")
    fi

    echo "worker: launching: ${wcmd[*]}"
    "${wcmd[@]}" || true
    ray status >/dev/null 2>&1 || true
  fi
}
