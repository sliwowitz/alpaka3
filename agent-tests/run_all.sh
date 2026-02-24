#!/bin/bash
# Master run script for all agent tests.
# Runs 4 programs x 3 agents x 2 backends, records performance,
# and writes results to README.md.
#
# Usage:
#   ./run_all.sh                # run all available backends
#   ./run_all.sh --openmp       # OpenMP only
#   ./run_all.sh --cuda         # CUDA only
#   ./run_all.sh --runs N       # repeat each benchmark N times (default: 3)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
README="${SCRIPT_DIR}/README.md"
NUM_RUNS=3
TIMEOUT=300   # seconds per run

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[RUN]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; }
hdr()  { echo -e "\n${CYAN}========== $* ==========${NC}"; }

# --- parse arguments ---
RUN_OMP=true
RUN_CUDA=true

while [ $# -gt 0 ]; do
    case "$1" in
        --openmp) RUN_CUDA=false ;;
        --cuda)   RUN_OMP=false ;;
        --runs)   shift; NUM_RUNS="$1" ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p "$RESULTS_DIR"

# ===================================================================
#  Executable map: agent -> (test_label, executable_name, args)
#
#  test labels:  vectorAdd  matrixAdd  matMul  matMulOpt
# ===================================================================

#                   test_label   executable_name           args
BLABLADOR_TESTS=(
    "vectorAdd      vectorAdd           "
    "matrixAdd      matrixAdd           "
    "matMul         matMul              1000 1000 1000"
    "matMulOpt      matMulOpt           1000 1000 1000"
)

CLAUDE_TESTS=(
    "vectorAdd      vectorAdd10M        "
    "matrixAdd      matrixAdd           "
    "matMul         matMul              1000 1000 1000"
    "matMulOpt      matMulOpt           1000 1000 1000"
)

VIBE_TESTS=(
    "vectorAdd      vectorAdd10M        "
    "matrixAdd      matrixAdd           "
    "matMul         matrixMul           1000 1000 1000"
    "matMulOpt      matrixMulOptimized  1000 1000 1000"
)

# ===================================================================
#  Kernel-time extraction
#  Handles all three agents' output formats:
#    "Kernel time: 123.45 ms"           (blablador, claude)
#    "Kernel execution time: 123.45 ms" (vibe)
# ===================================================================

extract_kernel_ms() {
    local output="$1"
    # CpuSerial executor is disabled at build time, so each program
    # produces output for exactly one backend (OmpBlocks or GpuCuda).
    local ms
    ms=$(echo "$output" | grep -oP '(?i)kernel\s*(execution\s*)?time:\s*\K[0-9]+(\.[0-9]+)?' | head -1)
    echo "${ms:-N/A}"
}

extract_total_ms() {
    local output="$1"
    local ms
    ms=$(echo "$output" | grep -oP '(?i)(total\s*(time\s*\(including transfers\)|computation time)):\s*\K[0-9]+(\.[0-9]+)?' | head -1)
    echo "${ms:-N/A}"
}

extract_validation() {
    local output="$1"
    if echo "$output" | grep -qi "PASSED\|completed successfully\|Validation.*ok"; then
        echo "PASS"
    elif echo "$output" | grep -qi "FAILED\|MISMATCH\|error"; then
        echo "FAIL"
    else
        echo "OK"
    fi
}

# ===================================================================
#  Run a single benchmark, return best kernel time over NUM_RUNS
# ===================================================================

# Global associative arrays for results
declare -A RESULT_KERNEL   # key = "agent:backend:test"
declare -A RESULT_TOTAL
declare -A RESULT_WALL
declare -A RESULT_STATUS
declare -A RESULT_VALID

run_benchmark() {
    local agent="$1"
    local backend="$2"    # openmp or cuda
    local test_label="$3"
    local exe_name="$4"
    local args="$5"

    local build_dir="${SCRIPT_DIR}/${agent}/build-${backend}"
    local exe="${build_dir}/${exe_name}"
    local key="${agent}:${backend}:${test_label}"
    local log_file="${RESULTS_DIR}/${agent}-${backend}-${test_label}.log"

    if [ ! -x "$exe" ]; then
        warn "  ${exe_name} not found — skipping"
        RESULT_KERNEL[$key]="N/A"
        RESULT_TOTAL[$key]="N/A"
        RESULT_WALL[$key]="N/A"
        RESULT_STATUS[$key]="SKIP"
        RESULT_VALID[$key]="-"
        return
    fi

    log "  ${agent}/${backend}/${test_label} (${NUM_RUNS} runs)"

    local best_kernel="N/A"
    local best_total="N/A"
    local best_wall=999999
    local last_output=""
    local status="OK"
    local run_ok=0

    for ((r = 1; r <= NUM_RUNS; r++)); do
        local t_start t_end wall_s
        t_start=$(date +%s%N)

        local output
        if output=$(timeout "$TIMEOUT" "$exe" $args 2>&1); then
            t_end=$(date +%s%N)
            wall_s=$(awk "BEGIN {printf \"%.2f\", ($t_end - $t_start) / 1000000000}")
            last_output="$output"
            run_ok=1

            local km tm
            km=$(extract_kernel_ms "$output")
            tm=$(extract_total_ms "$output")

            # Keep the best (lowest) kernel time
            if [ "$km" != "N/A" ]; then
                if [ "$best_kernel" = "N/A" ] || awk "BEGIN{exit(!($km < $best_kernel))}"; then
                    best_kernel="$km"
                fi
            fi

            if [ "$tm" != "N/A" ]; then
                if [ "$best_total" = "N/A" ] || awk "BEGIN{exit(!($tm < $best_total))}"; then
                    best_total="$tm"
                fi
            fi

            if awk "BEGIN{exit(!($wall_s < $best_wall))}"; then
                best_wall="$wall_s"
            fi
        else
            t_end=$(date +%s%N)
            last_output="$output"
            status="FAIL"
        fi
    done

    if [ $run_ok -eq 0 ]; then
        status="FAIL"
    fi

    # If the program has no built-in timing, use wall time in ms
    if [ "$best_kernel" = "N/A" ] && [ "$status" != "FAIL" ]; then
        best_kernel=$(awk "BEGIN {printf \"%.2f\", $best_wall * 1000}")
        best_kernel="${best_kernel} (wall)"
    fi

    local valid
    valid=$(extract_validation "$last_output")

    RESULT_KERNEL[$key]="$best_kernel"
    RESULT_TOTAL[$key]="$best_total"
    RESULT_WALL[$key]="$best_wall"
    RESULT_STATUS[$key]="$status"
    RESULT_VALID[$key]="$valid"

    # Save full output of last run
    echo "$last_output" > "$log_file"
}

# ===================================================================
#  Run all benchmarks for one agent
# ===================================================================

run_agent() {
    local agent="$1"
    local backend="$2"
    shift 2
    local -n tests=$1

    local build_dir="${SCRIPT_DIR}/${agent}/build-${backend}"
    if [ ! -d "$build_dir" ]; then
        warn "${agent}/build-${backend} not found — skipping"
        return
    fi

    log "${agent} / ${backend}"

    for entry in "${tests[@]}"; do
        local test_label exe_name args
        test_label=$(echo "$entry" | awk '{print $1}')
        exe_name=$(echo "$entry" | awk '{print $2}')
        args=$(echo "$entry" | awk '{$1=""; $2=""; print $0}' | xargs)
        run_benchmark "$agent" "$backend" "$test_label" "$exe_name" "$args"
    done
}

# ===================================================================
#  Main execution
# ===================================================================

BACKENDS=()
$RUN_OMP  && BACKENDS+=(openmp)
$RUN_CUDA && BACKENDS+=(cuda)

for backend in "${BACKENDS[@]}"; do
    hdr "Running ${backend^^} benchmarks"
    run_agent blablador "$backend" BLABLADOR_TESTS
    run_agent claude    "$backend" CLAUDE_TESTS
    run_agent vibe      "$backend" VIBE_TESTS
done

# ===================================================================
#  Generate README.md
# ===================================================================

hdr "Generating README.md"

AGENTS=(blablador claude vibe)
TESTS=(vectorAdd matrixAdd matMul matMulOpt)
TEST_DESCS=(
    "Vector Addition (10M elements)"
    "Matrix Addition (2000x5000)"
    "Matrix Multiplication (1000x1000, naive)"
    "Matrix Multiplication (1000x1000, optimized)"
)

{
cat <<'HEADER'
# Agent Tests — Performance Comparison

Comparison of alpaka benchmark implementations by three AI coding agents:
**blablador**, **claude**, and **vibe**.

Each agent independently wrote four programs:

| Program | Description |
|---------|-------------|
| vectorAdd | Addition of two 10M-element float vectors |
| matrixAdd | Element-wise addition of 2000x5000 float matrices |
| matMul | Naive O(n³) matrix multiplication (1000x1000) |
| matMulOpt | Optimized matrix multiplication (1000x1000) |

All programs are built with alpaka and tested with both OpenMP (CPU) and CUDA (GPU) backends.

HEADER

echo "## Test Environment"
echo ""
echo "- **Date:** $(date '+%Y-%m-%d %H:%M')"
echo "- **Host:** $(hostname 2>/dev/null || echo unknown)"
echo "- **CPU:** $(lscpu 2>/dev/null | grep 'Model name' | sed 's/.*:\s*//' | head -1 || echo unknown)"
echo "- **Cores:** $(nproc 2>/dev/null || echo unknown)"

if command -v nvidia-smi &>/dev/null; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "- **GPU:** ${gpu_name:-unknown}"
fi

echo "- **Runs per benchmark:** ${NUM_RUNS} (best of N)"
echo ""

# --- Per-backend tables ---
for backend in "${BACKENDS[@]}"; do
    echo "## ${backend^^} Results"
    echo ""
    echo "Kernel times in milliseconds (best of ${NUM_RUNS} runs). Lower is better."
    echo ""

    # Table header
    printf "| %-40s |" "Test"
    for agent in "${AGENTS[@]}"; do
        printf " %-14s |" "$agent"
    done
    echo ""

    # Separator
    printf "|%-42s|" "$(printf -- '-%.0s' {1..42})"
    for agent in "${AGENTS[@]}"; do
        printf "%-16s|" "$(printf -- '-%.0s' {1..16})"
    done
    echo ""

    # Rows
    for i in "${!TESTS[@]}"; do
        test="${TESTS[$i]}"
        desc="${TEST_DESCS[$i]}"
        printf "| %-40s |" "$desc"

        for agent in "${AGENTS[@]}"; do
            key="${agent}:${backend}:${test}"
            km="${RESULT_KERNEL[$key]:-N/A}"
            st="${RESULT_STATUS[$key]:-SKIP}"
            vl="${RESULT_VALID[$key]:--}"

            if [ "$st" = "FAIL" ]; then
                cell="FAIL"
            elif [ "$st" = "SKIP" ]; then
                cell="—"
            else
                cell="$km"
            fi

            printf " %-14s |" "$cell"
        done
        echo ""
    done

    echo ""

    # Validation table
    echo "### Validation"
    echo ""
    printf "| %-40s |" "Test"
    for agent in "${AGENTS[@]}"; do
        printf " %-14s |" "$agent"
    done
    echo ""
    printf "|%-42s|" "$(printf -- '-%.0s' {1..42})"
    for agent in "${AGENTS[@]}"; do
        printf "%-16s|" "$(printf -- '-%.0s' {1..16})"
    done
    echo ""

    for i in "${!TESTS[@]}"; do
        test="${TESTS[$i]}"
        desc="${TEST_DESCS[$i]}"
        printf "| %-40s |" "$desc"
        for agent in "${AGENTS[@]}"; do
            key="${agent}:${backend}:${test}"
            vl="${RESULT_VALID[$key]:--}"
            st="${RESULT_STATUS[$key]:-SKIP}"
            if [ "$st" = "SKIP" ]; then vl="—"; fi
            printf " %-14s |" "$vl"
        done
        echo ""
    done
    echo ""
done

# --- Notes section ---
cat <<'NOTES'
## Notes

- **Kernel time** = GPU/CPU compute time only (excludes memory transfers), as reported
  by each program. If a program does not report kernel time, wall-clock time is shown
  with a "(wall)" suffix.
- **Total time** includes host-device memory transfers where reported.
- All matrix multiplications use `float` (single precision) and 1000x1000 matrices.
- Each benchmark is run multiple times; the **best** (lowest) kernel time is reported.

## Algorithms

### blablador
- Uses compile-time `#ifdef` backend selection (OpenMP vs CUDA)
- 1D flat indexing with manual grid/block mapping
- matMulOpt: 16x16 tiled blocked multiplication with register accumulation

### claude
- Uses `executeForEachIfHasDevice` for runtime backend enumeration
- Native 2D alpaka `makeIdxMap` iteration patterns
- matMulOpt: shared-memory tiling (64x64 output tiles, 16-wide K-tiles)
  with 4x4 register micro-tiles and shared-memory accumulator

### vibe
- Uses compile-time `#ifdef` backend selection
- 1D flat indexing with manual block/grid decomposition
- matrixMulOptimized: 16x16 tiled multiplication with local tile copies

## Directory Structure

```
agent-tests/
├── build_all.sh          # master build script
├── run_all.sh            # master run + benchmark script
├── README.md             # this file (auto-generated)
├── results/              # raw output logs
├── blablador/            # blablador agent's implementation
│   ├── CMakeLists.txt
│   ├── vectorAdd.cpp
│   ├── matrixAdd.cpp
│   ├── matMul.cpp
│   └── matMulOpt.cpp
├── claude/               # claude agent's implementation
│   ├── CMakeLists.txt
│   └── src/
│       ├── vectorAdd10M.cpp
│       ├── matrixAdd.cpp
│       ├── matMul.cpp
│       └── matMulOpt.cpp
└── vibe/                 # vibe agent's implementation
    ├── CMakeLists.txt
    ├── vectorAdd.cpp
    ├── matrixAddSimple.cpp
    ├── matrixMulProper.cpp
    └── matrixMulOptimized.cpp
```
NOTES

} > "$README"

log "Results written to ${README}"
log "Raw logs in ${RESULTS_DIR}/"

# --- Print summary to terminal ---
hdr "Summary"
for backend in "${BACKENDS[@]}"; do
    echo ""
    echo "  ${backend^^}:"
    printf "  %-12s" ""
    for agent in "${AGENTS[@]}"; do
        printf "%-16s" "$agent"
    done
    echo ""
    printf "  %-12s" ""
    for agent in "${AGENTS[@]}"; do
        printf "%-16s" "$(printf -- '-%.0s' {1..14})"
    done
    echo ""

    for test in "${TESTS[@]}"; do
        printf "  %-12s" "$test"
        for agent in "${AGENTS[@]}"; do
            key="${agent}:${backend}:${test}"
            km="${RESULT_KERNEL[$key]:-—}"
            st="${RESULT_STATUS[$key]:-SKIP}"
            if [ "$st" = "FAIL" ]; then km="FAIL"; fi
            if [ "$st" = "SKIP" ]; then km="—"; fi
            printf "%-16s" "$km"
        done
        echo ""
    done
done
echo ""
