#!/bin/bash
# Master build script for all agent tests.
# Builds 4 programs x 3 agents x 2 backends (OpenMP + CUDA).
#
# Usage:
#   ./build_all.sh              # build all available backends
#   ./build_all.sh --openmp     # OpenMP only
#   ./build_all.sh --cuda       # CUDA only
#   ./build_all.sh --clean      # remove all build directories first
#
# Environment:
#   CUDA_ARCH  — CUDA compute capability (default: 86)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NPROC=$(nproc 2>/dev/null || echo 4)
CUDA_ARCH="${CUDA_ARCH:-86}"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[BUILD]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; }
hdr()  { echo -e "\n${CYAN}========== $* ==========${NC}"; }

# --- parse arguments ---
BUILD_OMP=true
BUILD_CUDA=true
DO_CLEAN=false

for arg in "$@"; do
    case "$arg" in
        --openmp)  BUILD_CUDA=false ;;
        --cuda)    BUILD_OMP=false ;;
        --clean)   DO_CLEAN=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# --- detect backends ---
HAS_CUDA=false
if command -v nvcc &>/dev/null; then
    HAS_CUDA=true
    log "CUDA detected: $(nvcc --version 2>&1 | grep -i release | head -1)"
fi

if $BUILD_CUDA && ! $HAS_CUDA; then
    warn "CUDA requested but nvcc not found — skipping CUDA builds"
    BUILD_CUDA=false
fi

# --- clean ---
if $DO_CLEAN; then
    log "Cleaning all build directories..."
    rm -rf "${SCRIPT_DIR}"/blablador/build-{openmp,cuda}
    rm -rf "${SCRIPT_DIR}"/claude/build-{openmp,cuda}
    rm -rf "${SCRIPT_DIR}"/vibe/build-{openmp,cuda}
    log "Clean complete"
fi

# ===================================================================
#  Build helper — runs cmake configure + build, propagates errors
# ===================================================================

do_build() {
    local src_dir="$1"
    local build_dir="$2"
    shift 2
    # remaining args are cmake flags (serial executor always disabled)

    mkdir -p "$build_dir"

    local log_file="${build_dir}/build.log"

    if ! cmake -S "$src_dir" -B "$build_dir" -Dalpaka_EXEC_CpuSerial=OFF "$@" > "$log_file" 2>&1; then
        tail -20 "$log_file"
        return 1
    fi

    if ! cmake --build "$build_dir" -j"$NPROC" >> "$log_file" 2>&1; then
        tail -30 "$log_file"
        return 1
    fi

    return 0
}

# ===================================================================
#  Per-agent build functions
# ===================================================================

build_blablador() {
    local backend=$1            # OPENMP or CUDA
    local tag="${backend,,}"    # lowercase
    local build_dir="${SCRIPT_DIR}/blablador/build-${tag}"

    log "blablador / ${backend}"

    local cmake_flags=(
        -DBACKEND="${backend}"
        -DCMAKE_BUILD_TYPE=Release
    )
    if [ "$backend" = "OPENMP" ]; then
        cmake_flags+=( -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc )
    else
        cmake_flags+=( -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
                        -Dalpaka_DEP_OMP=OFF )
    fi

    do_build "${SCRIPT_DIR}/blablador" "$build_dir" "${cmake_flags[@]}"
    log "blablador / ${backend} — done"
}

build_claude() {
    local backend=$1
    local tag="${backend,,}"
    local build_dir="${SCRIPT_DIR}/claude/build-${tag}"

    log "claude / ${backend}"

    local cmake_flags=(
        -Dalpaka_USE_SOURCE_TREE=ON
        -Dalpaka_BUILD_EXAMPLES=OFF
        -Dalpaka_BUILD_BENCHMARKS=OFF
        -Dalpaka_TESTING=OFF
        -DCMAKE_BUILD_TYPE=Release
    )
    if [ "$backend" = "OPENMP" ]; then
        cmake_flags+=( -Dalpaka_DEP_OMP=ON -Dalpaka_DEP_CUDA=OFF
                        -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc )
    else
        cmake_flags+=( -Dalpaka_DEP_CUDA=ON -Dalpaka_DEP_OMP=OFF
                        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" )
    fi

    do_build "${SCRIPT_DIR}/claude" "$build_dir" "${cmake_flags[@]}"
    log "claude / ${backend} — done"
}

build_vibe() {
    local backend=$1
    local tag="${backend,,}"
    local build_dir="${SCRIPT_DIR}/vibe/build-${tag}"

    log "vibe / ${backend}"

    local cmake_flags=(
        -DBACKEND="${backend}"
        -DCMAKE_BUILD_TYPE=Release
    )
    if [ "$backend" = "OPENMP" ]; then
        cmake_flags+=( -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc )
    else
        cmake_flags+=( -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
                        -Dalpaka_DEP_OMP=OFF )
    fi

    do_build "${SCRIPT_DIR}/vibe" "$build_dir" "${cmake_flags[@]}"
    log "vibe / ${backend} — done"
}

# ===================================================================
#  Run builds
# ===================================================================

FAIL=0

build_one() {
    # $1 = agent  $2 = backend
    if "build_$1" "$2"; then
        return 0
    else
        err "$1 / $2 FAILED"
        FAIL=1
        return 1
    fi
}

if $BUILD_OMP; then
    hdr "OpenMP builds"
    build_one blablador OPENMP || true
    build_one claude    OPENMP || true
    build_one vibe      OPENMP || true
fi

if $BUILD_CUDA; then
    hdr "CUDA builds"
    build_one blablador CUDA || true
    build_one claude    CUDA || true
    build_one vibe      CUDA || true
fi

hdr "Summary"
for agent in blablador claude vibe; do
    for tag in openmp cuda; do
        dir="${SCRIPT_DIR}/${agent}/build-${tag}"
        if [ -d "$dir" ]; then
            n=$(find "$dir" -maxdepth 1 -type f -executable 2>/dev/null | wc -l)
            log "${agent}/build-${tag}: ${n} executables"
        fi
    done
done

if [ $FAIL -ne 0 ]; then
    err "Some builds failed — check output above"
    exit 1
fi

log "All builds completed successfully"
