#!/bin/bash
# Build the benchmark venv on NERSC Perlmutter (AMD Zen3).

set -u
PG=${SPKNN_PLAYGROUND_ROOT:-/global/homes/d/dhaura/repos/SpKNN/spknn-playground}
cd "$PG"
source "$PG/common/bench_env_perlmutter.sh"

ARCH="$SPKNN_ARCH"

module load python/3.11-24.1.0 2>/dev/null
module load rust/stable 2>/dev/null

export RUSTUP_HOME=$SCRATCH/.rustup
export CARGO_HOME=$SCRATCH/.cargo
export PATH="$CARGO_HOME/bin:$PATH"

echo "=================== BUILD PROVENANCE ==================="
echo "date   : $(date -Is)"
echo "host   : $(hostname)"
echo "cpu    : $(lscpu | sed -n 's/^Model name: *//p')"
echo "avx    : $(lscpu | grep -o -E 'avx[0-9a-z_]*' | sort -u | tr '\n' ' ')"
echo "python : $(python3 -V 2>&1)  ($(which python3))"
echo "gcc    : $(gcc --version | head -1)"
echo "rustc  : $(rustc --version 2>/dev/null)"
echo "arch   : $ARCH"
echo "venv   : $SPKNN_VENV"
echo "========================================================"

if ! timeout 30 curl -sSI https://pypi.org/simple/ >/dev/null 2>&1; then
    echo "FATAL: no outbound network from $(hostname)." >&2
    echo "Run this on a Perlmutter login node, not a compute node." >&2
    exit 1
fi

rm -rf "$SPKNN_VENV"
mkdir -p "$(dirname "$SPKNN_VENV")"
python3 -m venv "$SPKNN_VENV"
source "$SPKNN_VENV/bin/activate"
pip install -q --upgrade pip setuptools wheel

declare -A STATUS

step() {  # step <name> <command...>
    local name=$1; shift
    echo; echo "=== $name ==="
    if "$@"; then STATUS[$name]=OK; else STATUS[$name]=FAILED; fi
}

step "base-stack" pip install -q numpy scipy pandas matplotlib pybind11

# --- kANNolo + SEISMIC (Rust) ---------------------------------------------
build_rust_pkgs() {
    rustup toolchain install nightly --profile minimal || return 1
    export RUSTUP_TOOLCHAIN=nightly       # kANNolo pins nightly
    export RUSTFLAGS="-C target-cpu=$ARCH"
    export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-16}"
    pip install --no-deps --no-binary :all: --no-cache-dir \
        kannolo==0.7.0 pyseismic-lsr==0.5.1
}
step "kannolo+seismic" build_rust_pkgs

# --- GrassRMA sparse_hnswlib (pybind11) ------------------------------------
build_grassrma() {
    cd "$SPKNN_GRASSRMA_SRC" || return 1
    export HNSWLIB_NO_NATIVE=1            # stop setup.py forcing -march=native
    export CFLAGS="-march=$ARCH -mtune=$ARCH"
    export CXXFLAGS="$CFLAGS"
    pip install --no-deps --no-cache-dir .
}
step "grassRMA" build_grassrma

# --- PyANNS (pybind11) -----------------------------------------------------
build_pyanns() {
    cd "$SPKNN_PYANNS_SRC/python" || return 1
    rm -rf build dist
    python setup.py bdist_wheel >/dev/null || return 1
    pip install --no-deps --force-reinstall --no-cache-dir dist/pyanns-*.whl
}
step "pyanns" build_pyanns

# --- report ----------------------------------------------------------------
echo
echo "=================== BUILD SUMMARY ======================"
for k in base-stack kannolo+seismic grassRMA pyanns; do
    printf '%-20s %s\n' "$k" "${STATUS[$k]:-SKIPPED}"
done
echo "--------------------------------------------------------"
echo "compiled artefacts (zmm refs must be 0 -- Zen3 has no AVX-512):"
SITE=$(python -c "import site;print(site.getsitepackages()[0])")
for so in $(find "$SITE" -maxdepth 2 -name "*.so" 2>/dev/null | head -20); do
    ymm=$(objdump -d --no-show-raw-insn "$so" 2>/dev/null | grep -coE '%ymm[0-9]+')
    zmm=$(objdump -d --no-show-raw-insn "$so" 2>/dev/null | grep -coE '%zmm[0-9]+')
    printf '  %-46s ymm=%-8s zmm=%s\n' "$(basename "$so")" "$ymm" "$zmm"
done
echo "========================================================"
