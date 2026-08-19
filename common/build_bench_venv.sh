#!/bin/bash

set -euo pipefail

VENV=/scratch/user/dhaura/benchmarks/SpKNN/bench-venv
GRASSRMA_SRC=/scratch/user/dhaura/benchmarks/SpKNN/GrassRMA
PYANNS_SRC=/scratch/user/dhaura/repos/pyanns
ARCH=cascadelake

module purge
module load GCCcore/13.2.0
module load Python/3.11.5

echo "=================== BUILD PROVENANCE ==================="
echo "date   : $(date -Is)"
echo "host   : $(hostname)"
echo "cpu    : $(lscpu | sed -n 's/^Model name: *//p')"
echo "python : $(python3 -V 2>&1)  ($(which python3))"
echo "gcc    : $(gcc --version | head -1)"
echo "arch   : $ARCH"
echo "========================================================"

if ! timeout 30 curl -sSI https://pypi.org/simple/ >/dev/null 2>&1; then
    echo "FATAL: no outbound network from $(hostname)." >&2
    echo "Grace compute nodes have no egress -- run this on a login node." >&2
    exit 1
fi

rm -rf "$VENV"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip setuptools wheel

echo
echo "=== base scientific stack ==="
pip install -q numpy scipy pandas matplotlib pybind11

echo
echo "=== kANNolo + SEISMIC (Rust, target-cpu=$ARCH) ==="
export PATH="$HOME/.cargo/bin:$PATH"
export RUSTUP_TOOLCHAIN=nightly          # kANNolo pins nightly in rust-toolchain.toml
export RUSTFLAGS="-C target-cpu=$ARCH"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"   # capped: shared login node
pip install --no-deps --no-binary :all: --no-cache-dir \
    kannolo==0.7.0 pyseismic-lsr==0.5.1

echo
echo "=== GrassRMA sparse_hnswlib (pybind11, -march=$ARCH) ==="
(
  cd "$GRASSRMA_SRC"
  export HNSWLIB_NO_NATIVE=1
  export CFLAGS="-march=$ARCH -mtune=$ARCH"
  export CXXFLAGS="$CFLAGS"
  pip install --no-deps --no-cache-dir .
)

echo
echo "=== PyANNS (pybind11, -march=$ARCH) ==="
(
  cd "$PYANNS_SRC/python"
  export PYANNS_ARCH="$ARCH"
  rm -rf build dist
  python setup.py bdist_wheel >/dev/null
  pip install --no-deps --force-reinstall --no-cache-dir dist/pyanns-*.whl
)

echo
echo "=== compiled artefacts ==="
SITE=$(python -c "import site;print(site.getsitepackages()[0])")
for so in $(find "$SITE" -maxdepth 2 -name "*.so" \
            | grep -viE "numpy|scipy|pandas|matplotlib|pillow|PIL|kiwi|contour|fontTools|_cython" \
            | sort); do
    printf '%-46s ymm=%-7s zmm=%-6s sha=%s\n' "$(basename "$so")" \
        "$(objdump -d --no-show-raw-insn "$so" | grep -oE '%ymm[0-9]+' | wc -l)" \
        "$(objdump -d --no-show-raw-insn "$so" | grep -oE '%zmm[0-9]+' | wc -l)" \
        "$(sha256sum "$so" | cut -c1-16)"
done

echo
echo "=== import smoke test ==="
python -c "
import sys, numpy, scipy, pandas
print('python  ', sys.version.split()[0])
import sparse_hnswlib;               print('GrassRMA OK')
from kannolo import SparsePlainHNSW; print('kANNolo  OK')
from seismic import SeismicIndexRaw; print('SEISMIC  OK')
import pyanns;                       print('PyANNS   OK')
print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| pandas', pandas.__version__)
"

echo
echo "bench-venv ready at $VENV"
echo "Every ymm count above must be non-zero; a zero means that module was"
echo "built unoptimized and must not be benchmarked."
