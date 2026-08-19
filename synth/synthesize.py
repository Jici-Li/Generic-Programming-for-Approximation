# Run a Verilog module through yosys's generic (technology-independent)
# synth pass and return its real gate count -- a real EDA tool's output,
# not a guessed formula. No standard-cell library is installed in this
# environment, so this reports cell COUNT (2-input primitive gates:
# AND/OR/NAND/NOR/XOR/XNOR/ANDNOT/ORNOT after generic tech-mapping), not
# um^2 -- still a real, unbiased area proxy, just not calibrated to a
# specific process node the way the Booth paper's TSMC 5nm numbers are.
import re
import subprocess
import tempfile
import os

_CELL_LINE = re.compile(r'^\s*(\d+)\s+cells$')

# Keyed on the exact Verilog source -- GP reinjects/archives the same
# individuals across many generations (seeds, hall-of-fame, crossover
# reuse), so most fitness evaluations during a real run hit this cache
# instead of spawning yosys again. Process-lifetime only, not persisted.
_CACHE = {}


def synthesize_cells(verilog_src, top="top"):
    if verilog_src in _CACHE:
        return _CACHE[verilog_src]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
        f.write(verilog_src)
        path = f.name
    try:
        script = f"read_verilog {path}; synth -top {top}; stat"
        result = subprocess.run(
            ["yosys", "-p", script],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"yosys failed:\n{result.stdout}\n{result.stderr}")
        # `stat` prints one "N cells" line per submodule (in `=== name ===`
        # sections) BEFORE the real top-level total, which only appears in
        # the final "=== design hierarchy ===" section (counting cells
        # "including submodules"). For single-module designs there's only
        # one such line and first==last, but for designs with submodules
        # (e.g. minifloat_hardware's sigmul_/roundadd_/input_ieee_) taking
        # the FIRST match silently returns just one submodule's own cell
        # count instead of the whole design's -- so take the LAST match.
        cells = None
        for line in result.stdout.splitlines():
            m = _CELL_LINE.match(line)
            if m:
                cells = int(m.group(1))
        if cells is None:
            raise RuntimeError(f"could not find cell count in yosys output:\n{result.stdout}")
        _CACHE[verilog_src] = cells
        return cells
    finally:
        os.unlink(path)


def cache_stats():
    return {'entries': len(_CACHE)}
