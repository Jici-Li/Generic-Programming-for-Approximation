"""
Area model for typed GP primitives.

Each node's area cost is derived from its bit-width, parsed directly
from the primitive name (e.g. 'add_4_5', 'mul_4_4', 'trunc_8_6').

Total area = sum of node costs.

Cost formulas are placeholders. Tune later based on real hardware
synthesis or published data.
"""

import os


# ==========================================================
# Op cost formulas (tunable)
# ==========================================================
# Each entry is a lambda that takes the relevant bit-width params
# and returns an integer area cost.
OP_COST_FN_QUADRATIC = {
    'add':    lambda bits_out: bits_out,                # ~one FA per bit
    'sub':    lambda bits_out: bits_out,                # same as add
    'mul':    lambda bits_x, bits_y: bits_x * bits_y,   # ~area square
    'and':    lambda bits_out: bits_out,                # one AND per bit
    'or':     lambda bits_out: bits_out,                # one OR per bit
    'lshift': lambda bits_in, k: 0,                     # hardwired shift ~ free
    'rshift': lambda bits_in, k: 0,                     # same
    'ite':    lambda bits_out: bits_out,                # one MUX per bit
    'trunc':  lambda src, dst: 0,                       # truncation = drop wires
}

# ---- symbolic/ranked cost model (diagnostic experiment) ----
# Tests whether GP ever discovers add/sub/ite-based compound rewrites once
# mul's cost stops being *quadratic* -- ranked purely by op "expense" (mul
# most expensive, then add/sub, then trunc/rcast, then ite, then shift=free)
# with LINEAR (not quadratic) width-scaling for every op, so narrowing still
# trades off against area (a flat, width-independent cost per op would make
# the single-node exact mul unbeatable for every alpha and collapse the
# whole experiment -- verified degenerate before choosing this instead).
# Weights are placeholders chosen only to preserve the requested ranking
# mul > add/sub > trunc/rcast > ite > shift(=0); not derived from synthesis.
_C_MUL   = int(os.environ.get('GP_C_MUL', 4))
_C_ADD, _C_TRUNC, _C_ITE = 3, 2, 1
OP_COST_FN_SYMBOLIC = {
    'add':    lambda bits_out: _C_ADD * bits_out,
    'sub':    lambda bits_out: _C_ADD * bits_out,
    'mul':    lambda bits_x, bits_y: _C_MUL * (bits_x + bits_y),  # linear, not x*y
    'and':    lambda bits_out: _C_ITE * bits_out,
    'or':     lambda bits_out: _C_ITE * bits_out,
    'lshift': lambda bits_in, k: 0,
    'rshift': lambda bits_in, k: 0,
    'ite':    lambda bits_out: _C_ITE * bits_out,
    'trunc':  lambda src, dst: _C_TRUNC,             # flat per-cast cost, not 0
}

SYMBOLIC_COST = os.environ.get('GP_SYMBOLIC_COST', '0') == '1'
OP_COST_FN = OP_COST_FN_SYMBOLIC if SYMBOLIC_COST else OP_COST_FN_QUADRATIC


# ==========================================================
# Parse primitive names into (kind, bit-width params)
# ==========================================================
def parse_op_name(name):
    """
    Parse a primitive/terminal name into (kind, params_tuple).

    Examples:
      'add_4_5'       -> ('add',   (4, 5))
      'sub_6_6'       -> ('sub',   (6, 6))
      'mul_4_4'       -> ('mul',   (4, 4))
      'and_4_4'       -> ('and',   (4, 4))
      'or_5_5'        -> ('or',    (5, 5))
      'lshift_4_by3'  -> ('lshift',(4, 3))
      'rshift_4_by1'  -> ('rshift',(4, 1))
      'trunc_8_6'     -> ('trunc', (8, 6))
      'ite_5'         -> ('ite',   (5,))
      'zero_4'        -> ('const', (4,))
      'one_4'         -> ('const', (4,))
      'ma', 'mb'      -> ('input', ())
    """
    # inputs
    if name in ('ma', 'mb', 'ARG0', 'ARG1'):
        return ('input', ())

    # constants
    if name.startswith('zero_') or name.startswith('one_'):
        w = int(name.split('_')[1])
        return ('const', (w,))
    
    # small constants: const_{value}_{width}
    if name.startswith('const_'):
        _, _val, w = name.split('_')
        return ('const', (int(w),))

    # trunc first (before add_/sub_ prefix match steal it)
    if name.startswith('trunc_'):
        src, dst = map(int, name[len('trunc_'):].split('_'))
        return ('trunc', (src, dst))

    # rcast: saturating round-to-nearest narrow (benchmarks/integer.py's
    # _make_rcast). Costed as free, same as trunc -- consistent with this
    # project's established "casts/shifts are free wires" convention used
    # for every other benchmark's cast primitive (mxint's scast, log's
    # scast, minifloat's fcast), even though a *real* round-then-clip
    # needs an adder + comparator unlike a pure bit-mask trunc. Kept
    # consistent on purpose so this experiment tests "does round beat
    # floor at equal nominal cost", not a different question about
    # rounding's own hardware cost.
    if name.startswith('rcast_'):
        src, dst = map(int, name[len('rcast_'):].split('_'))
        return ('rcast', (src, dst))

    # block_mul_k: blockmul_K_MA_MB -- narrowed-operand multiply, one
    # atomic primitive per k (benchmarks/integer.py's _make_block_mul)
    if name.startswith('blockmul_'):
        k, ma, mb = map(int, name[len('blockmul_'):].split('_'))
        return ('blockmul', (k, ma, mb))

    # cross_mul_k: crossmul_hilo_K_MA_MB (aH*bL) / crossmul_lohi_K_MA_MB
    # (aL*bH) -- the two terms block_mul_k drops. Kept as distinct kinds
    # since their multiplied widths differ when ma_bits != mb_bits.
    # (benchmarks/integer.py's _make_cross_mul_hi_lo/_make_cross_mul_lo_hi)
    for prefix, kind in (('crossmul_hilo_', 'crossmul_hilo'),
                        ('crossmul_lohi_', 'crossmul_lohi')):
        if name.startswith(prefix):
            k, ma, mb = map(int, name[len(prefix):].split('_'))
            return (kind, (k, ma, mb))

    # ite (only one bit-width parameter)
    if name.startswith('ite_'):
        w = int(name[len('ite_'):])
        return ('ite', (w,))

    # shifts: lshift_X_byK  / rshift_X_byK
    for prefix in ('lshift_', 'rshift_'):
        if name.startswith(prefix):
            kind = prefix.rstrip('_')
            rest = name[len(prefix):]
            x_str, k_str = rest.split('_by')
            return (kind, (int(x_str), int(k_str)))

    # binary ops: add_X_Y / sub_X_Y / and_X_Y / or_X_Y / mul_X_Y
    for prefix in ('add_', 'sub_', 'and_', 'or_', 'mul_'):
        if name.startswith(prefix):
            kind = prefix.rstrip('_')
            x, y = map(int, name[len(prefix):].split('_'))
            return (kind, (x, y))

    raise ValueError(f"Unknown primitive name: {name!r}")


# ==========================================================
# Cost of one node
# ==========================================================
def node_cost(name):
    """
    Compute the area cost of a single node given its primitive name.
    """
    kind, params = parse_op_name(name)

    # zero-cost cases
    if kind in ('input', 'const'):
        return 0

    # ------- binary ops that produce max(x,y) or max(x,y)+1 -------
    if kind in ('add', 'sub'):
        bits_out = max(params) + 1
        return OP_COST_FN[kind](bits_out)

    if kind in ('and', 'or'):
        bits_out = max(params)
        return OP_COST_FN[kind](bits_out)

    # ------- multiplication -------
    if kind == 'mul':
        return OP_COST_FN['mul'](*params)

    # ------- block_mul_k: same 'area ~ bits_x*bits_y' rule as mul, but on
    # the narrowed (ma-k, mb-k) operand widths actually being multiplied --
    if kind == 'blockmul':
        k, ma, mb = params
        return OP_COST_FN['mul'](ma - k, mb - k)

    # ------- cross_mul_k: aH*bL is an (ma-k)x k multiply, aL*bH is a
    # k x (mb-k) multiply -- same 'bits_x*bits_y' rule, different widths --
    if kind == 'crossmul_hilo':
        k, ma, mb = params
        return OP_COST_FN['mul'](ma - k, k)
    if kind == 'crossmul_lohi':
        k, ma, mb = params
        return OP_COST_FN['mul'](k, mb - k)

    # ------- shifts (parameterized by shift amount k) -------
    if kind in ('lshift', 'rshift'):
        return OP_COST_FN[kind](*params)

    # ------- ite (branch bit-width) -------
    if kind == 'ite':
        return OP_COST_FN['ite'](*params)

    # ------- trunc -------
    if kind == 'trunc':
        return OP_COST_FN['trunc'](*params)

    # ------- rcast (see parse_op_name's rcast branch for why this is
    # deliberately costed the same as trunc) -------
    if kind == 'rcast':
        return OP_COST_FN['trunc'](*params)

    raise ValueError(f"Unhandled kind: {kind}")


# ==========================================================
# Total area of a GP tree
# ==========================================================
def count_area(individual):
    """
    Sum up the area cost of all nodes in a DEAP GP tree (individual).
    """
    total = 0
    for node in individual:
        name = getattr(node, 'name', str(node))
        total += node_cost(name)
    return total


# ==========================================================
# Quick self-test
# ==========================================================
if __name__ == "__main__":
    # Check a bunch of primitive names to make sure parsing works
    tests = [
        # (name, expected_kind, expected_params, expected_cost)
        ('ma',            'input',  (),         0),
        ('mb',            'input',  (),         0),
        ('zero_4',        'const',  (4,),       0),
        ('one_5',         'const',  (5,),       0),
        ('add_4_5',       'add',    (4, 5),     6),   # max(4,5)+1 = 6
        ('sub_6_6',       'sub',    (6, 6),     7),   # max+1 = 7
        ('and_4_4',       'and',    (4, 4),     4),
        ('or_5_5',        'or',     (5, 5),     5),
        ('mul_4_4',       'mul',    (4, 4),     16),  # 4*4
        ('mul_4_8',       'mul',    (4, 8),     32),
        ('lshift_4_by3',  'lshift', (4, 3),     0),   # free
        ('rshift_5_by2',  'rshift', (5, 2),     0),   # free
        ('trunc_8_6',     'trunc',  (8, 6),     0),   # free
        ('ite_5',         'ite',    (5,),       5),
    ]

    print(f"{'name':<18} {'kind':<8} {'params':<12} {'cost':>4}")
    print("-" * 50)
    all_ok = True
    for name, exp_kind, exp_params, exp_cost in tests:
        kind, params = parse_op_name(name)
        cost = node_cost(name)
        ok = (kind == exp_kind and params == exp_params and cost == exp_cost)
        all_ok = all_ok and ok
        mark = "OK" if ok else "FAIL"
        print(f"{name:<18} {kind:<8} {str(params):<12} {cost:>4}  {mark}")

    print()
    if all_ok:
        print("[SUCCESS] All parse and cost tests passed.")
    else:
        print("[FAIL] Some tests failed. Check the failed lines.")