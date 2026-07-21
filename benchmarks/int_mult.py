import random
from deap import gp


# ==========================================================
# Dynamic bitwidth types
# ==========================================================
_bv_types = {}

def BV(n):
    if n not in _bv_types:
        _bv_types[n] = type(f"BV{n}", (), {})
    return _bv_types[n]


# ==========================================================
# Python-level implementations (for GP evaluation)
# ==========================================================
def _mask(n):
    return (1 << n) - 1


def _make_mul(x_bits, y_bits):
    out_bits = x_bits + y_bits
    def mul(a, b):
        return (a * b) & _mask(out_bits)
    return mul


def _make_add(x_bits, y_bits):
    out_bits = max(x_bits, y_bits) + 1
    def add(a, b):
        return (a + b) & _mask(out_bits)
    return add


def _make_sub(x_bits, y_bits):
    out_bits = max(x_bits, y_bits) + 1
    def sub(a, b):
        return (a - b) & _mask(out_bits)
    return sub


def _make_and(x_bits, y_bits):
    out_bits = max(x_bits, y_bits)
    def and_op(a, b):
        return (a & b) & _mask(out_bits)
    return and_op


def _make_or(x_bits, y_bits):
    out_bits = max(x_bits, y_bits)
    def or_op(a, b):
        return (a | b) & _mask(out_bits)
    return or_op


def _make_left_shift(x_bits, k):
    out_bits = x_bits + k
    def lshift(a):
        return (a << k) & _mask(out_bits)
    return lshift


def _make_right_shift(x_bits, k):
    def rshift(a):
        return (a >> k) & _mask(x_bits)
    return rshift


def _make_ite(val_bits):
    def ite_fn(cond, a, b):
        return a if cond else b
    return ite_fn


def _make_trunc(src_bits, dst_bits):
    def trunc(a):
        return a & _mask(dst_bits)
    return trunc


# ==========================================================
# Primitive set builder
# ==========================================================
def make_pset(ma_bits, mb_bits, output_bits, max_shift=None):
    """
    Build typed PrimitiveSet for approximate multiplier synthesis.
    ma:  BV(ma_bits)
    mb:  BV(mb_bits)
    out: BV(output_bits)
    """
    if max_shift is None:
        max_shift = max(ma_bits, mb_bits)

    # Bitwidths we need to support
    needed = set()
    needed.add(1)
    needed.add(ma_bits)
    needed.add(mb_bits)
    needed.add(output_bits)
    needed.add(ma_bits + mb_bits)  # mul output

    # left_shift outputs
    for k in range(1, max_shift + 1):
        needed.add(ma_bits + k)
        needed.add(mb_bits + k)

    # Iteratively expand for add/sub outputs
    prev = None
    cap = max(output_bits, ma_bits + mb_bits, ma_bits + max_shift) + 2
    while prev != needed:
        prev = set(needed)
        for x in list(prev):
            for y in list(prev):
                out = max(x, y) + 1
                if out <= cap:
                    needed.add(out)
        needed = {b for b in needed if b <= cap}

    pset = gp.PrimitiveSetTyped(
        "MAIN",
        [BV(ma_bits), BV(mb_bits)],
        BV(output_bits),
    )
    pset.renameArguments(ARG0="ma", ARG1="mb")

    # ---- mul: BV(x) * BV(y) -> BV(x+y) ----
    for x in needed:
        for y in needed:
            out = x + y
            if out in needed:
                pset.addPrimitive(_make_mul(x, y), [BV(x), BV(y)], BV(out),
                                  name=f"mul_{x}_{y}")

    # ---- add, sub ----
    for x in needed:
        for y in needed:
            out = max(x, y) + 1
            if out in needed:
                pset.addPrimitive(_make_add(x, y), [BV(x), BV(y)], BV(out),
                                  name=f"add_{x}_{y}")
                pset.addPrimitive(_make_sub(x, y), [BV(x), BV(y)], BV(out),
                                  name=f"sub_{x}_{y}")

    # ---- and, or ----
    for x in needed:
        for y in needed:
            out = max(x, y)
            if out in needed:
                pset.addPrimitive(_make_and(x, y), [BV(x), BV(y)], BV(out),
                                  name=f"and_{x}_{y}")
                pset.addPrimitive(_make_or(x, y), [BV(x), BV(y)], BV(out),
                                  name=f"or_{x}_{y}")

    # ---- left_shift ----
    for x in needed:
        for k in range(1, max_shift + 1):
            out = x + k
            if out in needed:
                pset.addPrimitive(_make_left_shift(x, k), [BV(x)], BV(out),
                                  name=f"lshift_{x}_by{k}")

    # ---- logic_right_shift ----
    for x in needed:
        for k in range(1, max_shift + 1):
            pset.addPrimitive(_make_right_shift(x, k), [BV(x)], BV(x),
                              name=f"rshift_{x}_by{k}")

    # ---- ite ----
    for w in needed:
        pset.addPrimitive(_make_ite(w),
                          [BV(1), BV(w), BV(w)], BV(w),
                          name=f"ite_{w}")

    # ---- trunc ----
    for src in needed:
        for dst in needed:
            if dst < src and dst >= 1:
                pset.addPrimitive(_make_trunc(src, dst), [BV(src)], BV(dst),
                                  name=f"trunc_{src}_{dst}")

# ---- Terminals: small constants for every bitwidth ----
    # 0 and 1 keep their historical names; 2..7 use const_{value}_{width}
    # so compensation constants are directly reachable by GP.
    for w in needed:
        pset.addTerminal(0, BV(w), name=f"zero_{w}")
        pset.addTerminal(1, BV(w), name=f"one_{w}")
        for c in range(2, min(8, 1 << w)):
            pset.addTerminal(c, BV(w), name=f"const_{c}_{w}")

    return pset


# ==========================================================
# Seed: simple truncated multiplier
# ==========================================================
def make_seed_str(ma_bits, mb_bits, output_bits):
    """
    Seed = trunc(mul(ma, mb), output_bits).
    Precise starting point; GP explores by reducing precision.
    """
    mul_out = ma_bits + mb_bits
    if mul_out > output_bits:
        return f"trunc_{mul_out}_{output_bits}(mul_{ma_bits}_{mb_bits}(ma, mb))"
    elif mul_out == output_bits:
        return f"mul_{ma_bits}_{mb_bits}(ma, mb)"
    else:
        raise ValueError(
            f"output_bits ({output_bits}) larger than mul output "
            f"({mul_out}). Extend seed if this configuration is needed."
        )


# ==========================================================
# Training data
# ==========================================================
def make_data(ma_bits, mb_bits, n_samples=100, seed=42, mantissa_domain=False):
    rng = random.Random(seed)
    lo_a = (1 << (ma_bits - 1)) if mantissa_domain else 0
    lo_b = (1 << (mb_bits - 1)) if mantissa_domain else 0
    data = []
    for _ in range(n_samples):
        ma = rng.randint(lo_a, (1 << ma_bits) - 1)
        mb = rng.randint(lo_b, (1 << mb_bits) - 1)
        data.append((ma, mb, ma * mb))
    return data

def make_block_seed_strs(ma_bits, mb_bits, output_bits):
    seeds = []
    n = ma_bits
    if not (ma_bits == mb_bits and n % 2 == 0 and output_bits == 2 * n):
        return seeds
    h = n // 2

    aH = f"trunc_{n}_{h}(rshift_{n}_by{h}(ma))"
    aL = f"trunc_{n}_{h}(ma)"
    bH = f"trunc_{n}_{h}(rshift_{n}_by{h}(mb))"
    bL = f"trunc_{n}_{h}(mb)"

    hh = f"lshift_{n}_by{n}(mul_{h}_{h}({aH}, {bH}))"          # BV(2n)
    hl = f"lshift_{n}_by{h}(mul_{h}_{h}({aH}, {bL}))"          # BV(n+h)
    lh = f"lshift_{n}_by{h}(mul_{h}_{h}({aL}, {bH}))"          # BV(n+h)

    # seed "high-only": keep only aH*bH (cheapest corner of the space)
    seeds.append(hh)

    # seed "drop-LL": keep three terms, drop aL*bL
    s = f"add_{2*n}_{n+h}({hh}, {hl})"                         # BV(2n+1)
    s = f"add_{2*n+1}_{n+h}({s}, {lh})"                        # BV(2n+2)
    seeds.append(f"trunc_{2*n+2}_{2*n}({s})")

    return seeds