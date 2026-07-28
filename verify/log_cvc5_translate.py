"""Translate a typed log-domain (benchmarks/log.py) GP tree into a CVC5
BitVector term.

The main operator is sadd_split{K}_{bits}: split both `bits`-wide two's-
complement operands into exactly two chunks -- the low K bits and the
high bits-K bits -- add each chunk pair *at chunk width* -- CVC5's
BITVECTOR_ADD wraps modulo 2**width by definition, which is exactly "drop
the carry crossing the split point" -- then CONCAT the two chunk results
back into a `bits`-wide result. K >= bits is the exact signed add (no
split at all): wrapping mod 2**bits with nowhere to drop a carry is just
ordinary two's-complement addition.

neg_{bits} is a saturating negate (only present so DEAP's typed tree
generation always has *something* to place at an SInt(exponent_bits)
internal-node position -- see benchmarks/log.py's make_pset).

scast/terminals reuse the same semantics as verify/mxint_hardware_cvc5_translate.py
(round-then-shift narrowing widen-as-sign-extend), duplicated here rather
than imported since log's scast has no clip step worth sharing code
over (narrowing an exponent by only enough to reach a slightly wider
output type never needs the saturating clamp mxint's scast does for its
much deeper multiplier-operand narrowings).
"""

from cvc5 import Kind

from verify.integer_cvc5_translate import cast_bv


def _signed_const(slv, width, value):
    return slv.mkBitVector(width, value & ((1 << width) - 1))


def gp_to_cvc5(individual, ea_bv, eb_bv, solver):
    nodes = list(individual)

    def convert(idx):
        node = nodes[idx]
        name = getattr(node, 'name', str(node))

        if not hasattr(node, 'arity') or node.arity == 0:
            if name in ('ea', 'ARG0'):
                return ea_bv, idx + 1
            if name in ('eb', 'ARG1'):
                return eb_bv, idx + 1
            if name.startswith('minus_one_'):
                w = int(name[len('minus_one_'):])
                return _signed_const(solver, w, -1), idx + 1
            if name.startswith('zero_'):
                w = int(name[len('zero_'):])
                return solver.mkBitVector(w, 0), idx + 1
            if name.startswith('one_'):
                w = int(name[len('one_'):])
                return solver.mkBitVector(w, 1), idx + 1
            raise ValueError(f"Unknown terminal: {name!r}")

        next_idx = idx + 1

        if name.startswith('sadd_split'):
            rest = name[len('sadd_split'):]
            split_str, bits_str = rest.split('_')
            split_bits, bits = int(split_str), int(bits_str)
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            a = cast_bv(solver, left, bits)
            b = cast_bv(solver, right, bits)
            if split_bits >= bits:
                # no split at all -- exact add, the carry has nowhere else to go.
                return solver.mkTerm(Kind.BITVECTOR_ADD, a, b), next_idx
            # exactly two chunks: low split_bits bits, high bits-split_bits
            # bits, each added *at that chunk's own width* (so CVC5's
            # native mod-2**width wraparound drops the carry crossing the
            # split point), then CONCAT back together -- matches
            # benchmarks/log.py's _make_sadd_split exactly.
            a_lo = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, split_bits - 1, 0), a)
            b_lo = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, split_bits - 1, 0), b)
            lo_sum = solver.mkTerm(Kind.BITVECTOR_ADD, a_lo, b_lo)
            a_hi = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, bits - 1, split_bits), a)
            b_hi = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, bits - 1, split_bits), b)
            hi_sum = solver.mkTerm(Kind.BITVECTOR_ADD, a_hi, b_hi)
            return solver.mkTerm(Kind.BITVECTOR_CONCAT, hi_sum, lo_sum), next_idx

        if name.startswith('scast_'):
            x, y = map(int, name[len('scast_'):].split('_'))
            child, next_idx = convert(next_idx)
            return cast_bv(solver, child, y), next_idx

        if name.startswith('neg_'):
            # saturating negate, matching benchmarks/log.py's
            # _clip_signed(-int(v), bits) -- plain two's-complement
            # negate wraps the most-negative value back to itself
            # instead of saturating, so this needs the explicit ITE
            # rather than a bare BITVECTOR_NEG.
            w = int(name[len('neg_'):])
            child, next_idx = convert(next_idx)
            child = cast_bv(solver, child, w)
            min_val = _signed_const(solver, w, -(1 << (w - 1)))
            max_val = _signed_const(solver, w, (1 << (w - 1)) - 1)
            is_min = solver.mkTerm(Kind.EQUAL, child, min_val)
            neg = solver.mkTerm(Kind.BITVECTOR_NEG, child)
            return solver.mkTerm(Kind.ITE, is_min, max_val, neg), next_idx

        raise ValueError(f"Unknown operator: {name!r}")

    result, _ = convert(0)
    return result
