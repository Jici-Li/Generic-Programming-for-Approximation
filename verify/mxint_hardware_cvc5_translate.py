from cvc5 import Kind

from verify.integer_cvc5_translate import cast_bv


def _signed_const(slv, width, value):
    """A width-bit BitVector constant for a possibly-negative Python int,
    via its two's-complement bit pattern."""
    return slv.mkBitVector(width, value & ((1 << width) - 1))


def _clip_signed_bv(slv, wide_val, dst_bits):
    """Clamp a wide signed BitVector term into the representable range of
    a narrower signed dst_bits width, then truncate -- mirrors
    benchmarks/mxint_hardware.py's _clip_signed. Safe to truncate after
    clamping since the clamped value is provably within dst_bits' range."""
    wide_bits = wide_val.getSort().getBitVectorSize()
    min_dst = -(1 << (dst_bits - 1))
    max_dst = (1 << (dst_bits - 1)) - 1
    min_c = _signed_const(slv, wide_bits, min_dst)
    max_c = _signed_const(slv, wide_bits, max_dst)
    below = slv.mkTerm(Kind.BITVECTOR_SLT, wide_val, min_c)
    above = slv.mkTerm(Kind.BITVECTOR_SGT, wide_val, max_c)
    clamped = slv.mkTerm(Kind.ITE, below, min_c,
                         slv.mkTerm(Kind.ITE, above, max_c, wide_val))
    return slv.mkTerm(slv.mkOp(Kind.BITVECTOR_EXTRACT, dst_bits - 1, 0), clamped)


def gp_to_cvc5(individual, ma_bv, mb_bv, solver):
    """Walk an mxint GP tree and produce a CVC5 BitVector term. ma_bv,
    mb_bv are the CVC5 constants for the two (signed) operands."""
    nodes = list(individual)

    def convert(idx):
        node = nodes[idx]
        name = getattr(node, 'name', str(node))

        if not hasattr(node, 'arity') or node.arity == 0:
            if name in ('ma', 'ARG0'):
                return ma_bv, idx + 1
            if name in ('mb', 'ARG1'):
                return mb_bv, idx + 1
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

        if name.startswith('smul_'):
            x, y = map(int, name[len('smul_'):].split('_'))
            out_bits = x + y
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            l = cast_bv(solver, left, out_bits)
            r = cast_bv(solver, right, out_bits)
            return solver.mkTerm(Kind.BITVECTOR_MULT, l, r), next_idx

        if name.startswith('sadd_') or name.startswith('ssub_'):
            op = name[:4]
            x, y = map(int, name[len(op) + 1:].split('_'))
            out_bits = max(x, y) + 1
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            l = cast_bv(solver, left, out_bits)
            r = cast_bv(solver, right, out_bits)
            kind = Kind.BITVECTOR_ADD if op == 'sadd' else Kind.BITVECTOR_SUB
            return solver.mkTerm(kind, l, r), next_idx

        if name.startswith('scast_'):
            x, y = map(int, name[len('scast_'):].split('_'))
            child, next_idx = convert(next_idx)
            if y >= x:
                return cast_bv(solver, child, y), next_idx
            shift = x - y
            wide_bits = x + 2  # headroom so the rounding bias can't overflow
            wide = cast_bv(solver, child, wide_bits)
            bias = _signed_const(solver, wide_bits, 1 << (shift - 1))
            biased = solver.mkTerm(Kind.BITVECTOR_ADD, wide, bias)
            shift_amount = solver.mkBitVector(wide_bits, shift)
            rounded = solver.mkTerm(Kind.BITVECTOR_ASHR, biased, shift_amount)
            return _clip_signed_bv(solver, rounded, y), next_idx

        if name.startswith('ashr_'):
            rest = name[len('ashr_'):]
            x_str, k_str = rest.split('_by')
            x, k = int(x_str), int(k_str)
            child, next_idx = convert(next_idx)
            shift_amount = solver.mkBitVector(x, k)
            return solver.mkTerm(Kind.BITVECTOR_ASHR, child, shift_amount), next_idx

        if name.startswith('sshl_'):
            rest = name[len('sshl_'):]
            x_str, k_str = rest.split('_by')
            x, k = int(x_str), int(k_str)
            out_bits = x + k
            child, next_idx = convert(next_idx)
            child_ext = cast_bv(solver, child, out_bits)
            shift_amount = solver.mkBitVector(out_bits, k)
            return solver.mkTerm(Kind.BITVECTOR_SHL, child_ext, shift_amount), next_idx

        raise ValueError(f"Unknown operator: {name!r}")

    result, _ = convert(0)
    return result
