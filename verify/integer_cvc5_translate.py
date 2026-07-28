"""
Translate a typed GP tree into a CVC5 BitVector term.

Each primitive's name encodes its bitwidth semantics, e.g.:
  mul_4_4       -> BITVECTOR_MULT with output width 4+4=8
  add_4_5       -> BITVECTOR_ADD  with output width max(4,5)+1=6
  lshift_4_by2  -> BITVECTOR_SHL by 2, output width 6
  trunc_8_6     -> BITVECTOR_EXTRACT [5:0]
  ite_6         -> BITVECTOR_ITE with cond promoted to bool
  zero_6/one_6  -> constants of the given width
"""

from cvc5 import Kind


def cast_bv(slv, x, n, signed=True):
    """Cast BitVector term x to width n (extend or extract)."""
    if n <= 0:
        raise ValueError("target width n must be positive")
    src_width = x.getSort().getBitVectorSize()
    if src_width == n:
        return x
    if src_width > n:
        return slv.mkTerm(slv.mkOp(Kind.BITVECTOR_EXTRACT, n - 1, 0), x)
    extension = n - src_width
    kind = Kind.BITVECTOR_SIGN_EXTEND if signed else Kind.BITVECTOR_ZERO_EXTEND
    return slv.mkTerm(slv.mkOp(kind, extension), x)


def gp_to_cvc5(individual, ma_bv, mb_bv, solver):
    """
    Walk the typed GP tree and produce a CVC5 BitVector term.
    ma_bv, mb_bv are the CVC5 constants representing the inputs.
    Returns the final term.
    """
    nodes = list(individual)

    def convert(idx):
        node = nodes[idx]
        name = getattr(node, 'name', str(node))

        # ---- leaves ----
        if not hasattr(node, 'arity') or node.arity == 0:
            if name in ('ma', 'ARG0'):
                return ma_bv, idx + 1
            if name in ('mb', 'ARG1'):
                return mb_bv, idx + 1
            if name.startswith('zero_'):
                w = int(name[len('zero_'):])
                return solver.mkBitVector(w, 0), idx + 1
            if name.startswith('one_'):
                w = int(name[len('one_'):])
                return solver.mkBitVector(w, 1), idx + 1
            if name.startswith('const_'):
                _, c_str, w_str = name.split('_')
                return solver.mkBitVector(int(w_str), int(c_str)), idx + 1
            raise ValueError(f"Unknown terminal: {name!r}")

        next_idx = idx + 1

        # ---- mul_x_y ----
        if name.startswith('mul_'):
            x, y = map(int, name[len('mul_'):].split('_'))
            out_bits = x + y
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            l = cast_bv(solver, left, out_bits, signed=False)
            r = cast_bv(solver, right, out_bits, signed=False)
            return solver.mkTerm(Kind.BITVECTOR_MULT, l, r), next_idx

        # ---- blockmul_K_MA_MB: drop low K bits of each operand, multiply,
        # shift back into position -- matches _make_block_mul in
        # benchmarks/integer.py: ((a>>K)*(b>>K)) << 2K on the
        # ma_bits+mb_bits output width.
        if name.startswith('blockmul_'):
            k_str, ma_str, mb_str = name[len('blockmul_'):].split('_')
            k, ma_bits, mb_bits = int(k_str), int(ma_str), int(mb_str)
            out_bits = ma_bits + mb_bits
            a, next_idx = convert(next_idx)
            b, next_idx = convert(next_idx)
            a_h = solver.mkTerm(Kind.BITVECTOR_LSHR, a, solver.mkBitVector(ma_bits, k))
            b_h = solver.mkTerm(Kind.BITVECTOR_LSHR, b, solver.mkBitVector(mb_bits, k))
            a_h = cast_bv(solver, a_h, out_bits, signed=False)
            b_h = cast_bv(solver, b_h, out_bits, signed=False)
            prod = solver.mkTerm(Kind.BITVECTOR_MULT, a_h, b_h)
            shift_amount = solver.mkBitVector(out_bits, 2 * k)
            return solver.mkTerm(Kind.BITVECTOR_SHL, prod, shift_amount), next_idx

        # ---- crossmul_hilo_K_MA_MB: aH*bL, shifted to line up with
        # blockmul_K's aH*bH term -- one of the two terms blockmul_K drops.
        if name.startswith('crossmul_hilo_'):
            k_str, ma_str, mb_str = name[len('crossmul_hilo_'):].split('_')
            k, ma_bits, mb_bits = int(k_str), int(ma_str), int(mb_str)
            out_bits = ma_bits + mb_bits
            a, next_idx = convert(next_idx)
            b, next_idx = convert(next_idx)
            a_h = solver.mkTerm(Kind.BITVECTOR_LSHR, a, solver.mkBitVector(ma_bits, k))
            mask = solver.mkBitVector(mb_bits, (1 << k) - 1)
            b_l = solver.mkTerm(Kind.BITVECTOR_AND, b, mask)
            a_h = cast_bv(solver, a_h, out_bits, signed=False)
            b_l = cast_bv(solver, b_l, out_bits, signed=False)
            prod = solver.mkTerm(Kind.BITVECTOR_MULT, a_h, b_l)
            shift_amount = solver.mkBitVector(out_bits, k)
            return solver.mkTerm(Kind.BITVECTOR_SHL, prod, shift_amount), next_idx

        # ---- crossmul_lohi_K_MA_MB: aL*bH, the other cross term. ----
        if name.startswith('crossmul_lohi_'):
            k_str, ma_str, mb_str = name[len('crossmul_lohi_'):].split('_')
            k, ma_bits, mb_bits = int(k_str), int(ma_str), int(mb_str)
            out_bits = ma_bits + mb_bits
            a, next_idx = convert(next_idx)
            b, next_idx = convert(next_idx)
            mask = solver.mkBitVector(ma_bits, (1 << k) - 1)
            a_l = solver.mkTerm(Kind.BITVECTOR_AND, a, mask)
            b_h = solver.mkTerm(Kind.BITVECTOR_LSHR, b, solver.mkBitVector(mb_bits, k))
            a_l = cast_bv(solver, a_l, out_bits, signed=False)
            b_h = cast_bv(solver, b_h, out_bits, signed=False)
            prod = solver.mkTerm(Kind.BITVECTOR_MULT, a_l, b_h)
            shift_amount = solver.mkBitVector(out_bits, k)
            return solver.mkTerm(Kind.BITVECTOR_SHL, prod, shift_amount), next_idx

        # ---- add_x_y ----
        if name.startswith('add_'):
            x, y = map(int, name[len('add_'):].split('_'))
            out_bits = max(x, y) + 1
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            l = cast_bv(solver, left, out_bits, signed=False)
            r = cast_bv(solver, right, out_bits, signed=False)
            return solver.mkTerm(Kind.BITVECTOR_ADD, l, r), next_idx

        # ---- sub_x_y ----
        if name.startswith('sub_'):
            x, y = map(int, name[len('sub_'):].split('_'))
            out_bits = max(x, y) + 1
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            l = cast_bv(solver, left, out_bits, signed=False)
            r = cast_bv(solver, right, out_bits, signed=False)
            return solver.mkTerm(Kind.BITVECTOR_SUB, l, r), next_idx

        # ---- and_x_y ----
        if name.startswith('and_'):
            x, y = map(int, name[len('and_'):].split('_'))
            out_bits = max(x, y)
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            l = cast_bv(solver, left, out_bits, signed=False)
            r = cast_bv(solver, right, out_bits, signed=False)
            return solver.mkTerm(Kind.BITVECTOR_AND, l, r), next_idx

        # ---- or_x_y ----
        if name.startswith('or_'):
            x, y = map(int, name[len('or_'):].split('_'))
            out_bits = max(x, y)
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            l = cast_bv(solver, left, out_bits, signed=False)
            r = cast_bv(solver, right, out_bits, signed=False)
            return solver.mkTerm(Kind.BITVECTOR_OR, l, r), next_idx

        # ---- lshift_X_byK ----
        if name.startswith('lshift_'):
            rest = name[len('lshift_'):]
            x_str, k_str = rest.split('_by')
            x, k = int(x_str), int(k_str)
            out_bits = x + k
            child, next_idx = convert(next_idx)
            child_ext = cast_bv(solver, child, out_bits, signed=False)
            shift_amount = solver.mkBitVector(out_bits, k)
            return solver.mkTerm(Kind.BITVECTOR_SHL, child_ext, shift_amount), next_idx

        # ---- rshift_X_byK ----
        if name.startswith('rshift_'):
            rest = name[len('rshift_'):]
            x_str, k_str = rest.split('_by')
            x, k = int(x_str), int(k_str)
            child, next_idx = convert(next_idx)
            shift_amount = solver.mkBitVector(x, k)
            return solver.mkTerm(Kind.BITVECTOR_LSHR, child, shift_amount), next_idx

        # ---- ite_W ----
        if name.startswith('ite_'):
            w = int(name[len('ite_'):])
            cond, next_idx = convert(next_idx)
            a, next_idx = convert(next_idx)
            b, next_idx = convert(next_idx)
            zero1 = solver.mkBitVector(1, 0)
            cond_bool = solver.mkTerm(Kind.DISTINCT, cond, zero1)
            a_cast = cast_bv(solver, a, w, signed=False)
            b_cast = cast_bv(solver, b, w, signed=False)
            return solver.mkTerm(Kind.ITE, cond_bool, a_cast, b_cast), next_idx

        # ---- trunc_SRC_DST ----
        if name.startswith('trunc_'):
            src, dst = map(int, name[len('trunc_'):].split('_'))
            child, next_idx = convert(next_idx)
            return cast_bv(solver, child, dst, signed=False), next_idx

        # ---- rcast_SRC_DST: saturating round-to-nearest (unlike trunc,
        # which wraps) -- matches benchmarks/integer.py's _make_rcast
        # exactly: add half-ulp at extra headroom width (so the add itself
        # can't overflow), shift right, then clamp to the destination's max
        # instead of letting the extra bit silently wrap. ----
        if name.startswith('rcast_'):
            src, dst = map(int, name[len('rcast_'):].split('_'))
            child, next_idx = convert(next_idx)
            shift = src - dst
            bias = 1 << (shift - 1)
            wide = cast_bv(solver, child, src + 1, signed=False)
            biased = solver.mkTerm(Kind.BITVECTOR_ADD, wide,
                                   solver.mkBitVector(src + 1, bias))
            shift_amount = solver.mkBitVector(src + 1, shift)
            rounded = solver.mkTerm(Kind.BITVECTOR_LSHR, biased, shift_amount)
            # the true value fits in dst+1 bits (rounding a src-bit value
            # can overflow the dst-bit range by at most 1)
            narrow = solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, dst, 0), rounded)
            max_val = solver.mkBitVector(dst + 1, (1 << dst) - 1)
            overflowed = solver.mkTerm(Kind.BITVECTOR_UGT, narrow, max_val)
            clamped = solver.mkTerm(Kind.ITE, overflowed, max_val, narrow)
            return solver.mkTerm(solver.mkOp(Kind.BITVECTOR_EXTRACT, dst - 1, 0), clamped), next_idx

        raise ValueError(f"Unknown operator: {name!r}")

    result, _ = convert(0)
    return result