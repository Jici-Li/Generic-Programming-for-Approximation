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

        raise ValueError(f"Unknown operator: {name!r}")

    result, _ = convert(0)
    return result