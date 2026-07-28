from cvc5 import Kind, RoundingMode

from benchmarks.minifloat import encode as _py_encode


_RM = RoundingMode.ROUND_NEAREST_TIES_TO_EVEN


def _fmt_of(ret_type):
    """FP_e4_m3 -> (4, 3) -- inverse of benchmarks.minifloat.FP()."""
    name = ret_type.__name__  # "FP_e4_m3"
    e_part, m_part = name[3:].split('_m')
    return int(e_part[1:]), int(m_part)


def _parse_tag(tag):
    e_str, m_str = tag[1:].split('m')
    return int(e_str), int(m_str)


class UnsupportedFormat(Exception):
    """Raised for m == 0: CVC5's FloatingPoint sort requires a significand
    width (m + 1) of at least 2, so m == 0 formats -- a real, used part of
    make_pset's narrowing grid -- have no native-FP representation at all.
    Caught by verify_minifloat_error_bound_native and turned into a clear
    {'reason': 'unsupported_format', ...} result instead of a cryptic CVC5
    C++ error message."""


def decode_native(slv, bits_bv, e, m):
    """Native FP(e, m+1) interpretation of bits_bv. Identical to
    benchmarks.minifloat.decode(bits, e, m) for every bit pattern
    except exp_field == 2**e - 1 (see module docstring)."""
    if m == 0:
        raise UnsupportedFormat(f"m=0 has no native FP representation (e={e})")
    return slv.mkTerm(slv.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_IEEE_BV, e, m + 1), bits_bv)


def widen(slv, fp_val, e, m):
    """Re-round fp_val into FP(e, m+1) with RNE. Exact/lossless when (e, m)
    is at least as wide as fp_val's own format (true for fmul/fadd/fsub's
    operand-widening use below); genuinely rounds (correctly) when used
    for fcast's narrowing case -- FLOATINGPOINT_TO_FP_FROM_FP handles both
    directions, this is just a thin wrapper for either use."""
    if m == 0:
        raise UnsupportedFormat(f"m=0 has no native FP representation (e={e})")
    return slv.mkTerm(slv.mkOp(Kind.FLOATINGPOINT_TO_FP_FROM_FP, e, m + 1), slv.mkRoundingMode(_RM), fp_val)


def _assert_finite(slv, fp_val, domain_exclusions):
    bad = slv.mkTerm(Kind.OR,
                     slv.mkTerm(Kind.FLOATINGPOINT_IS_INF, fp_val),
                     slv.mkTerm(Kind.FLOATINGPOINT_IS_NAN, fp_val))
    domain_exclusions.append(bad)


def gp_to_cvc5_native(individual, ma_bv, mb_bv, solver, domain_exclusions):
    """Walk a minifloat GP tree and produce a CVC5 native FloatingPoint
    term. ma_bv, mb_bv are raw BitVector constants for the two operands.
    domain_exclusions collects one (is_inf OR is_nan) term per
    intermediate/output node -- the caller must assert NOT(OR(*this list))
    so the proof only covers the domain where native FP and our
    saturating format agree (see module docstring)."""
    nodes = list(individual)

    def convert(idx):
        node = nodes[idx]
        name = getattr(node, 'name', str(node))

        if not hasattr(node, 'arity') or node.arity == 0:
            if name in ('ma', 'ARG0'):
                e, m = _fmt_of(node.ret)
                return decode_native(solver, ma_bv, e, m), idx + 1
            if name in ('mb', 'ARG1'):
                e, m = _fmt_of(node.ret)
                return decode_native(solver, mb_bv, e, m), idx + 1
            if name.startswith('zero_') or name.startswith('one_'):
                is_zero = name.startswith('zero_')
                e, m = _parse_tag(name.split('_', 1)[1])
                width = 1 + e + m
                value = 0 if is_zero else _py_encode(1.0, e, m)
                bits = solver.mkBitVector(width, value)
                return decode_native(solver, bits, e, m), idx + 1
            raise ValueError(f"Unknown terminal: {name!r}")

        next_idx = idx + 1

        if name.startswith('fmul_'):
            tag_a, tag_b = name[len('fmul_'):].split('_')
            ea, ma = _parse_tag(tag_a)
            eb, mb = _parse_tag(tag_b)
            eo, mo = ea + eb, ma + mb + 1
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            a_wide = widen(solver, left, eo, mo)
            b_wide = widen(solver, right, eo, mo)
            result = solver.mkTerm(Kind.FLOATINGPOINT_MULT, solver.mkRoundingMode(_RM), a_wide, b_wide)
            _assert_finite(solver, result, domain_exclusions)
            return result, next_idx

        if name.startswith('fadd_') or name.startswith('fsub_'):
            op = name[:4]
            tag_a, tag_b = name[len(op) + 1:].split('_')
            ea, ma = _parse_tag(tag_a)
            eb, mb = _parse_tag(tag_b)
            eo, mo = max(ea, eb), max(ma, mb) + 1
            left, next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            a_wide = widen(solver, left, eo, mo)
            b_wide = widen(solver, right, eo, mo)
            kind = Kind.FLOATINGPOINT_ADD if op == 'fadd' else Kind.FLOATINGPOINT_SUB
            result = solver.mkTerm(kind, solver.mkRoundingMode(_RM), a_wide, b_wide)
            _assert_finite(solver, result, domain_exclusions)
            return result, next_idx

        if name.startswith('fcast_'):
            tag_src, tag_dst = name[len('fcast_'):].split('_')
            ed, md = _parse_tag(tag_dst)
            child, next_idx = convert(next_idx)
            result = widen(solver, child, ed, md)
            _assert_finite(solver, result, domain_exclusions)
            return result, next_idx

        raise ValueError(f"Unknown operator: {name!r}")

    result, _ = convert(0)
    return result
