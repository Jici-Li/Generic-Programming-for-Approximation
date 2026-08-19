# Translate a typed GP tree (benchmarks/mxint_hardware.py's / benchmarks/
# block_fp.py's shared primitive grammar) into synthesizable Verilog,
# mirroring verify/mxint_hardware_verify.py node-for-node (same
# validation methodology as synth/verilog_translate.py for benchmarks/
# integer.py). All wires are declared plain (unsigned) [n-1:0]; signed
# semantics (sign-extend, arithmetic shift, signed compare/clamp) are
# built by hand from two's-complement bit manipulation rather than
# relying on Verilog `signed` wire declarations, so the translation is
# unambiguous across synthesis tools. This is valid because +, -, *, and
# unary - on N-bit two's complement bit patterns are correct regardless
# of any "signed" qualifier as long as operands are pre-sign-extended to
# a width wide enough to hold the exact result; only right-shift and
# ordered comparison actually depend on signedness, and those are
# implemented explicitly below (sign-filled shift, offset-binary compare).


def _sext(expr, src, dst):
    if src == dst:
        return expr
    if src > dst:
        return f"{expr}[{dst - 1}:0]"
    pad = dst - src
    return "{" + f"{{{pad}{{{expr}[{src - 1}]}}}}, {expr}" + "}"


def _zext(expr, src, dst):
    if src == dst:
        return expr
    if src > dst:
        return f"{expr}[{dst - 1}:0]"
    pad = dst - src
    return "{" + f"{{{pad}{{1'b0}}}}, {expr}" + "}"


def _biased_expr(expr, width):
    # offset-binary form (flip the sign bit): unsigned order of this form
    # matches signed order of the original two's-complement value, so a
    # plain `<`/`>` on this form implements a signed compare without
    # needing a `signed` wire declaration.
    if width == 1:
        return f"(~{expr})"
    return "{" + f"~{expr}[{width - 1}], {expr}[{width - 2}:0]" + "}"


def _biased_const(value, width):
    bits = value & ((1 << width) - 1)
    return bits ^ (1 << (width - 1))


def gp_to_verilog(individual, element_bits, output_bits, module_name="top"):
    nodes = list(individual)
    lines = []
    counter = [0]

    def emit(width, expr):
        name = f"t{counter[0]}"
        counter[0] += 1
        lines.append(f"  wire [{width - 1}:0] {name} = {expr};")
        return name, width

    def clip_signed(wide_expr, wide_width, dst_bits):
        min_dst = -(1 << (dst_bits - 1))
        max_dst = (1 << (dst_bits - 1)) - 1
        min_c = f"{wide_width}'d{min_dst & ((1 << wide_width) - 1)}"
        max_c = f"{wide_width}'d{max_dst & ((1 << wide_width) - 1)}"
        biased = _biased_expr(wide_expr, wide_width)
        below = f"({biased} < {wide_width}'d{_biased_const(min_dst, wide_width)})"
        above = f"({biased} > {wide_width}'d{_biased_const(max_dst, wide_width)})"
        clamped, _ = emit(wide_width, f"({below}) ? {min_c} : (({above}) ? {max_c} : {wide_expr})")
        if dst_bits == wide_width:
            return clamped, wide_width
        t, tw = emit(dst_bits, f"{clamped}[{dst_bits - 1}:0]")
        return t, tw

    def clip_unsigned(expr, width, dst_bits):
        max_dst = (1 << dst_bits) - 1
        above = f"({expr} > {width}'d{max_dst})"
        clamped, _ = emit(width, f"({above}) ? {width}'d{max_dst} : {expr}")
        t, tw = emit(dst_bits, f"{clamped}[{dst_bits - 1}:0]")
        return t, tw

    def convert(idx):
        node = nodes[idx]
        name = getattr(node, 'name', str(node))

        if not hasattr(node, 'arity') or node.arity == 0:
            if name in ('ma', 'ARG0'):
                return 'ma', element_bits, idx + 1
            if name in ('mb', 'ARG1'):
                return 'mb', element_bits, idx + 1
            if name.startswith('minus_one_'):
                w = int(name[len('minus_one_'):])
                t, tw = emit(w, f"{w}'d{(1 << w) - 1}")
                return t, tw, idx + 1
            if name.startswith('uzero_'):
                w = int(name[len('uzero_'):])
                t, tw = emit(w, f"{w}'d0")
                return t, tw, idx + 1
            if name.startswith('uone_'):
                w = int(name[len('uone_'):])
                t, tw = emit(w, f"{w}'d1")
                return t, tw, idx + 1
            if name.startswith('zero_'):
                w = int(name[len('zero_'):])
                t, tw = emit(w, f"{w}'d0")
                return t, tw, idx + 1
            if name.startswith('one_'):
                w = int(name[len('one_'):])
                t, tw = emit(w, f"{w}'d1")
                return t, tw, idx + 1
            if name == 'positive_sign':
                t, tw = emit(1, "1'b0")
                return t, tw, idx + 1
            if name == 'negative_sign':
                t, tw = emit(1, "1'b1")
                return t, tw, idx + 1
            raise ValueError(f"Unknown terminal: {name!r}")

        next_idx = idx + 1

        if name.startswith('smul_'):
            x, y = map(int, name[len('smul_'):].split('_'))
            out_bits = x + y
            le, lw, next_idx = convert(next_idx)
            re, rw, next_idx = convert(next_idx)
            l = _sext(le, lw, out_bits)
            r = _sext(re, rw, out_bits)
            t, tw = emit(out_bits, f"{l} * {r}")
            return t, tw, next_idx

        if name.startswith('sadd_') or name.startswith('ssub_'):
            op = name[:4]
            x, y = map(int, name[len(op) + 1:].split('_'))
            out_bits = max(x, y) + 1
            le, lw, next_idx = convert(next_idx)
            re, rw, next_idx = convert(next_idx)
            l = _sext(le, lw, out_bits)
            r = _sext(re, rw, out_bits)
            opsym = '+' if op == 'sadd' else '-'
            t, tw = emit(out_bits, f"{l} {opsym} {r}")
            return t, tw, next_idx

        if name.startswith('scast_'):
            x, y = map(int, name[len('scast_'):].split('_'))
            ce, cw, next_idx = convert(next_idx)
            if y >= x:
                t, tw = emit(y, _sext(ce, cw, y))
                return t, tw, next_idx
            shift = x - y
            wide_bits = x + 2
            wide, _ = emit(wide_bits, _sext(ce, cw, wide_bits))
            bias = 1 << (shift - 1)
            biased, _ = emit(wide_bits, f"{wide} + {wide_bits}'d{bias}")
            rounded, _ = emit(wide_bits,
                "{" + f"{{{shift}{{{biased}[{wide_bits - 1}]}}}}, {biased}[{wide_bits - 1}:{shift}]" + "}")
            t, tw = clip_signed(rounded, wide_bits, y)
            return t, tw, next_idx

        if name.startswith('ashr_'):
            rest = name[len('ashr_'):]
            x_str, k_str = rest.split('_by')
            x, k = int(x_str), int(k_str)
            ce, cw, next_idx = convert(next_idx)
            t, tw = emit(x, "{" + f"{{{k}{{{ce}[{x - 1}]}}}}, {ce}[{x - 1}:{k}]" + "}")
            return t, tw, next_idx

        if name.startswith('sshl_'):
            rest = name[len('sshl_'):]
            x_str, k_str = rest.split('_by')
            x, k = int(x_str), int(k_str)
            out_bits = x + k
            ce, cw, next_idx = convert(next_idx)
            ext = _sext(ce, cw, out_bits)
            t, tw = emit(out_bits, f"{ext} << {k}")
            return t, tw, next_idx

        if name.startswith('umul_'):
            x, y = map(int, name[len('umul_'):].split('_'))
            out_bits = x + y
            le, lw, next_idx = convert(next_idx)
            re, rw, next_idx = convert(next_idx)
            l = _zext(le, lw, out_bits)
            r = _zext(re, rw, out_bits)
            t, tw = emit(out_bits, f"{l} * {r}")
            return t, tw, next_idx

        if name.startswith('uadd_'):
            x, y = map(int, name[len('uadd_'):].split('_'))
            out_bits = max(x, y) + 1
            le, lw, next_idx = convert(next_idx)
            re, rw, next_idx = convert(next_idx)
            l = _zext(le, lw, out_bits)
            r = _zext(re, rw, out_bits)
            t, tw = emit(out_bits, f"{l} + {r}")
            return t, tw, next_idx

        if name.startswith('ucast_'):
            x, y = map(int, name[len('ucast_'):].split('_'))
            ce, cw, next_idx = convert(next_idx)
            if y >= x:
                t, tw = emit(y, _zext(ce, cw, y))
                return t, tw, next_idx
            t, tw = clip_unsigned(ce, cw, y)
            return t, tw, next_idx

        if name.startswith('uhigh_'):
            src, dst = map(int, name[len('uhigh_'):].split('_'))
            shift = src - dst
            ce, cw, next_idx = convert(next_idx)
            shifted, _ = emit(src, f"{ce} >> {shift}")
            t, tw = emit(dst, f"{shifted}[{dst - 1}:0]")
            return t, tw, next_idx

        if name.startswith('ulow_'):
            src, dst = map(int, name[len('ulow_'):].split('_'))
            ce, cw, next_idx = convert(next_idx)
            t, tw = emit(dst, f"{ce}[{dst - 1}:0]")
            return t, tw, next_idx

        if name.startswith('ulshift_'):
            rest = name[len('ulshift_'):]
            x_str, k_str = rest.split('_by')
            x, k = int(x_str), int(k_str)
            out_bits = x + k
            ce, cw, next_idx = convert(next_idx)
            ext = _zext(ce, cw, out_bits)
            t, tw = emit(out_bits, f"{ext} << {k}")
            return t, tw, next_idx

        if name.startswith('abs_s'):
            n = int(name[len('abs_s'):].split('_u')[0])
            ce, cw, next_idx = convert(next_idx)
            neg, _ = emit(n, f"-{ce}")
            t, tw = emit(n, f"({ce}[{n - 1}]) ? {neg} : {ce}")
            return t, tw, next_idx

        if name.startswith('is_negative_'):
            bits = int(name[len('is_negative_'):])
            ce, cw, next_idx = convert(next_idx)
            t, tw = emit(1, f"{ce}[{bits - 1}]")
            return t, tw, next_idx

        if name == 'xor_sign':
            le, lw, next_idx = convert(next_idx)
            re, rw, next_idx = convert(next_idx)
            t, tw = emit(1, f"{le} ^ {re}")
            return t, tw, next_idx

        if name.startswith('apply_sign_'):
            bits = int(name[len('apply_sign_'):])
            se, sw, next_idx = convert(next_idx)
            me, mw, next_idx = convert(next_idx)
            mag_wide, _ = emit(bits + 1, _zext(me, mw, bits + 1))
            neg_wide, _ = emit(bits + 1, f"-{mag_wide}")
            wide_signed, _ = emit(bits + 1, f"({se}) ? {neg_wide} : {mag_wide}")
            t, tw = clip_signed(wide_signed, bits + 1, bits)
            return t, tw, next_idx

        raise ValueError(f"Unknown operator: {name!r}")

    root_expr, root_width, _ = convert(0)
    assert root_width == output_bits, (root_width, output_bits)

    body = "\n".join(lines)
    module = (
        f"module {module_name}(\n"
        f"  input [{element_bits - 1}:0] ma,\n"
        f"  input [{element_bits - 1}:0] mb,\n"
        f"  output [{output_bits - 1}:0] y\n"
        f");\n"
        f"{body}\n"
        f"  assign y = {root_expr};\n"
        f"endmodule\n"
    )
    return module


# real-synthesis area cost, same convention as synth/integer_synth.py
from synth.synthesize import synthesize_cells

_FAILURE_COST = 999999.0

def count_area_synth(individual, element_bits, output_bits):
    try:
        verilog = gp_to_verilog(individual, element_bits, output_bits)
        return float(synthesize_cells(verilog))
    except Exception:
        return _FAILURE_COST
