# Translate a typed GP tree (benchmarks/integer.py's primitive grammar)
# into a synthesizable Verilog module, mirroring verify/integer_cvc5_
# translate.py node-for-node but emitting Verilog instead of CVC5 terms.
# Unlike area_model.py's per-node additive cost (which prices each
# primitive independently and sums), this lets a real synthesis tool
# (yosys) optimize/share logic across the WHOLE expression -- the only way
# to honestly test whether a decomposed (add/blockmul/crossmul-based)
# circuit is actually cheaper than a plain narrow-then-multiply one.


def _cast_expr(expr, src_width, dst_width):
    if src_width == dst_width:
        return expr
    if src_width > dst_width:
        return f"{expr}[{dst_width - 1}:0]"
    pad = dst_width - src_width
    return "{" + f"{{{pad}{{1'b0}}}}, {expr}" + "}"


def gp_to_verilog(individual, ma_bits, mb_bits, output_bits, module_name="top"):
    nodes = list(individual)
    lines = []
    counter = [0]

    def emit(width, expr):
        name = f"t{counter[0]}"
        counter[0] += 1
        lines.append(f"  wire [{width - 1}:0] {name} = {expr};")
        return name, width

    def convert(idx):
        node = nodes[idx]
        name = getattr(node, 'name', str(node))

        if not hasattr(node, 'arity') or node.arity == 0:
            if name in ('ma', 'ARG0'):
                return 'ma', ma_bits, idx + 1
            if name in ('mb', 'ARG1'):
                return 'mb', mb_bits, idx + 1
            if name.startswith('zero_'):
                w = int(name[len('zero_'):])
                t, tw = emit(w, f"{w}'d0")
                return t, tw, idx + 1
            if name.startswith('one_'):
                w = int(name[len('one_'):])
                t, tw = emit(w, f"{w}'d1")
                return t, tw, idx + 1
            if name.startswith('const_'):
                _, c_str, w_str = name.split('_')
                w = int(w_str)
                t, tw = emit(w, f"{w}'d{c_str}")
                return t, tw, idx + 1
            raise ValueError(f"Unknown terminal: {name!r}")

        next_idx = idx + 1

        if name.startswith('mul_'):
            x, y = map(int, name[len('mul_'):].split('_'))
            out_bits = x + y
            le, lw, next_idx = convert(next_idx)
            re, rw, next_idx = convert(next_idx)
            l = _cast_expr(le, lw, out_bits)
            r = _cast_expr(re, rw, out_bits)
            t, tw = emit(out_bits, f"{l} * {r}")
            return t, tw, next_idx

        if name.startswith('blockmul_'):
            k_str, ma_str, mb_str = name[len('blockmul_'):].split('_')
            k, mab, mbb = int(k_str), int(ma_str), int(mb_str)
            out_bits = mab + mbb
            ae, aw, next_idx = convert(next_idx)
            be, bw, next_idx = convert(next_idx)
            ah, ahw = emit(mab, f"{ae} >> {k}")
            bh, bhw = emit(mbb, f"{be} >> {k}")
            ah_c = _cast_expr(ah, ahw, out_bits)
            bh_c = _cast_expr(bh, bhw, out_bits)
            prod, pw = emit(out_bits, f"{ah_c} * {bh_c}")
            t, tw = emit(out_bits, f"{prod} << {2 * k}")
            return t, tw, next_idx

        if name.startswith('crossmul_hilo_'):
            k_str, ma_str, mb_str = name[len('crossmul_hilo_'):].split('_')
            k, mab, mbb = int(k_str), int(ma_str), int(mb_str)
            out_bits = mab + mbb
            ae, aw, next_idx = convert(next_idx)
            be, bw, next_idx = convert(next_idx)
            ah, ahw = emit(mab, f"{ae} >> {k}")
            mask = (1 << k) - 1
            bl, blw = emit(mbb, f"{be} & {mbb}'d{mask}")
            ah_c = _cast_expr(ah, ahw, out_bits)
            bl_c = _cast_expr(bl, blw, out_bits)
            prod, pw = emit(out_bits, f"{ah_c} * {bl_c}")
            t, tw = emit(out_bits, f"{prod} << {k}")
            return t, tw, next_idx

        if name.startswith('crossmul_lohi_'):
            k_str, ma_str, mb_str = name[len('crossmul_lohi_'):].split('_')
            k, mab, mbb = int(k_str), int(ma_str), int(mb_str)
            out_bits = mab + mbb
            ae, aw, next_idx = convert(next_idx)
            be, bw, next_idx = convert(next_idx)
            mask = (1 << k) - 1
            al, alw = emit(mab, f"{ae} & {mab}'d{mask}")
            bh, bhw = emit(mbb, f"{be} >> {k}")
            al_c = _cast_expr(al, alw, out_bits)
            bh_c = _cast_expr(bh, bhw, out_bits)
            prod, pw = emit(out_bits, f"{al_c} * {bh_c}")
            t, tw = emit(out_bits, f"{prod} << {k}")
            return t, tw, next_idx

        if name.startswith('add_'):
            x, y = map(int, name[len('add_'):].split('_'))
            out_bits = max(x, y) + 1
            le, lw, next_idx = convert(next_idx)
            re, rw, next_idx = convert(next_idx)
            l = _cast_expr(le, lw, out_bits)
            r = _cast_expr(re, rw, out_bits)
            t, tw = emit(out_bits, f"{l} + {r}")
            return t, tw, next_idx

        if name.startswith('sub_'):
            x, y = map(int, name[len('sub_'):].split('_'))
            out_bits = max(x, y) + 1
            le, lw, next_idx = convert(next_idx)
            re, rw, next_idx = convert(next_idx)
            l = _cast_expr(le, lw, out_bits)
            r = _cast_expr(re, rw, out_bits)
            t, tw = emit(out_bits, f"{l} - {r}")
            return t, tw, next_idx

        if name.startswith('and_'):
            x, y = map(int, name[len('and_'):].split('_'))
            out_bits = max(x, y)
            le, lw, next_idx = convert(next_idx)
            re, rw, next_idx = convert(next_idx)
            l = _cast_expr(le, lw, out_bits)
            r = _cast_expr(re, rw, out_bits)
            t, tw = emit(out_bits, f"{l} & {r}")
            return t, tw, next_idx

        if name.startswith('or_'):
            x, y = map(int, name[len('or_'):].split('_'))
            out_bits = max(x, y)
            le, lw, next_idx = convert(next_idx)
            re, rw, next_idx = convert(next_idx)
            l = _cast_expr(le, lw, out_bits)
            r = _cast_expr(re, rw, out_bits)
            t, tw = emit(out_bits, f"{l} | {r}")
            return t, tw, next_idx

        if name.startswith('lshift_'):
            rest = name[len('lshift_'):]
            x_str, k_str = rest.split('_by')
            x, k = int(x_str), int(k_str)
            out_bits = x + k
            ce, cw, next_idx = convert(next_idx)
            c = _cast_expr(ce, cw, out_bits)
            t, tw = emit(out_bits, f"{c} << {k}")
            return t, tw, next_idx

        if name.startswith('rshift_'):
            rest = name[len('rshift_'):]
            x_str, k_str = rest.split('_by')
            x, k = int(x_str), int(k_str)
            ce, cw, next_idx = convert(next_idx)
            t, tw = emit(x, f"{ce} >> {k}")
            return t, tw, next_idx

        if name.startswith('ite_'):
            w = int(name[len('ite_'):])
            cnde, cndw, next_idx = convert(next_idx)
            ae, aw, next_idx = convert(next_idx)
            be, bw, next_idx = convert(next_idx)
            a_c = _cast_expr(ae, aw, w)
            b_c = _cast_expr(be, bw, w)
            t, tw = emit(w, f"({cnde} != {cndw}'d0) ? {a_c} : {b_c}")
            return t, tw, next_idx

        if name.startswith('trunc_'):
            src, dst = map(int, name[len('trunc_'):].split('_'))
            ce, cw, next_idx = convert(next_idx)
            e = _cast_expr(ce, cw, dst)
            t, tw = emit(dst, e)
            return t, tw, next_idx

        if name.startswith('rcast_'):
            src, dst = map(int, name[len('rcast_'):].split('_'))
            ce, cw, next_idx = convert(next_idx)
            shift = src - dst
            bias = 1 << (shift - 1)
            wide = _cast_expr(ce, cw, src + 1)
            widew, _ = emit(src + 1, wide)
            biased, _ = emit(src + 1, f"{widew} + {src + 1}'d{bias}")
            rounded, _ = emit(src + 1, f"{biased} >> {shift}")
            narrow, _ = emit(dst + 1, f"{rounded}[{dst}:0]")
            max_val = (1 << dst) - 1
            clamped, _ = emit(dst + 1,
                f"({narrow} > {dst + 1}'d{max_val}) ? {dst + 1}'d{max_val} : {narrow}")
            t, tw = emit(dst, f"{clamped}[{dst - 1}:0]")
            return t, tw, next_idx

        raise ValueError(f"Unknown operator: {name!r}")

    root_expr, root_width, _ = convert(0)
    assert root_width == output_bits, (root_width, output_bits)

    body = "\n".join(lines)
    module = (
        f"module {module_name}(\n"
        f"  input [{ma_bits - 1}:0] ma,\n"
        f"  input [{mb_bits - 1}:0] mb,\n"
        f"  output [{output_bits - 1}:0] y\n"
        f");\n"
        f"{body}\n"
        f"  assign y = {root_expr};\n"
        f"endmodule\n"
    )
    return module


# Real-synthesis area cost: whole-individual Verilog synthesis via yosys,
# used as a drop-in replacement for search/area_model.py's additive
# per-node count_area(). The additive model sums each primitive's cost
# independently and can never see cross-node sharing (confirmed: a hand-
# built shift-and-add multiplier synthesizes to about the same real gate
# count as a single atomic mul, 366 vs 384 cells at 8x8, but the additive
# model prices the decomposition at ~3x the atomic mul) -- so any fitness
# function built on it is structurally blind to decomposition-based wins
# regardless of how well its per-primitive constants are calibrated. This
# puts the real number in the loop instead.
from synth.synthesize import synthesize_cells

_FAILURE_COST = 999999.0


def count_area_synth(individual, ma_bits, mb_bits, output_bits):
    try:
        verilog = gp_to_verilog(individual, ma_bits, mb_bits, output_bits)
        return float(synthesize_cells(verilog))
    except Exception:
        return _FAILURE_COST
