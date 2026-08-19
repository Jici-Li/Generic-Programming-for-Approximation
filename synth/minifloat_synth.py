# Translate a typed GP tree (benchmarks/minifloat_hardware.py's decomposed
# FP-multiplier primitive grammar) into synthesizable Verilog, mirroring
# synth/mxint_verilog_translate.py's approach (plain unsigned wires, no
# reliance on Verilog `signed` semantics). Unlike mxint, everything here
# is genuinely unsigned magnitude arithmetic plus a single 1-bit sign/
# carry/comparison "Sign" value per node -- no two's-complement handling
# needed at all, so this translator is considerably simpler than the
# mxint one.


def _zext(expr, src, dst, emit_forced=None):
    if src == dst:
        return expr
    if src > dst:
        # Verilog can only part-select ([x:y]) a named signal, not an
        # arbitrary parenthesized expression -- when the caller might pass
        # an inlined (non-wire) expression, force it through a real wire
        # first.
        if emit_forced is not None:
            expr, _ = emit_forced(src, expr)
        return f"{expr}[{dst - 1}:0]"
    pad = dst - src
    return "{" + f"{{{pad}{{1'b0}}}}, {expr}" + "}"


def _clip_unsigned_verilog(emit, emit_forced, expr, width, dst_bits):
    max_dst = (1 << dst_bits) - 1
    above = f"({expr} > {width}'d{max_dst})"
    # MUST be a real named wire (not inlined), regardless of the caller's
    # inline mode: Verilog can only part-select ([x:y]) a named signal, not
    # an arbitrary parenthesized expression.
    clamped, _ = emit_forced(width, f"({above}) ? {width}'d{max_dst} : {expr}")
    t, tw = emit(dst_bits, f"{clamped}[{dst_bits - 1}:0]")
    return t, tw


def gp_to_verilog(individual, src_format=(4, 3), out_format=(5, 4), module_name="top", inline=False,
                  split_classify=True):
    e, m = src_format
    eo, mo = out_format
    in_width = e + m + 1
    out_width = eo + mo + 1
    # matches benchmarks/minifloat_hardware.py's make_add_pset: a single
    # FIXED lzc output width (based on sum_w = m+3) applied to every
    # lzc_u{x} primitive regardless of x -- only relevant for Add trees.
    lzc_out_w = (m + 3).bit_length()

    nodes = list(individual)
    lines = []
    counter = [0]
    needed_classify_modules = set()
    needed_input_ieee_modules = set()
    needed_sigmul_modules = set()
    needed_roundadd_modules = set()
    instance_counter = [0]
    submodule_call_memo = {}
    # A DEAP primitive tree has no notion of a shared value (it's a tree,
    # not a DAG) -- make_exact_seed_str reuses the SAME Python string for a
    # sub-expression at multiple positions (e.g. the significand product
    # read by carry/carry1_view/carry0_view), which after
    # gp.PrimitiveTree.from_string becomes several textually-identical but
    # positionally-distinct subtrees. Left alone this synthesizes several
    # redundant copies of the same logic (verified: 100+ duplicate
    # significand-multiplier instances once those were made real submodule
    # instances instead of inline `*`, since yosys's own CSE doesn't merge
    # separate submodule instances the way it merges identical inline
    # expressions). Memoizing every emitted wire by its exact (width, expr)
    # collapses these back to one computation, bottom-up, matching FloPoCo's
    # real structure where a VHDL signal is genuinely computed once and read
    # by however many downstream expressions need it.
    wire_memo = {}

    def emit(width, expr):
        if inline:
            # skip naming a wire per node -- return the expression itself so
            # the caller keeps building one larger inline expression, instead
            # of chopping the tree into one named signal per primitive call.
            # Tested against a hand-transliterated reference: named-wire-per-
            # node synthesized meaningfully larger than FloPoCo's own real
            # circuit; letting yosys's frontend see the whole expression at
            # once (as a hand-written description naturally would) closes
            # most of that gap.
            return f"({expr})", width
        key = (width, expr)
        if key in wire_memo:
            return wire_memo[key]
        name = f"t{counter[0]}"
        counter[0] += 1
        lines.append(f"  wire [{width - 1}:0] {name} = {expr};")
        wire_memo[key] = (name, width)
        return name, width

    def emit_forced(width, expr):
        # ALWAYS a real named wire, ignoring `inline` -- Verilog can't
        # part-select ([x:y]) an arbitrary parenthesized expression, only a
        # named signal, so anything that needs to bit-select its own result
        # (_clip_unsigned_verilog) must go through this instead of emit().
        key = (width, expr)
        if key in wire_memo:
            return wire_memo[key]
        name = f"t{counter[0]}"
        counter[0] += 1
        lines.append(f"  wire [{width - 1}:0] {name} = {expr};")
        wire_memo[key] = (name, width)
        return name, width

    def emit_instance(module_name_, port_map, out_port, out_width):
        name = f"t{counter[0]}"
        counter[0] += 1
        instance_counter[0] += 1
        conns = ", ".join(f".{p}({v})" for p, v in port_map.items())
        lines.append(f"  wire [{out_width - 1}:0] {name};")
        lines.append(f"  {module_name_} {module_name_}_inst_{instance_counter[0]} ({conns}, .{out_port}({name}));")
        return name, out_width

    def convert(idx):
        node = nodes[idx]
        name = getattr(node, 'name', str(node))

        if not hasattr(node, 'arity') or node.arity == 0:
            if name in ('ma', 'ARG0'):
                return 'ma', in_width, idx + 1
            if name in ('mb', 'ARG1'):
                return 'mb', in_width, idx + 1
            if name.startswith('const_'):
                _, c_str, w_str = name.split('_')
                w = int(w_str)
                t, tw = emit(w, f"{w}'d{c_str}")
                return t, tw, idx + 1
            if name.startswith('uzero_'):
                w = int(name[len('uzero_'):])
                t, tw = emit(w, f"{w}'d0")
                return t, tw, idx + 1
            if name.startswith('uone_'):
                w = int(name[len('uone_'):])
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

        if name.startswith('sign_e') and 'm' in name:
            ce, cw, next_idx = convert(next_idx)
            t, tw = emit(1, f"{ce}[{e + m}]")
            return t, tw, next_idx

        if name.startswith('expf_e'):
            ce, cw, next_idx = convert(next_idx)
            t, tw = emit(e, f"{ce}[{e + m - 1}:{m}]")
            return t, tw, next_idx

        if name.startswith('input_ieee_e'):
            # 'input_ieee_e{e}m{m}' -- matches FloPoCo's real InputIEEE
            # structurally: computed ONCE per operand into a single packed
            # tag+sign+exp+frac register that downstream logic re-slices,
            # not re-derived independently every time a piece of it is
            # needed. The GP tree itself has no notion of a shared value
            # (it's a tree, not a DAG), so the same primitive call can
            # appear at multiple positions in the tree for the same
            # argument (e.g. once to extract sign, once for the exponent,
            # once for the fraction, once for the tag) -- memoizing by
            # (primitive, argument-signal) here is what actually collapses
            # those back into ONE instance in the generated Verilog,
            # matching FloPoCo's real one-InputIEEE-call-per-operand
            # structure instead of synthesizing several redundant copies.
            rest = name[len('input_ieee_e'):]
            e_val_str, m_val_str = rest.split('m')
            e_val, m_val = int(e_val_str), int(m_val_str)
            ce, cw, next_idx = convert(next_idx)
            memo_key = (name, ce)
            if memo_key in submodule_call_memo:
                cached_t, cached_tw = submodule_call_memo[memo_key]
                return cached_t, cached_tw, next_idx
            mod_name = f"input_ieee_e{e_val}m{m_val}"
            needed_input_ieee_modules.add((e_val, m_val))
            out_w = 2 + e_val + m_val + 1
            t, tw = emit_instance(mod_name, {"bits": ce}, "out", out_w)
            submodule_call_memo[memo_key] = (t, tw)
            return t, tw, next_idx

        if name.startswith('classify_e'):
            # 'classify_e{e}m{m}' -- deliberately emitted as a SUBMODULE
            # INSTANCE, not inline wire logic, unlike every other primitive
            # here. This mirrors FloPoCo's real architecture: InputIEEE
            # computes this tag ONCE per operand as a genuinely separate
            # component, and FPMult only ever reads it -- the module
            # boundary itself (not just the logic) is part of matching
            # FloPoCo's real synthesis structure, since ABC's optimization
            # scope can differ across a module boundary even for identical
            # logic content.
            rest = name[len('classify_e'):]
            e_str, m_str = rest.split('m')
            e_val, m_val = int(e_str), int(m_str)
            ce, cw, next_idx = convert(next_idx)
            if split_classify:
                mod_name = f"classify_e{e_val}m{m_val}"
                needed_classify_modules.add((e_val, m_val))
                t, tw = emit_instance(mod_name, {"bits": ce}, "tag", 2)
                return t, tw, next_idx
            # inline comparison, for isolating the module-boundary's own
            # effect from other changes when A/B testing.
            ce, _ = emit_forced(cw, ce)
            max_exp = (1 << e_val) - 1
            exp_field = f"{ce}[{e_val + m_val - 1}:{m_val}]"
            frac_field = f"{ce}[{m_val - 1}:0]"
            is_exp_zero = f"({exp_field} == {e_val}'d0)"
            is_exp_infty = f"({exp_field} == {e_val}'d{max_exp})"
            is_frac_zero = f"({frac_field} == {m_val}'d0)"
            is_repr_subnormal = f"{ce}[{m_val - 1}]"
            is_zero = f"({is_exp_zero} && !({is_repr_subnormal}))"
            is_infinity = f"({is_exp_infty} && {is_frac_zero})"
            is_nan = f"({is_exp_infty} && !({is_frac_zero}))"
            expr = f"{is_zero} ? 2'd0 : ({is_infinity} ? 2'd2 : ({is_nan} ? 2'd3 : 2'd1))"
            t, tw = emit(2, expr)
            return t, tw, next_idx

        if name.startswith('sig_e'):
            ce, cw, next_idx = convert(next_idx)
            # matches _make_sig's Python logic exactly (benchmarks/
            # minifloat_hardware.py): when the exponent field is 0 AND the
            # fraction's top bit is 1, this is FloPoCo's reinterpreted
            # extended-subnormal case -- drop the leading fraction bit (it
            # becomes the hidden bit) and shift the rest left by 1 with a
            # trailing 0, instead of unconditionally prepending a hidden 1.
            exp_field = f"{ce}[{e + m - 1}:{m}]"
            frac_field = f"{ce}[{m - 1}:0]"
            top_bit = f"{ce}[{m - 1}]"
            if m >= 2:
                is_ext_subnormal = f"(({exp_field} == {e}'d0) && {top_bit})"
                shifted = "{" + f"{ce}[{m - 2}:0], 1'b0" + "}"
                sig_expr = f"({is_ext_subnormal}) ? {{1'b1, {shifted}}} : {{1'b1, {frac_field}}}"
            else:
                # m==1: no bits remain below the dropped leading bit
                is_ext_subnormal = f"(({exp_field} == {e}'d0) && {top_bit})"
                sig_expr = f"({is_ext_subnormal}) ? {{1'b1, 1'b0}} : {{1'b1, {frac_field}}}"
            t, tw = emit(m + 1, sig_expr)
            return t, tw, next_idx

        if name.startswith('encode_e'):
            se, sw, next_idx = convert(next_idx)
            ee, ew, next_idx = convert(next_idx)
            fe, fw, next_idx = convert(next_idx)
            t, tw = emit(out_width, "{" + f"{se}, {ee}, {fe}" + "}")
            return t, tw, next_idx

        if name.startswith('output_ieee_e'):
            # FloPoCo's real OutputIEEE_..._comb.vhdl, the third pipeline
            # stage (InputIEEE -> FPMult -> OutputIEEE) -- not just a tag-
            # to-canonical-bit-pattern swap: when the tag says "normal" but
            # the internal exponent field is exactly 0, that's FloPoCo's
            # extended-subnormal codespace and has to be re-packed into an
            # honest IEEE subnormal bit pattern (drop the low fraction bit,
            # prefix the implicit 1), not passed through raw.
            rest = name[len('output_ieee_e'):]
            e_val_str, m_val_str = rest.split('m')
            eo_val, mo_val = int(e_val_str), int(m_val_str)
            xe, xw, next_idx = convert(next_idx)
            se, sw, next_idx = convert(next_idx)
            ee, ew, next_idx = convert(next_idx)
            fe, fw, next_idx = convert(next_idx)
            xe, _ = emit_forced(xw, xe)
            ee, _ = emit_forced(ew, ee)
            fe, _ = emit_forced(fw, fe)
            exp_zero = f"({ee} == {eo_val}'d0)"
            out_sign = f"(({xe}==2'd0||{xe}==2'd1||{xe}==2'd2) ? {se} : 1'b0)"
            if mo_val >= 2:
                boundary_frac = "{" + f"1'b1, {fe}[{mo_val - 1}:1]" + "}"
            else:
                boundary_frac = "1'b1"
            frac_r = (
                f"({xe}==2'd0) ? {mo_val}'d0 : "
                f"(({exp_zero}) && {xe}==2'd1) ? {boundary_frac} : "
                f"({xe}==2'd1) ? {fe} : "
                "{" + f"{{{mo_val - 1}{{1'b0}}}}, {xe}[0]" + "}"
            )
            exp_r = (
                f"({xe}==2'd0) ? {eo_val}'d0 : "
                f"({xe}==2'd1) ? {ee} : "
                "{" + f"{eo_val}{{1'b1}}" + "}"
            )
            t, tw = emit(out_width, "{" + f"{out_sign}, ({exp_r}), ({frac_r})" + "}")
            return t, tw, next_idx

        if name == 'xor_sign':
            le, lw, next_idx = convert(next_idx)
            re, rw, next_idx = convert(next_idx)
            t, tw = emit(1, f"{le} ^ {re}")
            return t, tw, next_idx

        if name == 'bit_to_u1':
            ce, cw, next_idx = convert(next_idx)
            t, tw = emit(1, ce)
            return t, tw, next_idx

        if name.startswith('msb_u'):
            n = int(name[len('msb_u'):])
            ce, cw, next_idx = convert(next_idx)
            ce, _ = emit_forced(cw, ce)
            t, tw = emit(1, f"{ce}[{n - 1}]")
            return t, tw, next_idx

        if name.startswith('ugt_u'):
            n = int(name[len('ugt_u'):])
            le, lw, next_idx = convert(next_idx)
            re, rw, next_idx = convert(next_idx)
            t, tw = emit(1, f"({le} > {re})")
            return t, tw, next_idx

        if name.startswith('ueq_u'):
            rest = name[len('ueq_u'):]
            w_str, val_str = rest.split('_is')
            n = int(w_str)
            val = int(val_str)
            ce, cw, next_idx = convert(next_idx)
            t, tw = emit(1, f"({ce} == {n}'d{val})")
            return t, tw, next_idx

        if name == 'exc_lookup':
            # Matches FloPoCo's real `with excSel select exc <= "00" when
            # "0000"|"0001"|"0100", "01" when "0101", "10" when
            # "0110"|"1001"|"1010", "11" when others` as ONE case-style
            # expression, instead of a chain of separate equality-
            # comparator primitives OR'd together.
            ce, cw, next_idx = convert(next_idx)
            ce, _ = emit_forced(cw, ce)
            expr = (
                f"({ce}==4'd0||{ce}==4'd1||{ce}==4'd4) ? 2'd0 : "
                f"({ce}==4'd5) ? 2'd1 : "
                f"({ce}==4'd6||{ce}==4'd9||{ce}==4'd10) ? 2'd2 : 2'd3"
            )
            t, tw = emit(2, expr)
            return t, tw, next_idx

        if name == 'excpostnorm_lookup':
            # Matches FloPoCo's real `with expSigPostRound(w-1 downto w-2)
            # select excPostNorm <= "01" when "00", "10" when "01", "00"
            # when "11"|"10", "11" when others` as ONE case-style
            # expression.
            ce, cw, next_idx = convert(next_idx)
            ce, _ = emit_forced(cw, ce)
            expr = f"({ce}==2'd0) ? 2'd1 : ({ce}==2'd1) ? 2'd2 : 2'd0"
            t, tw = emit(2, expr)
            return t, tw, next_idx

        if name.startswith('ite_u'):
            n = int(name[len('ite_u'):])
            cnde, cndw, next_idx = convert(next_idx)
            ae, aw, next_idx = convert(next_idx)
            be, bw, next_idx = convert(next_idx)
            t, tw = emit(n, f"({cnde}) ? {ae} : {be}")
            return t, tw, next_idx

        if name.startswith('sigmul_'):
            # Same computation as umul_, but emitted as a genuinely separate
            # submodule instance -- matches FloPoCo's real significand
            # multiplier being generated as its own IntMultiplier entity
            # (`SignificandMultiplication: IntMultiplier_... port map(...)`),
            # not an inline `*` expression. `sigProd` is a single VHDL
            # signal FloPoCo reads three times (`norm<=sigProd(7)`,
            # `sigProdExt<=...sigProd(6:0)...sigProd(5:0)...`) -- the GP
            # tree has no notion of a shared value, so the same primitive
            # call can appear at multiple positions for the same two
            # arguments; memoize by (primitive, arg signals) the same way
            # input_ieee_e is memoized above, so those collapse back into
            # ONE instance instead of one per occurrence.
            x, y = map(int, name[len('sigmul_'):].split('_'))
            out = x + y
            le, lw, next_idx = convert(next_idx)
            re, rw, next_idx = convert(next_idx)
            memo_key = (name, le, re)
            if memo_key in submodule_call_memo:
                cached_t, cached_tw = submodule_call_memo[memo_key]
                return cached_t, cached_tw, next_idx
            needed_sigmul_modules.add((x, y))
            mod_name = f"sigmul_{x}_{y}"
            t, tw = emit_instance(mod_name, {"X": le, "Y": re}, "R", out)
            submodule_call_memo[memo_key] = (t, tw)
            return t, tw, next_idx

        if name.startswith('roundadd_'):
            # Same computation as uaddc_, but emitted as a genuinely
            # separate submodule instance -- matches FloPoCo's real
            # rounding step being its own IntAdder entity (`RoundingAdder:
            # IntAdder_N port map(Cin=>round, X=>expSig, Y=>zeros,
            # R=>expSigPostRound)`), not an inline `+` expression. Memoized
            # for the same reason as sigmul_ above, in case a future seed
            # references the rounded result more than once. Output is
            # max(x,y) bits (no extra carry-out bit), matching FloPoCo's
            # real IntAdder_N port width exactly.
            x, y = map(int, name[len('roundadd_'):].split('_'))
            out = max(x, y)
            le, lw, next_idx = convert(next_idx)
            re, rw, next_idx = convert(next_idx)
            ce, cw, next_idx = convert(next_idx)
            memo_key = (name, le, re, ce)
            if memo_key in submodule_call_memo:
                cached_t, cached_tw = submodule_call_memo[memo_key]
                return cached_t, cached_tw, next_idx
            needed_roundadd_modules.add((x, y))
            mod_name = f"roundadd_{x}_{y}"
            t, tw = emit_instance(mod_name, {"X": le, "Y": re, "Cin": ce}, "R", out)
            submodule_call_memo[memo_key] = (t, tw)
            return t, tw, next_idx

        if name.startswith('umul_'):
            x, y = map(int, name[len('umul_'):].split('_'))
            out = x + y
            le, lw, next_idx = convert(next_idx)
            re, rw, next_idx = convert(next_idx)
            l = _zext(le, lw, out, emit_forced)
            r = _zext(re, rw, out, emit_forced)
            t, tw = emit(out, f"{l} * {r}")
            return t, tw, next_idx

        if name.startswith('uconcat_'):
            x, y = map(int, name[len('uconcat_'):].split('_'))
            out = x + y
            le, lw, next_idx = convert(next_idx)
            re, rw, next_idx = convert(next_idx)
            t, tw = emit(out, "{" + f"{le}, {re}" + "}")
            return t, tw, next_idx

        if name.startswith('uadd_'):
            x, y = map(int, name[len('uadd_'):].split('_'))
            out = max(x, y) + 1
            le, lw, next_idx = convert(next_idx)
            re, rw, next_idx = convert(next_idx)
            l = _zext(le, lw, out, emit_forced)
            r = _zext(re, rw, out, emit_forced)
            t, tw = emit(out, f"{l} + {r}")
            return t, tw, next_idx

        if name.startswith('uaddc_'):
            x, y = map(int, name[len('uaddc_'):].split('_'))
            out = max(x, y) + 1
            le, lw, next_idx = convert(next_idx)
            re, rw, next_idx = convert(next_idx)
            ce, cw, next_idx = convert(next_idx)  # carry-in, Sign (1 bit)
            l = _zext(le, lw, out, emit_forced)
            r = _zext(re, rw, out, emit_forced)
            cin_ext = _zext(ce, cw, out, emit_forced)
            t, tw = emit(out, f"{l} + {r} + {cin_ext}")
            return t, tw, next_idx

        if name.startswith('ucast_'):
            x, y = map(int, name[len('ucast_'):].split('_'))
            ce, cw, next_idx = convert(next_idx)
            if y >= x:
                t, tw = emit(y, _zext(ce, cw, y, emit_forced))
                return t, tw, next_idx
            t, tw = _clip_unsigned_verilog(emit, emit_forced, ce, cw, y)
            return t, tw, next_idx

        if name.startswith('uhigh_'):
            src, dst = map(int, name[len('uhigh_'):].split('_'))
            shift = src - dst
            ce, cw, next_idx = convert(next_idx)
            # shifted MUST be a real named wire before it can be part-selected.
            shifted, _ = emit_forced(src, f"{ce} >> {shift}")
            t, tw = emit(dst, f"{shifted}[{dst - 1}:0]")
            return t, tw, next_idx

        if name.startswith('ulow_'):
            src, dst = map(int, name[len('ulow_'):].split('_'))
            ce, cw, next_idx = convert(next_idx)
            # ce may be an inlined (non-wire) expression from a child node;
            # it must be materialized before it can be part-selected.
            ce, _ = emit_forced(cw, ce)
            t, tw = emit(dst, f"{ce}[{dst - 1}:0]")
            return t, tw, next_idx

        if name.startswith('ulshift_'):
            rest = name[len('ulshift_'):]
            x_str, k_str = rest.split('_by')
            x, k = int(x_str), int(k_str)
            out = x + k
            ce, cw, next_idx = convert(next_idx)
            ext = _zext(ce, cw, out, emit_forced)
            t, tw = emit(out, f"{ext} << {k}")
            return t, tw, next_idx

        if name.startswith('usub_'):
            x, y = map(int, name[len('usub_'):].split('_'))
            out = max(x, y) + 1
            le, lw, next_idx = convert(next_idx)
            re, rw, next_idx = convert(next_idx)
            l = _zext(le, lw, out, emit_forced)
            r = _zext(re, rw, out, emit_forced)
            # floors at 0 (never wraps) -- matches _make_usub exactly.
            t, tw = emit(out, f"({l} >= {r}) ? ({l} - {r}) : {out}'d0")
            return t, tw, next_idx

        if name.startswith('usubwrap_'):
            x, y = map(int, name[len('usubwrap_'):].split('_'))
            out = max(x, y) + 1
            le, lw, next_idx = convert(next_idx)
            re, rw, next_idx = convert(next_idx)
            l = _zext(le, lw, out, emit_forced)
            r = _zext(re, rw, out, emit_forced)
            # genuine wrapping subtractor: (l - r) mod 2**out, matching
            # _make_uwrapsub -- unlike usub_ above, this one never floors.
            t, tw = emit(out, f"{l} - {r}")
            return t, tw, next_idx

        if name.startswith('ushr_'):
            x, y = map(int, name[len('ushr_'):].split('_'))
            ve, vw, next_idx = convert(next_idx)
            ae, aw, next_idx = convert(next_idx)
            # Verilog's >> on an unsigned value already zero-fills and
            # naturally yields 0 once the shift amount >= width, matching
            # Python's `>>` on an arbitrary-precision int -- no extra
            # out-of-range handling needed (same as the CVC5 translation).
            t, tw = emit(x, f"{ve} >> {ae}")
            return t, tw, next_idx

        if name.startswith('ushl_'):
            x, y = map(int, name[len('ushl_'):].split('_'))
            ve, vw, next_idx = convert(next_idx)
            ae, aw, next_idx = convert(next_idx)
            # matches _make_ushl's saturating-clip (not modular wraparound)
            # semantics -- shift in a wide-enough register first, then clip.
            wide = x + y
            v_wide = _zext(ve, vw, wide, emit_forced)
            shifted, _ = emit(wide, f"{v_wide} << {ae}")
            t, tw = _clip_unsigned_verilog(emit, emit_forced, shifted, wide, x)
            return t, tw, next_idx

        if name.startswith('lzc_u'):
            n = int(name[len('lzc_u'):])
            ce, cw, next_idx = convert(next_idx)
            ce, _ = emit_forced(cw, ce)
            out_bits = lzc_out_w
            # nested-ternary priority chain, built innermost (all-zero-
            # input default) first and wrapped outward so testing bit
            # (n-1) -- the last one wrapped -- ends up outermost / checked
            # first, matching _make_lzc's MSB-down scan exactly.
            expr = f"{out_bits}'d{n}"
            for pos in range(0, n):
                expr = f"({ce}[{pos}]) ? {out_bits}'d{n - 1 - pos} : ({expr})"
            t, tw = emit(out_bits, expr)
            return t, tw, next_idx

        if name.startswith('sticky_'):
            x, y = map(int, name[len('sticky_'):].split('_'))
            ve, vw, next_idx = convert(next_idx)
            ae, aw, next_idx = convert(next_idx)
            shr, _ = emit(x, f"{ve} >> {ae}")
            shl_back, _ = emit(x, f"{shr} << {ae}")
            t, tw = emit(1, f"({shl_back} != {ve})")
            return t, tw, next_idx

        if name.startswith('unot_'):
            x = int(name[len('unot_'):])
            ce, cw, next_idx = convert(next_idx)
            t, tw = emit(x, f"~{ce}")
            return t, tw, next_idx

        if name.startswith('orbit_u'):
            x = int(name[len('orbit_u'):])
            ve, vw, next_idx = convert(next_idx)
            fe, fw, next_idx = convert(next_idx)
            f_ext = _zext(fe, fw, x, emit_forced)
            t, tw = emit(x, f"{ve} | {f_ext}")
            return t, tw, next_idx

        if name == 'ite_sign':
            cnde, cndw, next_idx = convert(next_idx)
            ae, aw, next_idx = convert(next_idx)
            be, bw, next_idx = convert(next_idx)
            t, tw = emit(1, f"({cnde}) ? {ae} : {be}")
            return t, tw, next_idx

        raise ValueError(f"Unknown operator: {name!r}")

    root_expr, root_width, _ = convert(0)
    assert root_width == out_width, (root_width, out_width)

    body = "\n".join(lines)
    module = (
        f"module {module_name}(\n"
        f"  input [{in_width - 1}:0] ma,\n"
        f"  input [{in_width - 1}:0] mb,\n"
        f"  output [{out_width - 1}:0] y\n"
        f");\n"
        f"{body}\n"
        f"  assign y = {root_expr};\n"
        f"endmodule\n"
    )

    helper_defs = []
    for e_val, m_val in sorted(needed_classify_modules):
        max_exp = (1 << e_val) - 1
        w = e_val + m_val
        helper_defs.append(
            f"module classify_e{e_val}m{m_val}(input [{w}:0] bits, output [1:0] tag);\n"
            f"  wire [{e_val - 1}:0] expf = bits[{w - 1}:{m_val}];\n"
            f"  wire [{m_val - 1}:0] frac = bits[{m_val - 1}:0];\n"
            f"  wire is_exp_zero = (expf == {e_val}'d0);\n"
            f"  wire is_exp_infty = (expf == {e_val}'d{max_exp});\n"
            f"  wire is_frac_zero = (frac == {m_val}'d0);\n"
            f"  wire is_repr_subnormal = frac[{m_val - 1}];\n"
            f"  wire is_zero = is_exp_zero && !is_repr_subnormal;\n"
            f"  wire is_infinity = is_exp_infty && is_frac_zero;\n"
            f"  wire is_nan = is_exp_infty && !is_frac_zero;\n"
            f"  assign tag = is_zero ? 2'd0 : (is_infinity ? 2'd2 : (is_nan ? 2'd3 : 2'd1));\n"
            f"endmodule\n"
        )
    for e_val, m_val in sorted(needed_input_ieee_modules):
        max_exp = (1 << e_val) - 1
        w = e_val + m_val
        out_w = 2 + w + 1
        if m_val >= 2:
            sfrac_expr = (
                f"(is_exp_zero && is_repr_subnormal) "
                f"? {{frac[{m_val - 2}:0], 1'b0}} : frac"
            )
        else:
            sfrac_expr = "(is_exp_zero && is_repr_subnormal) ? {1'b0} : frac"
        helper_defs.append(
            f"module input_ieee_e{e_val}m{m_val}(input [{w}:0] bits, output [{out_w - 1}:0] out);\n"
            f"  wire [{e_val - 1}:0] expf = bits[{w - 1}:{m_val}];\n"
            f"  wire [{m_val - 1}:0] frac = bits[{m_val - 1}:0];\n"
            f"  wire sign = bits[{w}];\n"
            f"  wire is_exp_zero = (expf == {e_val}'d0);\n"
            f"  wire is_exp_infty = (expf == {e_val}'d{max_exp});\n"
            f"  wire is_frac_zero = (frac == {m_val}'d0);\n"
            f"  wire is_repr_subnormal = frac[{m_val - 1}];\n"
            f"  wire [{m_val - 1}:0] sfrac = {sfrac_expr};\n"
            f"  wire is_zero = is_exp_zero && !is_repr_subnormal;\n"
            f"  wire is_infinity = is_exp_infty && is_frac_zero;\n"
            f"  wire is_nan = is_exp_infty && !is_frac_zero;\n"
            f"  wire [1:0] tag = is_zero ? 2'd0 : (is_infinity ? 2'd2 : (is_nan ? 2'd3 : 2'd1));\n"
            f"  assign out = {{tag, sign, expf, sfrac}};\n"
            f"endmodule\n"
        )
    for x_val, y_val in sorted(needed_sigmul_modules):
        out_val = x_val + y_val
        helper_defs.append(
            f"module sigmul_{x_val}_{y_val}(input [{x_val - 1}:0] X, "
            f"input [{y_val - 1}:0] Y, output [{out_val - 1}:0] R);\n"
            f"  assign R = X * Y;\n"
            f"endmodule\n"
        )
    for x_val, y_val in sorted(needed_roundadd_modules):
        # max(x,y) bits, no extra carry-out bit -- matches FloPoCo's real
        # IntAdder_N (`Rtmp <= X + Y + Cin; R <= Rtmp;` with Rtmp declared
        # exactly N bits); Verilog's implicit truncation on assignment to
        # the narrower R port gives the same mod-2**out_val wrap.
        out_val = max(x_val, y_val)
        helper_defs.append(
            f"module roundadd_{x_val}_{y_val}(input [{x_val - 1}:0] X, "
            f"input [{y_val - 1}:0] Y, input Cin, output [{out_val - 1}:0] R);\n"
            f"  assign R = X + Y + Cin;\n"
            f"endmodule\n"
        )
    return "".join(helper_defs) + module


# real-synthesis area cost, same convention as synth/integer_synth.py
from synth.synthesize import synthesize_cells

_FAILURE_COST = 999999.0

def count_area_synth(individual, src_format=(4, 3), out_format=(5, 4)):
    try:
        verilog = gp_to_verilog(individual, src_format, out_format)
        return float(synthesize_cells(verilog))
    except Exception:
        return _FAILURE_COST
