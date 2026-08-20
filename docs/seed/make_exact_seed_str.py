def make_exact_seed_str(src_format=(4, 3), out_format=(5, 4), pset=None,
                        with_exceptions=False):
    e, m = src_format
    eo, mo = out_format
    bias_in, bias_out = _bias(e), _bias(eo)
    rebias = bias_out - 2 * bias_in

    raw_w = 2 * (m + 1)
    def _not_sign(x):
        return f"ite_sign({x}, positive_sign, negative_sign)"

    def _and_sign(x, y):
        return f"ite_sign({x}, {y}, positive_sign)"

    def _or_sign(x, y):
        return f"ite_sign({x}, negative_sign, {y})"
    in_width = e + m + 1
    tag_width = 2 + in_width

    tagged_a = f"input_ieee_e{e}m{m}(ma)"
    tagged_b = f"input_ieee_e{e}m{m}(mb)"

    rest_a = f"ulow_{tag_width}_{in_width}({tagged_a})"
    rest_b = f"ulow_{tag_width}_{in_width}({tagged_b})"

    sign_a = f"msb_u{in_width}({rest_a})"
    sign_b = f"msb_u{in_width}({rest_b})"
    sign_out = f"xor_sign({sign_a}, {sign_b})"

    exp_frac_a = f"ulow_{in_width}_{e + m}({rest_a})"
    exp_frac_b = f"ulow_{in_width}_{e + m}({rest_b})"
    expf_a_expr = f"uhigh_{e + m}_{e}({exp_frac_a})"
    expf_b_expr = f"uhigh_{e + m}_{e}({exp_frac_b})"
    frac_adj_a = f"ulow_{e + m}_{m}({exp_frac_a})"
    frac_adj_b = f"ulow_{e + m}_{m}({exp_frac_b})"

    sig_a = f"uconcat_1_{m}(uone_1, {frac_adj_a})"
    sig_b = f"uconcat_1_{m}(uone_1, {frac_adj_b})"
    raw = f"sigmul_{m+1}_{m+1}({sig_a}, {sig_b})"

    tag_a = f"uhigh_{tag_width}_2({tagged_a})"
    tag_b = f"uhigh_{tag_width}_2({tagged_b})"

    carry = f"msb_u{raw_w}({raw})"#
    carry1_view = f"ulshift_{raw_w - 1}_by1(ulow_{raw_w}_{raw_w - 1}({raw}))"
    carry0_view = f"ulshift_{raw_w - 2}_by2(ulow_{raw_w}_{raw_w - 2}({raw}))"
    sig_prod_ext = f"ite_u{raw_w}({carry}, {carry1_view}, {carry0_view})"

    def _maybe_ucast(expr, src_w, dst_w):
        return expr if src_w == dst_w else f"ucast_{src_w}_{dst_w}({expr})"

    def rounded_frac_nearest_even_parts(aligned):
        shift = raw_w - mo
        if shift <= 0:
            src = f"ulow_{raw_w}_{mo}({aligned})"
            kept = f"ulshift_{mo}_by{mo - raw_w}({src})" if mo > raw_w else src
            return kept, "positive_sign"
        kept = f"uhigh_{raw_w}_{mo}({aligned})"          
        discarded = f"ulow_{raw_w}_{shift}({aligned})"  
        guard = f"msb_u{shift}({discarded})"         
        lsb_sign = f"msb_u1(ulow_{mo}_1({kept}))"
        if shift >= 2:
            sticky_region = f"ulow_{shift}_{shift - 1}({discarded})"
            sticky_nonzero = f"ugt_u{shift - 1}({sticky_region}, uzero_{shift - 1})"
            round_inc = _and_sign(guard, _or_sign(_and_sign(sticky_nonzero, _not_sign(lsb_sign)), lsb_sign))
        else:
            round_inc = _and_sign(guard, lsb_sign)
        return kept, round_inc

    kept, round_inc = rounded_frac_nearest_even_parts(sig_prod_ext)

    max_expf = (1 << e) - 2
    max_exp_sum = 2 * max_expf
    w_fixed = e + 2
    safe_min = max(1, (max_exp_sum + max(0, rebias) + 2).bit_length())
    w_fixed = max(w_fixed, eo + 2, safe_min)

    expf_a_wide = _maybe_ucast(expf_a_expr, e, w_fixed)
    expf_b_wide = _maybe_ucast(expf_b_expr, e, w_fixed)
    exp_sum_full = f"uadd_{w_fixed}_{w_fixed}({expf_a_wide}, {expf_b_wide})"
    exp_sum_wide = f"ulow_{w_fixed + 1}_{w_fixed}({exp_sum_full})"

    bias_tag = make_const_terminal(pset, (-rebias) % (1 << w_fixed), w_fixed)
    exp_after_bias_wide = f"usubwrap_{w_fixed}_{w_fixed}({exp_sum_wide}, {bias_tag})"
    exp_after_bias = f"ulow_{w_fixed + 1}_{w_fixed}({exp_after_bias_wide})"

    carry_u1 = f"bit_to_u1({carry})"
    exp_post_norm_wide = f"uadd_{w_fixed}_1({exp_after_bias}, {carry_u1})"
    exp_post_norm = f"ulow_{w_fixed + 1}_{w_fixed}({exp_post_norm_wide})"

    combined_w = w_fixed + mo
    combined = f"uconcat_{w_fixed}_{mo}({exp_post_norm}, {kept})"
    zero_pad_tag = f"uzero_{combined_w}"
    combined_rounded = f"roundadd_{combined_w}_{combined_w}({combined}, {zero_pad_tag}, {round_inc})"

    top2 = f"uhigh_{combined_w}_2({combined_rounded})"
    excpostnorm = f"excpostnorm_lookup({top2})"

    rest_w = eo + mo
    rest = f"ulow_{combined_w}_{rest_w}({combined_rounded})"
    exp_narrow = f"uhigh_{rest_w}_{eo}({rest})"
    frac_final = f"ulow_{rest_w}_{mo}({rest})"

    r_width = 2 + 1 + rest_w

    def _pack_then_unpack_R(exc_tag_expr):
        sign_u1 = f"bit_to_u1({sign_out})"
        exp_frac = f"uconcat_{eo}_{mo}({exp_narrow}, {frac_final})"          # expX & fracX
        tagged = f"uconcat_1_{rest_w}({sign_u1}, {exp_frac})"                # sX & expX & fracX
        R = f"uconcat_2_{1 + rest_w}({exc_tag_expr}, {tagged})"              # exnX & sX & expX & fracX

        unpacked_exc = f"uhigh_{r_width}_2({R})"                            # exnX <= R(10:9)
        rest_bits = f"ulow_{r_width}_{1 + rest_w}({R})"
        unpacked_sign = f"msb_u1(uhigh_{1 + rest_w}_1({rest_bits}))"         # sX <= R(8)
        exp_frac_bits = f"ulow_{1 + rest_w}_{rest_w}({rest_bits})"
        unpacked_exp = f"uhigh_{rest_w}_{eo}({exp_frac_bits})"               # expX <= R(7:4)
        unpacked_frac = f"ulow_{rest_w}_{mo}({exp_frac_bits})"               # fracX <= R(3:0)
        return unpacked_exc, unpacked_sign, unpacked_exp, unpacked_frac

    if not with_exceptions:
        exc2, sign2, exp2, frac2 = _pack_then_unpack_R(excpostnorm)
        return f"output_ieee_e{eo}m{mo}({exc2}, {sign2}, {exp2}, {frac2})"

    exc_sel = f"uconcat_2_2({tag_a}, {tag_b})"
    exc = f"exc_lookup({exc_sel})"
    is_tags_normal = f"{make_eq_primitive(pset, 2, 1)}({exc})"
    final_exc_tag = f"ite_u2({is_tags_normal}, {excpostnorm}, {exc})"
    exc2, sign2, exp2, frac2 = _pack_then_unpack_R(final_exc_tag)
    return f"output_ieee_e{eo}m{mo}({exc2}, {sign2}, {exp2}, {frac2})"