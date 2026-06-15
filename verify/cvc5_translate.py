"""
verify/cvc5_translate.py

Translates a DEAP PrimitiveTree (GP individual) into a CVC5 BitVector expression.

Supported operations (matching benchmarks/int_mult.py primitives):
    add, sub, and_, or_, left_shift, logic_right_shift, ite
    terminals: ma, mb, integer constants
"""

from cvc5 import Kind


def gp_to_cvc5(individual, ma_bv, mb_bv, solver, bv_width):
    """
    Translate a DEAP PrimitiveTree into a CVC5 BitVector term.

    Args:
        individual : DEAP PrimitiveTree (GP individual)
        ma_bv      : CVC5 BitVector constant for input ma
        mb_bv      : CVC5 BitVector constant for input mb
        solver     : cvc5.Solver instance
        bv_width   : bit width used for all intermediate values

    Returns:
        CVC5 term representing the expression
    """
    nodes = list(individual)

    def make_bv(v):
        """Make a CVC5 BitVector constant of bv_width bits."""
        return solver.mkBitVector(bv_width, int(v) & ((1 << bv_width) - 1))

    def convert(idx):
        """
        Recursively convert node at idx.
        Returns (cvc5_term, next_idx).
        """
        node = nodes[idx]
        name = getattr(node, 'name', str(node))

        # ── Leaf nodes ──
        if not hasattr(node, 'arity') or node.arity == 0:
            if name == 'ARG0':
                return ma_bv, idx + 1
            if name == 'ARG1':
                return mb_bv, idx + 1
            # Integer constant
            try:
                v = int(node.value) if hasattr(node, 'value') else int(name)
                return make_bv(v), idx + 1
            except ValueError:
                raise ValueError(f"Unknown leaf node: {name!r}")

        # ── Internal nodes: recurse into children first ──
        next_idx = idx + 1

        if name == 'add':
            left,  next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            return solver.mkTerm(Kind.BITVECTOR_ADD, left, right), next_idx

        if name == 'sub':
            left,  next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            return solver.mkTerm(Kind.BITVECTOR_SUB, left, right), next_idx

        if name == 'and_':
            left,  next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            return solver.mkTerm(Kind.BITVECTOR_AND, left, right), next_idx

        if name == 'or_':
            left,  next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            return solver.mkTerm(Kind.BITVECTOR_OR, left, right), next_idx

        if name == 'left_shift':
            left,  next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            return solver.mkTerm(Kind.BITVECTOR_SHL, left, right), next_idx

        if name == 'logic_right_shift':
            left,  next_idx = convert(next_idx)
            right, next_idx = convert(next_idx)
            return solver.mkTerm(Kind.BITVECTOR_LSHR, left, right), next_idx

        if name == 'ite':
            cond,      next_idx = convert(next_idx)
            true_val,  next_idx = convert(next_idx)
            false_val, next_idx = convert(next_idx)
            # ite condition: cond != 0
            zero     = make_bv(0)
            cond_bool = solver.mkTerm(Kind.DISTINCT, cond, zero)
            return solver.mkTerm(Kind.ITE, cond_bool, true_val, false_val), next_idx

        raise ValueError(f"Unknown operator: {name!r}")

    result, _ = convert(0)
    return result
