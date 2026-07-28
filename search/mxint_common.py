"""Shared GP search machinery for MXInt-family element multipliers (signed
integer mantissa, GP searches the element-level smul/sadd/ssub/scast tree).

Extracted out of search/mxint_hardware_search.py so search/block_fp_search.py (MASE's
block_fp.py / MSFP -- same signed-integer-mantissa element grammar, only a
different shared-exponent formula in the benchmarks module) can reuse it
without duplicating run_gp/cx_combine_sides/the area model/the fitness
function. Anything here that's specific to *how mantissas were quantised*
(shared-exponent selection, rounding) stays out of this file on purpose --
that's each benchmarks/*.py module's own job, passed in here only as
already-quantised MXIntBlock data.
"""

import re
from operator import attrgetter

from deap import algorithms, base, creator, gp, tools


# ==========================================================
# Area model
# ==========================================================
_BINARY_WIDTHS = re.compile(r"^s(?:mul|add|sub)_(\d+)_(\d+)$")
_UNSIGNED_BINARY_WIDTHS = re.compile(r"^u(?:mul|add)_(\d+)_(\d+)$")
_APPLY_SIGN = re.compile(r"^apply_sign_(\d+)$")
_ABS_MAGNITUDE = re.compile(r"^abs_s(\d+)_u(\d+)$")


def node_cost(name):
    match = _BINARY_WIDTHS.match(name)
    if match:
        a_bits, b_bits = map(int, match.groups())
        if name.startswith("smul_"):
            return a_bits * b_bits
        return max(a_bits, b_bits)
    match = _UNSIGNED_BINARY_WIDTHS.match(name)
    if match:
        a_bits, b_bits = map(int, match.groups())
        if name.startswith("umul_"):
            return a_bits * b_bits
        return max(a_bits, b_bits)
    # apply_sign/abs_magnitude are a conditional negate each -- real hardware
    # (XOR + increment on the two's-complement bits), the same order of cost
    # as sadd/ssub at that width, not free. Left at 0 before, this let GP
    # build circuits that never even read ma/mb (e.g. apply_sign_12(is_
    # negative_2(minus_one_2), ucast_3_12(ucast_7_3(uzero_7))), a disguised
    # constant) and have them look free on the area proxy -- caught because
    # a run with zero real training signal (see _block_error) couldn't tell
    # the constant from a real multiplier, but the area hole is real on its
    # own and worth closing independently of that.
    match = _APPLY_SIGN.match(name)
    if match:
        return int(match.group(1))
    match = _ABS_MAGNITUDE.match(name)
    if match:
        return int(match.group(1))
    if name == "xor_sign":
        return 1
    # Casts and fixed shifts are treated as wires in this first model.
    # is_negative is a free wire tap of the sign bit. A synthesis-calibrated
    # model comes later.
    return 0


def count_area(individual):
    return sum(
        node_cost(getattr(node, "name", str(node)))
        for node in individual
    )


# ==========================================================
# Fitness
# ==========================================================
def _block_error(approx_values, target_values, a_mantissas, b_mantissas, domain_fraction=None):
    """domain_fraction: skip an element unless BOTH abs(ma) >= domain_fraction
    * max(abs(a_mantissas)) AND abs(mb) >= domain_fraction * max(abs(b_
    mantissas)) -- i.e. each operand must be large *relative to its own
    block's own max*, independently.

    Two earlier versions of this filter didn't hold up:
      1. Both operands clearing a fixed *absolute* cutoff at the same
         position: a and b blocks are drawn independently, so this hit zero
         qualifying elements across 100/100 training blocks in practice.
      2. Thresholding on the *target*'s magnitude relative to its own
         block's max target: this has good element coverage, but doesn't
         protect the thing that actually matters here -- narrowing a
         specific *operand* (scast) is unreliable in relative-error terms
         whenever that operand's own value is small, regardless of whether
         the resulting product happens to look large (verified: ma=1 with
         mb=-11 gives a "large" product -11, but scast narrowing ma=1 is
         still relative-error catastrophic on ma's own contribution).
    This version filters each operand against its own within-block max, so
    it directly targets "is narrowing *this* operand's value safe", and
    still self-normalizes (each side's own max element always qualifies on
    its own axis) -- but a block whose *entire* a (or b) side is uniformly
    tiny (e.g. every element in {-1,0,1}, happens when the block's real-
    value magnitude falls outside choose_shared_exponent's unclamped zone)
    has no large element to compare against either; each benchmarks module's
    make_data is expected to pick its magnitude draw range to keep that
    case rare rather than trying to filter it out after the fact.

    domain_fraction=None disables the filter entirely."""
    max_error = 0.0
    a_cutoff = b_cutoff = None
    if domain_fraction is not None:
        a_cutoff = domain_fraction * (max((abs(m) for m in a_mantissas), default=0) or 1)
        b_cutoff = domain_fraction * (max((abs(m) for m in b_mantissas), default=0) or 1)
    for result, target, ma, mb in zip(approx_values, target_values, a_mantissas, b_mantissas):
        if target == 0:
            continue
        if a_cutoff is not None and (abs(ma) < a_cutoff or abs(mb) < b_cutoff):
            continue
        error = abs(result - target) / abs(target)
        max_error = max(max_error, error)
    return max_error


def make_evaluate(alpha, pset, data, seed_area, dequantize_block, domain_fraction=None):
    """dequantize_block: the benchmarks module's own dequantize_block
    function (MXInt-family blocks all share the same MXIntBlock shape, but
    each module owns its own dequantize_block so this stays decoupled from
    any one module's import)."""
    def evaluate(individual):
        try:
            approx_mul = gp.compile(individual, pset)
        except Exception:
            return (99999.0,)

        max_error = 0.0
        try:
            for a, b, target in data:
                approx_mantissas = tuple(
                    int(approx_mul(ma, mb))
                    for ma, mb in zip(a.mantissas, b.mantissas)
                )
                scale = 2.0 ** (a.exponent + b.exponent)
                approx_values = tuple(value * scale for value in approx_mantissas)
                target_values = dequantize_block(target)
                max_error = max(
                    max_error,
                    _block_error(approx_values, target_values, a.mantissas, b.mantissas, domain_fraction),
                )
        except Exception:
            max_error = 100.0

        area = count_area(individual)
        if max_error > alpha:
            penalty = seed_area * (1.0 + max_error / max(alpha, 1e-6))
            return (float(penalty),)
        return (float(area),)

    return evaluate


def eval_true(individual, pset, data, dequantize_block, domain_fraction=None):
    approx_mul = gp.compile(individual, pset)
    max_error = 0.0
    zeroed_nonzero = 0

    for a, b, target in data:
        approx_mantissas = tuple(
            int(approx_mul(ma, mb))
            for ma, mb in zip(a.mantissas, b.mantissas)
        )
        scale = 2.0 ** (a.exponent + b.exponent)
        approx_values = tuple(value * scale for value in approx_mantissas)
        target_values = dequantize_block(target)
        max_error = max(max_error, _block_error(
            approx_values, target_values, a.mantissas, b.mantissas, domain_fraction))
        zeroed_nonzero += sum(
            result == 0 and target_value != 0
            for result, target_value in zip(approx_values, target_values)
        )

    return max_error, count_area(individual), zeroed_nonzero


# ==========================================================
# Directed crossover: recombine the ma-side and mb-side of the tree's
# single smul_X_Y node, instead of a blind random-point swap.
# ==========================================================
def _sint_width(ret_type):
    """SInt(n) classes are named 'SIntn' by benchmarks.mxint_hardware.SInt --
    pull n back out instead of threading a parallel (type -> width) table."""
    return int(ret_type.__name__[4:])


def _find_single_smul(tree):
    """Index of the tree's one smul_X_Y node, or None if there isn't
    exactly one (mirrors search/integer_search._find_single_mul)."""
    idxs = [i for i, node in enumerate(tree)
            if getattr(node, 'name', '').startswith('smul_')]
    return idxs[0] if len(idxs) == 1 else None


def _split_at_smul(tree, idx):
    a_start = idx + 1
    a_slice = tree.searchSubtree(a_start)
    b_start = a_slice.stop
    b_slice = tree.searchSubtree(b_start)
    a_w = _sint_width(tree[a_start].ret)
    b_w = _sint_width(tree[b_start].ret)
    return a_slice, b_slice, a_w, b_w


def cx_combine_sides(ind1, ind2, pset, element_bits, output_bits):
    """Take the ma-side subtree from one parent and the mb-side subtree
    from the other, and rebuild with whichever smul_W1_W2 exists for the
    combined widths, restoring scale with a compensating sshl and casting
    to output_bits if needed. Falls back to cxOnePoint whenever the tree
    shape doesn't match or a needed primitive isn't in the pset."""
    idx1 = _find_single_smul(ind1)
    idx2 = _find_single_smul(ind2)
    if idx1 is None or idx2 is None:
        return gp.cxOnePoint(ind1, ind2)

    a1_slice, b1_slice, a1_w, b1_w = _split_at_smul(ind1, idx1)
    a2_slice, b2_slice, a2_w, b2_w = _split_at_smul(ind2, idx2)

    def try_build(src_a, a_slice, a_w, src_b, b_slice, b_w):
        smul_name = f"smul_{a_w}_{b_w}"
        if smul_name not in pset.mapping:
            return None
        cur = a_w + b_w
        head = [pset.mapping[smul_name]]
        total_shift = 2 * element_bits - cur
        if total_shift > 0:
            shl_name = f"sshl_{cur}_by{total_shift}"
            if shl_name not in pset.mapping:
                return None
            head = [pset.mapping[shl_name]] + head
            cur += total_shift
        if cur != output_bits:
            cast_name = f"scast_{cur}_{output_bits}"
            if cast_name not in pset.mapping:
                return None
            head = [pset.mapping[cast_name]] + head
        new_nodes = head + list(src_a[a_slice]) + list(src_b[b_slice])
        return creator.Individual(new_nodes)

    off1 = try_build(ind1, a1_slice, a1_w, ind2, b2_slice, b2_w)
    off2 = try_build(ind2, a2_slice, a2_w, ind1, b1_slice, b1_w)

    if off1 is None or off2 is None:
        return gp.cxOnePoint(ind1, ind2)
    return off1, off2


# ==========================================================
# GP search
# ==========================================================
def run_gp(pset, data, seed_area, seed_str, alpha, ngen, pop_size,
           dequantize_block, extra_seed_strs=(), element_bits=None,
           output_bits=None, domain_fraction=None):
    seed_tree = gp.PrimitiveTree.from_string(seed_str, pset)
    max_height = max(12, seed_tree.height + 3)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", cx_combine_sides,
                     pset=pset, element_bits=element_bits, output_bits=output_bits)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.decorate(
        "mate",
        gp.staticLimit(key=attrgetter("height"), max_value=max_height),
    )
    toolbox.decorate(
        "mutate",
        gp.staticLimit(key=attrgetter("height"), max_value=max_height),
    )
    toolbox.register("evaluate", make_evaluate(alpha, pset, data, seed_area, dequantize_block, domain_fraction))

    population = []
    attempts = 0
    while len(population) < pop_size and attempts < pop_size * 10:
        try:
            population.append(toolbox.individual())
        except Exception:
            pass
        attempts += 1

    if not population:
        population = [
            creator.Individual(gp.PrimitiveTree.from_string(seed_str, pset))
            for _ in range(pop_size)
        ]

    # Reinject the seed strings every generation, and explicitly precompute
    # every pairwise recombination too -- a lone fresh copy per generation
    # isn't enough for random selection/pairing to reliably cross two
    # *different* seeds within a normal generation budget.
    reinject_strs = [seed_str] + list(extra_seed_strs)
    reinject_inds = []
    for s in reinject_strs:
        try:
            ind = creator.Individual(gp.PrimitiveTree.from_string(s, pset))
        except Exception:
            continue
        ind.fitness.values = toolbox.evaluate(ind)
        reinject_inds.append(ind)

    # Merit-filtered pairwise combos: only keep a combo if it's feasible
    # (fitness <= seed_area is a clean proxy for "meets alpha", since
    # make_evaluate's penalty for exceeding alpha is always > seed_area).
    base_seeds = list(reinject_inds)
    for i in range(len(base_seeds)):
        for j in range(i + 1, len(base_seeds)):
            try:
                c1, c2 = cx_combine_sides(toolbox.clone(base_seeds[i]),
                                          toolbox.clone(base_seeds[j]),
                                          pset, element_bits, output_bits)
            except Exception:
                continue
            for c in (c1, c2):
                c.fitness.values = toolbox.evaluate(c)
                if c.fitness.values[0] <= seed_area:
                    reinject_inds.append(c)

    # Archive: not just the seeds/combos we already knew about
    # (reinject_inds) -- ANY individual that turns out good gets tracked
    # here too, and is reinjected every generation the same way.
    archive = tools.HallOfFame(20)
    archive.update(population)
    for _ in range(ngen):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=0.5, mutpb=0.2)

        slot = 0
        for ind in reinject_inds:
            if slot >= len(offspring):
                break
            offspring[slot] = toolbox.clone(ind)
            slot += 1
        for ind in archive:
            if slot >= len(offspring):
                break
            offspring[slot] = toolbox.clone(ind)
            slot += 1

        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        archive.update(offspring)
        population[:] = offspring

    return archive[0]
