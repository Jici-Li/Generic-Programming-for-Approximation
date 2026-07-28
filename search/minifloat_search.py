"""
GP search for approximate minifloat multiplier synthesis.

Mirrors search/integer_search.py's evolutionary loop, including the CVC5
formal-verification / CEGIS loop -- via verify/verify_minifloat_error_bound
_native.py's native-FloatingPoint-theory translation, not the generic
nonlinear Real arithmetic in verify/minifloat_cvc5_translate.py, which is
validated correct but can't prove universal properties in reasonable time
(see that module's docstring). "VERIFIED" here covers every input except
the reserved top exponent field / m=0 formats / any computation that
overflows -- see verify/minifloat_cvc5_translate.py's docstring for exactly
why and what that excludes.

Fitness: minimize area subject to a max relative-error constraint alpha,
same shape as search/integer_search.py's make_evaluate.
"""

import math
import re
from operator import attrgetter

import matplotlib.pyplot as plt
from deap import base, creator, tools, gp, algorithms

from benchmarks.minifloat import (make_pset, make_seed_str, make_narrow_seed_strs,
                                       make_data, decode)
from verify.verify_minifloat_error_bound import verify_error_bound


# ==========================================================
# Configuration
# ==========================================================
MA_FORMAT     = (4, 3)   # E4M3-style FP8 input
MB_FORMAT     = (4, 3)
OUTPUT_FORMAT = (5, 4)   # wider output format to land approximations in

ERRORS    = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
POP_SIZE  = 300
NGEN      = 60
N_DATA    = 100
MAX_CEGIS = 5
USE_CEGIS = True

EXCLUDE_SUBNORMAL_INPUTS = True   # flip after supervisor confirms domain
INCLUDE_ADDSUB           = False   # flip after supervisor confirms grammar


def alpha_floor(output_format):
    """Physical lower bound on achievable worst-case relative error:
    RNE into m mantissa bits can be off by up to half an ulp, i.e.
    2^-(m+1), even for the exact multiplier. Alphas below this are
    mathematically infeasible for this output format."""
    return 2.0 ** -(output_format[1] + 1)


# ==========================================================
# Area model: crude cost parsed straight from primitive names
# (placeholder, same spirit as search/area_model.py -- tune later
#  against real hardware synthesis numbers)
# ==========================================================
_FMT_RE = re.compile(r"e(\d+)m(\d+)")


def _parse_fmt(tag):
    e, m = _FMT_RE.match(tag).groups()
    return int(e), int(m)


def node_cost(name):
    if name.startswith("fmul_"):
        x_tag, y_tag = name[len("fmul_"):].split("_")
        ex, mx = _parse_fmt(x_tag)
        ey, my = _parse_fmt(y_tag)
        return (mx + 1) * (my + 1)          # mantissa multiplier ~ area square
    if name.startswith(("fadd_", "fsub_")):
        prefix = "fadd_" if name.startswith("fadd_") else "fsub_"
        x_tag, y_tag = name[len(prefix):].split("_")
        ex, mx = _parse_fmt(x_tag)
        ey, my = _parse_fmt(y_tag)
        return max(ex, ey) + max(mx, my) + 1  # aligner + adder ~ linear in width
    if name.startswith("fcast_"):
        return 0                              # narrowing = drop wires, free
    return 0                                  # inputs / zero_ / one_ terminals


def count_area(individual):
    total = 0
    for node in individual:
        name = getattr(node, "name", str(node))
        total += node_cost(name)
    return total


# ==========================================================
# Fitness
# ==========================================================
def make_evaluate(alpha, pset, data, output_format, seed_area):
    eo, mo = output_format

    def evaluate(individual):
        try:
            func = gp.compile(individual, pset)
        except Exception:
            return (99999.0,)

        max_err = 0.0
        for a_bits, b_bits, target in data:
            try:
                result_bits = func(a_bits, b_bits)
                result = decode(result_bits, eo, mo)
                if target == 0:
                    continue
                # math.isnan(err) can't happen from a NaN *target* (inputs
                # are restricted to finite bit patterns, and finite*finite
                # is never NaN), but result can still be NaN from internal
                # overflow (e.g. inf - inf inside an fadd/fsub chain) --
                # nan > x is always False in Python, so leaving this
                # unchecked would silently skip capping/counting it as an
                # error at all instead of the maximal-error violation it is.
                err = abs(result - target) / abs(target)
                if math.isnan(err) or err > 100:
                    err = 100.0
            except Exception:
                err = 100.0
            if err > max_err:
                max_err = err

        area = count_area(individual)

        if max_err > alpha:
            penalty = seed_area * (1.0 + max_err / max(alpha, 1e-6))
            return (float(penalty),)
        return (float(area),)

    return evaluate


def eval_true(individual, pset, data, output_format):
    """Re-check an individual's true (error, area), ignoring the penalty."""
    eo, mo = output_format
    func = gp.compile(individual, pset)
    max_err = 0.0
    for a_bits, b_bits, target in data:
        result = decode(func(a_bits, b_bits), eo, mo)
        if target == 0:
            continue
        err = abs(result - target) / abs(target)
        if math.isnan(err):
            err = 100.0
        if err > max_err:
            max_err = err
    return max_err, count_area(individual)


_FP_CLASS_RE = re.compile(r"FP_e(\d+)_m(\d+)")


def _fmt_of(cls):
    m = _FP_CLASS_RE.match(cls.__name__)
    return (int(m.group(1)), int(m.group(2)))


def _find_single_fmul(tree):
    """Index of the tree's one fmul_* node, or None if there isn't
    exactly one (e.g. INCLUDE_ADDSUB trees can have several -- this
    operator only knows how to split a tree with exactly one multiply)."""
    idxs = [i for i, node in enumerate(tree)
            if getattr(node, 'name', '').startswith('fmul_')]
    return idxs[0] if len(idxs) == 1 else None


def _split_at_fmul(tree, idx):
    """Given the index of an fmul_X_Y node, return the two argument
    subtree slices and their (e,m) formats."""
    a_start = idx + 1
    a_slice = tree.searchSubtree(a_start)
    b_start = a_slice.stop
    b_slice = tree.searchSubtree(b_start)
    a_fmt = _fmt_of(tree[a_start].ret)
    b_fmt = _fmt_of(tree[b_start].ret)
    return a_slice, b_slice, a_fmt, b_fmt


def cx_combine_sides(ind1, ind2, pset, output_format):
    """Custom crossover for the 'exactly one fmul' tree shape: take the
    ma-side subtree from one parent and the mb-side subtree from the
    other, and rebuild using whichever fmul_X_Y (and fcast into
    output_format, if needed) primitive matches the combined formats.

    Standard cxOnePoint can't do this -- differently-narrowed parents
    (e.g. "narrow ma" vs "narrow mb") typically share no type except the
    root and the bare ma/mb argument type, so there's no valid swap point
    that would recombine their two narrowing choices (verified earlier
    this session). This operator targets exactly that recombination
    directly instead of hoping a type-matched swap point exists.

    Falls back to standard cxOnePoint whenever the tree shape doesn't
    match or the needed primitive isn't in the pset -- never leaves the
    pair worse off than plain cxOnePoint would.
    """
    idx1 = _find_single_fmul(ind1)
    idx2 = _find_single_fmul(ind2)
    if idx1 is None or idx2 is None:
        return gp.cxOnePoint(ind1, ind2)

    a1_slice, b1_slice, a1_fmt, b1_fmt = _split_at_fmul(ind1, idx1)
    a2_slice, b2_slice, a2_fmt, b2_fmt = _split_at_fmul(ind2, idx2)

    def tag(fmt):
        return f"e{fmt[0]}m{fmt[1]}"

    def try_build(src_a, a_slice, a_fmt, src_b, b_slice, b_fmt):
        fmul_name = f"fmul_{tag(a_fmt)}_{tag(b_fmt)}"
        if fmul_name not in pset.mapping:
            return None
        mul_fmt = (a_fmt[0] + b_fmt[0], a_fmt[1] + b_fmt[1] + 1)
        if mul_fmt == output_format:
            head = [pset.mapping[fmul_name]]
        else:
            fcast_name = f"fcast_{tag(mul_fmt)}_{tag(output_format)}"
            if fcast_name not in pset.mapping:
                return None
            head = [pset.mapping[fcast_name], pset.mapping[fmul_name]]
        new_nodes = head + list(src_a[a_slice]) + list(src_b[b_slice])
        return creator.Individual(new_nodes)

    off1 = try_build(ind1, a1_slice, a1_fmt, ind2, b2_slice, b2_fmt)
    off2 = try_build(ind2, a2_slice, a2_fmt, ind1, b1_slice, b1_fmt)

    if off1 is None or off2 is None:
        return gp.cxOnePoint(ind1, ind2)
    return off1, off2


# ==========================================================
# GP search
# ==========================================================
def run_gp(pset, data, output_format, seed_area, seed_str, alpha, ngen, pop_size,
           extra_seed_strs=()):
    seed_tree = gp.PrimitiveTree.from_string(seed_str, pset)
    max_h = max(20, seed_tree.height + 2)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", cx_combine_sides, pset=pset, output_format=output_format)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=attrgetter("height"), max_value=max_h))
    toolbox.decorate("mutate", gp.staticLimit(key=attrgetter("height"), max_value=max_h))
    toolbox.register("evaluate", make_evaluate(alpha, pset, data, output_format, seed_area))

    pop = []
    attempts = 0
    while len(pop) < pop_size and attempts < pop_size * 5:
        try:
            pop.append(toolbox.individual())
        except Exception:
            pass
        attempts += 1

    if not pop:
        pop = [creator.Individual(gp.PrimitiveTree.from_string(seed_str, pset))
               for _ in range(pop_size)]

    seed_ind = creator.Individual(gp.PrimitiveTree.from_string(seed_str, pset))
    seed_ind.fitness.values = toolbox.evaluate(seed_ind)
    pop.append(seed_ind)

    for extra_str in extra_seed_strs:
        try:
            extra_ind = creator.Individual(gp.PrimitiveTree.from_string(extra_str, pset))
        except Exception:
            continue
        extra_ind.fitness.values = toolbox.evaluate(extra_ind)
        pop.append(extra_ind)

    # Reinject the seed strings every generation instead of only at gen 0 --
    # with tied/near-tied fitness, tournament selection lets one candidate
    # drift to fixation and the other go extinct within ~10 generations
    # purely from sampling noise (verified: two seeds tied at 1 copy each,
    # one is gone by gen 10). But reinjecting isn't enough by itself either
    # (verified): with only 1 fresh copy of each seed among pop_size
    # individuals, the odds that random selection+pairing ever puts two
    # *different* seeds next to each other for crossover in the same
    # generation are too low to matter within a normal generation budget.
    # So compute every pairwise recombination explicitly and inject those
    # too, instead of leaving pairing to chance.
    # base seeds are always kept -- there are few of them, and even one
    # that's individually infeasible can still be useful raw material for
    # a combo that turns out feasible.
    reinject_strs = [seed_str] + list(extra_seed_strs)
    reinject_inds = []
    for s in reinject_strs:
        try:
            ind = creator.Individual(gp.PrimitiveTree.from_string(s, pset))
        except Exception:
            continue
        ind.fitness.values = toolbox.evaluate(ind)
        reinject_inds.append(ind)

    # pairwise combos are O(n^2) -- for a 15-seed grid that's 105 pairs,
    # 210 offspring, more than the whole population, guaranteeing all of
    # them a spot every generation was the actual cost of the earlier
    # slowdown. Admit on merit instead: only keep a combo if it's feasible
    # (fitness <= seed_area is a clean proxy for "meets alpha", since
    # make_evaluate's penalty for exceeding alpha is always > seed_area,
    # while a feasible individual's fitness is exactly its own area).
    base_seeds = list(reinject_inds)
    for i in range(len(base_seeds)):
        for j in range(i + 1, len(base_seeds)):
            try:
                c1, c2 = cx_combine_sides(toolbox.clone(base_seeds[i]),
                                          toolbox.clone(base_seeds[j]),
                                          pset, output_format)
            except Exception:
                continue
            for c in (c1, c2):
                c.fitness.values = toolbox.evaluate(c)
                if c.fitness.values[0] <= seed_area:
                    reinject_inds.append(c)

    # archive: not just the seeds/combos we already knew about (reinject_inds)
    # -- ANY individual that turns out to be good gets tracked here too, and
    # a sample of the archive is reinjected every generation the same way,
    # so an unanticipated improvement that crossover/mutation stumbles onto
    # doesn't just vanish to drift the moment nothing is deliberately
    # protecting it. reinject_inds are what we knew to protect in advance;
    # the archive is what turned out to be worth protecting along the way.
    archive = tools.HallOfFame(20)
    archive.update(pop)
    for _ in range(ngen):
        offspring = toolbox.select(pop, len(pop))
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
        pop[:] = offspring

    return archive[0]


# ==========================================================
# Main experiment
# ==========================================================
def main():
    print(f"\n{'='*60}")
    print(f"  ma=FP{MA_FORMAT}  mb=FP{MB_FORMAT}  output=FP{OUTPUT_FORMAT}")
    print(f"{'='*60}\n")

    floor = alpha_floor(OUTPUT_FORMAT)
    errors = [a for a in ERRORS if a >= floor]
    skipped = [a for a in ERRORS if a < floor]
    if skipped:
        print(f"[feasibility] output FP{OUTPUT_FORMAT} bounds worst-case "
              f"error at {floor:.3%}; skipping infeasible alphas {skipped}")

    pset = make_pset(MA_FORMAT, MB_FORMAT, OUTPUT_FORMAT,
                     include_addsub=INCLUDE_ADDSUB)
    data = make_data(MA_FORMAT, MB_FORMAT, n_samples=N_DATA,
                     exclude_subnormal_inputs=EXCLUDE_SUBNORMAL_INPUTS)

    if hasattr(creator, "FitnessMin"): del creator.FitnessMin
    if hasattr(creator, "Individual"): del creator.Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    seed_str = make_seed_str(MA_FORMAT, MB_FORMAT, OUTPUT_FORMAT)
    seed_tree = gp.PrimitiveTree.from_string(seed_str, pset)
    seed_area = count_area(seed_tree)
    # single_axis_only=True: every seed narrows ma OR mb, never both -- the
    # jointly-narrowed seeds are withheld, so cx_combine_sides has to
    # construct them itself instead of just selecting one it was handed.
    narrow_seed_strs = make_narrow_seed_strs(MA_FORMAT, MB_FORMAT, OUTPUT_FORMAT,
                                             single_axis_only=True)
    print(f"Seed:      {seed_str}")
    print(f"Seed area: {seed_area}")
    print(f"Extra seeds: {len(narrow_seed_strs)} single-axis mantissa narrowings (joint combos withheld)\n")

    results = []
    max_iters = MAX_CEGIS if USE_CEGIS else 1

    for alpha in errors:
        print(f"\n  alpha={alpha}")
        current_data = list(data)
        best = None
        verified = False

        for cegis_iter in range(max_iters):
            best = run_gp(pset, current_data, OUTPUT_FORMAT, seed_area, seed_str,
                          alpha, NGEN, POP_SIZE, extra_seed_strs=narrow_seed_strs)
            gp_err, gp_area = eval_true(best, pset, current_data, OUTPUT_FORMAT)
            print(f"    iter {cegis_iter+1}: gp_err={gp_err:.1%} area={gp_area} "
                  f"data_size={len(current_data)}")

            if gp_err > alpha:
                print(f"    GP missed constraint, stopping CEGIS")
                break

            verified, ce = verify_error_bound(best, pset, MA_FORMAT, MB_FORMAT,
                                              OUTPUT_FORMAT, alpha,
                                              exclude_subnormal_inputs=EXCLUDE_SUBNORMAL_INPUTS)
            if verified:
                print(f"    VERIFIED after {cegis_iter+1} GP round(s)")
                break
            if ce and 'ma' in ce:
                ma_ce, mb_ce = ce['ma'], ce['mb']
                target = decode(ma_ce, *MA_FORMAT) * decode(mb_ce, *MB_FORMAT)
                current_data.append((ma_ce, mb_ce, target))
                print(f"    FV failed ce=ma={ma_ce},mb={mb_ce}, "
                      f"error={ce['error']:.1%} -> added to data")
            else:
                print(f"    FV inconclusive ({ce}), stopping")
                break

        final_err, final_area = eval_true(best, pset, data, OUTPUT_FORMAT)
        results.append((alpha, final_err, final_area, verified))
        print(f"  -> final: error={final_err:.1%} area={final_area} "
              f"verified={verified}")

    fig, ax = plt.subplots(figsize=(9, 6))
    verified_pts = [(a, e, ar) for (a, e, ar, v) in results if v]
    failed_pts   = [(a, e, ar) for (a, e, ar, v) in results if not v]
    if verified_pts:
        vs = sorted(verified_pts, key=lambda t: t[0])
        ax.plot([r[0] * 100 for r in vs], [r[2] for r in vs],
                'o-', color='#2E7D32', label='VERIFIED', markersize=10)
    if failed_pts:
        ax.scatter([r[0] * 100 for r in failed_pts], [r[2] for r in failed_pts],
                   c='#E53935', marker='x', s=100, label='FV failed / GP missed')
    ax.axhline(seed_area, color='gray', linestyle='--',
               label=f'seed area={seed_area}')
    ax.set_xlabel('alpha (error tolerance threshold, %)')
    ax.set_ylabel('Area (placeholder cost)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fname = (f'outcome/pareto_minifloat_ma{MA_FORMAT}_mb{MB_FORMAT}'
             f'_out{OUTPUT_FORMAT}.png')
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    print(f"\nSaved {fname}")

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"     alpha    error    area    FV")
    print(f"  {'-'*40}")
    for alpha, err, area, verified in results:
        tag = "VERIFIED" if verified else "  FAILED"
        print(f"    {alpha:>5}   {err*100:>5.1f}%   {area:>4}   {tag}")


if __name__ == "__main__":
    main()
