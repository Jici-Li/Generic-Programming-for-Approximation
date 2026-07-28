"""
Pareto experiment for typed approximate multiplier synthesis.

For each user-specified error tolerance alpha:
  1. GP searches for a candidate that meets alpha on training samples
  2. CVC5 verifies formally over the full input range
  3. If verification fails, add the counterexample to training data
     and re-run GP (CEGIS)
  4. Record the (error, area) point for the Pareto plot
"""

import gc
import os
import random
from operator import attrgetter

import matplotlib.pyplot as plt
from deap import base, creator, tools, gp, algorithms

from benchmarks.integer import (
    make_pset, make_seed_str, make_side_seed_strs, make_side_seed_strs_round, make_data,
)
from search.area_model import count_area
from verify.verify_integer_error_bound import verify_error_bound


# ==========================================================
# Configuration
# ==========================================================
# Per-port bitwidths (Jianyi's example: ma:4, mb:4, out:8). Overridable via
# env vars so the same script can test whether add/sub/shift/ite-based
# rewrites become competitive at larger scales (e.g. 8x8/16x16, where real
# Booth-style row reduction is known to start paying for its own overhead)
# without duplicating the whole file.
MA_BITS     = int(os.environ.get('GP_MA_BITS', 4))
MB_BITS     = int(os.environ.get('GP_MB_BITS', 4))
OUTPUT_BITS = int(os.environ.get('GP_OUTPUT_BITS', 8))

ERRORS      = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
POP_SIZE    = int(os.environ.get('GP_POP_SIZE', 300))
NGEN        = int(os.environ.get('GP_NGEN', 60))
N_DATA      = 100
MAX_CEGIS   = 5
USE_CEGIS   = True

MANTISSA_DOMAIN = True   # flip after confirming the real spec excludes
                          # small-magnitude ma/mb -- must match on both the
                          # make_data() training samples and the CVC5 domain
                          # restriction below, or "VERIFIED" is a promise
                          # over a narrower domain than GP was scored on.


# ==========================================================
# Fitness
# ==========================================================
def make_evaluate(alpha, pset, data, ma_bits, mb_bits, output_bits, seed_area):
    # output_bits can be narrower than the true product (ma_bits+mb_bits) --
    # the circuit's raw output is then an implicitly-rescaled approximation
    # (see make_seed_str), so it has to be shifted back up before comparing
    # against the true, unscaled target. Error is measured against reality,
    # not against a same-width "ideal" circuit -- a config whose seed alone
    # can't meet alpha is a real "this bit-width doesn't work" result, not
    # something to be hidden by comparing against an already-shrunk target.
    shift = max(0, (ma_bits + mb_bits) - output_bits)

    def evaluate(individual):
        try:
            func = gp.compile(individual, pset)
        except Exception:
            return (99999.0,)
        max_err = 0.0
        for ma, mb, target in data:
            try:
                result = func(ma, mb) << shift
                if target == 0:
                    # skip zero targets: not meaningful for relative error
                    continue
                err = abs(result - target) / target
                if err > 100:
                    err = 100.0
            except Exception:
                err = 100.0
            if err > max_err:
                max_err = err

        area = count_area(individual)

        # Continuous cost instead of a hard cliff at alpha: infeasible
        # candidates are still ranked by area, with overshoot added as a
        # steep but smooth penalty on top -- so a smaller-area individual
        # that's slightly over alpha can still beat a larger-area one that's
        # way over, giving mutation/crossover a gradient to climb instead of
        # every failing candidate looking equally bad regardless of area.
        overshoot = max(0.0, max_err - alpha) / max(alpha, 1e-6)
        cost = area + overshoot * seed_area * 1000.0
        return (float(cost),)
    return evaluate


def eval_true(individual, pset, data, ma_bits, mb_bits, output_bits):
    """Re-check an individual's true (error, area), ignoring the penalty."""
    shift = max(0, (ma_bits + mb_bits) - output_bits)
    func = gp.compile(individual, pset)
    max_err = 0.0
    for ma, mb, target in data:
        result = func(ma, mb) << shift
        if target == 0:
            continue
        err = abs(result - target) / target
        if err > max_err:
            max_err = err
    return max_err, count_area(individual)


def mutate_shift_amount(individual, pset):
    """Domain-informed mutation: nudge one rshift_X_byK node's K by +-1,
    in place, instead of regenerating a whole random subtree of the same
    type. rshift_X_byK's type is BV(X) for every K, so this edit is always
    type-safe without touching the rest of the tree. This targets exactly
    the truncation-depth tradeoff diagnosed as the useful move here, instead
    of relying on mutUniform to blindly stumble onto it among hundreds of
    same-type primitives most of which have nothing to do with that tradeoff.
    """
    candidates = [i for i, node in enumerate(individual)
                  if getattr(node, 'name', '').startswith('rshift_')]
    if not candidates:
        return individual,
    idx = random.choice(candidates)
    x_str, k_str = individual[idx].name[len('rshift_'):].split('_by')
    x, k = int(x_str), int(k_str)
    new_k = k + random.choice([-1, 1])
    new_name = f'rshift_{x}_by{new_k}'
    if new_name in pset.mapping:
        individual[idx] = pset.mapping[new_name]
    return individual,


def combined_mutate(individual, expr, pset):
    """Half the time, nudge a shift amount (targeted); otherwise fall back
    to standard mutUniform (generic exploration) -- keeps both instead of
    replacing blind search entirely with the domain-informed move."""
    if random.random() < 0.5:
        return mutate_shift_amount(individual, pset)
    return gp.mutUniform(individual, expr=expr, pset=pset)


def _bv_width(ret_type):
    """BV(n) classes are named 'BVn' by benchmarks.integer.BV -- pull n
    back out instead of threading a parallel (type -> width) table."""
    return int(ret_type.__name__[2:])


def _find_single_mul(tree):
    """Index of the tree's one mul_X_Y node, or None if there isn't
    exactly one (mirrors minifloat_search._find_single_fmul: this operator
    only knows how to split a tree with exactly one multiply)."""
    idxs = [i for i, node in enumerate(tree)
            if getattr(node, 'name', '').startswith('mul_')]
    return idxs[0] if len(idxs) == 1 else None


def _split_at_mul(tree, idx):
    """Given the index of a mul_X_Y node, return the two argument subtree
    slices and their BV widths."""
    a_start = idx + 1
    a_slice = tree.searchSubtree(a_start)
    b_start = a_slice.stop
    b_slice = tree.searchSubtree(b_start)
    a_w = _bv_width(tree[a_start].ret)
    b_w = _bv_width(tree[b_start].ret)
    return a_slice, b_slice, a_w, b_w


def cx_combine_sides(ind1, ind2, pset, output_bits):
    """Custom crossover for the 'exactly one mul' tree shape: take the
    ma-side subtree from one parent and the mb-side subtree from the
    other, and rebuild with whichever mul_W1_W2 (and an outer lshift back
    to output_bits, if needed) primitive matches the combined widths.

    Mirrors search/minifloat_search.cx_combine_sides -- same reasoning:
    two parents narrowed on different single axes typically share no type
    except the root and the bare ma/mb argument type, so plain cxOnePoint
    has no valid swap point that would recombine their two narrowing
    choices. This targets that recombination directly.

    Falls back to standard cxOnePoint whenever the tree shape doesn't
    match or the needed primitive isn't in the pset -- never leaves the
    pair worse off than plain cxOnePoint would.
    """
    idx1 = _find_single_mul(ind1)
    idx2 = _find_single_mul(ind2)
    if idx1 is None or idx2 is None:
        return gp.cxOnePoint(ind1, ind2)

    a1_slice, b1_slice, a1_w, b1_w = _split_at_mul(ind1, idx1)
    a2_slice, b2_slice, a2_w, b2_w = _split_at_mul(ind2, idx2)

    def try_build(src_a, a_slice, a_w, src_b, b_slice, b_w):
        mul_name = f"mul_{a_w}_{b_w}"
        if mul_name not in pset.mapping:
            return None
        mul_out = a_w + b_w
        shift = output_bits - mul_out
        if shift < 0:
            return None
        if shift == 0:
            head = [pset.mapping[mul_name]]
        else:
            lshift_name = f"lshift_{mul_out}_by{shift}"
            if lshift_name not in pset.mapping:
                return None
            head = [pset.mapping[lshift_name], pset.mapping[mul_name]]
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
def run_gp(pset, data, ma_bits, mb_bits, output_bits, seed_area,
           seed_str, alpha, ngen, pop_size, extra_seed_strs=()):

    seed_tree = gp.PrimitiveTree.from_string(seed_str, pset)
    max_h = max(20, seed_tree.height + 2)

    local_toolbox = base.Toolbox()
    local_toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    local_toolbox.register("individual", tools.initIterate,
                           creator.Individual, local_toolbox.expr)
    local_toolbox.register("population", tools.initRepeat,
                           list, local_toolbox.individual)
    local_toolbox.register("select", tools.selTournament, tournsize=3)
    local_toolbox.register("mate",   cx_combine_sides,
                           pset=pset, output_bits=output_bits)
    local_toolbox.register("mutate", combined_mutate,
                           expr=local_toolbox.expr, pset=pset)
    local_toolbox.decorate("mate",
        gp.staticLimit(key=attrgetter("height"), max_value=max_h))
    local_toolbox.decorate("mutate",
        gp.staticLimit(key=attrgetter("height"), max_value=max_h))
    local_toolbox.register("evaluate",
        make_evaluate(alpha, pset, data, ma_bits, mb_bits, output_bits,
                      seed_area))

    pop = []
    attempts = 0
    while len(pop) < pop_size and attempts < pop_size * 5:
        try:
            ind = local_toolbox.individual()
            pop.append(ind)
        except Exception:
            pass
        attempts += 1

    if not pop:
        # if random generation fails entirely, start from the seed
        pop = [creator.Individual(gp.PrimitiveTree.from_string(seed_str, pset))
               for _ in range(pop_size)]

    # Reinject the seed strings every generation instead of only at gen 0,
    # and explicitly precompute every pairwise recombination too -- a lone
    # fresh copy of each seed is not enough for random selection/pairing to
    # reliably cross two *different* seeds within a normal generation
    # budget (verified for minifloat this session; same tournament-
    # selection dynamics apply here unchanged).
    reinject_strs = [seed_str] + list(extra_seed_strs)
    reinject_inds = []
    for s in reinject_strs:
        try:
            ind = creator.Individual(gp.PrimitiveTree.from_string(s, pset))
        except Exception:
            continue
        ind.fitness.values = local_toolbox.evaluate(ind)
        reinject_inds.append(ind)

    # Merit-filtered pairwise combos: an O(n^2) all-pairs reinjection can
    # exceed population size for a large seed grid (measured on minifloat:
    # 105 pairs/210 offspring vs. 240s+ runtime). Only keep a combo that's
    # actually feasible (fitness <= seed_area is a clean proxy for "meets
    # alpha", since make_evaluate's overshoot penalty always pushes an
    # infeasible individual's cost above seed_area).
    base_seeds = list(reinject_inds)
    for i in range(len(base_seeds)):
        for j in range(i + 1, len(base_seeds)):
            try:
                c1, c2 = cx_combine_sides(local_toolbox.clone(base_seeds[i]),
                                          local_toolbox.clone(base_seeds[j]),
                                          pset, output_bits)
            except Exception:
                continue
            for c in (c1, c2):
                c.fitness.values = local_toolbox.evaluate(c)
                if c.fitness.values[0] <= seed_area:
                    reinject_inds.append(c)

    # Archive: not just the seeds/combos we already knew about
    # (reinject_inds) -- ANY individual that turns out good gets tracked
    # here too, and is reinjected every generation the same way, so an
    # unanticipated improvement crossover/mutation stumbles onto doesn't
    # just vanish to drift the moment nothing is deliberately protecting it.
    archive = tools.HallOfFame(20)
    archive.update(pop)
    for _ in range(ngen):
        offspring = local_toolbox.select(pop, len(pop))
        offspring = list(map(local_toolbox.clone, offspring))
        offspring = algorithms.varAnd(offspring, local_toolbox, cxpb=0.5, mutpb=0.2)

        slot = 0
        for ind in reinject_inds:
            if slot >= len(offspring):
                break
            offspring[slot] = local_toolbox.clone(ind)
            slot += 1
        for ind in archive:
            if slot >= len(offspring):
                break
            offspring[slot] = local_toolbox.clone(ind)
            slot += 1

        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = local_toolbox.evaluate(ind)

        archive.update(offspring)
        pop[:] = offspring

    return archive[0]


# ==========================================================
# Main experiment
# ==========================================================
def main():
    print(f"\n{'='*60}")
    print(f"  ma={MA_BITS}-bit, mb={MB_BITS}-bit, output={OUTPUT_BITS}-bit")
    print(f"{'='*60}\n")

    pset = make_pset(MA_BITS, MB_BITS, OUTPUT_BITS)
    base_data = make_data(MA_BITS, MB_BITS, n_samples=N_DATA,
                          mantissa_domain=MANTISSA_DOMAIN)

    if hasattr(creator, "FitnessMin"): del creator.FitnessMin
    if hasattr(creator, "Individual"): del creator.Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    seed_str = make_seed_str(MA_BITS, MB_BITS, OUTPUT_BITS)
    seed_tree = gp.PrimitiveTree.from_string(seed_str, pset)
    seed_area = count_area(seed_tree)
    # Single-axis only: every seed narrows ma OR mb, never both -- the
    # jointly-narrowed (both sides at once) circuits are withheld entirely,
    # so cx_combine_sides has to construct them itself by recombining two
    # single-axis seeds instead of just selecting one that was handed to
    # it. make_block_seed_strs's full grid (which includes joint seeds
    # directly) is no longer used here for that reason.
    # Both floor (trunc+rshift, wrap-based) and round (rcast, saturating)
    # single-axis narrowings are offered side by side -- round is NOT
    # redundant with floor (verified: at this config, round beats floor's
    # worst-case error at several k, e.g. k=2 drops from 27.3% to 20.0% at
    # the *same* area -- see the project report/discussion), so giving
    # only floor would silently steer GP away from a real, cheaper option.
    side_seeds = make_side_seed_strs(MA_BITS, MB_BITS, OUTPUT_BITS)
    side_seeds_round = make_side_seed_strs_round(MA_BITS, MB_BITS, OUTPUT_BITS)
    block_seed_strs = list(side_seeds.values()) + list(side_seeds_round.values())
    print(f"Seed:        {seed_str}")
    print(f"Seed height: {seed_tree.height}")
    print(f"Seed area:   {seed_area}")
    print(f"Single-axis seeds: {len(block_seed_strs)} "
          f"({len(side_seeds)} floor + {len(side_seeds_round)} round, joint combos withheld)")

    results = []
    max_iters = MAX_CEGIS if USE_CEGIS else 1

    for alpha in ERRORS:
        print(f"\n  alpha={alpha}")
        current_data = list(base_data)

        best = None
        verified = False
        ce = None

        for cegis_iter in range(max_iters):
            best = run_gp(pset, current_data, MA_BITS, MB_BITS, OUTPUT_BITS,
                          seed_area, seed_str, alpha, NGEN, POP_SIZE,
                          extra_seed_strs=block_seed_strs)
            gp_err, gp_area = eval_true(best, pset, current_data,
                                        MA_BITS, MB_BITS, OUTPUT_BITS)
            print(f"    iter {cegis_iter+1}: "
                  f"gp_err={gp_err:.1%} area={gp_area} data_size={len(current_data)}")

            if gp_err > alpha:
                print(f"    GP missed constraint, stopping CEGIS")
                break

            verified, ce = verify_error_bound(best, pset,
                                              MA_BITS, MB_BITS, OUTPUT_BITS,
                                              alpha, mantissa_domain=MANTISSA_DOMAIN)
            if verified:
                print(f"    VERIFIED after {cegis_iter+1} GP round(s)")
                break
            else:
                if ce and 'ma' in ce:
                    ma_ce, mb_ce = ce['ma'], ce['mb']
                    current_data.append((ma_ce, mb_ce, ma_ce * mb_ce))
                    print(f"    FV failed ce=ma={ma_ce},mb={mb_ce}, "
                          f"error={ce['error']:.1%} -> added to data")
                else:
                    print(f"    FV inconclusive ({ce}), stopping")
                    break

        final_err, final_area = eval_true(best, pset, base_data,
                                          MA_BITS, MB_BITS, OUTPUT_BITS)
        results.append((alpha, final_err, final_area, verified))
        print(f"  -> final: error={final_err:.1%} area={final_area} "
              f"verified={verified}")
        print(f"  -> tree: {str(best)}")

    # ---- Pareto plot: X = the alpha threshold you swept over (all ERRORS
    # values, never collapsed), Y = area GP found under that threshold. ----
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
    ax.axhline(seed_area, color='gray', linestyle='--', label=f'exact seed area={seed_area}')
    ax.set_xlabel('alpha (error tolerance threshold, %)')
    ax.set_ylabel('Area (node count)')
    ax.set_title(f'Accuracy-Area Trade-off '
                 f'(ma={MA_BITS}b, mb={MB_BITS}b, out={OUTPUT_BITS}b)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    tag = "cegis" if USE_CEGIS else "nocegis"
    fname = f'outcome/pareto_ma{MA_BITS}_mb{MB_BITS}_out{OUTPUT_BITS}_{tag}.png'
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    print(f"\n  Saved {fname}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"     alpha    error    area    FV")
    print(f"  {'-'*40}")
    for alpha, err, area, verified in results:
        tag = "VERIFIED" if verified else "  FAILED"
        print(f"    {alpha:>5}   {err*100:>5.1f}%   {area:>4}   {tag}")


if __name__ == "__main__":
    main()