"""GP search for an approximate Block Minifloat (BM) block element
multiplier -- DeepWok/mase's block_minifloat.py. Renamed from this
project's earlier "mxfp_search.py".

Mirrors search/mxint_hardware_search.py's block-vs-element split (block
quantisation and shared-exponent selection are fixed reference logic;
only the element multiplier is searched) and search/minifloat_search.py's
GP machinery (cx_combine_sides directed crossover, reinjection +
merit-filtered pairwise combos + HallOfFame archive, CVC5 CEGIS loop) --
since a block-minifloat element genuinely *is* a minifloat value, those
pieces are imported and reused directly rather than reimplemented.

Formal verification reuses verify/verify_minifloat_error_bound.py
unmodified: the block's shared exponent (bias) is a multiplicative
constant applied equally to the approximate and exact element results, so
it cancels out of the relative-error ratio exactly -- proving the element
multiplier meets alpha for every element bit-pattern pair (ignoring block
scale entirely) is mathematically equivalent to proving the rescaled
block-level relative error meets alpha. No block-aware verifier needed.
"""

import math
import random
from operator import attrgetter

import matplotlib.pyplot as plt
from deap import algorithms, base, creator, gp, tools

from benchmarks.minifloat import decode
from benchmarks.block_minifloat import (
    BlockMinifloatBlock, dequantize_block, make_data, make_exhaustive_element_data,
    make_narrow_seed_strs, make_pset, make_seed_str, reference_multiply_blocks,
    self_check,
)
from search.minifloat_search import alpha_floor, cx_combine_sides, count_area
from verify.verify_minifloat_error_bound import verify_error_bound


# ==========================================================
# Configuration
# ==========================================================
ELEMENT_FORMAT = (4, 3)   # E4M3-style FP8 element
# exponent width MUST stay >= the lossless product's own (ea+eb=8), not
# narrowed down to something like (5,4) the way flat (non-block) minifloat
# experiments used -- verified concretely: two elements near e4m3's own
# max (480 each) multiply to 230400, which overflows (5,4)'s max
# (126976) and silently saturates, corrupting the "exact" seed's baseline
# error. Flat minifloat's 100-sample random test rarely hit that corner;
# block quantisation puts elements near the block's own scale (and hence
# near the element format's max) far more often, so it isn't optional
# here. Only the mantissa (mo) is narrowed from the lossless mo=7 down to
# 4 -- the actual precision reduction GP searches over.
OUTPUT_FORMAT  = (8, 4)
BLOCK_SIZE     = 8
EXPONENT_BIAS_WIDTH = 3   # shared-bias field width for benchmarks.
                          # block_minifloat's choose_shared_exponent,
                          # matching MASE's block_minifloat.py exactly
                          # (per_block_bias = clamp(floor(log2(block_max)),
                          # 0, 2**exponent_bias_width - 1)) -- a knob
                          # separate from ELEMENT_FORMAT's own exponent
                          # width, same as MASE's real API.

ERRORS    = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
POP_SIZE  = 250
NGEN      = 50
N_BLOCKS  = 100
MAX_CEGIS = 3
RANDOM_SEED = 42

MANTISSA_DOMAIN = True   # exclude near-zero-magnitude block elements --
                          # narrowing them is catastrophic in *relative*
                          # error terms even though the underlying
                          # rounding is correct; see benchmarks/
                          # mxint_hardware.py's make_data docstring for the
                          # same reasoning applied there.


# ==========================================================
# Fitness (block-aware: rescale by the block's shared exponent before
# comparing against the dequantised target)
# ==========================================================
def _block_error(approx_values, target_values):
    max_error = 0.0
    for result, target in zip(approx_values, target_values):
        if target == 0:
            continue
        error = abs(result - target) / abs(target)
        if math.isnan(error):
            error = 100.0
        if error > max_error:
            max_error = error
    return max_error


def make_evaluate(alpha, pset, data, output_format, seed_area):
    eo, mo = output_format

    def evaluate(individual):
        try:
            func = gp.compile(individual, pset)
        except Exception:
            return (99999.0,)

        max_error = 0.0
        for a, b, target in data:
            try:
                scale = 2.0 ** (a.exponent + b.exponent)
                approx_values = tuple(
                    decode(func(x, y), eo, mo) * scale
                    for x, y in zip(a.elements, b.elements)
                )
                target_values = dequantize_block(target)
                err = _block_error(approx_values, target_values)
                if err > 100:
                    err = 100.0
            except Exception:
                err = 100.0
            if err > max_error:
                max_error = err

        area = count_area(individual)
        if max_error > alpha:
            penalty = seed_area * (1.0 + max_error / max(alpha, 1e-6))
            return (float(penalty),)
        return (float(area),)

    return evaluate


def eval_true(individual, pset, data, output_format):
    """Re-check an individual's true (error, area), ignoring the penalty."""
    eo, mo = output_format
    func = gp.compile(individual, pset)
    max_error = 0.0
    for a, b, target in data:
        scale = 2.0 ** (a.exponent + b.exponent)
        approx_values = tuple(
            decode(func(x, y), eo, mo) * scale
            for x, y in zip(a.elements, b.elements)
        )
        target_values = dequantize_block(target)
        err = _block_error(approx_values, target_values)
        if err > max_error:
            max_error = err
    return max_error, count_area(individual)


# ==========================================================
# GP search -- identical machinery to search/minifloat_search.py's
# run_gp, just wired to this file's block-aware make_evaluate.
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

    reinject_strs = [seed_str] + list(extra_seed_strs)
    reinject_inds = []
    for s in reinject_strs:
        try:
            ind = creator.Individual(gp.PrimitiveTree.from_string(s, pset))
        except Exception:
            continue
        ind.fitness.values = toolbox.evaluate(ind)
        reinject_inds.append(ind)

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
    self_check()
    random.seed(RANDOM_SEED)
    print(f"\n{'=' * 68}")
    print(f"  Block Minifloat block: size={BLOCK_SIZE}, element=FP{ELEMENT_FORMAT}, "
          f"output=FP{OUTPUT_FORMAT}")
    print(f"{'=' * 68}\n")

    floor = alpha_floor(OUTPUT_FORMAT)
    errors = [a for a in ERRORS if a >= floor]
    skipped = [a for a in ERRORS if a < floor]
    if skipped:
        print(f"[feasibility] output FP{OUTPUT_FORMAT} bounds worst-case "
              f"error at {floor:.3%}; skipping infeasible alphas {skipped}")

    pset = make_pset(ELEMENT_FORMAT, ELEMENT_FORMAT, OUTPUT_FORMAT)
    data = make_data(ELEMENT_FORMAT, EXPONENT_BIAS_WIDTH, block_size=BLOCK_SIZE,
                     n_blocks=N_BLOCKS, seed=RANDOM_SEED, mantissa_domain=MANTISSA_DOMAIN)

    if hasattr(creator, "FitnessMin"): del creator.FitnessMin
    if hasattr(creator, "Individual"): del creator.Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    seed_str = make_seed_str(ELEMENT_FORMAT, ELEMENT_FORMAT, OUTPUT_FORMAT)
    seed_tree = gp.PrimitiveTree.from_string(seed_str, pset)
    seed_area = count_area(seed_tree)
    # single_axis_only=True: only narrow-one-side seeds given, joint combos
    # withheld -- cx_combine_sides has to construct those itself.
    narrow_seed_strs = make_narrow_seed_strs(ELEMENT_FORMAT, ELEMENT_FORMAT,
                                             OUTPUT_FORMAT, single_axis_only=True)

    seed_err, _ = eval_true(seed_tree, pset, data, OUTPUT_FORMAT)
    print(f"Seed:      {seed_str}")
    print(f"Seed area: {seed_area}")
    print(f"Seed error (training, should be near 0): {seed_err:.2%}")
    print(f"Extra seeds: {len(narrow_seed_strs)} single-axis element narrowings "
          f"(joint combos withheld)")
    print(f"Training:  {len(data)} blocks / {len(data) * BLOCK_SIZE} element products\n")

    results = []
    for alpha in errors:
        print(f"\n  alpha={alpha}")
        current_data = list(data)
        best = None
        verified = False

        for cegis_iter in range(MAX_CEGIS):
            best = run_gp(pset, current_data, OUTPUT_FORMAT, seed_area, seed_str,
                         alpha, NGEN, POP_SIZE, extra_seed_strs=narrow_seed_strs)
            gp_err, gp_area = eval_true(best, pset, current_data, OUTPUT_FORMAT)
            print(f"    iter {cegis_iter + 1}: gp_err={gp_err:.1%} area={gp_area} "
                  f"data_size={len(current_data)}")

            if gp_err > alpha:
                print(f"    GP missed constraint, stopping CEGIS")
                break

            verified, ce = verify_error_bound(best, pset, ELEMENT_FORMAT, ELEMENT_FORMAT,
                                              OUTPUT_FORMAT, alpha,
                                              exclude_subnormal_inputs=MANTISSA_DOMAIN)
            if verified:
                print(f"    VERIFIED after {cegis_iter + 1} round(s)")
                break
            if ce and 'ma' in ce:
                ma_ce, mb_ce = ce['ma'], ce['mb']
                # a single-element, zero-exponent block reproduces exactly
                # the element-level counterexample CVC5 found.
                a_ce = BlockMinifloatBlock(0, (ma_ce,), ELEMENT_FORMAT)
                b_ce = BlockMinifloatBlock(0, (mb_ce,), ELEMENT_FORMAT)
                current_data.append((a_ce, b_ce, reference_multiply_blocks(a_ce, b_ce)))
                print(f"    FV failed ce=ma={ma_ce},mb={mb_ce}, "
                      f"error={ce.get('error', float('nan')):.1%} -> added to data")
            else:
                print(f"    FV inconclusive ({ce}), stopping")
                break

        final_err, final_area = eval_true(best, pset, data, OUTPUT_FORMAT)
        results.append((alpha, final_err, final_area, verified))
        print(f"  -> final: error={final_err:.1%} area={final_area} verified={verified}")

    fig, ax = plt.subplots(figsize=(9, 6))
    verified_pts = [(a, e, ar) for (a, e, ar, v) in results if v]
    failed_pts = [(a, e, ar) for (a, e, ar, v) in results if not v]
    if verified_pts:
        vs = sorted(verified_pts, key=lambda t: t[0])
        ax.plot([r[0] * 100 for r in vs], [r[2] for r in vs],
                'o-', color='#2E7D32', label='VERIFIED', markersize=10)
    if failed_pts:
        ax.scatter([r[0] * 100 for r in failed_pts], [r[2] for r in failed_pts],
                   c='#E53935', marker='x', s=100, label='FV failed / GP missed')
    ax.axhline(seed_area, color='gray', linestyle='--', label=f'seed area={seed_area}')
    ax.set_xlabel('alpha (error tolerance threshold, %)')
    ax.set_ylabel('Area (element multiplier node count)')
    ax.set_title(f'Block Minifloat minimal-hint discovery Pareto\n'
                f'block={BLOCK_SIZE}, element=FP{ELEMENT_FORMAT}, output=FP{OUTPUT_FORMAT}')
    ax.grid(True, alpha=0.3)
    ax.legend()

    fname = f'outcome/pareto_block_minifloat_block{BLOCK_SIZE}_elem{ELEMENT_FORMAT}_out{OUTPUT_FORMAT}.png'
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    print(f"\nSaved {fname}")

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    print(f"     alpha    error    area    FV")
    print(f"  {'-' * 40}")
    for alpha, err, area, verified in results:
        tag = "VERIFIED" if verified else "  FAILED"
        print(f"    {alpha:>5}   {err * 100:>5.1f}%   {area:>4}   {tag}")


if __name__ == "__main__":
    main()
