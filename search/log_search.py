"""GP search for an approximate log-domain (LNS) exponent adder --
DeepWok/mase's log.py.

Structurally different from every other search/*.py in this project: the
"multiplier" here is really an adder (log(a*b) = log(a)+log(b)), and the
approximation axis is WHERE to place a single carry-chain truncation
point, not how much to narrow an operand -- see benchmarks/log.py's
module docstring for why operand narrowing has no usable gentle regime
for log-domain values (any absolute exponent error becomes an exponential
relative error), and for why *multiple simultaneous* truncation points
were dropped from the design entirely (they can compound into unbounded
error -- verified, not a hypothetical) in favour of exactly one split per
candidate.

Because there's only one axis (where to place the one split, out of a
handful of same-area choices) rather than two independently narrowable
operands, there is no "single-axis hint, withhold joint combos, directed
crossover recombines them" story to run here the way every other
benchmark's minimal-hint experiment does -- every safe split point is
handed over directly as a hint, and plain (undirected) crossover/mutation
is enough since there's nothing structural to combine.
"""

import random
from operator import attrgetter

import matplotlib.pyplot as plt
from deap import algorithms, base, creator, gp, tools

from benchmarks.log import (
    make_exhaustive_exponent_data,
    make_pset,
    make_seed_str,
    make_seg_seed_strs as make_split_seed_strs,
    self_check,
)
from verify.verify_log_error_bound import verify_error_bound


# ==========================================================
# Configuration
# ==========================================================
EXPONENT_BITS = 6
OUTPUT_BITS = 7  # exponent_bits + 1, see benchmarks/log.py's make_pset

# Error grows explosively with any unsafe segmentation and is otherwise
# bounded well under 100% for every safe one (see the project report/
# discussion for the exact verified thresholds), so this sweep spans a
# much narrower, denser-near-the-bottom range than the other benchmarks'
# alpha sweeps.
ERRORS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
POP_SIZE = 100
NGEN = 30
RANDOM_SEED = 42
MAX_CEGIS = 3
USE_CEGIS = True


# ==========================================================
# Area model: sadd_split{K}_{bits} costs bits-1 for any real split
# (K < bits, one carry link removed) and bits for no split at all
# (K >= bits, the exact add). scast/terminals/neg are free wires, same
# convention as every other benchmarks module here.
# ==========================================================
import re

_SADD_SPLIT = re.compile(r"^sadd_split(\d+)_(\d+)$")


def node_cost(name):
    match = _SADD_SPLIT.match(name)
    if match:
        split_bits, bits = map(int, match.groups())
        return bits - 1 if split_bits < bits else bits
    return 0


def count_area(individual):
    return sum(node_cost(getattr(node, "name", str(node))) for node in individual)


# ==========================================================
# Fitness
# ==========================================================
def eval_true(individual, pset, data):
    func = gp.compile(individual, pset)
    max_error = 0.0
    for ea, eb, exact_exp in data:
        approx_exp = int(func(ea, eb))
        d = approx_exp - exact_exp
        err = (2.0 ** d - 1.0) if d >= 0 else (1.0 - 2.0 ** d)
        max_error = max(max_error, err)
    return max_error, count_area(individual)


def make_evaluate(alpha, pset, data, seed_area):
    def evaluate(individual):
        try:
            func = gp.compile(individual, pset)
        except Exception:
            return (99999.0,)

        max_error = 0.0
        try:
            for ea, eb, exact_exp in data:
                approx_exp = int(func(ea, eb))
                d = approx_exp - exact_exp
                err = (2.0 ** d - 1.0) if d >= 0 else (1.0 - 2.0 ** d)
                max_error = max(max_error, err)
        except Exception:
            max_error = 1e9

        area = count_area(individual)
        if max_error > alpha:
            penalty = seed_area * (1.0 + max_error / max(alpha, 1e-6))
            return (float(penalty),)
        return (float(area),)

    return evaluate


# ==========================================================
# GP search -- no directed crossover: a single-axis choice (which safe
# segmentation to use) has nothing structural to recombine, ordinary
# cxOnePoint/mutUniform is enough. Reinjection + HallOfFame archive kept
# for consistency with every other search/*.py here.
# ==========================================================
def run_gp(pset, data, seed_area, seed_str, alpha, ngen, pop_size, extra_seed_strs=(),
           evaluate_fn=None):
    """evaluate_fn: override the fitness function -- defaults to this
    module's own make_evaluate (flat, single (ea,eb,exact) triples per
    data item), but search/block_log_search.py passes its own block-aware
    version (blocks of LogValue triples) through here instead of
    duplicating the rest of this loop."""
    seed_tree = gp.PrimitiveTree.from_string(seed_str, pset)
    max_height = max(6, seed_tree.height + 2)
    if evaluate_fn is None:
        evaluate_fn = make_evaluate(alpha, pset, data, seed_area)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=attrgetter("height"), max_value=max_height))
    toolbox.decorate("mutate", gp.staticLimit(key=attrgetter("height"), max_value=max_height))
    toolbox.register("evaluate", evaluate_fn)

    population = []
    attempts = 0
    while len(population) < pop_size and attempts < pop_size * 10:
        try:
            population.append(toolbox.individual())
        except Exception:
            pass
        attempts += 1
    if not population:
        population = [creator.Individual(gp.PrimitiveTree.from_string(seed_str, pset))
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

    archive = tools.HallOfFame(10)
    archive.update(population)
    for _ in range(ngen):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=0.5, mutpb=0.3)

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


# ==========================================================
# Main experiment
# ==========================================================
def main():
    self_check()
    random.seed(RANDOM_SEED)
    print(f"\n{'=' * 68}")
    print(f"  Log-domain (LNS) exponent adder: exponent={EXPONENT_BITS}-bit, "
          f"output={OUTPUT_BITS}-bit")
    print(f"{'=' * 68}\n")

    pset = make_pset(EXPONENT_BITS, OUTPUT_BITS)
    data = make_exhaustive_exponent_data(EXPONENT_BITS)  # small domain, exact -- see module docstring

    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
    if hasattr(creator, "Individual"):
        del creator.Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    seed_str = make_seed_str(EXPONENT_BITS, OUTPUT_BITS)
    seed_tree = gp.PrimitiveTree.from_string(seed_str, pset)
    seed_area = count_area(seed_tree)
    seg_seeds = make_split_seed_strs(EXPONENT_BITS, OUTPUT_BITS)

    seed_error, _ = eval_true(seed_tree, pset, data)
    if seed_error != 0.0:
        raise AssertionError(f"exact seed does not match the log reference: error={seed_error}")

    print(f"Exact seed:  {seed_str}")
    print(f"Seed area:   {seed_area}")
    print(f"Safe split seeds ({len(seg_seeds)}, all offered directly -- single axis, no combos to withhold):")
    for s in seg_seeds:
        tree = gp.PrimitiveTree.from_string(s, pset)
        err, area = eval_true(tree, pset, data)
        print(f"  area={area:>2} err={err:>8.1%}  {s}")
    print(f"Exhaustive domain: {len(data)} exponent pairs\n")

    results = []
    max_iters = MAX_CEGIS if USE_CEGIS else 1

    for alpha in ERRORS:
        print(f"\n  alpha={alpha}")
        current_data = list(data)
        best = None
        verified = False

        for cegis_iter in range(max_iters):
            best = run_gp(pset, current_data, seed_area, seed_str, alpha, NGEN, POP_SIZE,
                         extra_seed_strs=seg_seeds)
            gp_error, gp_area = eval_true(best, pset, current_data)
            print(f"    iter {cegis_iter + 1}: gp_err={gp_error:.1%} area={gp_area} "
                  f"data_size={len(current_data)}")

            if gp_error > alpha:
                print(f"    GP missed constraint, stopping CEGIS")
                break

            verified, ce = verify_error_bound(best, pset, EXPONENT_BITS, OUTPUT_BITS, alpha)
            if verified:
                print(f"    VERIFIED after {cegis_iter + 1} round(s)")
                break
            if ce and 'ea' in ce:
                ea_ce, eb_ce = ce['ea'], ce['eb']
                current_data.append((ea_ce, eb_ce, ea_ce + eb_ce))
                print(f"    FV failed ce=ea={ea_ce},eb={eb_ce}, "
                      f"error={ce.get('error', float('nan')):.1%} -> added to data")
            else:
                print(f"    FV inconclusive ({ce}), stopping")
                break

        final_error, final_area = eval_true(best, pset, data)
        reduction = 100.0 * (1.0 - final_area / seed_area)
        results.append((alpha, final_error, final_area, verified, str(best)))
        print(f"  -> final: error={final_error:.1%} area={final_area} "
              f"({reduction:+.1f}% area) verified={verified} tree={str(best)[:80]}")

    fig, axis = plt.subplots(figsize=(9, 6))
    verified_pts = [(a, e, ar) for a, e, ar, v, _ in results if v]
    failed_pts = [(a, e, ar) for a, e, ar, v, _ in results if not v]
    if verified_pts:
        vs = sorted(verified_pts, key=lambda t: t[0])
        axis.plot([r[0] * 100 for r in vs], [r[2] for r in vs],
                  'o-', color='#2E7D32', label='VERIFIED', markersize=10)
    if failed_pts:
        axis.scatter([r[0] * 100 for r in failed_pts], [r[2] for r in failed_pts],
                     c='#E53935', marker='x', s=100, label='FV failed / GP missed')
    axis.axhline(seed_area, color='gray', linestyle='--', label=f'exact seed area={seed_area}')
    axis.set_xlabel('alpha (error tolerance threshold, %)')
    axis.set_ylabel('Area (adder carry-truncation proxy cost)')
    axis.set_title(f'Log-domain (LNS) GP: exponent={EXPONENT_BITS}-bit -- CVC5 verified')
    axis.grid(True, alpha=0.3)
    axis.legend()

    filename = f"outcome/pareto_log_exp{EXPONENT_BITS}_out{OUTPUT_BITS}.png"
    fig.tight_layout()
    fig.savefig(filename, dpi=140)
    print(f"\nSaved {filename}")

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    print(f"     alpha    error    area    FV")
    print(f"  {'-' * 40}")
    for alpha, err, area, verified, _ in results:
        tag = "VERIFIED" if verified else "  FAILED"
        print(f"    {alpha:>5}   {err * 100:>5.1f}%   {area:>4}   {tag}")


if __name__ == "__main__":
    main()
