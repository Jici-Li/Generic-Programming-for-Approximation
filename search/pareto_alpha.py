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

from benchmarks.int_mult import make_pset, make_seed_str, make_data
from search.area_model import count_area
from verify.verify_error_bound import verify_error_bound


# ==========================================================
# Configuration
# ==========================================================
# Per-port bitwidths (Jianyi's example: ma:4, mb:4, out:8)
MA_BITS     = 4
MB_BITS     = 4
OUTPUT_BITS = 8

ERRORS      = [0.01, 0.02, 0.05, 0.10, 0.20]
POP_SIZE    = 300
NGEN        = 60
N_DATA      = 100
MAX_CEGIS   = 5
USE_CEGIS   = True


# ==========================================================
# Fitness
# ==========================================================
def make_evaluate(alpha, pset, data, ma_bits, mb_bits, output_bits, seed_area):
    def evaluate(individual):
        try:
            func = gp.compile(individual, pset)
        except Exception:
            return (99999.0,)
        max_err = 0.0
        for ma, mb, target in data:
            try:
                result = func(ma, mb)
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

        if max_err > alpha:
            penalty = seed_area * (1.0 + max_err / max(alpha, 1e-6))
            return (float(penalty),)
        return (float(area),)
    return evaluate


def eval_true(individual, pset, data, ma_bits, mb_bits, output_bits):
    """Re-check an individual's true (error, area), ignoring the penalty."""
    func = gp.compile(individual, pset)
    max_err = 0.0
    for ma, mb, target in data:
        result = func(ma, mb)
        if target == 0:
            continue
        err = abs(result - target) / target
        if err > max_err:
            max_err = err
    return max_err, count_area(individual)


# ==========================================================
# GP search
# ==========================================================
def run_gp(pset, data, ma_bits, mb_bits, output_bits, seed_area,
           seed_str, alpha, ngen, pop_size):

    seed_tree = gp.PrimitiveTree.from_string(seed_str, pset)
    max_h = max(20, seed_tree.height + 2)

    local_toolbox = base.Toolbox()
    local_toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    local_toolbox.register("individual", tools.initIterate,
                           creator.Individual, local_toolbox.expr)
    local_toolbox.register("population", tools.initRepeat,
                           list, local_toolbox.individual)
    local_toolbox.register("select", tools.selTournament, tournsize=3)
    local_toolbox.register("mate",   gp.cxOnePoint)
    local_toolbox.register("mutate", gp.mutUniform,
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

    # inject the seed into the population
    seed_ind = creator.Individual(gp.PrimitiveTree.from_string(seed_str, pset))
    seed_ind.fitness.values = local_toolbox.evaluate(seed_ind)
    pop.append(seed_ind)

    hof = tools.HallOfFame(1)
    pop, _ = algorithms.eaSimple(
        pop, local_toolbox, cxpb=0.5, mutpb=0.2,
        ngen=ngen, halloffame=hof, verbose=False
    )
    return hof[0]


# ==========================================================
# Main experiment
# ==========================================================
def main():
    print(f"\n{'='*60}")
    print(f"  ma={MA_BITS}-bit, mb={MB_BITS}-bit, output={OUTPUT_BITS}-bit")
    print(f"{'='*60}\n")

    pset = make_pset(MA_BITS, MB_BITS, OUTPUT_BITS)
    base_data = make_data(MA_BITS, MB_BITS, n_samples=N_DATA,
                          mantissa_domain=False)

    if hasattr(creator, "FitnessMin"): del creator.FitnessMin
    if hasattr(creator, "Individual"): del creator.Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    seed_str = make_seed_str(MA_BITS, MB_BITS, OUTPUT_BITS)
    seed_tree = gp.PrimitiveTree.from_string(seed_str, pset)
    seed_area = count_area(seed_tree)
    print(f"Seed:        {seed_str}")
    print(f"Seed height: {seed_tree.height}")
    print(f"Seed area:   {seed_area}")

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
                          seed_area, seed_str, alpha, NGEN, POP_SIZE)
            gp_err, gp_area = eval_true(best, pset, current_data,
                                        MA_BITS, MB_BITS, OUTPUT_BITS)
            print(f"    iter {cegis_iter+1}: "
                  f"gp_err={gp_err:.1%} area={gp_area} data_size={len(current_data)}")

            if gp_err > alpha:
                print(f"    GP missed constraint, stopping CEGIS")
                break

            verified, ce = verify_error_bound(best, pset,
                                              MA_BITS, MB_BITS, OUTPUT_BITS,
                                              alpha)
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

    # ---- Pareto plot ----
    fig, ax = plt.subplots(figsize=(9, 6))
    verified_pts = [(a, e, ar) for (a, e, ar, v) in results if v]
    failed_pts   = [(a, e, ar) for (a, e, ar, v) in results if not v]
    if verified_pts:
        vs = sorted(verified_pts, key=lambda t: t[2])
        ax.plot([r[2] for r in vs], [r[1] * 100 for r in vs],
                'o-', color='#2E7D32', label='VERIFIED', markersize=10)
    if failed_pts:
        ax.scatter([r[2] for r in failed_pts], [r[1] * 100 for r in failed_pts],
                   c='#E53935', marker='x', s=100, label='FV failed / GP missed')
    ax.set_xlabel('Area (node count)')
    ax.set_ylabel('Error (%)')
    ax.set_title(f'Accuracy-Area Trade-off '
                 f'(ma={MA_BITS}b, mb={MB_BITS}b, out={OUTPUT_BITS}b)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    tag = "cegis" if USE_CEGIS else "nocegis"
    fname = f'pareto_ma{MA_BITS}_mb{MB_BITS}_out{OUTPUT_BITS}_{tag}.png'
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