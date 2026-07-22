"""
GP search for approximate minifloat multiplier synthesis.

Mirrors search/pareto_alpha.py's evolutionary loop, but for the minifloat
grammar in benchmarks/minifloat_mult.py instead of the integer one, and
without the CVC5 formal-verification / CEGIS loop (todo: add once the
minifloat -> CVC5 translation exists, analogous to verify/cvc5_translate.py).

Fitness: minimize area subject to a max relative-error constraint alpha,
same shape as search/pareto_alpha.py's make_evaluate.
"""

import re
from operator import attrgetter

import matplotlib.pyplot as plt
from deap import base, creator, tools, gp, algorithms

from benchmarks.minifloat_mult import (make_pset, make_seed_str, make_narrow_seed_strs,
                                       make_data, decode)


# ==========================================================
# Configuration
# ==========================================================
MA_FORMAT     = (4, 3)   # E4M3-style FP8 input
MB_FORMAT     = (4, 3)
OUTPUT_FORMAT = (5, 4)   # wider output format to land approximations in

ERRORS    = [0.01, 0.02, 0.05, 0.10, 0.20]
POP_SIZE  = 300
NGEN      = 60
N_DATA    = 100

EXCLUDE_SUBNORMAL_INPUTS = False   # flip after supervisor confirms domain
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
                err = abs(result - target) / abs(target)
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
        if err > max_err:
            max_err = err
    return max_err, count_area(individual)


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
    toolbox.register("mate", gp.cxOnePoint)
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

    hof = tools.HallOfFame(1)
    pop, _ = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                 ngen=ngen, halloffame=hof, verbose=False)
    return hof[0]


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
    narrow_seed_strs = make_narrow_seed_strs(MA_FORMAT, MB_FORMAT, OUTPUT_FORMAT)
    print(f"Seed:      {seed_str}")
    print(f"Seed area: {seed_area}")
    print(f"Extra seeds: {len(narrow_seed_strs)} narrowed mantissa combinations\n")

    results = []
    for alpha in errors:
        best = run_gp(pset, data, OUTPUT_FORMAT, seed_area, seed_str,
                      alpha, NGEN, POP_SIZE, extra_seed_strs=narrow_seed_strs)
        err, area = eval_true(best, pset, data, OUTPUT_FORMAT)
        results.append((alpha, err, area))
        reduction = 100.0 * (1.0 - area / seed_area)
        print(f"  alpha={alpha:<5}  best_err={err:.1%}  area={area:>4}  "
              f"({reduction:+.1f}% vs seed area={seed_area})  "
              f"tree={str(best)[:80]}")

    fig, ax = plt.subplots(figsize=(9, 6))
    pts = sorted(results, key=lambda t: t[2])
    ax.plot([r[2] for r in pts], [r[1] * 100 for r in pts], 'o-', label='best found')
    ax.axvline(seed_area, color='gray', linestyle='--',
               label=f'seed area={seed_area}')
    ax.set_xlabel('Area (placeholder cost)')
    ax.set_ylabel('Worst-case relative error (%)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fname = (f'pareto_minifloat_ma{MA_FORMAT}_mb{MB_FORMAT}'
             f'_out{OUTPUT_FORMAT}.png')
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    print(f"\nSaved {fname}")


if __name__ == "__main__":
    main()
