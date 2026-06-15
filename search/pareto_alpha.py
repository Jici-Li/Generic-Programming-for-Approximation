import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms, gp
from operator import attrgetter

from verify.verify_error_bound import verify_error_bound
from benchmarks.int_mult import make_pset, make_seed_str, make_data
from search.area_model import count_area_nodes, eval_true

BIT_WIDTHS = [4, 8, 16, 32]
ALPHAS     = [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
NGEN_SMALL = 100
NGEN_LARGE = 200
POP_SIZE   = 499
N_DATA     = 100
N_DATA_32  = 500
MAX_CEGIS  = 5

def make_evaluate(alpha, pset, data, input_bits, seed_area):
    def evaluate(individual):
        func = gp.compile(individual, pset)
        try:
            results = []
            for ma, mb, target in data:
                try:
                    result = func(ma, mb)
                    results.append(abs(result - target) / target)
                except:
                    results.append(1.0)
            avg_error = sum(results) / len(results)
            nodes = list(individual)
            area, _ = count_area_nodes(nodes, input_bits)
            if avg_error > alpha:
                penalty = seed_area * (1.0 + avg_error / alpha)
                return float(penalty),
            else:
                return float(area),
        except:
            return 99999,
    return evaluate

def run_gp(pset, data, input_bits, seed_area, seed_str,
           alpha, ngen, pop_size):
    """每次调用都建一个新的local toolbox，避免状态污染。"""

    # 计算seed的树高，用于设置高度限制
    seed_tree = gp.PrimitiveTree.from_string(seed_str, pset)
    max_h = max(20, seed_tree.height + 2)

    local_toolbox = base.Toolbox()
    local_toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
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
        make_evaluate(alpha, pset, data, input_bits, seed_area))

    random.seed(42)
    pop = local_toolbox.population(n=pop_size)

    seed_ind = creator.Individual(
        gp.PrimitiveTree.from_string(seed_str, pset)
    )
    seed_ind.fitness.values = local_toolbox.evaluate(seed_ind)
    pop.append(seed_ind)

    pop, _ = algorithms.eaSimple(
        pop, local_toolbox, cxpb=0.5, mutpb=0.2,
        ngen=ngen, verbose=False
    )

    return tools.selBest(pop, 1)[0]


all_results = {}

for INPUT_BITS in BIT_WIDTHS:
    print(f"\n{'='*55}")
    print(f"  INT{INPUT_BITS}")
    print(f"{'='*55}")

    pset      = make_pset(INPUT_BITS)
    n_data    = N_DATA_32 if INPUT_BITS >= 32 else N_DATA
    base_data = make_data(INPUT_BITS, n_data, seed=42)

    seed_str  = make_seed_str(INPUT_BITS + 1)
    seed_tree = gp.PrimitiveTree.from_string(seed_str, pset)
    seed_area, _ = count_area_nodes(list(seed_tree), INPUT_BITS)
    print(f"  Seed height: {seed_tree.height}")
    print(f"  Seed area:   {seed_area}")

    # DEAP creator setup (once per bit width)
    if hasattr(creator, "FitnessMin"):  del creator.FitnessMin
    if hasattr(creator, "Individual"): del creator.Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    pareto = []
    ngen   = NGEN_LARGE if INPUT_BITS >= 16 else NGEN_SMALL

    # ── GP + CEGIS loop ──
    for alpha in ALPHAS:
        print(f"\n  alpha={alpha}")

        current_data = list(base_data)
        best     = None
        verified = False
        ce       = None

        for cegis_iter in range(MAX_CEGIS):
            # GP search with current_data
            best = run_gp(pset, current_data, INPUT_BITS, seed_area,
                          seed_str, alpha, ngen, POP_SIZE)

            true_err, true_area = eval_true(best, pset, current_data, INPUT_BITS)
            gp_status = "OK" if true_err <= alpha else "MISS"
            print(f"    iter {cegis_iter+1}: error={true_err:.1%} "
                  f"area={true_area} [{gp_status}] "
                  f"data_size={len(current_data)}", flush=True)

            if true_err > alpha:
                print(f"    GP missed constraint, stopping CEGIS")
                break

            # FV verification
            verified, ce = verify_error_bound(best, pset, INPUT_BITS, alpha)

            if verified:
                print(f"    VERIFIED after {cegis_iter+1} GP round(s)")
                break
            else:
                ma_ce = ce['ma']
                mb_ce = ce['mb']
                current_data.append((ma_ce, mb_ce, ma_ce * mb_ce))
                print(f"    FV failed ce=ma={ma_ce},mb={mb_ce},"
                      f"error={ce['error']:.1%} → added to data")

        # Evaluate final best on original base_data
        final_err, final_area = eval_true(best, pset, base_data, INPUT_BITS)
        pareto.append((alpha, final_err, final_area, str(best), best, verified))
        print(f"  → final: error={final_err:.1%} area={final_area} "
              f"verified={verified}")

    all_results[INPUT_BITS] = pareto

    # ── Plot ──
    gp_ok_fv_ok   = [(a,e,ar) for a,e,ar,_,__,v in pareto
                     if e<=a and v == True]
    gp_ok_fv_fail = [(a,e,ar) for a,e,ar,_,__,v in pareto
                     if e<=a and v == False]
    gp_miss       = [(a,e,ar) for a,e,ar,_,__,v in pareto if e>a]

    fig, ax = plt.subplots(figsize=(9, 5))

    if gp_ok_fv_ok:
        vr = [x[2] for x in gp_ok_fv_ok]
        ve = [x[1] for x in gp_ok_fv_ok]
        va = [x[0] for x in gp_ok_fv_ok]
        ax.scatter(vr, ve, color='steelblue', s=100, zorder=5,
                   label='GP OK + FV verified')
        for i in range(len(gp_ok_fv_ok)):
            ax.annotate(f'a={va[i]}', (vr[i], ve[i]),
                        textcoords='offset points', xytext=(5, 4), fontsize=8)
        if len(gp_ok_fv_ok) > 1:
            sv = sorted(gp_ok_fv_ok, key=lambda x: x[2])
            ax.plot([x[2] for x in sv], [x[1] for x in sv],
                    color='steelblue', linewidth=1.5, alpha=0.5)

    if gp_ok_fv_fail:
        ax.scatter([x[2] for x in gp_ok_fv_fail],
                   [x[1] for x in gp_ok_fv_fail],
                   color='orange', s=100, marker='D', zorder=5,
                   label='GP OK but FV failed')
        for x in gp_ok_fv_fail:
            ax.annotate(f'a={x[0]}', (x[2], x[1]),
                        textcoords='offset points', xytext=(5, 4), fontsize=8)

    if gp_miss:
        ax.scatter([x[2] for x in gp_miss],
                   [x[1] for x in gp_miss],
                   color='tomato', s=100, marker='x', zorder=5,
                   label='GP constraint missed')
        for x in gp_miss:
            ax.annotate(f'a={x[0]}', (x[2], x[1]),
                        textcoords='offset points', xytext=(5, 4), fontsize=8)

    ax.axvline(x=seed_area, color='gray', linestyle='--',
               alpha=0.5, label=f'seed area={seed_area}')
    ax.set_xlabel('Area (bitwidth-weighted add/sub)', fontsize=12)
    ax.set_ylabel('Avg Relative Error', fontsize=12)
    ax.set_title(
        f'Accuracy-Area Tradeoff (INT{INPUT_BITS}, GP+CEGIS+FV)',
        fontsize=13)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f'pareto_int{INPUT_BITS}_cegis.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"\n  Saved {fname}")

# ── Summary ──
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for bits, pareto in all_results.items():
    print(f"\nINT{bits}:")
    print(f"  {'alpha':>8} {'error':>10} {'area':>8} {'GP':>6} {'FV':>10}")
    print(f"  {'-'*46}")
    for alpha, err, area, _, __, verified in pareto:
        gp_met = "OK"   if err <= alpha else "MISS"
        fv_str = "VERIFIED" if verified else ("FAILED" if err <= alpha else "-")
        print(f"  {alpha:>8} {err:>10.1%} {area:>8} {gp_met:>6} {fv_str:>10}")