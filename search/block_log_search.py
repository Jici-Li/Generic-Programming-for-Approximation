"""GP search for an approximate block log-domain (LNS) exponent adder --
DeepWok/mase's block_log.py.

Mirrors search/log_search.py exactly at the element level (same grammar,
same run_gp/node_cost, same CVC5 verifier) -- see benchmarks/block_log.py's
module docstring for why: the block's shared exponent bias is folded
directly into each element's own quantized exponent at data-generation
time, so the elementwise operation GP searches over (and what CVC5
verifies) is completely unaffected by it. Only the training-data
generation is block-aware here.
"""

import random

import matplotlib.pyplot as plt
from deap import base, creator, gp

from benchmarks.block_log import (
    LogBlock,
    make_data,
    reference_multiply_blocks,
    self_check,
)
from benchmarks.log import (
    LogValue,
    make_exhaustive_exponent_data,
    make_pset,
    make_seed_str,
    make_seg_seed_strs as make_split_seed_strs,
)
from search.log_search import count_area, run_gp
from verify.verify_log_error_bound import verify_error_bound


# ==========================================================
# Configuration
# ==========================================================
WIDTH = 8  # 1 sign bit + 7 exponent bits
EXPONENT_BITS = WIDTH - 1
OUTPUT_BITS = EXPONENT_BITS + 1
EXPONENT_BIAS_WIDTH = 4  # shared-bias field width, MASE's own separate knob
BLOCK_SIZE = 4
N_BLOCKS = 100
RANDOM_SEED = 42

ERRORS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
POP_SIZE = 100
NGEN = 30
MAX_CEGIS = 3
USE_CEGIS = True


# ==========================================================
# Fitness (block-aware: elementwise over each block, exact reference
# exponents are pulled straight out of each already-quantised LogValue --
# no re-deriving the shared bias needed since it's baked into each
# element already).
# ==========================================================
def eval_true(individual, pset, data):
    func = gp.compile(individual, pset)
    max_error = 0.0
    for a, b, target in data:
        for av, bv, tv in zip(a.values, b.values, target.values):
            approx_exp = int(func(av.exponent, bv.exponent))
            exact_exp = tv.exponent
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
            for a, b, target in data:
                for av, bv, tv in zip(a.values, b.values, target.values):
                    approx_exp = int(func(av.exponent, bv.exponent))
                    d = approx_exp - tv.exponent
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
# Main experiment
# ==========================================================
def main():
    self_check()
    random.seed(RANDOM_SEED)
    print(f"\n{'=' * 68}")
    print(f"  Block log-domain (LNS) adder: width={WIDTH}, block={BLOCK_SIZE}, "
          f"exponent_bias_width={EXPONENT_BIAS_WIDTH}")
    print(f"{'=' * 68}\n")

    pset = make_pset(EXPONENT_BITS, OUTPUT_BITS)
    data = make_data(WIDTH, EXPONENT_BIAS_WIDTH, block_size=BLOCK_SIZE,
                     n_blocks=N_BLOCKS, seed=RANDOM_SEED)
    exhaustive_data = make_exhaustive_exponent_data(EXPONENT_BITS)

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
        raise AssertionError(f"exact seed does not match the block_log reference: error={seed_error}")

    print(f"Exact seed:  {seed_str}")
    print(f"Seed area:   {seed_area}")
    print(f"Safe split seeds ({len(seg_seeds)}):")
    for s in seg_seeds:
        tree = gp.PrimitiveTree.from_string(s, pset)
        train_err, area = eval_true(tree, pset, data)
        # also check against the exhaustive raw-exponent domain -- the
        # elementwise operation is identical to flat log's, so the
        # worst case should match search/log_search.py's numbers exactly.
        func = gp.compile(tree, pset)
        full_err = 0.0
        for ea, eb, exact in exhaustive_data:
            d = int(func(ea, eb)) - exact
            err = (2.0 ** d - 1.0) if d >= 0 else (1.0 - 2.0 ** d)
            full_err = max(full_err, err)
        print(f"  area={area:>2} train={train_err:>8.1%} full={full_err:>8.1%}  {s}")
    print(f"Training: {len(data)} blocks / {len(data) * BLOCK_SIZE} element products\n")

    results = []
    max_iters = MAX_CEGIS if USE_CEGIS else 1

    for alpha in ERRORS:
        print(f"\n  alpha={alpha}")
        current_data = list(data)
        best = None
        verified = False

        for cegis_iter in range(max_iters):
            best = run_gp(pset, current_data, seed_area, seed_str, alpha, NGEN, POP_SIZE,
                         extra_seed_strs=seg_seeds,
                         evaluate_fn=make_evaluate(alpha, pset, current_data, seed_area))
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
                a_ce = LogBlock((LogValue(False, ea_ce),))
                b_ce = LogBlock((LogValue(False, eb_ce),))
                current_data.append((a_ce, b_ce, reference_multiply_blocks(a_ce, b_ce)))
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
    axis.set_title(f'Block log-domain (LNS) GP: width={WIDTH}, block={BLOCK_SIZE} -- CVC5 verified')
    axis.grid(True, alpha=0.3)
    axis.legend()

    filename = f"outcome/pareto_block_log_w{WIDTH}_block{BLOCK_SIZE}.png"
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
