"""GP search for an approximate MSFP (Microsoft Floating Point) block
mantissa multiplier -- DeepWok/mase's block_fp.py.

Structurally identical to search/mxint_hardware_search.py (same MXIntBlock shape,
same element grammar, same CVC5 verifier -- verify/verify_mxint_error_bound
.py operates purely on mantissa integers and never looks at how the shared
exponent was chosen, so it's reused completely unchanged here). All shared
GP machinery lives in search/mxint_common.py. The only real difference is
benchmarks/block_fp.py's own choose_shared_exponent/make_data, which
matches MASE's real block_fp.py formula instead of mxint_hardware.py's.
"""

import random

import matplotlib.pyplot as plt
from deap import base, creator, gp

from benchmarks.mxint_hardware import MXIntBlock
from benchmarks.block_fp import (
    dequantize_block,
    make_data,
    make_exhaustive_mantissa_data,
    make_narrow_seed_strs,
    make_partial_seed_strs,
    make_pset,
    make_seed_str,
    reference_multiply_blocks,
    self_check,
)
from search.mxint_common import count_area, eval_true, run_gp
from verify.verify_mxint_hardware_error_bound import verify_error_bound


# ==========================================================
# Configuration
# ==========================================================
ELEMENT_BITS = 6
OUTPUT_BITS = 12
BLOCK_SIZE = 4
EXPONENT_WIDTH = 3   # shared-exponent field width for benchmarks.block_fp's
                      # choose_shared_exponent, matching MASE's block_fp.py
                      # exactly (exponent_bias=2**(ew-1)-1, standard IEEE
                      # bias -- unlike mxint_hardware.py's, this formula
                      # explicitly scales by 2**(element_bits-1) after
                      # normalizing to the block's own max, so it uses the
                      # mantissa's full dynamic range regardless of how
                      # EXPONENT_WIDTH and ELEMENT_BITS are paired).

ERRORS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
POP_SIZE = 300
NGEN = 60
N_BLOCKS = 100
RANDOM_SEED = 42
MAX_CEGIS = 3
USE_CEGIS = True
MANTISSA_DOMAIN = True   # see search/mxint_hardware_search.py's config docstring;
                          # same reasoning, must match verify_error_bound's
                          # mantissa_domain param below.


# ==========================================================
# Main experiment
# ==========================================================
def main():
    self_check()
    random.seed(RANDOM_SEED)
    print(f"\n{'=' * 68}")
    print(
        f"  block_fp (MSFP) block: size={BLOCK_SIZE}, element={ELEMENT_BITS}-bit, "
        f"product={OUTPUT_BITS}-bit"
    )
    print(f"{'=' * 68}\n")

    pset = make_pset(ELEMENT_BITS, OUTPUT_BITS)
    data = make_data(
        ELEMENT_BITS,
        EXPONENT_WIDTH,
        block_size=BLOCK_SIZE,
        n_blocks=N_BLOCKS,
        seed=RANDOM_SEED,
    )
    exhaustive_data = make_exhaustive_mantissa_data(ELEMENT_BITS, mantissa_domain=MANTISSA_DOMAIN)
    domain_fraction = 0.5 if MANTISSA_DOMAIN else None

    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
    if hasattr(creator, "Individual"):
        del creator.Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    seed_str = make_seed_str(ELEMENT_BITS, OUTPUT_BITS)
    seed_tree = gp.PrimitiveTree.from_string(seed_str, pset)
    seed_area = count_area(seed_tree)
    narrow_seeds = make_narrow_seed_strs(ELEMENT_BITS, OUTPUT_BITS, single_axis_only=True)
    exact_partial_seed, _partial_approx_seeds = make_partial_seed_strs(
        ELEMENT_BITS,
        OUTPUT_BITS,
    )
    helper_seeds = list(narrow_seeds)

    seed_error, _, seed_zeroed = eval_true(seed_tree, pset, data, dequantize_block, domain_fraction)
    seed_full_error, _, _ = eval_true(seed_tree, pset, exhaustive_data, dequantize_block, domain_fraction)
    if seed_error != 0.0:
        raise AssertionError(
            f"exact seed does not match the block_fp reference: error={seed_error}"
        )
    if exact_partial_seed is not None:
        partial_tree = gp.PrimitiveTree.from_string(exact_partial_seed, pset)
        partial_full_error, _, _ = eval_true(
            partial_tree,
            pset,
            exhaustive_data,
            dequantize_block,
            domain_fraction,
        )
        if partial_full_error != 0.0:
            raise AssertionError(
                "exact partial-product seed does not match the block_fp "
                f"reference: error={partial_full_error}"
            )

    print(f"Exact seed:   {seed_str}")
    print(f"Seed area:    {seed_area}")
    print(f"Seed error:   {seed_error:.1%}")
    print(f"Seed full:    {seed_full_error:.1%} over {len(exhaustive_data)} pairs")
    print(f"Seed zeroed:  {seed_zeroed}")
    print(f"Narrow seeds: {len(narrow_seeds)} (single-axis only, joint combos withheld)")
    print(f"Training:     {len(data)} blocks / {len(data) * BLOCK_SIZE} products\n")

    print("Helper seed diagnostics:")
    for label, candidate_seed in [
        *[(f"narrow-{index + 1}", value)
          for index, value in enumerate(narrow_seeds)],
    ]:
        tree = gp.PrimitiveTree.from_string(candidate_seed, pset)
        train_error, area, _ = eval_true(tree, pset, data, dequantize_block, domain_fraction)
        full_error, _, full_zeroed = eval_true(tree, pset, exhaustive_data, dequantize_block, domain_fraction)
        print(
            f"  {label:<17} area={area:>3} "
            f"train={train_error:>7.1%} full={full_error:>7.1%} "
            f"zeroed={full_zeroed:>3}"
        )
    print()

    results = []
    max_iters = MAX_CEGIS if USE_CEGIS else 1

    for alpha in ERRORS:
        print(f"\n  alpha={alpha}")
        current_data = list(data)
        best = None
        verified = False

        for cegis_iter in range(max_iters):
            best = run_gp(
                pset,
                current_data,
                seed_area,
                seed_str,
                alpha,
                NGEN,
                POP_SIZE,
                dequantize_block,
                extra_seed_strs=helper_seeds,
                element_bits=ELEMENT_BITS,
                output_bits=OUTPUT_BITS,
                domain_fraction=domain_fraction,
            )
            gp_error, gp_area, _ = eval_true(best, pset, current_data, dequantize_block, domain_fraction)
            print(f"    iter {cegis_iter + 1}: gp_err={gp_error:.1%} area={gp_area} "
                  f"data_size={len(current_data)}")

            if gp_error > alpha:
                print(f"    GP missed constraint, stopping CEGIS")
                break

            verified, ce = verify_error_bound(best, pset, ELEMENT_BITS, OUTPUT_BITS,
                                              alpha, mantissa_domain=MANTISSA_DOMAIN)
            if verified:
                print(f"    VERIFIED after {cegis_iter + 1} round(s)")
                break
            if ce and 'ma' in ce:
                ma_ce, mb_ce = ce['ma'], ce['mb']
                a_ce = MXIntBlock(0, (ma_ce,), ELEMENT_BITS)
                b_ce = MXIntBlock(0, (mb_ce,), ELEMENT_BITS)
                current_data.append((a_ce, b_ce, reference_multiply_blocks(a_ce, b_ce)))
                print(f"    FV failed ce=ma={ma_ce},mb={mb_ce}, "
                      f"error={ce.get('error', float('nan')):.1%} -> added to data")
            else:
                print(f"    FV inconclusive ({ce}), stopping")
                break

        train_error, area, zeroed = eval_true(best, pset, data, dequantize_block, domain_fraction)
        full_error, _, full_zeroed = eval_true(best, pset, exhaustive_data, dequantize_block, domain_fraction)
        reduction = 100.0 * (1.0 - area / seed_area)
        results.append(
            (alpha, train_error, full_error, area, zeroed, full_zeroed, verified, str(best))
        )
        print(
            f"  -> final: train={train_error:>7.1%} "
            f"full={full_error:>7.1%} area={area:>3} "
            f"zeroed={zeroed:>3}/{full_zeroed:<3} ({reduction:+.1f}% area) "
            f"verified={verified} tree={str(best)[:90]}"
        )

    fig, axis = plt.subplots(figsize=(9, 6))
    verified_pts = [r for r in results if r[6]]
    failed_pts = [r for r in results if not r[6]]
    if verified_pts:
        vs = sorted(verified_pts, key=lambda r: r[0])
        axis.plot([r[0] * 100 for r in vs], [r[3] for r in vs],
                  'o-', color='#2E7D32', label='VERIFIED', markersize=10)
    if failed_pts:
        axis.scatter([r[0] * 100 for r in failed_pts], [r[3] for r in failed_pts],
                     c='#E53935', marker='x', s=100, label='FV failed / GP missed')
    axis.axhline(
        seed_area,
        color="gray",
        linestyle="--",
        label=f"exact seed area={seed_area}",
    )
    axis.set_xlabel("alpha (error tolerance threshold, %)")
    axis.set_ylabel("Area (stage-1 proxy cost)")
    axis.set_title(
        f"block_fp (MSFP) GP: block={BLOCK_SIZE}, element={ELEMENT_BITS}-bit -- CVC5 verified"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()

    filename = (
        f"outcome/pareto_block_fp_block{BLOCK_SIZE}_elem{ELEMENT_BITS}"
        f"_out{OUTPUT_BITS}.png"
    )
    fig.tight_layout()
    fig.savefig(filename, dpi=140)
    print(f"\nSaved {filename}")

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    print(f"     alpha    error    area    FV")
    print(f"  {'-' * 40}")
    for alpha, _, full_error, area, _, _, verified, _ in results:
        tag = "VERIFIED" if verified else "  FAILED"
        print(f"    {alpha:>5}   {full_error * 100:>5.1f}%   {area:>4}   {tag}")


if __name__ == "__main__":
    main()
