import os

from deap import base, creator, tools, gp, algorithms
from operator import attrgetter

import math

from benchmarks.minifloat import decode as _decode, bitsplit as _bitsplit, _bias as _bias_of
from synth.minifloat_synth import count_area_synth

REAL_SYNTH_COST = os.environ.get('GP_REAL_SYNTH_COST', '0') == '1'

def _placeholder_area(individual):
    return float(len(individual))

def area_of(individual, src_format=(4, 3), out_format=(5, 4)):
    if REAL_SYNTH_COST:
        return count_area_synth(individual, src_format, out_format)
    return _placeholder_area(individual)

def make_normal_domain(src_format=(4, 3), exclude_all_ones=True):
    e, m = src_format
    max_field = (1 << e) - 1
    bits_list = []
    for bits in range(1 << (e + m + 1)):
        _, expf, frac = _bitsplit(bits, e, m)
        # exp==0 covers BOTH exact zero (frac==0) and genuine subnormals
        # (frac!=0) -- only exclude the latter. Excluding exact zero too
        # (as this used to) meant "anything times/plus zero" was never in
        # the training/verification domain at all, so GP was free to get
        # it wrong for free (verified: a GP individual that outputs
        # garbage for a=0 still scored a perfect fitness, since the case
        # was never sampled or checked).
        if expf == 0 and frac != 0:
            continue
        if exclude_all_ones and expf == max_field:
            continue
        bits_list.append(bits)
    return bits_list

def _sample_normal_bits(rng, e, m, exclude_all_ones=True):
    max_field = (1 << e) - 1
    while True:
        bits = rng.getrandbits(e + m + 1)
        _, expf, frac = _bitsplit(bits, e, m)
        # see make_normal_domain's comment -- exclude subnormal (frac!=0),
        # keep exact zero (frac==0) in the sampleable domain.
        if expf == 0 and frac != 0:
            continue
        if exclude_all_ones and expf == max_field:
            continue
        return bits

def make_data(src_format=(4, 3), n_samples=120, seed=42):
    import random
    e, m = src_format
    rng = random.Random(seed)
    data = []
    for _ in range(n_samples):
        a = _sample_normal_bits(rng, e, m)
        b = _sample_normal_bits(rng, e, m)
        av, bv = _decode(a, e, m), _decode(b, e, m)
        data.append((a, b, av * bv))
    return data

def make_add_data(src_format=(4, 3), n_samples=120, seed=42):
    import random
    e, m = src_format
    rng = random.Random(seed)
    data = []
    for _ in range(n_samples):
        a = _sample_normal_bits(rng, e, m)
        b = _sample_normal_bits(rng, e, m)
        av, bv = _decode(a, e, m), _decode(b, e, m)
        data.append((a, b, av + bv))
    return data

def make_evaluate(alpha, pset, data, src_format, out_format, seed_area):
    eo, mo = out_format

    def evaluate(individual):
        try:
            func = gp.compile(individual, pset)
        except Exception:
            return (99999.0,)
        max_err = 0.0
        for a_bits, b_bits, exact in data:
            try:
                out_bits = int(func(a_bits, b_bits))
                got = _decode(out_bits, eo, mo)
                if exact == 0:
                    continue
                err = abs(got - exact) / abs(exact)
                if err > 100:
                    err = 100.0
            except Exception:
                err = 100.0
            if err > max_err:
                max_err = err
        area = area_of(individual, src_format, out_format)
        overshoot = max(0.0, max_err - alpha) / max(alpha, 1e-6)
        cost = area + overshoot * seed_area * 1000.0
        return (float(cost),)
    return evaluate

def eval_true(individual, pset, data, src_format, out_format):
    eo, mo = out_format
    func = gp.compile(individual, pset)
    max_err = 0.0
    for a_bits, b_bits, exact in data:
        out_bits = int(func(a_bits, b_bits))
        got = _decode(out_bits, eo, mo)
        if exact == 0:
            continue
        err = abs(got - exact) / abs(exact)
        if err > max_err:
            max_err = err
    return max_err, area_of(individual, src_format, out_format)

def relative_rmse(individual, pset, data, out_format):
    eo, mo = out_format
    func = gp.compile(individual, pset)
    sq_errs = []
    for a_bits, b_bits, exact in data:
        if exact == 0:
            continue
        try:
            out_bits = int(func(a_bits, b_bits))
            got = _decode(out_bits, eo, mo)
            err = (got - exact) / exact
        except Exception:
            err = 1.0
        sq_errs.append(err * err)
    if not sq_errs:
        return 0.0
    return (sum(sq_errs) / len(sq_errs)) ** 0.5


def make_evaluate_rmse(pset, data, src_format, out_format, rmse_weight, seed_area):
    def evaluate(individual):
        try:
            gp.compile(individual, pset)
        except Exception:
            return (99999.0,)
        try:
            rmse = relative_rmse(individual, pset, data, out_format)
        except Exception:
            rmse = 1.0
        area = area_of(individual, src_format, out_format)
        cost = area + rmse_weight * seed_area * rmse
        return (float(cost),)
    return evaluate


def run_gp_rmse(pset, data, src_format, out_format, seed_area, seed_str, rmse_weight, ngen, pop_size,
                extra_seed_strs=()):
    seed_tree = gp.PrimitiveTree.from_string(seed_str, pset)
    max_h = max(20, seed_tree.height + 2)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=attrgetter("height"), max_value=max_h))
    toolbox.decorate("mutate", gp.staticLimit(key=attrgetter("height"), max_value=max_h))
    toolbox.register("evaluate", make_evaluate_rmse(pset, data, src_format, out_format, rmse_weight, seed_area))

    pop = []
    attempts = 0
    while len(pop) < pop_size and attempts < pop_size * 5:
        try:
            pop.append(toolbox.individual())
        except Exception:
            pass
        attempts += 1
    if not pop:
        pop = [creator.Individual(gp.PrimitiveTree.from_string(seed_str, pset)) for _ in range(pop_size)]

    reinject_strs = [seed_str] + list(extra_seed_strs)
    reinject_inds = []
    for s in reinject_strs:
        try:
            ind = creator.Individual(gp.PrimitiveTree.from_string(s, pset))
        except Exception:
            continue
        ind.fitness.values = toolbox.evaluate(ind)
        reinject_inds.append(ind)

    archive = tools.HallOfFame(20)
    archive.update(pop)
    for _ in range(ngen):
        offspring = toolbox.select(pop, len(pop))
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
        pop[:] = offspring

    return archive[0]

def run_gp(pset, data, src_format, out_format, seed_area, seed_str, alpha, ngen, pop_size,
          extra_seed_strs=()):
    seed_tree = gp.PrimitiveTree.from_string(seed_str, pset)
    max_h = max(20, seed_tree.height + 2)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=attrgetter("height"), max_value=max_h))
    toolbox.decorate("mutate", gp.staticLimit(key=attrgetter("height"), max_value=max_h))
    toolbox.register("evaluate", make_evaluate(alpha, pset, data, src_format, out_format, seed_area))

    pop = []
    attempts = 0
    while len(pop) < pop_size and attempts < pop_size * 5:
        try:
            pop.append(toolbox.individual())
        except Exception:
            pass
        attempts += 1
    if not pop:
        pop = [creator.Individual(gp.PrimitiveTree.from_string(seed_str, pset)) for _ in range(pop_size)]

    reinject_strs = [seed_str] + list(extra_seed_strs)
    reinject_inds = []
    for s in reinject_strs:
        try:
            ind = creator.Individual(gp.PrimitiveTree.from_string(s, pset))
        except Exception:
            continue
        ind.fitness.values = toolbox.evaluate(ind)
        reinject_inds.append(ind)

    archive = tools.HallOfFame(20)
    archive.update(pop)
    for _ in range(ngen):
        offspring = toolbox.select(pop, len(pop))
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
        pop[:] = offspring

    return archive[0]

def _hw_max_finite(eo, mo):
    # Real max FINITE value for this hardware's encoding, i.e. excluding
    # the all-ones exponent field (reserved for inf/nan by InputIEEE/
    # OutputIEEE/classify_tag -- real IEEE754/FloPoCo convention). NOT the
    # same as benchmarks.minifloat.format_max_val, which treats every
    # exponent field (including all-ones) as an ordinary finite value --
    # that's a different, simpler format with no inf/nan concept at all,
    # used by other (non-hardware) benchmarks in this repo.
    bias = _bias_of(eo)
    max_normal_field = (1 << eo) - 2
    return (2.0 - 2.0 ** (-mo)) * (2.0 ** (max_normal_field - bias))

def _decode_with_inf(bits, e, m):
    # Like benchmarks.minifloat.decode, but recognizes the all-ones
    # exponent field as an overflow/infinity signal instead of reading it as
    # an ordinary (huge) finite exponent -- needed because the actual
    # hardware (InputIEEE/OutputIEEE) reserves that pattern, so a circuit
    # that correctly overflows to infinity must not be scored as "returned
    # a wrong huge finite number".
    #
    # Verified against real FloPoCo (GHDL simulation of its own generated
    # OutputIEEE, e.g. a=91,b=216 -> exact=-352): on overflow FloPoCo's own
    # R <= finalExc & sign & expSigPostRound(7:0) passes the RAW rounded
    # fraction bits through unchanged underneath the all-ones exponent --
    # it does NOT zero them out, so the fraction is leftover garbage, not a
    # real NaN payload. Our circuit reproduces that bit-for-bit. Since our
    # training/verification domain only ever feeds normal (non-NaN, non-
    # subnormal) operands, a genuine NaN can never legitimately occur here
    # (see make_exact_seed_str's excpostnorm_lookup, whose codomain is only
    # {zero, normal, inf}) -- so any all-ones exponent, garbage fraction or
    # not, means "overflowed to infinity", never "this is really NaN".
    sign, exp_field, _frac = _bitsplit(bits, e, m)
    if exp_field == (1 << e) - 1:
        return float('-inf') if sign else float('inf')
    return _decode(bits, e, m)

def _faithful_candidates(exact, out_format):
    eo, mo = out_format
    bias = _bias_of(eo)
    min_normal = 2.0 ** (1 - bias)
    if exact == 0:
        return (0.0, 0.0)
    sign = -1.0 if exact < 0 else 1.0
    v = abs(exact)
    if v < min_normal:
        return (math.copysign(0.0, exact), sign * min_normal)
    max_val = _hw_max_finite(eo, mo)
    # Real IEEE754 round-to-nearest overflow rule: only values within half a
    # ULP (at the top exponent) of max_val still round to max_val -- beyond
    # that, the correctly-rounded result is genuinely infinity, matching
    # what a real (and our) hardware overflow path does. The previous
    # version unconditionally clamped to max_val regardless of how far past
    # it `exact` was, which silently called a correct "overflowed to
    # infinity" answer wrong.
    top_ulp = 2.0 ** (((1 << eo) - 2 - bias) - mo)
    if v >= max_val + 0.5 * top_ulp:
        return (sign * float('inf'), sign * float('inf'))
    mant, exp2 = math.frexp(v)
    exp = exp2 - 1  # v in [2**exp, 2**(exp+1))
    ulp = 2.0 ** (exp - mo)
    floor_val = math.floor(v / ulp) * ulp
    ceil_val = floor_val + ulp
    floor_val = min(floor_val, max_val)
    ceil_val = min(ceil_val, max_val)
    return (sign * floor_val, sign * ceil_val)

def is_faithfully_rounded(got, exact, out_format):
    lo, hi = _faithful_candidates(exact, out_format)
    return got == lo or got == hi

def make_evaluate_faithful(pset, data, src_format, out_format, seed_area):
    eo, mo = out_format

    def evaluate(individual):
        try:
            func = gp.compile(individual, pset)
        except Exception:
            return (99999.0,)
        n_bad = 0
        n = 0
        for a_bits, b_bits, exact in data:
            n += 1
            try:
                out_bits = int(func(a_bits, b_bits))
                got = _decode_with_inf(out_bits, eo, mo)
                if not is_faithfully_rounded(got, exact, out_format):
                    n_bad += 1
            except Exception:
                n_bad += 1
        frac_bad = (n_bad / n) if n else 0.0
        area = area_of(individual, src_format, out_format)
        return (float(cost),)
    return evaluate

def eval_true_faithful(individual, pset, data, src_format, out_format):
    eo, mo = out_format
    func = gp.compile(individual, pset)
    n_ok = 0
    n_total = 0
    for a_bits, b_bits, exact in data:
        out_bits = int(func(a_bits, b_bits))
        got = _decode_with_inf(out_bits, eo, mo)
        n_total += 1
        if is_faithfully_rounded(got, exact, out_format):
            n_ok += 1
    frac_faithful = (n_ok / n_total) if n_total else 1.0
    return frac_faithful, area_of(individual, src_format, out_format)

def run_gp_faithful(pset, data, src_format, out_format, seed_area, seed_str, ngen, pop_size,
                    extra_seed_strs=()):
    seed_tree = gp.PrimitiveTree.from_string(seed_str, pset)
    max_h = max(20, seed_tree.height + 2)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=attrgetter("height"), max_value=max_h))
    toolbox.decorate("mutate", gp.staticLimit(key=attrgetter("height"), max_value=max_h))
    toolbox.register("evaluate", make_evaluate_faithful(pset, data, src_format, out_format, seed_area))

    pop = []
    attempts = 0
    while len(pop) < pop_size and attempts < pop_size * 5:
        try:
            pop.append(toolbox.individual())
        except Exception:
            pass
        attempts += 1
    if not pop:
        pop = [creator.Individual(gp.PrimitiveTree.from_string(seed_str, pset)) for _ in range(pop_size)]

    reinject_strs = [seed_str] + list(extra_seed_strs)
    reinject_inds = []
    for s in reinject_strs:
        try:
            ind = creator.Individual(gp.PrimitiveTree.from_string(s, pset))
        except Exception:
            continue
        ind.fitness.values = toolbox.evaluate(ind)
        reinject_inds.append(ind)

    archive = tools.HallOfFame(20)
    archive.update(pop)
    for _ in range(ngen):
        offspring = toolbox.select(pop, len(pop))
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
        pop[:] = offspring

    return archive[0]


def _ulp_distance(got, exact, out_format):
    lo, hi = _faithful_candidates(exact, out_format)
    if hi == lo:
        return 0.0 if got == lo else float('inf')
    dist_lo = abs(exact - lo)
    dist_hi = abs(exact - hi)
    ulp_size = abs(hi - lo)
    if dist_lo == dist_hi:
        # Exact tie: round-to-nearest-EVEN, matching the hardware's real
        # rounding mode (verified exhaustively against FloPoCo) -- NOT
        # "always round toward hi", which silently scored the seed's own
        # correct output as 1 ULP off on every halfway point.
        k = round(abs(lo) / ulp_size)   # lo/hi's shared mantissa integer +/- 1
        round_nearest = lo if k % 2 == 0 else hi
    else:
        round_nearest = lo if dist_lo < dist_hi else hi
    return abs(got - round_nearest) / ulp_size

def make_evaluate_ulp(threshold, pset, data, src_format, out_format, seed_area):
    eo, mo = out_format

    def evaluate(individual):
        try:
            func = gp.compile(individual, pset)
        except Exception:
            return (99999.0,)
        max_ulp_err = 0.0
        for a_bits, b_bits, exact in data:
            # exact==0 (e.g. one operand is exactly zero) used to be
            # skipped here -- but _faithful_candidates/_ulp_distance both
            # handle it correctly (0 ULP iff the circuit also outputs
            # exact zero, inf otherwise), so skipping it just meant
            # "anything times zero" was never fitness-checked at all.
            try:
                out_bits = int(func(a_bits, b_bits))
                got = _decode_with_inf(out_bits, eo, mo)
                ulp_err = _ulp_distance(got, exact, out_format)
                if ulp_err > 1e6:
                    ulp_err = 1e6
            except Exception:
                ulp_err = 1e6
            if ulp_err > max_ulp_err:
                max_ulp_err = ulp_err
        area = area_of(individual, src_format, out_format)
        overshoot = max(0.0, max_ulp_err - threshold) / max(threshold, 1e-6)
        cost = area + overshoot * seed_area * 1000.0
        return (float(cost),)
    return evaluate

def eval_true_ulp(individual, pset, data, src_format, out_format):
    eo, mo = out_format
    func = gp.compile(individual, pset)
    max_ulp_err = 0.0
    for a_bits, b_bits, exact in data:
        # see make_evaluate_ulp's comment -- exact==0 is no longer skipped.
        out_bits = int(func(a_bits, b_bits))
        got = _decode_with_inf(out_bits, eo, mo)
        ulp_err = _ulp_distance(got, exact, out_format)
        if ulp_err > max_ulp_err:
            max_ulp_err = ulp_err
    return max_ulp_err, area_of(individual, src_format, out_format)

def run_gp_ulp(threshold, pset, data, src_format, out_format, seed_area, seed_str, ngen, pop_size,
              extra_seed_strs=(), archive_size=20, tournsize=3, cxpb=0.5, mutpb=0.3,
              reinject_interval=1):
    seed_tree = gp.PrimitiveTree.from_string(seed_str, pset)
    max_h = max(20, seed_tree.height + 2)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=attrgetter("height"), max_value=max_h))
    toolbox.decorate("mutate", gp.staticLimit(key=attrgetter("height"), max_value=max_h))
    toolbox.register("evaluate", make_evaluate_ulp(threshold, pset, data, src_format, out_format, seed_area))

    pop = []
    attempts = 0
    while len(pop) < pop_size and attempts < pop_size * 5:
        try:
            pop.append(toolbox.individual())
        except Exception:
            pass
        attempts += 1
    if not pop:
        pop = [creator.Individual(gp.PrimitiveTree.from_string(seed_str, pset)) for _ in range(pop_size)]

    reinject_strs = [seed_str] + list(extra_seed_strs)
    reinject_inds = []
    for s in reinject_strs:
        try:
            ind = creator.Individual(gp.PrimitiveTree.from_string(s, pset))
        except Exception:
            continue
        ind.fitness.values = toolbox.evaluate(ind)
        reinject_inds.append(ind)

    archive = tools.HallOfFame(archive_size)
    archive.update(pop)
    for gen in range(ngen):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=cxpb, mutpb=mutpb)

        if gen % reinject_interval == 0:
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


def run_seed_robustness_trial(op, threshold, rng_seed, src_format=(4, 3),
                              mult_out_format=(5, 4), add_out_format=(4, 4),
                              pop_size=100, ngen=30, max_cegis=3, n_samples=120):
    import random
    import time
    from benchmarks.minifloat_hardware import (
        make_pset, make_exact_seed_str, make_add_pset, make_exact_add_seed_str,
    )
    from verify.minifloat_verify import verify_mult_ulp_bound, verify_add_ulp_bound

    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
    if hasattr(creator, "Individual"):
        del creator.Individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    random.seed(rng_seed)
    t0 = time.time()

    if op == 'mult':
        out_format = mult_out_format
        pset = make_pset(src_format, out_format)
        seed_str = make_exact_seed_str(src_format, out_format, pset=pset)
        data = make_data(src_format, n_samples=n_samples, seed=rng_seed)
        verify_fn = lambda ind, thr: verify_mult_ulp_bound(ind, src_format, out_format, thr)
    else:
        out_format = add_out_format
        pset = make_add_pset(src_format)
        seed_str = make_exact_add_seed_str(src_format, pset=pset)
        data = make_add_data(src_format, n_samples=n_samples, seed=rng_seed)
        verify_fn = lambda ind, thr: verify_add_ulp_bound(ind, src_format, out_format, thr)

    seed_tree = gp.PrimitiveTree.from_string(seed_str, pset)
    seed_area = area_of(seed_tree, src_format, out_format)

    current_data = list(data)
    best = None
    verified = False
    cegis_rounds_used = 0
    for it in range(max_cegis):
        cegis_rounds_used = it + 1
        best = run_gp_ulp(threshold, pset, current_data, src_format, out_format,
                          seed_area, seed_str, ngen, pop_size)
        gp_err, gp_area = eval_true_ulp(best, pset, current_data, src_format, out_format)
        if gp_err > threshold:
            break
        verified, ce = verify_fn(best, threshold)
        if verified:
            break
        if ce and 'ma' in ce:
            current_data.append((ce['ma'], ce['mb'], ce['exact']))
        else:
            break

    area = area_of(best, src_format, out_format)
    elapsed = time.time() - t0

    return {
        'op': op, 'threshold': threshold, 'rng_seed': rng_seed,
        'verified': bool(verified), 'cegis_rounds_used': cegis_rounds_used,
        'seed_area_cells': seed_area, 'best_area_cells': area,
        'time_s': elapsed, 'tree_len': len(best),
    }
