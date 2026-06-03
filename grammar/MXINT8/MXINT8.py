import operator
import random
import math
from deap import base, creator, tools, algorithms, gp
from operator import attrgetter


INPUT_BITS = 4  

def abs_(x):
    x = int(x)
    return -x if x < 0 else x

def is_negative(x):
    return 1 if int(x) < 0 else 0

def is_large(x):
    return 1 if int(x) >= 32 else 0

def logic_right_shift(x, n):
    n = int(n) & 0x1F
    x = int(x)
    if x < 0:
        x = x & 0xFF
    return x >> n

def arith_right_shift(x, n):
    n = int(n) & 0x1F
    return int(x) >> n

def ite(c, a, b):
    return a if int(c) != 0 else b

pset = gp.PrimitiveSet("MAIN", 2)
pset.renameArguments(ARG0="m1", ARG1="m2")

pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(abs_, 1)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(is_negative, 1)
pset.addPrimitive(is_large, 1)
pset.addPrimitive(arith_right_shift, 2)
pset.addPrimitive(logic_right_shift, 2)
pset.addPrimitive(operator.and_, 2)
pset.addPrimitive(ite, 3)

pset.addTerminal(0)
pset.addTerminal(1)
pset.addTerminal(2)
pset.addTerminal(3)
pset.addTerminal(4)
pset.addTerminal(15)
pset.addTerminal(32)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr",       gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat,  list, toolbox.individual)

def mxint8_mant_target(m1, m2):
    abs_p = abs(m1 * m2)
    shift = 3 if abs_p >= 32 else 2
    return (abs_p >> shift) & 0xF

data = [
    (m1, m2, mxint8_mant_target(m1, m2))
    for m1 in range(-8, 8)
    for m2 in range(-8, 8)
]

def get_bitwidth(individual, input_bits):
    nodes = list(individual)
    bitwidths = [0] * len(nodes)

    def calc_bw(idx):
        node = nodes[idx]
        if not hasattr(node, 'arity') or node.arity == 0:
            name = getattr(node, 'name', '')
            if name in ('m1', 'm2'):              
                return input_bits, idx + 1
            try:
                v = int(node.value) if hasattr(node, 'value') else int(name)
                return (1 if v == 0 else int(math.log2(abs(v))) + 1), idx + 1
            except Exception:
                return 1, idx + 1

        child_bws = []
        next_idx = idx + 1
        for _ in range(node.arity):
            bw, next_idx = calc_bw(next_idx)
            child_bws.append(bw)

        name = node.name
        if name == 'mul':
            out_bw = min(child_bws[0] + child_bws[1], 64)  
        elif name in ('neg', 'abs_'):
            out_bw = child_bws[0]
        elif name in ('is_negative', 'is_large'):
            out_bw = 1
        elif name == 'and_':
            out_bw = max(child_bws[0], child_bws[1])
        elif name in ('arith_right_shift', 'logic_right_shift'):
            out_bw = child_bws[0]
        elif name == 'ite':
            out_bw = max(child_bws[1], child_bws[2])
        else:
            out_bw = max(child_bws)

        bitwidths[idx] = out_bw
        return out_bw, next_idx

    root_bw, _ = calc_bw(0)
    return bitwidths, root_bw


def count_area(individual, input_bits):
    """Gate-count proxy: multiplier area ≈ output_bw (= bw_a + bw_b)."""
    nodes = list(individual)
    bws, root_bw = get_bitwidth(individual, input_bits)
    cost = 0
    for i, node in enumerate(nodes):
        name = getattr(node, 'name', '')
        if name == 'mul':                          
            cost += bws[i]
    return cost, root_bw

def evaluate(individual):
    func = gp.compile(individual, pset)
    errors = []
    for m1, m2, target in data:
        try:
            result = int(func(m1, m2)) & 0xF      
            errors.append(abs(result - target))
        except Exception:
            errors.append(15)
    avg_error = sum(errors) / len(errors)
    area, _ = count_area(individual, INPUT_BITS)   
    depth = individual.height
    return float(avg_error + area * 0.1 + depth * 0.001),


toolbox.register("evaluate", evaluate)
toolbox.register("select",   tools.selTournament, tournsize=3)
toolbox.register("mate",     gp.cxOnePoint)
toolbox.register("mutate",   gp.mutUniform, expr=toolbox.expr, pset=pset)
toolbox.decorate("mate",   gp.staticLimit(key=attrgetter("height"), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=attrgetter("height"), max_value=10))

seed_str = (
    "and_(ite(is_large(abs_(mul(m1, m2))),"
    " arith_right_shift(abs_(mul(m1, m2)), 3),"
    " arith_right_shift(abs_(mul(m1, m2)), 2)), 15)"
)

seed_tree = gp.PrimitiveTree.from_string(seed_str, pset)
seed_func = gp.compile(seed_tree, pset)

assert all(
    (int(seed_func(m1, m2)) & 0xF) == target
    for m1, m2, target in data
), "Seed tree does not reproduce ground truth — check seed_str."

random.seed(42)
pop = toolbox.population(n=499)
seed_ind = creator.Individual(seed_tree)
seed_ind.fitness.values = evaluate(seed_ind)
pop.append(seed_ind)

pop, logbook = algorithms.eaSimple(
    pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, verbose=True
)

best      = tools.selBest(pop, 1)[0]
best_func = gp.compile(best, pset)

# fixed: compute report values here, not inside evaluate
correct    = sum(1 for m1, m2, target in data if (int(best_func(m1, m2)) & 0xF) == target)
avg_error  = sum(abs((int(best_func(m1, m2)) & 0xF) - target) for m1, m2, target in data) / len(data)
area, output_bw = count_area(best, INPUT_BITS)
depth      = best.height

print(str(best))
print(f"accuracy:  {correct}/{len(data)}  ({correct / len(data):.1%})")
print(f"avg_error: {avg_error:.3f}")
print(f"area:      {area} (bitwidth-weighted)")
print(f"output_bw: {output_bw} bits")
print(f"depth:     {depth}")
print(f"fitness:   {best.fitness.values[0]}")
