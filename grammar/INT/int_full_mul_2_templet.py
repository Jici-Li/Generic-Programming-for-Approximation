import operator
import random
import math
from deap import base, creator, tools, algorithms, gp

def left_shift(x, n):
    n = int(n) & 0x1F
    return x << n

def logic_right_shift(x, n):
    n = int(n) & 0x1F
    return int(x) >> n

def ite(c, a, b):
    return a if c else b

INPUT_BITS = 2  

pset = gp.PrimitiveSet("MAIN", 2)
pset.renameArguments(ARG0="ma", ARG1="mb")
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.and_, 2)
pset.addPrimitive(operator.or_, 2)
pset.addPrimitive(left_shift, 2)
pset.addPrimitive(logic_right_shift, 2)
pset.addPrimitive(ite, 3)
pset.addTerminal(0)
pset.addTerminal(1)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

random.seed(42)
data = []
for _ in range(100):
    a = random.uniform(1.0, 2.0)
    b = random.uniform(1.0, 2.0)
    ma = int(a * 2**INPUT_BITS)
    mb = int(b * 2**INPUT_BITS)
    target = ma * mb
    data.append((ma, mb, target))

def get_bitwidth(individual, input_bits):
    nodes = list(individual)
    bitwidths = [0] * len(nodes)

    def calc_bw(idx):
        node = nodes[idx]
        if not hasattr(node,
                        'arity') or node.arity == 0:
            name = getattr(node, 'name', '')
            if name in ('ma', 'mb'):
                return input_bits, idx + 1
            try:
                v = int(node.value) if hasattr(node, 'value') else int(name)
                return (1 if v == 0 else int(math.log2(abs(v))) + 1), idx + 1
            except:
                return 1, idx + 1

        child_bws = []
        next_idx = idx + 1
        for _ in range(node.arity):
            bw, next_idx = calc_bw(next_idx)
            child_bws.append(bw)

        name = node.name
        if name in ('add', 'sub'):
            out_bw = max(child_bws[0], child_bws[1]) + 1
        elif name in ('and_', 'or_'):
            out_bw = max(child_bws[0], child_bws[1])
        elif name == 'left_shift':
            out_bw = min(child_bws[0] + (2**child_bws[1] - 1), 64)
        elif name == 'logic_right_shift':
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
    nodes = list(individual)
    bws, root_bw = get_bitwidth(individual, input_bits)
    cost = sum(bws[i] for i, node in enumerate(nodes)
               if hasattr(node, 'name') and node.name in ('add', 'sub'))
    return cost, root_bw

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
        area, _ = count_area(individual, INPUT_BITS)
        depth = individual.height
        return float(avg_error * 100 + area * 0.1 + depth * 0.01),
    except:
        return 99999,

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

from operator import attrgetter
toolbox.decorate("mate", gp.staticLimit(key=attrgetter("height"), max_value=8))
toolbox.decorate("mutate", gp.staticLimit(key=attrgetter("height"), max_value=8))

def make_seed_str(n_bits):
    terms = ["ite(and_(mb, 1), ma, 0)"]
    for i in range(1, n_bits):
        terms.append(f"ite(and_(logic_right_shift(mb, {i}), 1), left_shift(ma, {i}), 0)")
    result = terms[0]
    for t in terms[1:]:
        result = f"add({result}, {t})"
    return result

seed_str = make_seed_str(INPUT_BITS + 1)

seed_tree = gp.PrimitiveTree.from_string(seed_str, pset)
seed_func = gp.compile(seed_tree, pset)

random.seed(42)
pop = toolbox.population(n=499)
seed_ind = creator.Individual(seed_tree)
seed_ind.fitness.values = evaluate(seed_ind)
pop.append(seed_ind)

pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, verbose=True)

best = tools.selBest(pop, 1)[0]
func = gp.compile(best, pset)

avg_error = sum(abs(func(ma,mb) - target)/target for ma,mb,target in data) / len(data)
area, output_bw = count_area(best, INPUT_BITS)
depth = best.height

print(str(best))
print(f"error:     {avg_error:.1%}")
print(f"area:      {area} (bitwidth-weighted)")
print(f"output_bw: {output_bw} bits")
print(f"depth:     {depth}")
print(f"fitness:   {best.fitness.values[0]}")