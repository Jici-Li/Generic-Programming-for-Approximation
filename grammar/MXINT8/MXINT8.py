import operator
import random
import math
import struct
from deap import base, creator, tools, algorithms, gp

def float_to_uint4(f):
    return struct.unpack("I", struct.pack("f", float(f)))[0]

def uint4_to_float(u):
    try:
        return struct.unpack("f", struct.pack("I", int(u) & 0xF))[0]
    except:
        return float("nan")

def left_shift(x, n):
    n = int(n) & 0x1F
    return x << n



def logic_right_shift(x, n):
    n = int(n) & 0x1F
    return int(x) >> n

def ite(c, a, b):
    return a if c else b

def get_sign(uint8):
    return (int(uint8) >> 7) & 1

def get_exp(uint8):
    return (int(uint8) >> 4) & 0xF

def get_mant(uint8):
    return int(uint8) & 0xF

def concat_fp8(s, e, m):
    s = int(s) & 0x1
    e = int(e) & 0xF
    m = int(m) & 0xF
    return (s << 7) | (e << 4) | m

def xor1(a, b):
    return (int(a) ^ int(b)) & 1

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
pset.addTerminal(30)
pset.addTerminal(31)
pset.addTerminal(32)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

random.seed(42)
data = []
for _ in range(100):
    a = random.uniform(0.0, 8.0)
    b = random.uniform(0.0, 8.0)
    m1 = float_to_uint4(a)
    m2 = float_to_uint4(b)
    target = abs(m1 * m2)
    data.append((m1, m2, target))

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
        depth = individual.height
        return float(avg_error * 100 + depth * 0.01),
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
depth = best.height

print(str(best))
print(f"error:     {avg_error:.1%}")
print(f"depth:     {depth}")
print(f"fitness:   {best.fitness.values[0]}")