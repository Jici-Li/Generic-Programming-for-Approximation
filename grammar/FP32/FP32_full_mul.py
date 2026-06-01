import operator
import random
import math
import struct
from deap import base, creator, tools, algorithms, gp

def left_shift(x, n):
    n = int(n) & 0x1F
    return (int(x) << n) & 0xFFFFFFFF

def logic_right_shift(x, n):
    n = int(n) & 0x1F
    return (int(x) & 0xFFFFFFFF) >> n

def ite(c, a, b):
    return a if c else b

def get_sign(uint32):
    return (int(uint32) >> 31) & 1

def get_exp(uint32):
    return (int(uint32) >> 23) & 0xFF

def get_mant(uint32):
    return int(uint32) & 0x7FFFFF

def concat_fp32(s, e, m):
    s = int(s) & 0x1
    e = int(e) & 0xFF
    m = int(m) & 0x7FFFFF
    return (s << 31) | (e << 23) | m

def add_hidden_bit(exp, mant):
    exp = int(exp) & 0xFF
    mant = int(mant) & 0x7FFFFF
    if exp == 0:
        return mant
    else:
        return (1 << 23) | mant  

def xor1(a, b):
    return (int(a) ^ int(b)) & 1

def float_to_uint32(f):
    return struct.unpack("I", struct.pack("f", float(f)))[0]

def uint32_to_float(u):
    try:
        return struct.unpack("f", struct.pack("I", int(u) & 0xFFFFFFFF))[0]
    except:
        return float("nan")

INPUT_BITS = 32

pset = gp.PrimitiveSet("MAIN", 2)
pset.renameArguments(ARG0="ma", ARG1="mb")

pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.and_, 2)
pset.addPrimitive(operator.or_, 2)
pset.addPrimitive(left_shift, 2)
pset.addPrimitive(logic_right_shift, 2)
pset.addPrimitive(ite, 3)

pset.addPrimitive(get_sign, 1)
pset.addPrimitive(get_exp, 1)
pset.addPrimitive(get_mant, 1)
pset.addPrimitive(concat_fp32, 3)
pset.addPrimitive(add_hidden_bit, 2)
pset.addPrimitive(xor1, 2)

# terminals
pset.addTerminal(0)
pset.addTerminal(1)
pset.addTerminal(127)  
pset.addTerminal(24)    
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

random.seed(42)
data = []
for _ in range(200):
    a = random.uniform(1.0, 2.0)
    b = random.uniform(1.0, 2.0)
    ma = float_to_uint32(a)
    mb = float_to_uint32(b)
    target_float = a * b
    data.append((ma, mb, target_float))

def get_bitwidth(individual, input_bits):
    nodes = list(individual)
    bitwidths = [0] * len(nodes)

    def calc_bw(idx):
        node = nodes[idx]
        if not hasattr(node, 'arity') or node.arity == 0:
            name = getattr(node, 'name', '')
            if name in ('ma', 'mb'):
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
        if name == 'get_sign':
            out_bw = 1
        elif name == 'get_exp':
            out_bw = 8
        elif name == 'get_mant':
            out_bw = 23
        elif name == 'concat_fp32':
            out_bw = 32
        elif name == 'add_hidden_bit':
            out_bw = 24
        elif name in ('add', 'sub'):
            out_bw = max(child_bws[0], child_bws[1]) + 1
        elif name in ('and_', 'or_'):
            out_bw = max(child_bws[0], child_bws[1])
        elif name == 'left_shift':
            out_bw = min(child_bws[0] + (2 ** max(0, child_bws[1]) - 1), 64)
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
    cost = 0
    for i, node in enumerate(nodes):
        name = getattr(node, 'name', '')
        if name in ('add', 'sub'):
            cost += bws[i]
    return cost, root_bw

def evaluate(individual):
    func = gp.compile(individual, pset)
    results = []
    for ma, mb, target in data:
        try:
            result_uint = func(ma, mb)
            if not isinstance(result_uint, int):
                results.append(1.0)
                continue
            result_uint = int(result_uint) & 0xFFFFFFFF
            result_float = uint32_to_float(result_uint)
            if math.isnan(result_float) or math.isinf(result_float):
                results.append(1.0)
                continue
            results.append(abs(result_float - target) / abs(target))
        except Exception:
            results.append(1.0)
    avg_error = sum(results) / len(results)
    area, _ = count_area(individual, INPUT_BITS)
    depth = individual.height
    return (avg_error * 100.0 + area * 0.1 + depth * 0.01,)

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

from operator import attrgetter
toolbox.decorate("mate", gp.staticLimit(key=attrgetter("height"), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=attrgetter("height"), max_value=10))

seed_str = "concat_fp32(xor1(get_sign(ma), get_sign(mb)), add(get_exp(ma), get_exp(mb)), 0)"
seed_tree = gp.PrimitiveTree.from_string(seed_str, pset)
seed_ind = creator.Individual(seed_tree)

random.seed(42)
pop = toolbox.population(n=200)
seed_ind.fitness.values = evaluate(seed_ind)
pop.append(seed_ind)

pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, verbose=True)

best = tools.selBest(pop, 1)[0]
func = gp.compile(best, pset)

avg_error = sum(
    (abs((uint32_to_float(func(ma, mb) & 0xFFFFFFFF)) - target) / abs(target))
    if (isinstance(func(ma, mb), int) and not math.isnan(uint32_to_float(func(ma, mb) & 0xFFFFFFFF)))
    else 1.0
    for ma, mb, target in data
) / len(data)

area, output_bw = count_area(best, INPUT_BITS)
depth = best.height

print("Best individual:", str(best))
print(f"error:     {avg_error:.4%}")
print(f"area:      {area} (heuristic)")
print(f"output_bw: {output_bw} bits")
print(f"depth:     {depth}")
print(f"fitness:   {best.fitness.values[0]}")