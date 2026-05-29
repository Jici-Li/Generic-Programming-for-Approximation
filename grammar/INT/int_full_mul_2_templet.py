import operator
import random
from deap import base, creator, tools, algorithms, gp

def left_shift(x, n):
    n = int(n) & 0x1F
    return x << n

def logic_right_shift(x, n):
    n = int(n) & 0x1F
    return int(x) >> n

def ite(c, a, b):
    return a if c else b

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
creator.create("Individual", gp.PrimitiveTree,
               fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf,
                 pset=pset, min_=1, max_=4)
toolbox.register("individual", tools.initIterate,
                 creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat,
                 list, toolbox.individual)

random.seed(42)
data = []
for _ in range(100):
    a = random.uniform(1.0, 2.0)
    b = random.uniform(1.0, 2.0)
    ma = int(a * 2**1)   
    mb = int(b * 2**1)
    target = ma * mb
    data.append((ma, mb, target))

def count_area(individual):
    cost = 0
    for node in individual:
        if hasattr(node, 'name'):
            if node.name in ('add', 'sub'):
                cost += 1
    return cost

def critical_path_depth(individual):  
    def get_depth(nodes, idx):
        node = nodes[idx]
        if not hasattr(node, 'arity') or node.arity == 0:
            return 0, idx + 1
        max_child_depth = 0
        next_idx = idx + 1
        for _ in range(node.arity):
            child_depth, next_idx = get_depth(nodes, next_idx)
            max_child_depth = max(max_child_depth, child_depth)
        if node.name in ('add', 'sub'):
            return max_child_depth + 1, next_idx
        else:
            return max_child_depth, next_idx
    depth, _ = get_depth(list(individual), 0)
    return depth

def evaluate(individual):
    func = gp.compile(individual, pset)
    try:
        results = []
        for ma, mb, target in data:
            try:
                result = func(ma, mb)
                rel_error = abs(result - target) / target
                results.append(rel_error)
            except:
                results.append(1.0)
        avg_error = sum(results) / len(results)
        area = count_area(individual)
        path_depth = critical_path_depth(individual)
        return float(avg_error + area * 0.001+ path_depth * 0.001),
    except:
        return 99999,

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform,
                 expr=toolbox.expr, pset=pset)

from operator import attrgetter
toolbox.decorate("mate", gp.staticLimit(
    key=attrgetter("height"), max_value=8))
toolbox.decorate("mutate", gp.staticLimit(
    key=attrgetter("height"), max_value=8))

def correct_mul(ma, mb):
    result = 0
    if (mb >> 0) & 1: result += ma         
    if (mb >> 1) & 1: result += ma << 1    
  
    return result

seed_str = (
    "add("
        "add("
            "ite(and_(mb, 1), ma, 0),"
            "ite(and_(logic_right_shift(mb, 1), 1), left_shift(ma, 1), 0)"
        "),"
)

random.seed(42)
pop = toolbox.population(n=499)

seed_tree = gp.PrimitiveTree.from_string(seed_str, pset)
seed_ind = creator.Individual(seed_tree)
seed_ind.fitness.values = evaluate(seed_ind)
pop.append(seed_ind)

pop, logbook = algorithms.eaSimple(
    pop, toolbox,
    cxpb=0.5, mutpb=0.2, ngen=200,
    verbose=True
)

best = tools.selBest(pop, 1)[0]
func = gp.compile(best, pset)

print( str(best))

avg_error = sum(abs(func(ma, mb) - target) / target
                for ma, mb, target in data) / len(data)
area = count_area(best)
print(f"error{avg_error:.1%}")
print(f"area {area}")
print(f"fitness{best.fitness.values[0]}")
print(f"path_depth {critical_path_depth(best)}")
