import operator
import random
from deap import base, creator, tools, algorithms, gp

def left_shift(x, n):
    return x << n

def logic_right_shift(x, n):
    return int(x) >> n

pset = gp.PrimitiveSet("MAIN", 2) 
pset.renameArguments(ARG0="ma", ARG1="mb")

pset.addPrimitive(operator.add, 2)          
pset.addPrimitive(operator.sub, 2)     
pset.addPrimitive(operator.and_, 2)   
pset.addPrimitive(operator.or_, 2)   
pset.addPrimitive(operator.xor, 2)           
pset.addPrimitive(left_shift, 2)            
pset.addPrimitive(logic_right_shift, 2)     
pset.addTerminal(0)                         

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create("Individual", gp.PrimitiveTree,
               fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("expr", gp.genHalfAndHalf,pset=pset, min_=1, max_=4)

toolbox.register("individual", tools.initIterate,creator.Individual, toolbox.expr)

toolbox.register("population", tools.initRepeat,
                 list, toolbox.individual)

data = []
for _ in range(100):
    a = random.uniform(1.0, 2.0)  
    b = random.uniform(1.0, 2.0)
    ma = int(a * 2**23)             
    mb = int(b * 2**23)
    target = int(a * b * 2**23)    
    data.append((ma, mb, target))

def evaluate(individual):
    func = gp.compile(individual, pset)

    try:
        results = []
        for ma, mb, target in data:
            try:
                result = func(ma, mb)
                rel_error = abs(result - target) / target
                results.append(rel_error)
            except (OverflowError, ValueError, ZeroDivisionError):
                results.append(1.0)  

        avg_error = sum(results) / len(results)

        size = len(individual)

        return float(avg_error + size * 0.001),

    except Exception:
        return 99999,

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform,expr=toolbox.expr, pset=pset)

from operator import attrgetter
toolbox.decorate("mate", gp.staticLimit(key=attrgetter("height"), max_value=8))
toolbox.decorate("mutate", gp.staticLimit(key=attrgetter("height"), max_value=8))

random.seed(42)
pop = toolbox.population(n=500)

pop, logbook = algorithms.eaSimple(
    pop, toolbox,
    cxpb=0.5,    
    mutpb=0.2,   
    ngen=100,    
    verbose=True 
)

best = tools.selBest(pop, 1)[0]
print("best", str(best))
print("fitness", best.fitness.values[0])
