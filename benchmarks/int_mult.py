import operator
import random
from deap import gp

def left_shift(x, n):
    n = int(n) & 0x1F
    return x << n

def logic_right_shift(x, n):
    n = int(n) & 0x1F
    return int(x) >> n

def ite(c, a, b):
    return a if c else b

def make_seed_str(n_bits):
    terms = ["ite(and_(mb, 1), ma, 0)"]
    for i in range(1, n_bits):
        terms.append(
            f"ite(and_(logic_right_shift(mb, {i}), 1), left_shift(ma, {i}), 0)"
        )
    result = terms[0]
    for t in terms[1:]:
        result = f"add({result}, {t})"
    return result

def make_pset(input_bits):
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
    for t in range(2, input_bits + 1):
        pset.addTerminal(t)
    return pset

def make_data(input_bits, n_data, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n_data):
        a = random.uniform(1.0, 2.0)
        b = random.uniform(1.0, 2.0)
        ma = int(a * 2**input_bits)
        mb = int(b * 2**input_bits)
        target = ma * mb
        data.append((ma, mb, target))
    return data
