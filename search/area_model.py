import math
from deap import gp


def count_area_nodes(nodes, input_bits):
    def calc_bw(idx):
        node = nodes[idx]
        if not hasattr(node, 'arity') or node.arity == 0:
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
            out_bw = min(child_bws[0] + (2 ** child_bws[1] - 1), 64)
        elif name == 'logic_right_shift':
            out_bw = child_bws[0]
        elif name == 'ite':
            out_bw = max(child_bws[1], child_bws[2])
        else:
            out_bw = max(child_bws)
        return out_bw, next_idx

    cost = 0
    def traverse(idx):
        nonlocal cost
        node = nodes[idx]
        name = getattr(node, 'name', '')
        bw, _ = calc_bw(idx)
        if name in ('add', 'sub'):
            cost += bw
        if not hasattr(node, 'arity') or node.arity == 0:
            return idx + 1
        next_idx = idx + 1
        for _ in range(node.arity):
            next_idx = traverse(next_idx)
        return next_idx

    bw_root, _ = calc_bw(0)
    traverse(0)
    return cost, bw_root


def eval_true(individual, pset, data, input_bits):
    func = gp.compile(individual, pset)
    results = []
    for ma, mb, target in data:
        try:
            result = func(ma, mb)
            results.append(abs(result - target) / target)
        except:
            results.append(1.0)
    avg_error = sum(results) / len(results)
    nodes = list(individual)
    area, _ = count_area_nodes(nodes, input_bits)
    return avg_error, area
