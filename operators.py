import bisect
import random


# Utils
def index_of_first_equal_or_greater(value_list, ref):
    for i in range(len(value_list)):
        if value_list[i] >= ref:
            return i


def calculate_distance(node_a, node_b):
    return (((node_a[1] - node_b[1]) ** 2) + ((node_a[2] - node_b[2]) ** 2)) ** 0.5

# SELECTION OPERATORS


# Roulette Wheel selection operator
def roulette(r, population, population_dict):
    fitness_values = [population_dict[i] for i in population]
    totals = []
    for value in fitness_values:
        totals.append(value + totals[-1] if totals else value)
    return [population[bisect.bisect(totals, random.uniform(0, totals[-1]))] for _ in range(r)]


def tournament(r, population, population_dict, k=6):
    selection = []
    for i in range(r):
        best = None
        best_fitness = None
        for j in range(k):
            individual = population[random.randrange(len(population))]
            individual_fitness = population_dict[individual]
            if best is None or individual_fitness > best_fitness:
                best = individual
                best_fitness = individual_fitness
        selection.append(best)
    return tuple(selection)


# Linear Ranking
# Receives pre-sorted population to reduce complexity
def rank(r, population, rate_of_worst_selected=0):
    def calculate_probability(index, min_prob, best_rank):
        max_prob = 2 - min_prob
        return (1 / best_rank) * (min_prob + ((max_prob - min_prob) * (index - 1) / (best_rank - 1)))

    assert 0 <= rate_of_worst_selected <= 1
    prob_sum = 0.0
    sum_array = []
    selected = []
    for i in range(1, len(population) + 1):
        sum_array.append(prob_sum + calculate_probability(i, rate_of_worst_selected, len(population)))
        prob_sum = sum_array[i-1]
    for _ in range(r):
        value = random.random()
        selected.append(population[index_of_first_equal_or_greater(sum_array, value)])
    return selected


# CROSSOVER OPERATORS


# Order Crossover (OX)
def order_crossover(p1, p2):
    length = len(p1)
    start_idx, end_idx = sorted(random.sample(range(len(p1)), 2))
    child = list(p1[start_idx:end_idx])
    used_set = set(child)
    i = end_idx
    j = 0
    while len(child) < length:
        value = p2[i % length]
        if value not in used_set:
            child.insert((end_idx + j) % length, value)
            j += 1
        i += 1
    return tuple(child)


# Partially Mapped Crossover (PMX)
def partially_mapped_crossover(p1, p2):
    length = len(p1)
    start_idx, end_idx = sorted(random.sample(range(length), 2))
    c1 = [0] * length
    c1[start_idx:end_idx] = list(p1[start_idx:end_idx])
    used_set = set(c1)
    for i in range(length):
        if c1[i] == 0:
            value = p2[i]
            if value not in used_set:
                c1[i] = value
                used_set.add(value)
            else:
                while value in used_set:
                    value = p2[p1.index(value)]
                c1[i] = value
                used_set.add(value)
    return tuple(c1)


# Edge Recombination Crossover (ERX)
def edge_recombination_crossover(p1, p2):
    def get_neighbor_with_least_neighbors(vertex, edge_dict):
        neighbors_by_size = {}
        for neighbor in edge_dict[vertex]:
            size = len(edge_dict[neighbor])
            if neighbors_by_size.get(size):
                neighbors_by_size[size].append(neighbor)
            else:
                neighbors_by_size[size] = [neighbor]
        # random tiebreaker
        return random.choice(neighbors_by_size[min(neighbors_by_size.keys())])

    length = len(p1)
    child = []
    used_set = set()
    edge_dict = {}
    for i in range(1, length + 1):
        p1_idx = p1.index(i)
        p2_idx = p2.index(i)
        edge_dict[i] = {p1[p1_idx - 1], p1[(p1_idx + 1) % length], p2[p2_idx - 1], p2[(p2_idx + 1) % length]}
    vertex = random.randrange(1, length + 1)
    child.append(vertex)
    used_set.add(vertex)
    while len(child) < length:
        for k in edge_dict:
            edge_dict[k].discard(vertex)
        if edge_dict[vertex] != set():
            vertex = get_neighbor_with_least_neighbors(vertex, edge_dict)
        else:
            vertex = random.choice([x for x in p1 if x not in used_set])
        child.append(vertex)
        used_set.add(vertex)
    return tuple(child)


# Edge Quality Aware Crossover (ER-Q)
def edge_quality_aware_crossover(p1, p2, problem_instance, beta=1):
    # Chitty et al. use this model with another representation of the CVRP.
    # Because of this, they set a probability of 1 for vertices which are a depot,
    # and a null probability of delivering to vertices which would exceed vehicle capacity.
    # In our implementation, doing that would lead to problems, so we are only considering
    # edge quality as 1 / distance to calculate the probabilities.
    def edge_probabilistic_model(vertex, neighbors, beta, problem_instance):
        probabilities = []
        for i in range(len(neighbors)):
            d = calculate_distance(problem_instance[vertex], problem_instance[neighbors[i]])
            probabilities.append((1 / d) ** beta)
        probabilities_sum = sum(probabilities)
        for i in range(len(neighbors)):
            probabilities[i] = (probabilities[i] / probabilities_sum) + (probabilities[i - 1] if i > 0 else 0)
        probabilities = list(zip(probabilities, neighbors))
        return probabilities

    length = len(p1)
    child = []
    used_set = set()
    edge_dict = dict()
    for i in range(1, length + 1):
        p1_idx = p1.index(i)
        p2_idx = p2.index(i)
        edge_dict[i] = {p1[p1_idx - 1], p1[(p1_idx + 1) % length], p2[p2_idx - 1], p2[(p2_idx + 1) % length]}
    vertex = random.randrange(1, length + 1)
    child.append(vertex)
    used_set.add(vertex)
    while len(child) < length:
        for k in edge_dict:
            edge_dict[k].discard(vertex)
        if edge_dict[vertex] != set():
            model = edge_probabilistic_model(vertex, list(edge_dict[vertex]), beta, problem_instance)
        else:
            matrix = [neighbors for neighbors in edge_dict.values()]
            all_available = list({item for row in matrix for item in row})
            model = edge_probabilistic_model(vertex, all_available, beta, problem_instance)
        probabilities = [x[0] for x in model]
        selection = random.random()
        vertex = model[index_of_first_equal_or_greater(probabilities, selection)][1]
        child.append(vertex)
        used_set.add(vertex)
    return tuple(child)


# Alternating Edges Crossover (AEX)
def alternating_edges_crossover(p1, p2):
    length = len(p1)
    child = [-1] * length
    child[:2] = p1[:2]
    used_set = set(child)
    current_parent = p2
    for i in range(length):
        if child[i] == -1:
            value = current_parent[(current_parent.index(child[i-1]) + 1) % length]
            if value in used_set:
                usable_values = [x for x in current_parent if x not in used_set]
                value = random.choice(usable_values)
            child[i] = value
            used_set.add(value)
            current_parent = p1 if current_parent is p2 else p2
    return tuple(child)

# MUTATION OPERATORS


# Swap
def swap(solution, pmut):
    if random.random() >= pmut:
        return solution

    # swap two random positions
    length = len(solution)
    mutated_solution = list(solution)
    i = random.randrange(0, length)
    j = random.randrange(0, length)
    temp = mutated_solution[i]
    mutated_solution[i] = mutated_solution[j]
    mutated_solution[j] = temp

    return tuple(mutated_solution)


# Inversion
def inversion(solution, pmut):
    if random.random() >= pmut:
        return solution

    length = len(solution)
    inverted_slice = []
    i, j = random.sample(range(length), 2)
    k = i
    while k % length != (j + 1) % length or k == i:
        inverted_slice.append(solution[k % length])
        k += 1
    inverted_slice.reverse()
    mutated_solution = list(solution)
    k = i
    while k % length != (j + 1) % length or k == i:
        mutated_solution[k % length] = inverted_slice[(k - i) % length]
        k += 1
    return tuple(mutated_solution)

