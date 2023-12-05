from operators import *
from problem_instances import *
import random
import time


def genetic_algorithm(problem_instance, population, fn_fitness, fn_select, fn_recombine, fn_mutate, thres=None, ngen=1000, pmut=0.1, show_progress=False, elitism_number=0):
    for i in range(ngen):
        new_population = []

        # dict with calculated fitness values
        population_dict = {individual: fn_fitness(individual) for individual in population}

        # sort population from worst to best fitness (important for some operators)
        population.sort(key=lambda p: population_dict.get(p))

        if show_progress:
            print(f'{i} - Best:{fn_fitness(population[-1], True)[0]:7.2f}; Average: {sum([fn_fitness(individual, True)[0] for individual in population])/len(population):7.2f}')

        # check if one of the individuals achieved a fitness of thres; if so, return it
        if fitness_threshold(population[-1], population_dict.get(population[-1]), thres):
            return population[-1]

        # if elitism_number > 0, keep best elitism_number individuals from previous generation
        if elitism_number > 0:
            new_population = population[-elitism_number:]

        # create len(population) individuals
        for _ in range(len(population) - elitism_number):
            # select the parents
            p1, p2 = fn_select(2, population, population_dict)

            # recombine the parents, thus producing the child
            if fn_recombine == edge_quality_aware_crossover:
                child = fn_recombine(p1, p2, problem_instance)
            else:
                child = fn_recombine(p1, p2)

            # mutate the child
            child = fn_mutate(child, pmut)

            # add the child to the new population
            new_population.append(child)

        # move to the new population
        population = new_population

    population.sort(key=fn_fitness)
    if show_progress:
        print(f'final:{fn_fitness(population[-1], True)[0]:7.2f}')

    # return the individual with the highest fitness
    return population[-1]


def fitness_threshold(individual, fitness, thres):
    if not thres:
        return False
    if fitness >= thres:
        return True
    return False


def init_population(instance_nodes, population_size):
    possible_values = list(instance_nodes.keys())[1:]
    return [tuple(random.sample(possible_values, len(possible_values))) for _ in range(population_size)]


# evaluation class
class EvaluateCVRP:
    def __init__(self, problem_instance, capacity):
        self.problem_instance = problem_instance
        self.capacity = capacity

    # fitness function
    def __call__(self, solution, show=False):
        route_capacity = self.capacity
        total_distance = 0
        vehicle_count = 0
        previous_node = self.problem_instance[0]
        for i in range(len(solution)):
            current_node = self.problem_instance[solution[i]]
            route_capacity -= current_node[0]
            # route_capacity < 0 means end of route
            if route_capacity < 0:
                # vehicle goes from previous_node to 0
                total_distance += calculate_distance(previous_node, self.problem_instance[0])
                vehicle_count += 1
                # new truck starts at 0 and goes to current_node
                route_capacity = self.capacity
                previous_node = self.problem_instance[0]
                route_capacity -= current_node[0]
            total_distance += calculate_distance(previous_node, current_node)
            if i == len(solution) - 1:
                total_distance += calculate_distance(current_node, self.problem_instance[0])
                vehicle_count += 1
            previous_node = current_node

        if show:
            return total_distance, vehicle_count
        else:
            return 1 / (total_distance * vehicle_count)


def calculate_distance(node_a, node_b):
    return (((node_a[1] - node_b[1]) ** 2) + ((node_a[2] - node_b[2]) ** 2)) ** 0.5


def run_all(instance_nodes, instance_capacity, instance_thres, instance_vehicle_count, repetitions):
    # Fixed
    fn_fitness = EvaluateCVRP(instance_nodes, instance_capacity)
    generations = 1000
    population_size = 150
    populations = [init_population(instance_nodes, population_size) for _ in range(repetitions)]

    selection_operators = [roulette, tournament, rank]
    recombination_operators = [order_crossover, partially_mapped_crossover, edge_recombination_crossover,
                               alternating_edges_crossover, edge_quality_aware_crossover]
    mutation_operators = [swap, inversion]
    pmut_values = [0.0, 0.02, 0.1]

    for select in selection_operators:
        for recombine in recombination_operators:
            for mutate in mutation_operators:
                for pmut_value in pmut_values:
                    scores = []
                    best_score = 999999999
                    worst_score = 0
                    times = []
                    for i in range(repetitions):
                        population = populations[i]
                        start = time.time()
                        solution = genetic_algorithm(instance_nodes, population, fn_fitness, select, recombine, mutate, thres=instance_thres, ngen=generations, pmut=pmut_value)
                        end = time.time()
                        times.append(end - start)
                        solution_score, vehicle_count = fn_fitness(solution, True)
                        if vehicle_count > instance_vehicle_count:
                            solution_score = 99999
                        scores.append(solution_score)
                        if solution_score < best_score:
                            best_score = solution_score
                        if solution_score > worst_score:
                            worst_score = solution_score
                    average_score = sum(scores) / len(scores)
                    average_distance_to_optimal = (average_score / (instance_thres - 1)) - 1
                    average_time = sum(times) / len(times)
                    csv_line = f'eil{len(instance_nodes)};{select};{recombine};{mutate};{pmut_value};{best_score:.2f};{worst_score:.2f};{average_score:.2f};{average_distance_to_optimal:.4f};{average_time:.2f}\n'
                    f = open("results.txt", "a")
                    f.write(csv_line)
                    f.close()


if __name__ == '__main__':
    population_size = 150
    instance = (nodes_eil33, capacity_eil33)
    population = init_population(instance[0], population_size)
    fn_fitness = EvaluateCVRP(instance[0], instance[1])
    fn_select = roulette
    fn_recombine = partially_mapped_crossover
    fn_mutate = swap
    thres = None
    ngen = 1000
    pmut = 0.1
    print('Running genetic algorithm!')
    start = time.time()
    solution = genetic_algorithm(instance[0], population, fn_fitness, fn_select, fn_recombine, fn_mutate, thres, ngen, pmut, show_progress=True, elitism_number=2)
    end = time.time()
    print(solution)
    print(fn_fitness(solution, True))
    print(end - start)
