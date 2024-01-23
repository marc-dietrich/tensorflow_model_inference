import csv
import random
import subprocess
import time
from collections import Counter

import numpy as np
from deap import base, creator, tools, algorithms

# Define the problem
creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness)

memoization_cache = {}


def generate_random_stage_assignment_list(target_sum, length):
    # Generate a list of random integers
    random_numbers = [random.uniform(0.1, 1) for _ in range(length)]

    # Calculate the scaling factor to ensure the sum equals the target
    scaling_factor = target_sum / sum(random_numbers)

    # Scale the numbers
    scaled_numbers = [int(round(num * scaling_factor)) for num in random_numbers]

    amount_deleted_numbers = 0
    while not sum(scaled_numbers) == target_sum:
        # Adjust the last element to make sure the sum is exactly the target
        scaled_numbers[-1] += target_sum - sum(scaled_numbers)
        if scaled_numbers[-1] < 0:
            scaled_numbers.pop(-1)
            amount_deleted_numbers += 1
    scaled_numbers += [0 for _ in range(amount_deleted_numbers)]

    return scaled_numbers


def create_individual(num_cores, num_stages, threshold=0.8):
    individual = creator.Individual()
    core_order = np.random.permutation(num_cores)
    random_assignment_list = generate_random_stage_assignment_list(num_stages, num_cores)
    for core_idx in core_order:
        num_assign_stages = random_assignment_list[core_idx]
        for _ in range(num_assign_stages):
            individual.append(core_idx)
    return individual


def create_population(amount, num_cores, num_stages):
    population = []
    for _ in range(amount):
        population.append(
            create_individual(num_cores, num_stages)
        )
    return population


def ind_to_assignment(ind):
    assignment = {}
    for core_id in ind:
        starting, ending = get_active_core_range(ind, core_id)
        assignment[core_id] = list(range(starting, ending + 1))

    print(ind)
    print(assignment)


def run_main_script(ind):
    # Define the path to your shell script
    script_path = './measure.sh'


    # Convert the list of integers to a list of strings
    params = [str(x) for x in ind]

    # Construct the command to run the shell script
    command = [script_path] + params

    # Run the shell script and capture the output
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        # Access the output
        script_output = result.stdout
        print("Output from the script:")
        print(script_output)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


    return random.random(), random.random()


# Define the evaluation function (replace with your problem's objectives)
def evaluate(individual):
    # todo: eval by called messure script and pass indivi. as list
    tup = tuple(individual)
    if tup in memoization_cache:
        return memoization_cache[tup]

    exec_time, energy = run_main_script(individual)

    memoization_cache[tup] = (exec_time, energy)

    return exec_time, energy


def swap_mutation(individual):
    i, j = random.sample(range(len(individual)), 2)
    individual[i], individual[j] = individual[j], individual[i]
    return individual


def get_active_core_range(ind, core_id):
    starting = -1
    ending = -1
    for stage_idx, core_idx in enumerate(ind):
        if core_idx == core_id and starting < 0:
            starting = stage_idx
        if core_idx != core_id and starting >= 0 and ending < 0:
            ending = stage_idx - 1
    if starting == -1:
        starting = num_stages - 1
    if ending == -1:
        ending = num_stages - 1
    return starting, ending


def shift_core_delete(ind):
    rdm_core_idx = random.randint(0, num_cores - 1)
    starting, ending = get_active_core_range(ind, rdm_core_idx)
    delete_idx = starting
    if starting % (num_stages - 1) == 0:
        delete_idx = ending

    if delete_idx == 0:
        ind[delete_idx] = ind[1]
    elif delete_idx == num_stages - 1:
        ind[delete_idx] = ind[-2]
    else:
        ind[delete_idx] = ind[delete_idx - 1]

    return ind


def shift_core_add(ind):
    c = Counter(ind)
    if len(c) == num_cores:
        return ind  # no free cores

    new_core = random.choice(
        list({core_id for core_id in range(num_cores)} - c.keys())
    )
    rdm_core_idx = random.randint(0, num_cores - 1)
    starting, ending = get_active_core_range(ind, rdm_core_idx)
    delete_idx = starting
    if starting % (num_stages - 1) == 0:
        delete_idx = ending
    ind[delete_idx] = new_core
    return ind


def mutate(ind):
    # todo: other mutations maybe:
    # add core usage --> also balance usage

    mutations = [
        shift_core_add,
        shift_core_delete
    ]

    ind = random.choice(mutations)(ind)

    return ind,


def sum_(ind):
    return sum(
        [sum(stages_per_core) for stages_per_core in ind]
    )


def none_func(ind1, ind2):
    return ind1, ind2

def contains_consecutive_values(ind:list, core_indicies):
    # fill
    idx_sublists = {c:[] for c in core_indicies}
    for idx in core_indicies:
        for stage_idx, core_idx in enumerate(ind):
            if idx == core_idx:
                idx_sublists[core_idx] += [stage_idx]

    for core_idx, mapped_stages in idx_sublists.items():
        if not mapped_stages:
            continue
        current = mapped_stages[0]
        for mapped_stage in mapped_stages[1:]:
            if not current + 1 == mapped_stage:
                return False, (core_idx, mapped_stages)
            current += 1
    return True, None

random.seed(42)

toolbox = base.Toolbox()
toolbox.register("evaluate", evaluate)
toolbox.register("mate", none_func)  # Blend crossover
toolbox.register("mutate", mutate)  # Gaussian mutation
toolbox.register("select", tools.selNSGA2)  # NSGA-II selection

# Create an initial population
num_stages = 23
amount = 100
num_cores = 32
population = create_population(amount=amount, num_stages=num_stages, num_cores=num_cores)
for ind in population:
    b, v = contains_consecutive_values(ind, list(range(num_cores)))
    if not b:
        print(v)

# Set up the statistics object
# stats = tools.Statistics(lambda ind: ind.fitness.values)
# stats.register("avg", tools.mean)
# stats.register("min", tools.min)

# Run the genetic algorithm
hof = tools.ParetoFront()
st = time.time()
counter = 0
while time.time() - st < 10:  # 60*60*24: # run 1 day
    counter += 1
    population, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=20, cxpb=0.01, mutpb=.99,
                                                    ngen=5,
                                                    halloffame=hof, verbose=False)
    for ind in population:
        b, v = contains_consecutive_values(ind, list(range(num_cores)))
        if not b:
            print(v)

print(counter)
print("time: ", time.time() - st)
#print(hof.keys)
#print(hof.items)

# Define the file name
csv_file_name = './outputs/pareto_solutions.csv'

ind_to_assignment(hof.items[0])

# Open the CSV file in write mode with a semicolon as the separator
with open(csv_file_name, 'w', newline='') as csvfile:
    # Create a CSV writer object with semicolon as the delimiter
    csv_writer = csv.writer(csvfile)

    # Write header
    header = ['exec_time', 'energy_consumption', 'Solutions']  # Adjust as needed
    csv_writer.writerow(header)

    # Write solutions and fitness values
    for ind in hof:
        fitness_values = ind.fitness.values
        solution_values = ','.join(map(str, ind))  # Convert list to string
        row_data = list(fitness_values) + [solution_values]
        csv_writer.writerow(row_data)
