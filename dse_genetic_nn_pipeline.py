import csv
import random
import subprocess
import time
from datetime import datetime

import numpy as np
from deap import base, creator, tools, algorithms
from ordered_set import OrderedSet

# Define the problem
creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0, -1.0, -1.0))
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


def get_core_types(ind):
    ind_ = sorted(ind)
    e_cores = [core for core in ind_ if 16 <= core <= 31]
    p_cores = [core for core in ind_ if 0 <= core <= 15]
    ht_count = 0
    #print(ind_)
    #print(p_cores)
    for core, next_core in zip(p_cores, p_cores[1:]):
        #print(core, next_core)
        if core == next_core - 1 and core % 2 == 0:
            ht_count += 1
    return len(e_cores), len(p_cores) - 2 * ht_count, ht_count


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

        # print(script_output)

        # Initialize a list to store the values
        values = []

        # Extract values using split and convert to float
        for line in script_output.split("\n"):
            if "cpu0_package_joules" in line or "cpu0_core_joules" in line:
                value = line.split("=")[-1].strip()
                values.append(float(value))
            if not ":" in line:
                continue
            value = line.split(":")[-1].strip()
            values.append(float(value))

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

    e_cores, st_cores, ht_cores = get_core_types(ind)
    values += [e_cores, st_cores, ht_cores]
    print(values)
    return tuple(values)


# Define the evaluation function (replace with your problem's objectives)
def evaluate(individual):
    # todo: eval by called messure script and pass indivi. as list
    tup = tuple(individual)
    if tup in memoization_cache:
        return memoization_cache[tup]

    exec_time, accuracy, latency, package_energy, cpu_energy, e_cores, st_cores, ht_cores = run_main_script(individual)
    #exec_time, accuracy, latency, package_energy, cpu_energy, e_cores, st_cores, ht_cores = (
    #    random.random(), random.random(), random.random(), random.random(), random.random(), random.random(),
    #    random.random(), random.random())

    memoization_cache[tup] = (exec_time, latency, package_energy, cpu_energy, e_cores, st_cores, ht_cores)

    #e_cores, st_cores, ht_cores = get_core_types(ind)
    #print(sorted(ind))
    #print(e_cores, st_cores, ht_cores)

    return exec_time, latency, package_energy, e_cores, st_cores, ht_cores


def consecutive_order_crossover(ind1, ind2):
    pass


def get_fixed_offspring_part(cutoff_point, ind):
    child = ind[:cutoff_point]
    child_offset = 0
    continuation_element = child[-1]
    for core_id in ind[cutoff_point:]:
        if core_id == continuation_element:
            child_offset += 1
    return child, child_offset


def extend_offspring_by_group_structure(offspring, x_parent, cutoff_point, x_offset):
    # collect all cores, used in x_parent: starting to collect from cutoff-point and then iter over x_parent
    used_cores_in_x_parent = OrderedSet(x_parent[cutoff_point + x_offset:] + x_parent[:cutoff_point + x_offset])
    # print("collection of all cores in 2nd parts: ", used_cores_in_second_parts)

    # focus now one 1 child:
    # delete cores, that are already used on corresponding fixed-child-part
    used_cores_in_x_parent -= OrderedSet(offspring)
    # print("filtered collection of cores in x_parent: ", used_cores_in_x_parent)

    # receive pattern of opposite parent
    collected_appearances = {}
    for core_id in x_parent[cutoff_point + x_offset:]:
        if core_id not in collected_appearances:
            collected_appearances[core_id] = 1
        else:
            collected_appearances[core_id] += 1
    # print("collected appearances: ", collected_appearances)

    # insert cores in collected order and cardinalities of received pattern
    # not sure whether order is maintained, if iteration is done with .values() --> .items() should maintain it
    counter = 0
    for _, cardinality in collected_appearances.items():
        unused_cores = list(set(range(num_cores)) - set(offspring))
        for _ in range(cardinality):
            if counter < len(used_cores_in_x_parent):
                new_core = used_cores_in_x_parent[counter]
            elif unused_cores:
                # used existing-unused core
                new_core = random.choice(unused_cores)
            else:
                # used artificial placeholder-core_id
                # num_stages is the upper ceiling of required additional values
                new_core = random.randint(num_cores, num_stages)
            offspring += [new_core]
        counter += 1

    return offspring


def extend_offsprings_by_offsets(offspring_1, offspring_2, offset_1, offset_2, cutoff_point):
    return (offspring_1 + [offspring_1[-1] for _ in range(offset_2)],
            offspring_2 + [offspring_2[-1] for _ in range(offset_1)])


def custom_crossover(ind1, ind2):
    # get cutoff point
    cutoff_point = random.randint(1, len(ind1) - 1)

    # get fixed part of childs # offsets
    offspring_1, offset_1 = get_fixed_offspring_part(cutoff_point, ind1)
    offspring_2, offset_2 = get_fixed_offspring_part(cutoff_point, ind2)

    # print("offsets: ", offset_1, offset_2, "\n")

    offspring_1, offspring_2 = extend_offsprings_by_offsets(offspring_1, offspring_2, offset_1, offset_2, cutoff_point)

    # print(offspring_1, offspring_2)

    # print("##############################")

    offspring_1 = extend_offspring_by_group_structure(offspring=offspring_1,
                                                      x_parent=ind2,
                                                      cutoff_point=cutoff_point,
                                                      x_offset=offset_2)

    # print("##############################")

    offspring_2 = extend_offspring_by_group_structure(offspring=offspring_2,
                                                      x_parent=ind1,
                                                      cutoff_point=cutoff_point,
                                                      x_offset=offset_1)

    offspring1 = creator.Individual()
    offspring2 = creator.Individual()

    for e1, e2 in zip(offspring_1, offspring_2):
        offspring1.append(e1)
        offspring2.append(e2)

    assert len(offspring1) == num_stages
    assert len(offspring2) == num_stages

    return offspring1, offspring2


def get_group_ranges(ind):
    ranges = {
        core_id: None for core_id in range(num_cores)
    }
    for core_id in range(num_cores):
        starting = 0
        for idx in ind:
            if core_id == idx:
                ranges[core_id] = [starting, 0]
                break
            starting += 1
    ind.reverse()
    for core_id in range(num_cores - 1, -1, -1):
        ending = num_stages - 1
        for idx in ind:
            if core_id == idx:
                ranges[core_id][1] = ending
                break
            ending -= 1
    ind.reverse()
    return ranges


# replace wit unused core
def mut_replace_core_group(ind):
    ranges = get_group_ranges(ind)
    used_cores = [core_id for core_id in range(num_cores) if ranges[core_id]]

    if len(used_cores) == num_cores:  # no core to replace with available
        return ind

    core_to_replace = random.choice(used_cores)
    unused_cores = [core_id for core_id in range(num_cores) if not ranges[core_id]]
    core_to_replace_with = random.choice(unused_cores)

    for idx in range(ranges[core_to_replace][0], ranges[core_to_replace][1] + 1):
        ind[idx] = core_to_replace_with

    return ind


# split used group and replace by unused core if possible
def mut_split_group(ind):
    ranges = get_group_ranges(ind)
    used_cores = [core_id for core_id in range(num_cores) if ranges[core_id]]

    if len(used_cores) == num_cores:  # no core to replace with available
        return ind

    # filter cores with group-size 1
    used_cores = [core_id for core_id in used_cores if ranges[core_id][0] != ranges[core_id][1]]

    if not used_cores:
        return ind

    core_group_to_split = random.choice(used_cores)
    unused_cores = [core_id for core_id in range(num_cores) if not ranges[core_id]]
    core_to_introduce = random.choice(unused_cores)

    # starting = random.randint(ranges[core_group_to_split], ranges[core_group_to_split] + 1) maybe random also possible
    new_starting = random.randint(ranges[core_group_to_split][0] + 1, ranges[core_group_to_split][1])
    '''
    math.floor(
        (ranges[core_group_to_split][0] + ranges[core_group_to_split][1] + 1) / 2 #replace by random
    )  # if core was just used once, in 1 stage, it will be just replaced
    '''

    for idx in range(new_starting, ranges[core_group_to_split][1] + 1):
        ind[idx] = core_to_introduce

    return ind


def mut_merge_groups(ind):
    dominant_core = random.choice(ind)
    merge_options = [-1, 1]
    if ind.index(dominant_core) == 0:  # first core
        # exclude merge-option with predecessor
        merge_options.remove(-1)
    ind.reverse()
    if ind.index(dominant_core) == 0:
        # exclude merge-option with successor
        merge_options.remove(1)
    ind.reverse()
    if not merge_options:
        # print(ind)
        # print("unforeseen result .. returning individual")
        return ind
    merge_decision = random.choice(merge_options)
    if merge_decision == -1:
        dom_core_idx = ind.index(dominant_core)
        core_to_merge = ind[dom_core_idx - 1]
    else:  # merge decision = 1
        ind.reverse()
        dom_core_idx = ind.index(dominant_core)
        core_to_merge = ind[dom_core_idx - 1]
        ind.reverse()
    ranges = get_group_ranges(ind)
    for core_id in range(ranges[core_to_merge][0], ranges[core_to_merge][1] + 1):
        ind[core_id] = dominant_core
    return ind


def mutate(ind):
    mutations = [
        mut_replace_core_group,
        mut_split_group,
        mut_merge_groups
    ]

    ind = random.choice(mutations)(ind)

    return ind,


def sum_(ind):
    return sum(
        [sum(stages_per_core) for stages_per_core in ind]
    )


def none_func(ind1, ind2):
    return ind1, ind2


def contains_consecutive_values(ind: list, core_indicies):
    # fill
    idx_sublists = {c: [] for c in core_indicies}
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

    for idx in ind:
        if idx not in list(range(num_cores)):
            return False, ind
    return True, None


random.seed(42)

toolbox = base.Toolbox()
toolbox.register("evaluate", evaluate)
toolbox.register("mate", custom_crossover)
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
while time.time() - st < 10:  # 60 * 60 * (24 + 18):  # 60*60*24: # run 1 day
    counter += 1
    population, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=20, cxpb=0.01, mutpb=.99,
                                                    ngen=5,
                                                    halloffame=hof, verbose=False)
    for ind in population:
        b, v = contains_consecutive_values(ind, list(range(num_cores)))
        if not b:
            print(v)

print(counter)
# for ind in population[:10]:
#    print(ind)
print("time: ", time.time() - st)
# print(hof.keys)
# print(hof.items)

# Define the file name
csv_file_name = './outputs/pareto_solutions'

# Open the CSV file in write mode with a semicolon as the separator
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
with open(f"{csv_file_name}_{timestamp}.csv", 'w', newline='') as csvfile:
    # Create a CSV writer object with semicolon as the delimiter
    csv_writer = csv.writer(csvfile)

    # Write header
    header = ['exec_time', "avg_latency", 'package_energy', 'e_cores', 'st_cores', 'ht_cores', 'Solutions']  # Adjust as needed
    csv_writer.writerow(header)

    # Write solutions and fitness values
    for ind in hof:
        fitness_values = ind.fitness.values
        solution_values = ','.join(map(str, ind))  # Convert list to string
        row_data = list(fitness_values) + [solution_values]
        csv_writer.writerow(row_data)
