import matplotlib.pyplot as plt

filename = "4,8,8,8,5 add 0.1 delete 0.07 nodes.txt"
data = {}
species_info = {}
current_generation = None

with open(filename, "r") as file:
    for line in file:
        if line.startswith(" ****** Running generation"):
            parts = line.split()
            current_generation = int(parts[-2])
            data[current_generation] = {}
        elif line.startswith("Best fitness in specie"):
            parts = line.split()
            specie = int(parts[4])
            fitness_str = parts[6].replace(",", "")
            fitness = float(fitness_str)
            structure_str = parts[7:]
            structure_str = ' '.join(structure_str)
            structure_str = structure_str.replace(".", "")
            structure_str = structure_str.replace("'", "")
            data[current_generation][specie] = fitness
            species_info.setdefault(specie, []).append(structure_str)

generations = list(data.keys())
species = set(specie for specie_data in data.values() for specie in specie_data.keys())

plt.figure(figsize=(12, 8))
for specie in species:
    fitness_values = [data[generation].get(specie, None) for generation in generations]
    plt.plot(generations, fitness_values, label=f"Specie {specie} {species_info[specie][0]} -> {species_info[specie][-1]}")
    # plt.plot(generations, fitness_values, label=f"Specie {specie} {species_info[specie][-1]}")
    start_generation = next(
        (generation for generation, fitness in zip(generations, fitness_values) if fitness is not None), None)
    start_fitness = data[start_generation][specie] if start_generation is not None else None
    if start_generation is not None and start_fitness is not None:
        plt.text(start_generation, start_fitness, f"{specie}", fontsize=8, verticalalignment='bottom')

    start_info = species_info[specie][0]
    end_info = species_info[specie][-1]
    legend_info = f"{start_info} → {end_info}"

plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Fitness Change of Different Species Across Generations")
plt.legend(bbox_to_anchor=(0.99, 1), loc='upper left', fontsize=7)
plt.gcf().set_size_inches(14, 8)
plt.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1)
plt.show()