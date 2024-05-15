import os
import matplotlib.pyplot as plt

txt_files = [file for file in os.listdir(".") if file.endswith(".txt")]

fitness_values = []

for file_name in txt_files:
    with open(file_name, "r") as file:
        lines = file.readlines()
        fitness_values.append([float(line.split("Best fitness: ")[1].split(" ")[0]) for line in lines if "Best fitness:" in line])

plt.figure()
for i, values in enumerate(fitness_values):
    file_name = os.path.splitext(txt_files[i])[0]
    plt.plot(range(1, len(values)+1), values, label=file_name)

plt.xlabel("Generations")
plt.ylabel("Best Fitness Value")
plt.legend()
plt.show()