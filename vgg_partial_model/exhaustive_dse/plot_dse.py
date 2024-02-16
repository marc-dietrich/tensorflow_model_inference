import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paretoset import paretoset

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d(e_cores, total_cores, energy):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(e_cores, total_cores, energy, c=energy, cmap='viridis', s=100)

    ax.set_xlabel('E Cores')
    ax.set_ylabel('Total Cores')
    ax.set_zlabel('Energy')

    # Add a color bar to show energy levels
    cbar = plt.colorbar(ax.scatter(e_cores, total_cores, energy, c=energy, cmap='viridis'))
    cbar.set_label('Energy')

    plt.title('3D Plot of E Cores, Total Cores, and Energy')
    plt.show()


def calculate_total_cores(df):
    # Calculate total cores based on the logic: st_cores count as 1, ht_cores count as 2
    total_cores = df['st_cores'] * 1 + df['ht_cores'] * 2

    # Add the total_cores as a new column to the DataFrame
    df['total_p_cores'] = total_cores

    return df


# Read the CSV file into a DataFrame
df = pd.read_csv("./exhaustive_dse_results_20240215123555.csv")

'''
# Find the index of the entry with the maximum value in column 'B'
max_index = df['package_energy'].idxmax()

# Drop the row with the maximum value in column 'B'
df = df.drop(max_index)
'''

df = calculate_total_cores(df)

# Extract columns for plotting
exec_time = df['exec_time']
package_energy = df['package_energy']

df_time_energy = df[["exec_time", "package_energy"]]

mask = paretoset(df_time_energy, sense=["min", "min"])
paretoset = df[mask]

print(paretoset)

# Plot exec_time against package_energy
plt.figure(figsize=(10, 6))
plt.scatter(exec_time, package_energy)
plt.scatter(paretoset["exec_time"], paretoset["package_energy"], color="red")
plt.title('Execution Time vs Package Energy')
plt.xlabel('Execution Time')
plt.ylabel('Package Energy')
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.show()

plot_3d(df['e_cores'], df['total_p_cores'], df['package_energy'])
