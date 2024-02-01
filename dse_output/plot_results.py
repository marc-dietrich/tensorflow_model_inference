import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path = 'pareto_solutions_dse_VGG_alice.csv'  # Replace 'your_file.csv' with the actual path to your CSV file
df = pd.read_csv(file_path)

# Extracting data
exec_time = df['exec_time']
avg_latency = df['avg_latency']
package_energy = df['package_energy']
cpu_energy = df['cpu_energy']

# Create a single large plot with subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Subplot 1: Time vs Energy1
axs[0, 0].scatter(exec_time, package_energy, label='Package Energy')
axs[0, 0].set_title('Time vs Package Energy')
axs[0, 0].set_xlabel('Execution Time (s)')
axs[0, 0].set_ylabel('Package Energy')
axs[0, 0].legend()

# Subplot 2: Time vs Energy2
axs[0, 1].scatter(exec_time, cpu_energy, label='CPU Energy', color='orange')
axs[0, 1].set_title('Time vs CPU Energy')
axs[0, 1].set_xlabel('Execution Time (s)')
axs[0, 1].set_ylabel('CPU Energy')
axs[0, 1].legend()

# Subplot 3: Energy1 vs Energy2
axs[1, 0].scatter(package_energy, cpu_energy)
axs[1, 0].set_title('Package Energy vs CPU Energy')
axs[1, 0].set_xlabel('Package Energy')
axs[1, 0].set_ylabel('CPU Energy')

# Subplot 4: Time vs Latency
axs[1, 1].scatter(exec_time, avg_latency, label='Average Latency', color='green')
axs[1, 1].set_title('Time vs Average Latency')
axs[1, 1].set_xlabel('Execution Time (s)')
axs[1, 1].set_ylabel('Average Latency')
axs[1, 1].legend()

# Adjust layout
plt.tight_layout()
plt.show()
