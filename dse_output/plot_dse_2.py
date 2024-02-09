import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path = 'pareto_solutions_dse_VGG_alice_2nd.csv'  # Replace 'your_file.csv' with the actual path to your CSV file
df = pd.read_csv(file_path)

# Filter out rows where the accuracy is not equal to 0.8465
df = df[df['avg_latency'] == 0.8465]
df = df[df['e_cores'] <= 16]
df = df[df['st_cores'] <= 16]

# Extracting data
exec_time = df['exec_time']
avg_latency = df['avg_latency']
package_energy = df['package_energy']
e_cores = df["e_cores"]
st_cores = df["st_cores"]
ht_cores = df["ht_cores"]

# Create the first plot figure with 4 plots
fig1, axs1 = plt.subplots(2, 2, figsize=(10, 8))

# Plot 2: Time vs Package Energy
axs1[0, 1].scatter(exec_time, package_energy)
axs1[0, 1].set_title('Execution Time vs Package Energy')
axs1[0, 1].set_xlabel('Execution Time (s)')
axs1[0, 1].set_ylabel('Package Energy')

# Plot 3: Time vs E-Cores
axs1[1, 0].scatter(exec_time, e_cores)
axs1[1, 0].set_title('Execution Time vs E-Cores')
axs1[1, 0].set_xlabel('Execution Time (s)')
axs1[1, 0].set_ylabel('E-Cores')

# Plot 4: Time vs ST-Cores
axs1[1, 1].scatter(exec_time, st_cores)
axs1[1, 1].set_title('Execution Time vs ST-Cores')
axs1[1, 1].set_xlabel('Execution Time (s)')
axs1[1, 1].set_ylabel('ST-Cores')

# Create the second plot figure with 4 plots
fig2, axs2 = plt.subplots(2, 2, figsize=(10, 8))

# Plot 5: Time vs HT-Cores
axs2[0, 0].scatter(exec_time, ht_cores)
axs2[0, 0].set_title('Execution Time vs HT-Cores')
axs2[0, 0].set_xlabel('Execution Time (s)')
axs2[0, 0].set_ylabel('HT-Cores')

# Plot 6: Package Energy vs E-Cores
axs2[0, 1].scatter(package_energy, e_cores)
axs2[0, 1].set_title('Package Energy vs E-Cores')
axs2[0, 1].set_xlabel('Package Energy')
axs2[0, 1].set_ylabel('E-Cores')

# Plot 7: Package Energy vs ST-Cores
axs2[1, 0].scatter(package_energy, st_cores)
axs2[1, 0].set_title('Package Energy vs ST-Cores')
axs2[1, 0].set_xlabel('Package Energy')
axs2[1, 0].set_ylabel('ST-Cores')

# Plot 8: Package Energy vs HT-Cores
axs2[1, 1].scatter(package_energy, ht_cores)
axs2[1, 1].set_title('Package Energy vs HT-Cores')
axs2[1, 1].set_xlabel('Package Energy')
axs2[1, 1].set_ylabel('HT-Cores')

# Adjust layout for both figures
plt.tight_layout()
plt.show()
