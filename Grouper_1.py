import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# -----------------------------------------------------------
# READ CSV & PREPARE DATA
# -----------------------------------------------------------
# Reading the CSV file (assuming no header, similar to readmatrix)
try:
    # header=None assumes the file is pure numbers. 
    # If your CSV has a text header, remove 'header=None' or skip the row.
    df = pd.read_csv("output_nodes.csv", header=None)
except FileNotFoundError:
    print("Error: 'output_nodes.csv' not found. Please ensure the file exists.")
    exit()

raw = df.values

# MATLAB: t = raw(:, 1) -> Python (0-indexed): Column 0
t = raw[:, 0]

# MATLAB: node_cols = raw(:, 3:end) -> Python: Column 2 to end
# (This implies we skip Column 1, just like the MATLAB script)
node_cols = raw[:, 2:]

node_names = [
    "N_0_0_1", "N_0_1_1", "N_0_2_1", "N_0_3_1",
    "N_1_0_1", "N_1_1_1", "N_1_2_1", "N_1_3_1",
    "N_2_0_1", "N_2_1_1", "N_2_2_1", "N_2_3_1",
    "N_3_0_1", "N_3_1_1", "N_3_2_1", "N_3_3_1"
]

# -----------------------------------------------------------
# PLOT NODES
# -----------------------------------------------------------
plt.figure(figsize=(10, 6))
# Plotting all node columns against time
plt.plot(t, node_cols, linewidth=1.1)
plt.grid(True)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("SPICE Output Node Voltages vs Time")
# Create legend using the list of names
plt.legend(node_names, loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# FUNCTION: Extract rising edges
# -----------------------------------------------------------
def get_rising_edges(time, voltage, threshold=0.5):
    """
    Finds time indices where voltage crosses threshold from below.
    Uses linear interpolation for precise timing.
    """
    rising = []
    # Loop through signal to find crossings
    # Vectorized approaches exist, but a loop is safer to strictly match 
    # the specific interpolation logic of the MATLAB script.
    for i in range(len(voltage) - 1):
        if voltage[i] < threshold and voltage[i+1] >= threshold:
            # Linear interpolation formula
            t_cross = time[i] + (threshold - voltage[i]) * \
                      (time[i+1] - time[i]) / (voltage[i+1] - voltage[i])
            rising.append(t_cross)
    
    return np.array(rising)

# -----------------------------------------------------------
# FUNCTION: Compute relative phase using last 5 cycles
# -----------------------------------------------------------
def compute_relative_phase(time, v_node, v_ref):
    rising_ref = get_rising_edges(time, v_ref)
    rising_node = get_rising_edges(time, v_node)
    
    # Check if we have enough edges (MATLAB script checks for < 6)
    if len(rising_ref) < 6 or len(rising_node) < 6:
        return np.nan
    
    SS = 5
    # Take the last 5 edges
    rising_ref_ss = rising_ref[-SS:]
    rising_node_ss = rising_node[-SS:]
    
    # Calculate Period (Median of diffs)
    T = np.median(np.diff(rising_ref_ss))
    
    t_ref_last = rising_ref_ss[-1]
    t_node_last = rising_node_ss[-1]
    
    # Calculate phase difference in degrees
    dphi = (t_node_last - t_ref_last) / T * 360.0
    
    # Wrap phase to [-180, 180]
    # Python modulo behaves differently with negatives than MATLAB's rem/mod logic
    # The formula below replicates: mod(dphi + 180, 360) - 180
    rel_phase_deg = ((dphi + 180) % 360) - 180
    
    return rel_phase_deg

# -----------------------------------------------------------
# STEP 1: Compute preliminary phases vs themselves
# (To detect valid nodes)
# -----------------------------------------------------------
num_nodes = node_cols.shape[1]
prelim_phases = np.zeros(num_nodes)

for k in range(num_nodes):
    # Pass slice as 1D array
    prelim_phases[k] = compute_relative_phase(t, node_cols[:, k], node_cols[:, k])

# Find indices that are NOT NaN
valid_idx = np.where(~np.isnan(prelim_phases))[0]

if len(valid_idx) == 0:
    raise ValueError("No valid nodes found for reference selection.")

# Select random reference from valid indices
ref_idx = random.choice(valid_idx)
ref_name = node_names[ref_idx]
print(f"Random reference node selected: {ref_name}")

v_ref = node_cols[:, ref_idx]

# -----------------------------------------------------------
# STEP 2: Compute final relative phases
# -----------------------------------------------------------
relative_phases = {}

for k in range(num_nodes):
    v = node_cols[:, k]
    rel_phase = compute_relative_phase(t, v, v_ref)
    relative_phases[node_names[k]] = rel_phase

print("Relative phases (degrees) vs chosen reference:")
for name, val in relative_phases.items():
    print(f"{name}: {val}")

# -----------------------------------------------------------
# APPLY SIGN(COS(relative phase))
# -----------------------------------------------------------
phase_signum = {}

for name, ph in relative_phases.items():
    if np.isnan(ph):
        phase_signum[name] = np.nan
    else:
        # np.cos takes radians, so convert from degrees
        phase_signum[name] = np.sign(np.cos(np.deg2rad(ph)))

print("\nSIGN(COS(relative phase)) per node:")
for name, val in phase_signum.items():
    print(f"{name}: {val}")

# -----------------------------------------------------------
# WRITE SIGNUM VALUES TO CSV
# -----------------------------------------------------------
output_path = "signum_output.csv"

# Prepare data for DataFrame
data_rows = []
for name in phase_signum:
    data_rows.append({"Node": name, "Signum": phase_signum[name]})

df_out = pd.DataFrame(data_rows)
# MATLAB writetable creates headers by default, so we do index=False to hide row numbers
df_out.to_csv(output_path, index=False)

print(f"CSV written to: {output_path}")