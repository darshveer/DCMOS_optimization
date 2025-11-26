import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# -----------------------------------------------------------
#  READ CSV & EXTRACT NODE NAMES AUTOMATICALLY
# -----------------------------------------------------------

file = "/Users/ahilnandankabilan/Downloads/new 2/output_nodes.csv"

df = pd.read_csv(file)

# First column = time
t = df.iloc[:, 0].values

# Remaining columns after column 2 → node voltages
node_cols = df.iloc[:, 2:].values

# Extract node names from header
nodeNames = df.columns[2:]


# -----------------------------------------------------------
#  PLOT NODE VOLTAGES
# -----------------------------------------------------------

plt.figure(figsize=(10, 5))
plt.plot(t, node_cols)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("SPICE Output Node Voltages vs Time")
plt.grid(True)
plt.legend(nodeNames, bbox_to_anchor=(1.04, 1))
plt.tight_layout()
plt.show()


# -----------------------------------------------------------
#  PHASE EXTRACTION
# -----------------------------------------------------------

def extract_phase(time, voltage, threshold=0.5):
    rising = []

    for i in range(len(voltage)-1):
        if voltage[i] < threshold and voltage[i+1] >= threshold:

            # Linear interpolation crossing time
            t_cross = time[i] + (threshold - voltage[i]) * \
                      ((time[i+1] - time[i]) / (voltage[i+1] - voltage[i]))
            rising.append(t_cross)

    # Need 6 cycles to compute smooth median frequency
    if len(rising) < 6:
        return float("nan"), float("nan")

    SS = 5
    rising_ss = rising[-(SS+1):]

    T = np.median(np.diff(rising_ss))
    t_last = rising_ss[-1]

    phase_rad = (t_last % T) / T * 2 * math.pi
    phase_deg = phase_rad * 180 / math.pi

    return phase_deg, T


def compute_all_phases(time, node_matrix, nodeNames):
    phases = {}
    for k, name in enumerate(nodeNames):
        v = node_matrix[:, k]
        ph_deg, T = extract_phase(time, v)
        phases[name] = ph_deg
    return phases


# -----------------------------------------------------------
#  RUN PHASE EXTRACTION
# -----------------------------------------------------------

phases = compute_all_phases(t, node_cols, nodeNames)
print("PHASES (degrees):")
print(phases)


# -----------------------------------------------------------
#  APPLY SIGNUM( COS(phase) )
# -----------------------------------------------------------

phase_signum = {}

for name, ph in phases.items():
    phase_signum[name] = np.sign(np.cos(np.deg2rad(ph))) if not np.isnan(ph) else np.nan

print("SIGN( COS(phase) ) per node:")
print(phase_signum)


# -----------------------------------------------------------
#  WRITE SIGNUM VALUES TO CSV
# -----------------------------------------------------------

output_path = "/Users/ahilnandankabilan/Downloads/new 2/signum_output.csv"

nodes = list(phase_signum.keys())
values = [phase_signum[n] for n in nodes]

out_df = pd.DataFrame({"Node": nodes, "Signum": values})
out_df.to_csv(output_path, index=False)

print(f"CSV written to: {output_path}")
