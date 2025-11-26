import re
import sys
import matplotlib.pyplot as plt
import csv
import math

# -------------------------------------------------------------------
# Load signum CSV (Node,Signum) — now handles NaN safely
# -------------------------------------------------------------------
def load_signum_csv(csv_file):
    signum = {}
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Node"]
            raw_val = row["Signum"].strip()

            # Node naming like N_2_3_1 → extract (2,3)
            m = re.match(r"N_(\d+)_(\d+)_\d+", name)
            if not m:
                continue

            r, c = map(int, m.groups())

            # ---------- NaN Handling ----------
            # If empty, nan, or not an int, store None → colored gray
            try:
                if raw_val.lower() == "nan" or raw_val == "":
                    val = None
                else:
                    v = float(raw_val)
                    val = int(v) if not math.isnan(v) else None
            except:
                val = None

            signum[(r, c)] = val

    return signum


# -------------------------------------------------------------------
# Parse enables from testbench file
# -------------------------------------------------------------------
def parse_enables(enable_file):
    ro_enable = {}
    c_enable = {}

    with open(enable_file, "r") as f:
        for line in f:
            line = line.strip()

            m = re.match(r"V_EN_RO_(\d+)_(\d+)\s+\S+\s+\S+\s+([01])$", line)
            if m:
                r, c, val = m.groups()
                ro_enable[(int(r), int(c))] = int(val)
                continue

            m = re.match(r"V_EN_C_(\d+)_(\d+)__(\d+)_(\d+)\s+\S+\s+\S+\s+([01])$", line)
            if m:
                r1, c1, r2, c2, val = m.groups()
                c_enable[(int(r1), int(c1), int(r2), int(c2))] = int(val)
                continue

    return ro_enable, c_enable


# -------------------------------------------------------------------
# Parse network netlist for RO nodes and edges
# -------------------------------------------------------------------
def parse_network(net_file):
    ro_positions = []
    edges = []

    with open(net_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("*"):
                continue

            m = re.match(r"XRO_(\d+)_(\d+)\b", line)
            if m:
                ro_positions.append(tuple(map(int, m.groups())))
                continue

            m = re.match(r"XCPL_(\d+)_(\d+)__(\d+)_(\d+)\b", line)
            if m:
                r1, c1, r2, c2 = map(int, m.groups())
                edges.append(((r1, c1), (r2, c2)))
                continue

    return ro_positions, edges


# -------------------------------------------------------------------
# Hex-grid coordinates
# -------------------------------------------------------------------
def hex_position(r, c, size=1.0):
    import math
    dx = (math.sqrt(3)) * size
    dy = 1.5 * size
    x = c * dx + (r % 2) * (dx / 2)
    y = r * dy
    return x, y


# -------------------------------------------------------------------
# Draw final network with signum-colored nodes (NaN → gray)
# -------------------------------------------------------------------
def draw_network(ro_positions, edges, ro_enable, c_enable, signum_map):
    plt.figure(figsize=(14, 10))

    # Enabled ROs only
    active_ros = [p for p in ro_positions if ro_enable.get(p, 0) == 1]

    pos_map = {p: hex_position(*p) for p in active_ros}

    # Draw enabled edges
    for a, b in edges:
        key = (a[0], a[1], b[0], b[1])
        rkey = (b[0], b[1], a[0], a[1])

        if c_enable.get(key, 0) == 1 or c_enable.get(rkey, 0) == 1:
            if a in pos_map and b in pos_map:
                x1, y1 = pos_map[a]
                x2, y2 = pos_map[b]
                plt.plot([x1, x2], [y1, y2], '-', color='black', linewidth=1)

    # Draw nodes
    for (r, c), (x, y) in pos_map.items():

        sig = signum_map.get((r, c), None)

        if sig == 1:            # +1 phase → red
            color = "red"
        elif sig == -1:         # -1 phase → blue
            color = "blue"
        else:                   # NaN or missing → gray
            color = "gray"

        plt.scatter([x], [y], s=240, color=color, edgecolors='black')
        plt.text(x, y, f"{r},{c}", ha='center', va='center', color='white')

    plt.title("Ring Oscillator Network Colored by sign(cos(phase))")
    plt.axis("equal")
    plt.axis("off")
    plt.show()


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage:")
        print("   python visualize_network.py <enable_file.sp> <network_file.sp> <signum_output.csv>")
        sys.exit(1)

    enable_file = sys.argv[1]
    network_file = sys.argv[2]
    signum_csv = sys.argv[3]

    ro_enable, c_enable = parse_enables(enable_file)
    ro_positions, edges = parse_network(network_file)
    signum_map = load_signum_csv(signum_csv)

    print("ROs found:", len(ro_positions))
    print("Enabled ROs:", sum(ro_enable.values()))
    print("Enabled couplers:", sum(c_enable.values()))
    print("Signum entries loaded:", len(signum_map))

    draw_network(ro_positions, edges, ro_enable, c_enable, signum_map)
