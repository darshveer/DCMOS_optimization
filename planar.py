#!/usr/bin/env python3

import random
import argparse

# ---------------- HEX NEIGHBOR RULE ----------------
def hex_neighbors(rows, cols, r, c):
    if r % 2 == 0:  # even row
        cand = [
            (r, c-1), (r, c+1),
            (r-1, c-1), (r-1, c),
            (r+1, c-1), (r+1, c)
        ]
    else:  # odd row
        cand = [
            (r, c-1), (r, c+1),
            (r-1, c), (r-1, c+1),
            (r+1, c), (r+1, c+1)
        ]
    return [(rr, cc) for rr, cc in cand if 0 <= rr < rows and 0 <= cc < cols]


# ---------------- GRAPH GENERATION ----------------
def generate_planar_graph(rows, cols, num_nodes, seed=None):
    if seed is not None:
        random.seed(seed)

    max_cells = rows * cols
    if num_nodes > max_cells:
        raise ValueError("num_nodes must be <= rows*cols")

    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    chosen = random.sample(all_cells, num_nodes)

    nodes = [str(i) for i in range(num_nodes)]
    pos_map = {nodes[i]: chosen[i] for i in range(num_nodes)}

    degree = {n: 0 for n in nodes}
    edges = set()
    maxdeg = 6

    def add_edge(a, b):
        if a == b: return False
        if degree[a] >= maxdeg or degree[b] >= maxdeg: return False
        e = tuple(sorted((a, b)))
        if e in edges: return False
        edges.add(e)
        degree[a] += 1
        degree[b] += 1
        return True

    # Build minimal spanning structure
    unvisited = set(nodes)
    visited = set()

    root = unvisited.pop()
    visited.add(root)

    while unvisited:
        nxt = unvisited.pop()
        r, c = pos_map[nxt]

        parents = []
        for v in visited:
            rv, cv = pos_map[v]
            if (r, c) in hex_neighbors(rows, cols, rv, cv):
                parents.append(v)

        if not parents:
            return generate_planar_graph(rows, cols, num_nodes, seed)

        parent = random.choice(parents)
        add_edge(parent, nxt)
        visited.add(nxt)

    # Add random planar edges
    for u in nodes:
        ru, cu = pos_map[u]
        for v in nodes:
            if u == v: continue
            rv, cv = pos_map[v]
            if (rv, cv) in hex_neighbors(rows, cols, ru, cu):
                if random.random() < 0.3:
                    add_edge(u, v)

    return nodes, list(edges), pos_map


# ---------------- TESTBENCH UTIL ----------------
def en_ro(r, c):
    return f"EN_RO_{r}_{c}"

def en_c(r1, c1, r2, c2):
    return f"EN_C_{r1}_{c1}__{r2}_{c2}"

def probe_node(r, c):
    return f"N_{r}_{c}_1"


def write_testbench(rows, cols, nodes, edges, pos_map, network_file, outname):
    # All RO enables
    all_ro_en = [en_ro(r, c) for r in range(rows) for c in range(cols)]

    # All possible coupler pairs
    all_coupler_pairs = []
    for r in range(rows):
        for c in range(cols):
            for rr, cc in hex_neighbors(rows, cols, r, c):
                if (rr, cc, r, c) not in all_coupler_pairs:
                    all_coupler_pairs.append((r, c, rr, cc))

    all_coupler_en = [en_c(r, c, rr, cc) for r, c, rr, cc in sorted(all_coupler_pairs)]

    probe_ports = [probe_node(r, c) for r in range(rows) for c in range(cols)]

    rev = {pos_map[n]: n for n in nodes}
    edge_set = set(tuple(sorted(e)) for e in edges)

    with open(outname, "w") as f:
        f.write("* Auto-testbench (planar graph mapped to RO grid)\n\n")
        f.write('.include "ptm_45nm_lp.l"\n')
        f.write('.include "inv.subckt"\n')
        f.write('.include "nand.subckt"\n')
        f.write('.include "ring_osc.subckt"\n')
        f.write('.include "coupling.subckt"\n')
        f.write(f'.include "{network_file}"\n\n')

        # Instance of the entire network subcircuit
        f.write("Xdut ")
        f.write(" ".join(all_ro_en + all_coupler_en + probe_ports))
        f.write(" vdd gnd RING_OSC_NETWORK\n\n")

        # RO enables: 1 for nodes in this network, 0 otherwise
        f.write("* RO enables\n")
        for r in range(rows):
            for c in range(cols):
                node = rev.get((r, c))
                val = 1 if node in nodes else 0
                f.write(f"V_{en_ro(r,c)} {en_ro(r,c)} gnd {val}\n")
        f.write("\n")

        # Coupler enables: 1 for edges in this network, 0 otherwise
        f.write("* Coupler enables\n")
        for r1, c1, r2, c2 in sorted(all_coupler_pairs):
            u = rev.get((r1, c1))
            v = rev.get((r2, c2))
            val = 1 if u and v and tuple(sorted((u, v))) in edge_set else 0
            f.write(f"V_{en_c(r1,c1,r2,c2)} {en_c(r1,c1,r2,c2)} gnd {val}\n")
        f.write("\n")

        f.write("VDD vdd gnd 1.0\n\n")

        # Control block
        f.write(".control\n")
        f.write("save time " + " ".join(probe_ports) + "\n")
        f.write("tran 0.1ns 5us uic\n")
        f.write("set filetype=ascii\n")
        f.write("set wr_singlescale\n")
        f.write("set wr_vecnames\n")
        f.write("set csvdelim=comma\n")
        f.write("wrdata output_nodes.csv time " + " ".join(probe_ports) + "\n")
        f.write("quit\n")
        f.write(".endc\n\n")

        f.write(".end\n")

    print(f"Testbench written to: {outname}")

def maxcut_bruteforce(nodes, edges):
    """
    Brute-force MaxCut for small graphs.
    nodes: list of node labels (e.g. ['0','1',...])
    edges: list of (u, v) pairs using those labels
    Returns: (best_cut_value, assignment_map)
      - best_cut_value: number of edges in the maximum cut
      - assignment_map: dict {node_label: 0 or 1}
    """
    n = len(nodes)
    label_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_edges = [(label_to_idx[u], label_to_idx[v]) for (u, v) in edges]

    best_cut_value = -1
    best_assignment = None  # list of 0/1

    for mask in range(1 << n):
        cut_val = 0
        for i, j in idx_edges:
            if ((mask >> i) & 1) != ((mask >> j) & 1):
                cut_val += 1
        if cut_val > best_cut_value:
            best_cut_value = cut_val
            best_assignment = [(mask >> k) & 1 for k in range(n)]

    assignment_map = {nodes[i]: best_assignment[i] for i in range(n)}
    return best_cut_value, assignment_map

# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", required=True, help="Path to 4x4 subckt network file")
    parser.add_argument("--out", default="testbench.cir", help="Output testbench file")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--num", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    nodes, edges, pos_map = generate_planar_graph(
        args.rows, args.cols, args.num, seed=args.seed
    )

    print("Nodes:", nodes)
    print("Edges:", edges)
    print("Positions:", pos_map)

    write_testbench(args.rows, args.cols, nodes, edges, pos_map, args.network, args.out)

    nodes, edges, pos_map = generate_planar_graph(
        args.rows, args.cols, args.num, seed=args.seed
    )

    print("Nodes:", nodes)
    print("Edges:", edges)
    print("Positions:", pos_map)

    # Exact MaxCut (brute-force) for comparison
    maxcut_value, maxcut_assign = maxcut_bruteforce(nodes, edges)
    print("MaxCut value:", maxcut_value)
    print("MaxCut assignment:", maxcut_assign)

    write_testbench(args.rows, args.cols, nodes, edges, pos_map, args.network, args.out)

if __name__ == "__main__":
    main()
