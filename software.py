#!/usr/bin/env python3

import random
import argparse
import itertools

# ============================================================
#                HEX NEIGHBOR RULE
# ============================================================
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


# ============================================================
#                GRAPH GENERATION
# ============================================================
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
        if a == b: 
            return False
        if degree[a] >= maxdeg or degree[b] >= maxdeg: 
            return False
        e = tuple(sorted((a, b)))
        if e in edges: 
            return False
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
            if u == v: 
                continue
            rv, cv = pos_map[v]
            if (rv, cv) in hex_neighbors(rows, cols, ru, cu):
                if random.random() < 0.3:
                    add_edge(u, v)

    return nodes, list(edges), pos_map


# ============================================================
#                MAXCUT SOLVER
# ===============================================================
def cut_value(spins, edges):
    """
    spins : dict {node: +1 or -1}
    edges : list of (u, v)
    Returns number of cut edges (sum of weights)
    """
    return sum(1 for u, v in edges if spins[u] != spins[v])


def maxcut_bruteforce(nodes, edges):
    """Exact solver: used if ≤ 20 nodes"""
    best_val = -1
    best_spins = None

    for assign in itertools.product([+1, -1], repeat=len(nodes)):
        spins = {nodes[i]: assign[i] for i in range(len(nodes))}
        val = cut_value(spins, edges)

        if val > best_val:
            best_val = val
            best_spins = spins

    return best_val, best_spins


def maxcut_local_search(nodes, edges, iters=5000):
    """Local-search MaxCut for large graphs"""
    spins = {n: random.choice([+1, -1]) for n in nodes}
    best_val = cut_value(spins, edges)

    for _ in range(iters):
        u = random.choice(nodes)
        spins[u] *= -1  # flip
        new_val = cut_value(spins, edges)

        if new_val >= best_val:
            best_val = new_val
        else:
            spins[u] *= -1  # undo

    return best_val, spins


def solve_maxcut(nodes, edges):
    """
    Returns:
        cut_weight  (int)
        spins       dict {node: ±1}
    """
    if len(nodes) <= 20:
        return maxcut_bruteforce(nodes, edges)
    else:
        return maxcut_local_search(nodes, edges)



# ============================================================
#                TESTBENCH UTIL
# ============================================================
def en_ro(r, c):
    return f"EN_RO_{r}_{c}"

def en_c(r1, c1, r2, c2):
    return f"EN_C_{r1}{c1}{r2}{c2}"

def probe_node(r, c):
    return f"N_{r}_{c}_1"


def write_testbench(rows, cols, nodes, edges, pos_map, network_file, outname):
    all_ro_en = [en_ro(r, c) for r in range(rows) for c in range(cols)]

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

        f.write("Xdut ")
        f.write(" ".join(all_ro_en + all_coupler_en + probe_ports))
        f.write(" vdd gnd RING_OSC_NETWORK\n\n")

        f.write("* RO enables\n")
        for r in range(rows):
            for c in range(cols):
                node = rev.get((r, c))
                val = 1 if node in nodes else 0
                f.write(f"V_{en_ro(r,c)} {en_ro(r,c)} gnd {val}\n")
        f.write("\n")

        f.write("* Coupler enables\n")
        for r1, c1, r2, c2 in sorted(all_coupler_pairs):
            u = rev.get((r1, c1))
            v = rev.get((r2, c2))
            val = 1 if u and v and tuple(sorted((u, v))) in edge_set else 0
            f.write(f"V_{en_c(r1,c1,r2,c2)} {en_c(r1,c1,r2,c2)} gnd {val}\n")
        f.write("\n")

        f.write("VDD vdd gnd 1.0\n\n")

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


# ============================================================
#                MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", required=True, help="Path to 4x4 subckt file")
    parser.add_argument("--out", default="testbench.cir", help="Output netlist")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--num", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    nodes, edges, pos_map = generate_planar_graph(
        args.rows, args.cols, args.num, seed=args.seed
    )

    print("\nNodes:", nodes)
    print("Edges:", edges)
    print("Positions:", pos_map)

    # --------------------------------------
    #        RUN MAXCUT SOLVER HERE
    # --------------------------------------
    cut, spins = solve_maxcut(nodes, edges)

    print("\n================ MAXCUT ================")
    print("Cut Weight (sum of weights):", cut)
    print("Spin Assignment (+1/-1):")
    for n in sorted(nodes, key=lambda x: int(x)):
        print(f" Node {n}: {spins[n]}")
    print("========================================\n")

    # write testbench
    write_testbench(args.rows, args.cols, nodes, edges, pos_map, args.network, args.out)


if __name__ == "__main__":
    main()