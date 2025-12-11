
# COP Oscillator: Ring Oscillator-Based Ising Solver

A VLSI implementation of a probabilistic compute fabric using coupled ring oscillators to solve combinatorial optimization problems (COPs) such as the Max-Cut problem.

## 📋 Overview

This project implements an **Ising computer** in 65nm CMOS technology using a network of coupled ring oscillators (ROSCs) to solve NP-hard combinatorial optimization problems. Unlike traditional von Neumann architectures and quantum annealing approaches, this system operates at room temperature and offers a scalable, energy-efficient alternative for solving optimization problems.

### Key Innovation

The core innovation maps the **Ising Hamiltonian** to circuit dynamics:
- Each ring oscillator represents a spin variable (si ∈ {-1, +1})
- Oscillator phase represents spin state (in-phase → same state, 180° out-of-phase → opposite state)
- Coupling between oscillators is implemented using back-to-back inverters (B2B inverters)
- The system naturally evolves to minimize the Hamiltonian energy: **Hs = Σ Jij·si·sj**

![Ising Model Mapping - Fully-connected graphs with spin representation](./Photos/Figure_1.png)

## 🏗️ Architecture

### Design Specifications

| Aspect | Details |
|--------|---------|
| **Array Configuration** | Hexagonal lattice arrangement |
| **Ring Oscillator Count** | 560 ROSCs (full design) / 4×4 testable prototype |
| **Coupling Topology** | Each oscillator couples to 6 nearest neighbors |
| **Technology Node** | 65nm CMOS |
| **Ring Oscillator Design** | 6 inverters + 1 NAND gate (acts as enable switch) |
| **Output Tap Point** | Output of 3rd inverter (coupling point) |

### Component Details

**Ring Oscillator Unit Cell:**
- 6 inverter stages
- 1 NAND gate for enable control
- Programmable coupling to neighbors
- Selectable steady-state output

**Coupling Mechanism:**
- Back-to-back inverters implement Jij term (ferromagnetic/antiferromagnetic)
- Selectable weight enables/disables coupling
- All weights set to -1 (antiferromagnetic) for Max-Cut problems

![Coupling between two ROSCs using B2B Inverters - Circuit implementation showing inverter chain](./Photos/Figure_12.png)

### Hexagonal Array Layout

The proposed 28×20 hexagonal array design demonstrates the full-scale architecture:


## 🛠️ Technology Stack

| Component | Tool/Library |
|-----------|--------------|
| **Simulation** | NGspice |
| **Automation** | Python 3 |
| **Post-processing** | MATLAB |
| **Visualization** | Python (matplotlib) |
| **Scripting** | Bash |

## 📁 Project Structure

```
cop_oscillator/
├── planar.py              # Graph generation and testbench automation
├── network_4_4.subckt     # Core SPICE netlist (4×4 array)
├── testbench.cir          # Auto-generated simulation testbench
├── Grouper.m              # Phase detection and post-processing
├── visualize_edges.py     # Results visualization
├── run_sim.sh             # Complete workflow orchestration
├── nodes.csv              # Raw simulation output
├── signum_output.csv      # Discretized phase states
├── Photos/                # Circuit diagrams and result plots
└── README.md              # This file
```

## 🚀 Workflow

The complete simulation pipeline follows this sequence:

```
planar.py 
   ↓ (generates random graph & enables)
testbench.cir + network_4_4.subckt
   ↓ (NGspice transient analysis)
nodes.csv (raw voltage/time data)
   ↓ (Grouper.m - edge detection & phase mapping)
signum_output.csv (binary spin states)
   ↓ (visualize_edges.py)
Final Result Plot (colored node graph)
```

![Horizontal workflow of the Ring Oscillator Ising Solver](./Photos/Figure_15.png)

### Step-by-Step Usage

1. **Generate Test Case:**
   ```bash
   python3 planar.py
   ```
   - Creates random planar graph mapped to hexagonal grid
   - Computes optimal Max-Cut solution (brute-force verification)
   - Generates `testbench.cir` with enable signals

2. **Run Simulation:**
   ```bash
   ngspice testbench.cir
   ```
   - Transient analysis: 5 seconds, 0.1ns time steps
   - Outputs voltage waveforms to `nodes.csv`

3. **Process Results:**
   ```bash
   matlab < Grouper.m
   ```
   - Detects rising edges in oscillator outputs
   - Computes relative phase differences
   - Discretizes to binary states using signum function
   - Generates `signum_output.csv`

4. **Visualize Solution:**
   ```bash
   python3 visualize_edges.py
   ```
   - Reconstructs graph topology from enables
   - Colors nodes: Red (-1) / Blue (+1)
   - Plots final Max-Cut partition

5. **Full Automation:**
   ```bash
   ./run_sim.sh
   ```
   - Executes complete workflow in sequence

## 📊 Results and Performance

### Current Implementation

**4×4 Oscillator Array Performance:**
- **Accuracy Range:** 65% - 100%
- **Test Cases:** 20 independent runs
- **Success Rate:** Highly variable due to lack of thermal noise

![Histogram of Accuracy for 4×4 array across 20 test cases](./Photos/Figure_17.png)

### Example Result - Max-Cut Solution

The following example demonstrates a successful solution with 100% accuracy on a graph with Max-Cut value of 11:

![Example Max-Cut problem graph with solution partition](./Photos/Figure_19.png)

The graph is partitioned into two sets:
- **Red nodes** represent spin state -1
- **Blue nodes** represent spin state +1
- **Edges between partitions** contribute to Max-Cut value (11 in this case)

### Probabilistic Behavior

The system exhibits probabilistic operation through:
- Inherent system noise and process variation
- Escape from local minima to explore energy landscape
- Multiple stable states achievable with same configuration

## 🔧 Script Functions

### `planar.py` - Graph Generation & Automation
- Generates random planar graphs
- Maps graphs to hexagonal grid
- Solves Max-Cut via brute-force (ground truth)
- Generates SPICE testbench with enable signals
- Creates network instantiations

### `network_4_4.subckt` - Hardware Netlist
- Defines 4×4 array structure
- Implements all ring oscillators
- Includes all possible couplings (selectively enabled)
- Acts as hardware substrate model

### `Grouper.m` - Signal Processing
- Reads simulation CSV output
- Performs rising edge detection
- Computes phase differences between oscillators
- Applies signum function for binary discretization
- Outputs processed states

### `visualize_edges.py` - Results Visualization
- Parses graph topology from enable signals
- Overlays computed spin states
- Creates hexagonal network plots
- Color codes final partition (Red = -1, Blue = +1)

## ⚙️ Configuration Parameters

### Simulation Settings
- **Time Step:** 0.1 ns
- **Total Simulation Time:** 5 seconds
- **Initial Oscillator State:** Always starts from 0
- **Graph Size:** Configurable (current: 4×4 = 16 nodes)

### Enable Signals
- **OSC_EN[n]:** Enable individual ring oscillators (1 per node)
- **WEIGHT_EN[i,j]:** Enable coupling between oscillators (1 per edge)
- **GLOBAL_EN:** Master enable for all couplers

## 🎯 Applications

This architecture is suited for solving:
- **Max-Cut Problems** (currently demonstrated)
- **Graph Partitioning**
- **QUBO (Quadratic Unconstrained Binary Optimization)**
- **Combinatorial Optimization** in general

## 📈 Advantages

✅ Room-temperature operation (vs. quantum annealing)
✅ CMOS-compatible (standard foundries)
✅ Scalable hexagonal topology
✅ Naturally probabilistic (noise aids exploration)
✅ Low power operation
✅ Parallel computation fabric

## ⚠️ Known Limitations

### Current Implementation

1. **Array Size:** Currently limited to 4×4 due to:
   - Python recursion depth constraints
   - Computational resource limits
   - Future work: Optimize code for 28×20 array (as in paper)

2. **Thermal Noise:** Not simulated due to:
   - Digital-only PDK limitations
   - NGspice constraints
   - Solution: Migrate to SKY130 PDK or Cadence tools

3. **Deterministic Output:** Lack of noise causes:
   - Same output for identical configurations
   - Cannot perform Monte-Carlo analysis
   - No stochastic exploration of solution space

## 🔮 Future Work

- [ ] Optimize code for larger arrays (28×20+)
- [ ] Implement thermal noise in simulation
- [ ] Migrate to SKY130 or advanced PDK
- [ ] Use Cadence/commercial simulators for accuracy
- [ ] Perform Monte-Carlo analysis
- [ ] Hardware tape-out and silicon validation
- [ ] Compare with quantum annealing and classical solvers

## 📚 Theory Reference

This implementation is based on:

**Paper:** A Probabilistic Compute Fabric Based on Coupled Ring Oscillators for Solving Combinatorial Optimization Problems

**Authors:** Ahmed, I., Chiu, P.-W., Moy, W., Kim, C. H.

**Journal:** IEEE Journal of Solid-State Circuits, Vol. 56, No. 9, 2021

**DOI:** [10.1109/JSSC.2021.3062821](https://doi.org/10.1109/JSSC.2021.3062821)

### Key Concepts
- **Ising Model:** Framework for mapping optimization to spin systems
- **Ring Oscillators:** Phase-domain computation using oscillatory dynamics
- **Coupled Oscillator Networks:** Multi-body synchronization dynamics
- **Probabilistic Computing:** Using noise for optimization

## 👥 Contributors

- Nivethitha B (MS2025007)
- Darsh Veer Singh (IMT2023543)
- Mihir S Kagalkar (IMT2023570)
- Ahilnandan Kabilan (IMT2023614)

**Course:** VLS503 - DCMOS VLSI Design
**Institution:** IIIT Bangalore

## 📝 License

This project is provided for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Code optimization for larger arrays
- Enhanced noise modeling
- Alternative coupling schemes
- Novel graph generation algorithms
- Performance benchmarking

## 📞 Support & Issues

For questions or issues:
1. Check existing documentation and examples
2. Review the paper for theoretical background
3. Open an issue on GitHub with detailed description

## 🔗 Quick Links

- **Paper:** [IEEE JSSC 2021](https://doi.org/10.1109/JSSC.2021.3062821)
- **GitHub:** [darshveer/cop_oscillator](https://github.com/darshveer/cop_oscillator)
- **Photos:** [All Circuit Diagrams and Results](./Photos/)
- **Institution:** [IIIT Bangalore](https://www.iiitb.ac.in/)

---

**Last Updated:** November 2025
**Project Status:** Active Development
