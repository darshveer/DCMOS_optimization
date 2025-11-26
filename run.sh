#!/bin/bash

# Define directories
SAVE_TB_DIR="Saved TB"
# Note: Using quotes to handle spaces in the path
IMG_DIR="/mnt/c/Users/Darsh Veer Singh/OneDrive - iiit-b/Sem 5/VLS503 - DCMOS VLSI Design/Project/DCMOS_optimization/Final/Photos"

# Create directories if they don't exist
mkdir -p "$SAVE_TB_DIR"
mkdir -p "$IMG_DIR"

# Loop from 8 to 15
for i in {21..30}
do
    echo "----------------------------------------"
    echo "Starting Iteration: $i"
    echo "----------------------------------------"

    # 1. Run the planner script
    python3 planar.py --network 'network_4_4.subckt' --rows 4 --cols 4 --num 10
    
    # 2. Run Ngspice
    # Added -b for batch mode to prevent interactive prompt blocking. 
    # Remove -b if you specifically need the interactive mode, but usually automation requires batch.
    ngspice -b testbench.cir
    
    wait
    
    # 3. Run MATLAB
    /usr/local/MATLAB/R2025b/bin/matlab -nodisplay -nosplash -nodesktop -r "run('Grouper.m');exit;" | tail -n +11
    
    wait
    
    # 4. Save the Testbench file
    echo "Copying testbench.cir to $SAVE_TB_DIR/testbench_${i}.cir"
    cp testbench.cir "$SAVE_TB_DIR/testbench_${i}.cir"

    # 5. Run Visualization and Save Image
    # We construct the output path here
    OUTPUT_IMG="$IMG_DIR/Figure_${i}.png"
    
    echo "Generating image: $OUTPUT_IMG"
    python3 Visualize_edges_1.py 'testbench.cir' 'network_4_4.subckt' 'signum_output.csv' "$OUTPUT_IMG"

    echo "Iteration $i completed."
    sleep 1
done

echo "All iterations (8-15) finished."