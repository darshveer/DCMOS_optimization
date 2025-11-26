python3 planar.py --network 'network_4_4.subckt' --rows 4 --cols 4 --num 10
ngspice testbench.cir
wait
/usr/local/MATLAB/R2025b/bin/matlab -nodisplay -nosplash -nodesktop -r "run('Grouper.m');exit;" | tail -n +11
wait
python3 Visualize_edges.py 'testbench.cir' 'network_4_4.subckt' 'signum_output.csv'
