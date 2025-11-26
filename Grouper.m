%%-----------------------------------------------------------
% READ CSV & PLOT NODES
%%-----------------------------------------------------------

raw = readmatrix("output_nodes.csv");

t = raw(:, 1);
node_cols = raw(:, 3:end);

nodeNames = { ...
"N_0_0_1","N_0_1_1","N_0_2_1","N_0_3_1", ...
"N_1_0_1","N_1_1_1","N_1_2_1","N_1_3_1", ...
"N_2_0_1","N_2_1_1","N_2_2_1","N_2_3_1", ...
"N_3_0_1","N_3_1_1","N_3_2_1","N_3_3_1" ...
};

figure;
plot(t, node_cols, 'LineWidth', 1.1);
grid on;
xlabel("Time (s)");
ylabel("Voltage (V)");
title("SPICE Output Node Voltages vs Time");
legend(nodeNames, 'Location', 'bestoutside');



%%-----------------------------------------------------------
% FUNCTION: Extract rising edges
%%-----------------------------------------------------------
function rising = get_rising_edges(time, voltage, threshold)

    if nargin < 3
        threshold = 0.5;
    end

    rising = [];
    for i = 1:length(voltage)-1
        if voltage(i) < threshold && voltage(i+1) >= threshold
            t_cross = time(i) + (threshold - voltage(i)) * ...
                     (time(i+1)-time(i)) / (voltage(i+1)-voltage(i));
            rising(end+1) = t_cross; %#ok<AGROW>
        end
    end
end



%%-----------------------------------------------------------
% FUNCTION: Compute relative phase using last 5 cycles
%%-----------------------------------------------------------
function rel_phase_deg = compute_relative_phase(time, v_node, v_ref)

    rising_ref  = get_rising_edges(time, v_ref);
    rising_node = get_rising_edges(time, v_node);

    if length(rising_ref) < 6 || length(rising_node) < 6
        rel_phase_deg = NaN;
        return;
    end

    SS = 5;
    rising_ref_ss  = rising_ref(end-SS:end);
    rising_node_ss = rising_node(end-SS:end);

    T = median(diff(rising_ref_ss));

    t_ref_last  = rising_ref_ss(end);
    t_node_last = rising_node_ss(end);

    dphi = (t_node_last - t_ref_last) / T * 360;

    rel_phase_deg = mod(dphi + 180, 360) - 180;
end



%%-----------------------------------------------------------
% STEP 1: Compute preliminary phases vs themselves
% (Only to detect which nodes are valid / non-NaN)
%%-----------------------------------------------------------
prelim_phases = zeros(1, size(node_cols,2));

for k = 1:size(node_cols, 2)
    prelim_phases(k) = compute_relative_phase(t, node_cols(:,k), node_cols(:,k));
end

valid_idx = find(~isnan(prelim_phases));

if isempty(valid_idx)
    error("No valid nodes found for reference selection.");
end

% Select random non-NaN reference
ref_idx = valid_idx(randi(length(valid_idx)));
ref_name = nodeNames{ref_idx};
fprintf("Random reference node selected: %s\n", ref_name);

v_ref = node_cols(:, ref_idx);



%%-----------------------------------------------------------
% STEP 2: Compute final relative phases using selected reference
%%-----------------------------------------------------------
relative_phases = struct();

for k = 1:size(node_cols, 2)
    v = node_cols(:, k);
    rel_phase = compute_relative_phase(t, v, v_ref);
    relative_phases.(nodeNames{k}) = rel_phase;
end

disp("Relative phases (degrees) vs chosen reference:");
disp(relative_phases);



%%-----------------------------------------------------------
% APPLY SIGN(COS(relative phase))
%%-----------------------------------------------------------
phase_signum = struct();
fields = fieldnames(relative_phases);

for i = 1:length(fields)
    name = fields{i};
    ph = relative_phases.(name);

    if isnan(ph)
        phase_signum.(name) = NaN;
    else
        phase_signum.(name) = sign(cosd(ph));
    end
end

disp("SIGN(COS(relative phase)) per node:");
disp(phase_signum);



%%-----------------------------------------------------------
% WRITE SIGNUM VALUES TO CSV
%%-----------------------------------------------------------
output_path = "signum_output.csv";

node_list = fieldnames(phase_signum);
values = zeros(length(node_list), 1);

for i = 1:length(node_list)
    values(i) = phase_signum.(node_list{i});
end

T = table(node_list, values, 'VariableNames', {'Node', 'Signum'});
writetable(T, output_path);

fprintf("CSV written to: %s\n", output_path);


