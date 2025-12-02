% bench_all.m
% Wrapper script to run a comprehensive suite of benchmarks using breach_benchmark.m.
% This script iterates through multiple trace lengths and approaches to generate
% data for performance analysis (e.g., Time vs Input Size).

% --- 1. CONFIGURATION ---
% Define the variations you want to test
approaches = {'online', 'classic', 'thom'};    % Compare specific implementations
% approaches = {'online'};
trace_lengths = [5000, 10000, 20000]; % Varying input sizes (N)
% trace_lengths = [1000]; % Varying input sizes (N)
formula_ids = 1:21;                  % Different formula complexities
reps_per_config = 50;                   % M: Number of repetitions for averaging

% Results file (to clear before starting)
csv_file = 'breach_benchmark_results.csv';
log_file = 'breach_benchmark_results.log';

% --- 2. CLEANUP ---
fprintf('Preparing benchmark environment...\n');
if exist(csv_file, 'file')
    delete(csv_file);
    fprintf('  - Deleted old CSV results: %s\n', csv_file);
end
if exist(log_file, 'file')
    delete(log_file);
    fprintf('  - Deleted old log file: %s\n', log_file);
end

% --- 3. EXECUTION LOOP ---
total_experiments = length(approaches) * length(trace_lengths) * length(formula_ids);
current_exp = 0;

fprintf('======================================================\n');
fprintf('STARTING BENCHMARK SUITE\n');
fprintf('Total Experiments: %d (Approaches: %d, Sizes: %d)\n', ...
    total_experiments, length(approaches), length(trace_lengths));
fprintf('======================================================\n');


%outer dummy loop to allow progress bar over all experiments
for f_id = 1:length(formula_ids)
    formula_id = formula_ids(f_id);
    fprintf('\n--- Running Benchmarks for Formula ID: %d ---\n', formula_id);

    phi = get_formula_by_id(formula_id);
    fprintf('Using formula: %s\n', phi);

    for i = 1:length(approaches)
        app = approaches{i};

        for j = 1:length(trace_lengths)
            sz = trace_lengths(j);

            current_exp = current_exp + 1;

            fprintf('\n[Experiment %d/%d] Approach: %s | N: %d \n', ...
                current_exp, total_experiments, app, sz);

            try
                % Call the benchmark function using Name-Value syntax
                % This calls the breach_benchmark.m function you created earlier
                breach_benchmark('approach', app, ...
                    'sizeN', sz, ...
                    'M', reps_per_config, ...
                    'suppressOutput', true, ...
                    'formula', phi); % Suppress detailed output for brevity

            catch ME
                fprintf(2, '  >> ERROR encountered in experiment %d: %s\n', current_exp, ME.message);
                % We continue to the next experiment even if one fails

            end
        end
    end
end

fprintf('\n======================================================\n');
fprintf('SUITE COMPLETED.\n');
fprintf('Results saved to: %s\n', csv_file);
fprintf('======================================================\n');

% Optional: Display a quick summary of the CSV if it exists
if exist(csv_file, 'file')
    fprintf('\n--- CSV Preview ---\n');
    type(csv_file);
end

function f = get_formula_by_id(formula_id)
phi1 = 'x[t]<0.5';
phi2 = 'x[t]>-0.5';
switch formula_id
    case 1 % and
        f = sprintf('(%s) and (%s)', phi1, phi2);
    case 2 % or
        f = sprintf('(%s) or (%s)', phi1, phi2);
    case 3 % not
        f = sprintf('not (%s)', phi1);
    case 4 % alw_[0,10]
        f = sprintf('alw_[0,10] (%s)', phi1);
    case 5 % alw_[0,100]
        f = sprintf('alw_[0,100] (%s)', phi1);
    case 6 % alw_[0,1000]
        f = sprintf('alw_[0,1000] (%s)', phi1);
    case 7 % ev_[0,10]
        f = sprintf('ev_[0,10] (%s)', phi1);
    case 8 % ev_[0,100]
        f = sprintf('ev_[0,100] (%s)', phi1);
    case 9 % ev_[0,1000]
        f = sprintf('ev_[0,1000] (%s)', phi1);
    case 10 % until_[0,10]
        f = sprintf('(%s) until_[0,10] (%s)', phi1, phi2);
    case 11 % until_[0,100]
        f = sprintf('(%s) until_[0,100] (%s)', phi1, phi2);
    case 12 % until_[0,1000]
        f = sprintf('(%s) until_[0,1000] (%s)', phi1, phi2);
    case 13 % branching, depth = 10
        f = nested_formula(10, 'branching');
    case 14 % branching, depth = 20
        f = nested_formula(20, 'branching');
    case 15 % branching, depth = 30
        f = nested_formula(30, 'branching');
    case 16 % alternating, depth = 10
        f = nested_formula(10, 'alternating');
    case 17 % alternating, depth = 20
        f = nested_formula(20, 'alternating');
    case 18 % alternating, depth = 30
        f = nested_formula(30, 'alternating');
    case 19 % until, depth = 10
        f = nested_formula(10, 'until');
    case 20 % until, depth = 20
        f = nested_formula(20, 'until');
    case 21 % until, depth = 30
        f = nested_formula(30, 'until');
    otherwise
        error('Unknown Formula ID: %d', formula_id);
end
end

function f = nested_formula(depth, pattern)
% NESTED_FORMULA Generates STL formulas with specified nesting depth.
%
% Usage:
%   f = nested_formula(5, 'alternating') % alw(ev(alw(...)))
%   f = nested_formula(5, 'branching')   % phi and (ev (phi ...))
%   f = nested_formula(5, 'until')       % phi until (phi until ...)
%
% Inputs:
%   depth   - (int) Depth of the nesting.
%             Depth 1 returns the atomic predicate.
%   pattern - (string)
%             'alternating' : Unary chain: alw_[0,10] ( ev_[0,10] (...) )
%             'branching'   : Binary logic chain: phi and ( ev_[0,10] (...) )
%             'until'       : Binary temporal chain: phi until_[0,10] ( ... )
%
% Returns:
%   f       - (string) The constructed STL formula string.

if depth < 1
    error('Depth must be at least 1.');
end

% Define atomic propositions (leaf nodes)
% Using two slightly different predicates can help debug trace logic
phi_A = '(x[t] > 0.0)';
phi_B = '(x[t] < 0.5)';

% Initialize f with the base case (Leaf)
f = phi_A;

% If depth is 1, return just the base predicate
if depth == 1
    return;
end

switch lower(pattern)
    case 'alternating'
        % PATTERN: Unary chain
        % alw_[0,10] ( ev_[0,10] ( ... ) )
        for i = 1:(depth - 1)
            if mod(i, 2) ~= 0
                f = sprintf('ev_[0,10] (%s)', f);
            else
                f = sprintf('alw_[0,10] (%s)', f);
            end
        end

    case 'branching'
        % PATTERN: Boolean + Temporal
        % phi and ( ev_[0,10] ( ... ) )
        for i = 1:(depth - 1)
            f = sprintf('%s and (ev_[0,10] (%s))', phi_B, f);
        end

    case 'until'
        % PATTERN: Binary Temporal Chain
        % phi_A until_[0,10] ( phi_A until_[0,10] ( ... ) )
        %
        % This tests "Depth + Height" because Until is binary.
        % It forces the solver to track the validity of LHS (phi_B)
        % while searching for the satisfaction of the nested RHS.
        for i = 1:(depth - 1)
            f = sprintf('(%s) until_[0,10] (%s)', phi_B, f);
        end

    otherwise
        error('Unknown pattern: "%s". Use "alternating", "branching", or "until".', pattern);
end
end