function breach_benchmark(varargin)
% BREACH_BENCHMARK Run incremental evaluation benchmarks for STL formulas.
%
% Usage Examples:
%   breach_benchmark('sizeN', 2000, 'approach', 'classic')
%   breach_benchmark('-M', 10, '-freq', 100)
%   breach_benchmark() % Runs with all defaults
%
% Arguments (Name-Value pairs):
%   'sizeN'    : Number of samples (Default: 1000)
%   'freq'     : Samples per second (Default: 1)
%   'M'        : Number of repetitions (Default: 5)
%   'approach' : 'online' or 'classic' (Default: 'online')

% --- 1. PARSE INPUTS ---
p = inputParser;
p.CaseSensitive = false;

% Define defaults
defaultSizeN = 1000;
% defaultFreq = 1;
defaultM = 5;
defaultApproach = 'online';

% Add parameters to parser
addParameter(p, 'sizeN', defaultSizeN);
% addParameter(p, 'freq', defaultFreq);
addParameter(p, 'M', defaultM);
addParameter(p, 'approach', defaultApproach);
addParameter(p, 'suppressOutput', false); % Optional flag to suppress output
addParameter(p, 'formula', '');

% Handle hyphens if user inputs them (e.g., '-sizeN' -> 'sizeN')
cleanInputs = varargin;
for k = 1:2:numel(cleanInputs)
    if ischar(cleanInputs{k}) && startsWith(cleanInputs{k}, '-')
        cleanInputs{k} = cleanInputs{k}(2:end);
    end
end

% Parse the inputs
parse(p, cleanInputs{:});

% Assign to variables
sizeN = p.Results.sizeN;
% freq = p.Results.freq;
M = p.Results.M;
approach = p.Results.approach;

% --- 2. LOGGING SETUP ---
log_file = 'breach_benchmark_results.log';
fid = fopen(log_file, 'a');
fprintf(fid, '--- New Benchmark Run ---\n');
fprintf(fid, 'Timestamp: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, 'Parameters: sizeN=%d, M=%d, approach=''%s''\n', sizeN, M, approach);
fclose(fid); % Close immediately after write to save

% --- 3. DEFINE FORMULA ---
% G[30, 100] ((((x < 30) && (x > -30)) && ((x < 0.5) && (x > -0.5))) -> (F[0, 50](G[0, 20]((x < 0.5) && (x > -0.5)))))
if ~isempty(p.Results.formula)
    phi_long = p.Results.formula;
else
    % phi_long = 'alw_[30, 100] ((((x[t] < 30) and (x[t] > -30)) and ((x[t] < 0.5) and (x[t] > -0.5))) => (ev_[0, 50](alw_[0, 20]((x[t] < 0.5) and (x[t] > -0.5)))))';
    phi_long = '(x[t]<0.5) and (x[t]>-0.5)';
end

phi = STL_Formula('phi', phi_long);

horizon = get_horizon(phi);
if ~p.Results.suppressOutput
    fprintf('Formula horizon: %.2f\n', horizon);
end

% force horizon to be at least 1
horizon = max(horizon, 1);

% Setup Breach System
Bdata = BreachTraceSystem({'x'});
P_online = CreateParamSet(Bdata.Sys);

% Generate Trace
% trace = get_long_signal(sizeN, freq, 'x');
trace = get_signal_from_file(sizeN);
N = size(trace,1);

% --- 4. RUN BENCHMARK ---
elapsed_times = zeros(M,1);
all_means = zeros(M,1);

for run = 1:M
    if ~p.Results.suppressOutput
        fprintf('Starting run %d/%d (Method: %s, N: %d)...\n', run, M, approach, sizeN);
    end

    % Start timing this run
    t_start = tic;

    for k = 2:N
        % Simulate incremental data arrival
        start = max(1, k - ceil(horizon)); % start index considering horizon

        partial = trace(start:k,:);

        % Extract partial signals
        t_partial = partial(:,1)';
        X_partial = partial(:,2:end)';

        % Build Trajectory Struct
        traj.time = t_partial;
        traj.X = X_partial;

        try
            % Evaluate
            if strcmpi(approach, 'classic')
                % Classic requires struct input, not cell
                STL_Eval(Bdata.Sys, phi, P_online, traj, approach);
            else
                % Online/Thom can handle cell or struct
                STL_Eval(Bdata.Sys, phi, P_online, {traj}, approach);
            end

        catch ME
            % Fail silently or warn during benchmark
            % if ~strcmp(ME.message, "Index exceeds the number of array elements. Index must not exceed 1.")
            warning('Eval failed at step %d: %s', k, ME.message);
            % end
        end
    end

    % Store timing
    elapsed_times(run) = toc(t_start);
    all_means(run) = elapsed_times(run);

    if ~p.Results.suppressOutput
        fprintf('Run %d/%d finished: elapsed %.6f s\n', run, M, elapsed_times(run));
    end
end

% --- 5. AGGREGATE & SAVE CSV ---
mean_elapsed = mean(elapsed_times);
std_elapsed = std(elapsed_times);


if ~p.Results.suppressOutput
    fprintf('\nFinished %d incremental evaluations.\n', M);
    fprintf('Average elapsed time: %.6f s (std %.6f s)\n', mean_elapsed, std_elapsed);
end

% Log file CSV
log_file_csv = 'breach_benchmark_results.csv';
write_header = ~exist(log_file_csv, 'file');
fid_csv = fopen(log_file_csv, 'a');

if write_header
    fprintf(fid_csv, 'timestamp,sizeN,M,approach,mean_elapsed_s,std_elapsed_s,formula,all_means\n');
end

timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
fprintf(fid_csv, '%s,%d,%d,%s,%.6f,%.6f,%s,"%s"\n', ...
    timestamp, sizeN, M, approach, mean_elapsed, std_elapsed, ['"' phi_long '"'], mat2str(all_means));

fclose(fid_csv);
end

% --- local helper functions ---
% function trace = get_long_signal(sizeN, freq, ~)
% t = (0:(sizeN-1)) / double(freq);
% vals = mod(0:(sizeN-1), 10);
% trace = [t' vals'];
% end

function trace = get_signal_from_file(sizeN)
% load from signals/signal_< sizeN >.csv
file_name = sprintf('signals/signal_%d.csv', sizeN);
if exist(file_name, 'file')
    data = readmatrix(file_name);
    trace = data(1:sizeN, :);
else
    error('Signal file %s does not exist.', file_name);
end
end