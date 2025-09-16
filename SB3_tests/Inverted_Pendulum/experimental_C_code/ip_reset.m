function obs = ip_reset(model, theta0, thetaDot0, dt, angle_block, angvel_block)
% Start/clean the sim and return initial obs.
% angle_block / angvel_block can be either:
%   - full path:  'pendulum_core/angle'
%   - short name: 'angle'  (we'll prefix model/ for you)

% --- ensure model is loaded ---
load_system(model);

% --- resolve block paths to full 'model/block' form (cheap + robust) ---
angle_block  = resolve_block_path(model, angle_block);
angvel_block = resolve_block_path(model, angvel_block);

% stash full paths for ip_step
assignin('base','ip_angle_block', angle_block);
assignin('base','ip_angvel_block', angvel_block);

% --- base workspace params ---
assignin('base','Ts',dt);
assignin('base','theta0',theta0);
assignin('base','thetaDot0',thetaDot0);
assignin('base','u_hold',0.0);

% --- run/pause/prime ---
if ~strcmp(get_param(model, 'SimulationStatus'), 'paused')
    % First run: configure model for fast restart
    set_param(model, 'StopTime','inf', 'FastRestart','on');
end
try, set_param(model,'SimulationCommand','stop'); end %#ok<TRYNC>
set_param(model,'SimulationCommand','start');
set_param(model,'SimulationCommand','pause');
set_param(model,'SimulationCommand','step');

% --- read initial obs from To Workspace blocks ---
var1 = get_param(angle_block, 'VariableName');
var2 = get_param(angvel_block, 'VariableName');
temp_ang  = evalin('base', var1);
temp_angv = evalin('base', var2);
ang       = temp_ang(end, end);  % Selects the signal value from the last row
angv      = temp_angv(end, end); % Selects the signal value from the last row
obs       = [ang; angv];
end

function full = resolve_block_path(model, blockPathOrName)
% Return a full path 'model/name' if only 'name' was provided; otherwise validate.
    full = blockPathOrName;
    if ~contains(full, '/')
        % looks like a bare block name; prefix with model/
        candidate = [model '/' full];
        if exist_block(candidate)
            full = candidate; return;
        end
    end
    % if it already contains '/', just validate; if validation fails, throw a clear error
    if ~exist_block(full)
        error("ip_reset:badBlock", "Invalid block path: '%s' (model: %s)", full, model);
    end
end

function tf = exist_block(pathstr)
% true if 'pathstr' is a valid block in memory
    try
        get_param(pathstr,'Handle');  %#ok<VUNUS>
        tf = true;
    catch
        tf = false;
    end
end