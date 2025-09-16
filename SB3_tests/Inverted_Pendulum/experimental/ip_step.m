function [obs, reward, done, t] = ip_step(model, u, angle_limit)
% One exact fixed-step tick while the model is paused.

% Update action
assignin('base','u_hold',double(u));

% Advance exactly one step
set_param(model,'SimulationCommand','step');

% Read paths saved by ip_reset
angle_block  = evalin('base','ip_angle_block');
angvel_block = evalin('base','ip_angvel_block');

% Read current values from the To Workspace blocks
var1 = get_param(angle_block, 'VariableName');
var2 = get_param(angvel_block, 'VariableName');
temp_theta  = evalin('base', var1);
temp_thetad = evalin('base', var2);
theta       = temp_theta(end, end);   % Selects the signal value from the last row
theta_dot   = temp_thetad(end, end);  % Selects the signal value from the last row

obs = [theta; theta_dot];
reward = cos(theta);
done = abs(theta) > angle_limit;
t = double(get_param(model, 'SimulationTime'));
end
