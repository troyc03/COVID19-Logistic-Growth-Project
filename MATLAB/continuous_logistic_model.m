function [P, dP] = continuous_logistic_model(t, P0, r, K)
%CONTINUOUS_LOGISTIC_MODEL Analytical solution and derivative of logistic ODE.
% Inputs: t vector, initial population P0, growth rate r, carrying capacity K.
validateattributes(P0, {'numeric'}, {'scalar','positive'});
validateattributes(r, {'numeric'}, {'scalar','positive'});
validateattributes(K, {'numeric'}, {'scalar','positive'});
if P0 >= K, error('Require P0 < K.'); end
t = double(t);
P = K ./ (1 + ((K-P0)./P0).*exp(-r.*t));
dP = r .* P .* (1-P./K);
end
