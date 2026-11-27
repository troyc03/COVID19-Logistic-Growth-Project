function [future_t, future_pred, params, covariance] = forecasting(t, y, horizon)
%FORECASTING Fit logistic curve and forecast horizon integer time steps.
if nargin<3, horizon=30; end
validateattributes(horizon, {'numeric'}, {'scalar','integer','nonnegative'});
[params,covariance]=parameter_estimation(t,y);
future_t=(max(t)+1:max(t)+horizon)';
K=params(1); P0=params(2); r=params(3);
future_pred=K./(1+((K-P0)./P0).*exp(-r.*future_t));
end
