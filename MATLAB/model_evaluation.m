function metrics = model_evaluation(y, yhat)
%MODEL_EVALUATION Regression metrics (MAE, MSE, RMSE, MAPE percent, R2).
y=y(:); yhat=yhat(:);
if numel(y)~=numel(yhat) || isempty(y) || any(~isfinite(y)) || any(~isfinite(yhat))
    error('Inputs must be finite vectors of equal nonzero length.');
end
e=y-yhat; mse=mean(e.^2);
metrics=struct('MAE',mean(abs(e)),'MSE',mse,'RMSE',sqrt(mse), ...
    'MAPE_percent',mean(abs(e(y~=0)./y(y~=0)))*100, ...
    'R2',1-sum(e.^2)/sum((y-mean(y)).^2));
if all(y==y(1)), metrics.R2=NaN; end
end
