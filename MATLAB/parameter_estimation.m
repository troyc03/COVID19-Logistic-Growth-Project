function [params, covariance, fitted] = parameter_estimation(t, y)
%PARAMETER_ESTIMATION Nonlinear least-squares estimate [K P0 r].
t=t(:); y=y(:);
if numel(t)~=numel(y) || numel(y)<4 || any(~isfinite(t)) || any(~isfinite(y)) || any(y<=0) || any(diff(t)<=0)
    error('Inputs must be matching finite arrays, >=4 positive observations, and increasing time.');
end
model=@(p,x) p(1)./(1+((p(1)-p(2))./p(2)).*exp(-p(3).*x));
p0=[max(1.2*max(y),1.5*y(1)),y(1),0.15];
objective=@(p) sum((model(p,t)-y).^2);
opts=optimset('MaxFunEvals',50000,'MaxIter',50000,'Display','off');
params=fminsearch(objective,p0,opts);
if params(1)<=max(y) || params(2)<=0 || params(2)>=params(1) || params(3)<=0
    error('Fit did not yield valid positive logistic parameters [K P0 r].');
end
fitted=model(params,t);
resid=y-fitted; dof=max(numel(y)-3,1);
J=zeros(numel(t),3);
for j=1:3
    delta=sqrt(eps)*max(abs(params(j)),1);
    pp=params; pm=params; pp(j)=pp(j)+delta; pm(j)=pm(j)-delta;
    J(:,j)=(model(pp,t)-model(pm,t))/(2*delta);
end
covariance=(sum(resid.^2)/dof)*pinv(J'*J);
end
