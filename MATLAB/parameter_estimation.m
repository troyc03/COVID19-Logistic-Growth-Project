function [params, covariance, fitted] = parameter_estimation(t, y)
%PARAMETER_ESTIMATION Fit logistic parameters [K P0 r], fixing P0 to y(1).
% Uses fminsearch on transformed K and r so both remain positive and K>max(y).
t=t(:); y=y(:);
if numel(t)~=numel(y) || numel(y)<4 || any(~isfinite(t)) || any(~isfinite(y)) || any(y<=0) || any(diff(t)<=0)
    error('Inputs must be matching finite arrays, >=4 positive observations, and increasing time.');
end
P0=y(1); ymax=max(y);
model=@(K,r,x) K./(1+((K-P0)./P0).*exp(-r.*x));
K0=max(1.2*ymax,ymax+1); r0=.15;
objective=@(z) sum((model(ymax+exp(z(1)),exp(z(2)),t)-y).^2);
z0=[log(K0-ymax),log(r0)];
opts=optimset('MaxFunEvals',50000,'MaxIter',50000,'Display','off');
z=fminsearch(objective,z0,opts);
K=ymax+exp(z(1)); r=exp(z(2));
params=[K,P0,r]; fitted=model(K,r,t);
resid=y-fitted; dof=max(numel(y)-2,1);
J=zeros(numel(t),2);
for j=1:2
    delta=sqrt(eps)*max(abs(z(j)),1);
    zp=z; zm=z; zp(j)=zp(j)+delta; zm(j)=zm(j)-delta;
    kp=ymax+exp(zp(1)); rp=exp(zp(2)); km=ymax+exp(zm(1)); rm=exp(zm(2));
    J(:,j)=(model(kp,rp,t)-model(km,rm,t))/(2*delta);
end
covariance=(sum(resid.^2)/dof)*pinv(J'*J);
end
