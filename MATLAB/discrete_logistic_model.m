function P = discrete_logistic_model(P0, r, K, t_max)
%DISCRETE_LOGISTIC_MODEL Euler-form logistic map over integer steps.
validateattributes(P0, {'numeric'}, {'scalar','nonnegative'});
validateattributes(r, {'numeric'}, {'scalar','finite'});
validateattributes(K, {'numeric'}, {'scalar','positive'});
validateattributes(t_max, {'numeric'}, {'scalar','integer','nonnegative'});
P = zeros(t_max+1,1); P(1)=P0;
for n=1:t_max
    P(n+1)=P(n)+r*P(n)*(1-P(n)/K);
end
end
