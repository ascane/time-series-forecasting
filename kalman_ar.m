function [t_pred, sigma_pred, x_pred, P_pred] = kalman_ar(t_tr, x_init, P_init, F, Q, R, n_tr, n_te)

n = n_tr + n_te;
p = size(x_init, 1);
t_pred = zeros(n,1);
sigma_pred = zeros(n,1);
for i = 1:n_tr
    t_pred(i) = t_tr(i);
end
x_pred = zeros(n, p);
P_pred = zeros(n, p, p);
x_post = x_init;
P_post = P_init;
for i = p+1:n
    if i <= n_tr
        x_prior = F * x_post;
        P_prior = F * P_post * F' + Q;
        H = t_tr(i-p:i-1)';
        K = P_prior * H' * (H * P_prior * H' + R)^(-1);
        x_post = x_prior + K * (t_tr(i) - H * x_prior);
        P_post = P_prior - K * H * P_prior;
    end
    
    x_pred(i, :) = x_post;
    P_pred(i, :, :) = P_post;
    if i > n_tr
        t_pred(i) = t_pred(i-p:i-1)' * x_post;
        sigma_pred(i) = sqrt(t_pred(i-p:i-1)' * P_post * t_pred(i-p:i-1) + R);
    end
end

end
