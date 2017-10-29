function [x_future] = predict_future_AR(a_coeff, p , x_init, stopafter)

x_init_future = zeros(stopafter + p, 1);
for i = 1 : p
    x_init_future(i) = x_init(i);
end

for t = p + 1 : p + stopafter
    for i = 1 : p
        x_init_future(t) = x_init_future(t) + x_init_future(t - i) * a_coeff(i); 
    end
end

x_future = x_init_future(p+1:end, 1);

end