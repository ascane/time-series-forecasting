% AIMS CDT Signal Processing End of Module Open Assignment
% Time series forecasting

%% 1.
% mg.mat
% Mackey-Glass chaotic system
% t_tr, t_te

%% Mackey-Glass Chaotic System --  Gaussian Process
n_tr = 800; n_te = 200;
x = (1:n_tr)'; y = t_tr;
xx = (n_tr+1:n_tr+n_te)'; yy = t_te;
mean_y = mean(y);

k1 = @covSEiso;
k2 = {@covProd, {@covPeriodic, @covSEisoU}};
k3 = @covRQiso;
k4 = {@covSum, {@covSEiso, @covNoise}};
covfunc = {@covSum, {k1, k2, k3, k4}};

hyp.cov = [1.96984395453684,5.36244414033112,-3.40777816380627e-14,-3.04853799261264e-07,0.986987132307878,3.99868897183809,1.80029837408984,4.52006594173325,0.797890187651848,-1.99999999957095,-1.57161390650341,-5.04403656039070]; hyp.lik = -2;

[hyp, fX, i] = minimize(hyp, @gp, -500, @infExact, [], covfunc, @likGauss, x, y-mean_y);

zz = xx;
[mu, s2] = gp(hyp, @infExact, [], covfunc, @likGauss, x, y-mean_y, zz);

hold on;
plot(x,y,'b.');
plot(xx,yy,'k.');

plot(zz, mu+mean_y, 'r.');
f = [mu+2*sqrt(s2); flip(mu-2*sqrt(s2),1)] + mean_y;
fill([zz; flip(zz,1)], f, 'r', 'linestyle','none');
alpha(0.25)
lgnd = legend('train', 'test', 'prediction', '±2SD', 'Location','southoutside');
set(lgnd, 'color', 'none');

n_te = size(yy,1);
yy_pred = mu(1:n_te)+mean_y;
rms_err = rms(yy_pred - yy);

%% Mackey-Glass Chaotic System --  Kalman filter
p = 60;
x_init = zeros(p,1);
P_init = eye(p);
n_tr = size(t_tr,1);
n_te = size(t_te,1);
F = eye(p);
mean_tr = mean(t_tr);

% experiments on F
% dt = 10^(-5);
% for i = 1:p-1
%     for j = i+1:p
%         F(i,j)=dt^(j-i);
%     end
% end

Q = 6*10^(-7);
R = 1.7;

[t_pred, sigma_pred, x_pred, P_pred] = kalman_ar(t_tr-mean_tr, x_init, P_init, F, Q, R, n_tr, n_te);
plot(1:n_tr, t_tr, 'b');
hold on
% plot(1:n_tr, t_tr_noisy, 'b+');
plot(n_tr+1:n_tr+n_te, t_te, 'k');
plot(n_tr+1:n_tr+n_te, t_pred(n_tr+1:n_tr+n_te)+mean_tr, 'r');
fill([n_tr+1:n_tr+n_te n_tr+n_te:-1:n_tr+1], [(t_pred(n_tr+1:n_tr+n_te)+mean_tr-1.96*sigma_pred(n_tr+1:n_tr+n_te))' (flipud(t_pred(n_tr+1:n_tr+n_te)+mean_tr+1.96*sigma_pred(n_tr+1:n_tr+n_te)))'],'r','linestyle','none');
alpha(0.25);
lgnd = legend('train', 'test', 'prediction', '±1.96SD', 'Location','southoutside');
set(lgnd, 'color', 'none');

yy_pred = t_pred(n_tr+1:n_tr+n_te)+mean_tr;
rms_err = rms(yy_pred - t_te);

%% Mackey-Glass Chaotic System -- AR
n_tr = 800; n_te = 200;
x = (1:n_tr)'; y = t_tr;
xx = (n_tr+1:n_tr+n_te)'; yy = t_te;

p = 100;
M_temp = lagmatrix(y(1:n_tr), -p+1:1:0);
M = M_temp(1:end-p, :);
X = y(p+1:n_tr);
a_emb = pinv(M)* X;

y_init = y(n_tr-p+1:n_tr);
[y_future_emb] = predict_future_AR(a_emb, p , y_init, n_te);

plot(x, y, 'b');
hold on;
plot(xx, yy, 'k');
plot(xx, y_future_emb, 'r');

lgnd = legend('train', 'test', 'prediction', 'Location','southoutside');
set(lgnd, 'color', 'none');

rms_err = rms(y_future_emb(1:n_te) - yy);

%% 2.
% sunspots.mat
% Sunspot data

%% Sunspot -- Gaussian Process
x = year(year<1950); y = activity(year<1950);
xx = year(year>1950); yy = activity(year>1950);
mean_y = mean(y);

k1 = @covSEiso;
% k2 = {@covProd, {@covPeriodic, @covSEisoU}};
k3 = @covRQiso;
% k4 = {@covSum, {@covSEiso, @covNoise}};
% k5 = @covMaterniso;
k6 = @covNoise;
% covfunc = {@covSum, {k1, k2, k3, k4}};
covfunc = {@covSum, {k1, k3, k6}};

hyp.cov = [4 4 0 0 -1 1]; hyp.lik = -2;

[hyp, fX, i] = minimize(hyp, @gp, -500, @infExact, [], covfunc, @likGauss, x, y-mean_y);

zz = (1950+1/24:1/12:2000-1/24)';
[mu, s2] = gp(hyp, @infExact, [], covfunc, @likGauss, x, y-mean_y, zz);


hold on;
plot(x,y,'b');
plot(xx,yy,'k');
plot(zz,mu+mean(y), 'r');
f = [mu+2*sqrt(s2); flip(mu-2*sqrt(s2),1)] + mean_y;
fill([zz; flip(zz,1)], f, 'r', 'linestyle','none');
alpha(0.25)
lgnd = legend('train', 'test', 'prediction', '±2SD', 'Location','southoutside');
set(lgnd, 'color', 'none');
xlabel('year')
ylabel('sunspot number')

n_te = size(yy,1);
yy_pred = mu(1:n_te)+mean_y;
rms_err = rms(yy_pred - yy);

%% Sunspot -- Kalman filter
x = year(year<1950); y = activity(year<1950);
xx = year(year>1950); yy = activity(year>1950);
n_tr = size(y,1);
n_te = size(yy,1);
p = 150;
a_init = zeros(p,1);
P_init = eye(p);
F = eye(p);
mean_tr = mean(y);

Q = 10^(-7);
R = 20;

[y_pred, sigma_pred, a_pred, P_pred] = kalman_ar(y-mean_tr, a_init, P_init, F, Q, R, n_tr, n_te);
plot(x, y, 'b');
hold on
plot(xx, yy, 'k');
plot(xx, max(0, y_pred(n_tr+1:n_tr+n_te)+mean_tr), 'r');
fill([xx' flip(xx)'], [(max(0,y_pred(n_tr+1:n_tr+n_te)+mean_tr-1.96*sigma_pred(n_tr+1:n_tr+n_te)))' (flipud(max(0,y_pred(n_tr+1:n_tr+n_te)+mean_tr)+1.96*sigma_pred(n_tr+1:n_tr+n_te)))'],'r','linestyle','none');
alpha(0.25);
lgnd = legend('train', 'test', 'prediction', '±1.96SD', 'Location','southoutside');
set(lgnd, 'color', 'none');
xlabel('year')
ylabel('sunspot number')

yy_pred = max(0,y_pred(n_tr+1:n_tr+n_te)+mean_tr);
rms_err = rms(yy_pred - yy);

%% Sunspot -- AR
x = year(year<1950); y = activity(year<1950);
xx = year(year>1950); yy = activity(year>1950);
n_tr = size(x,1); n_te = size(xx,1);

p = 800;
M_temp = lagmatrix(y(1:n_tr), -p+1:1:0);
M = M_temp(1:end-p, :);
X = y(p+1:n_tr);
a_emb = pinv(M)* X;

y_init = y(n_tr-p+1:n_tr);
[y_future_emb] = predict_future_AR(a_emb, p , y_init, n_te);

plot(x, y, 'b');
hold on;
plot(xx, yy, 'k');
plot(xx, y_future_emb, 'r');

lgnd = legend('train', 'test', 'prediction', 'Location','southoutside');
set(lgnd, 'color', 'none');
xlabel('year')

rms_err = rms(y_future_emb(1:n_te) - yy);

%% 3. CO2
% co2.mat
% First attempt (Does not work well)
% n = size(co2,1);
% co2_tbl = table((1:n)',co2);
% co2_tbl.Properties.VariableNames = {'time','co2'};
% 
% ts = zeros(n,1);
% for i = 1:n
%     ts(i) = i * i;
% end
% co2_tbl2 = table((1:n)',ts,co2);
% co2_tbl2.Properties.VariableNames = {'time', 'time_square','co2'};
% 
% sigma0 = 0.2;
% kparams0 = [3.5, 6.2];
% co2_gprMdl = fitrgp(co2_tbl,'co2','KernelFunction','squaredexponential');
% co2_gprMdl2 = fitrgp(co2_tbl,'co2','KernelFunction','squaredexponential',...
%      'KernelParameters',kparams0,'Sigma',sigma0);
% co2_gprMdl3 = fitrgp(co2_tbl2,'co2','KernelFunction','squaredexponential');
% co2_gprMdl4 = fitrgp(co2_tbl2,'co2','KernelFunction','squaredexponential','KernelParameters',kparams0,'Sigma',sigma0);
% co2_pred = resubPredict(co2_gprMdl);
% co2_pred2 = resubPredict(co2_gprMdl2);
% co2_pred3 = resubPredict(co2_gprMdl3);
% co2_pred4 = resubPredict(co2_gprMdl4);
% figure();
% plot(co2_tbl.co2,'r');
% hold on
% plot(co2_pred,'b');
% plot(co2_pred2,'g');
% plot(co2_pred3,'m');
% plot(co2_pred4,'y');
% 
% n_test = 100;
% test_tbl = table((n+1:n+n_test)');
% test_tbl.Properties.VariableNames = {'time'};
% ts_test = zeros(n_test,1);
% for i =1:n_test
%     ts_test(i) = (n + i)^2;
% end
% test_tbl2 = table((n+1:n+n_test)', ts_test);
% test_tbl2.Properties.VariableNames = {'time', 'time_square'};
% 
% [ypred, ysd] = predict(co2_gprMdl, test_tbl);
% [ypred2, ysd2] = predict(co2_gprMdl2, test_tbl);
% [ypred3, ysd3] = predict(co2_gprMdl3, test_tbl2);
% [ypred4, ysd4] = predict(co2_gprMdl4, test_tbl2);
% plot(n+1:n+n_test, ypred, 'b');
% plot(n+1:n+n_test, ypred2, 'g');
% plot(n+1:n+n_test, ypred3, 'm');
% plot(n+1:n+n_test, ypred4, 'y');

%% CO2 -- Gaussian Process
z = mauna(:,2) ~= -99.99;
year = mauna(z,1);
co2 = mauna(z,2);

x = year(year<2005); y = co2(year<2005);
xx = year(year>2005); yy = co2(year>2005);
mean_y = mean(y);

k1 = @covSEiso;
k2 = {@covProd, {@covPeriodic, @covSEisoU}};
k3 = @covRQiso;
k4 = {@covSum, {@covSEiso, @covNoise}};
covfunc = {@covSum, {k1, k2, k3, k4}};

hyp.cov = [4 4 0 0 1 4 0 0 -1 -2 -2 1]; hyp.lik = -2;

[hyp, fX, i] = minimize(hyp, @gp, -500, @infExact, [], covfunc, @likGauss, x, y-mean_y);

zz = (2005+1/24:1/12:2050-1/24)';
[mu, s2] = gp(hyp, @infExact, [], covfunc, @likGauss, x, y-mean_y, zz);

hold on;
plot(x,y,'b.');
plot(xx,yy,'k.');

plot(zz, mu+mean_y, 'r.');
f = [mu+2*sqrt(s2); flip(mu-2*sqrt(s2),1)] + mean_y;
fill([zz; flip(zz,1)], f, 'r', 'linestyle','none');
alpha(0.25)
lgnd = legend('train', 'test', 'prediction', '±2SD', 'Location','southoutside');
set(lgnd, 'color', 'none');
xlabel('year')
ylabel('CO2 average (ppm)')

n_te = size(yy,1);
yy_pred = mu(1:n_te)+mean_y;
rms_err = rms(yy_pred - yy);

%% CO2 -- Kalman filter
z = mauna(:,2) ~= -99.99;
year = mauna(z,1);
co2 = mauna(z,2);

x = year(year<2005); y = co2(year<2005);
xx = year(year>2005); yy = co2(year>2005);
n_tr = size(x,1); n_te = size(xx,1);
zz = (2005+1/24:1/12:2050-1/24)';
mean_tr = mean(y);

p = 100;
a_init = zeros(p,1);
P_init = eye(p);
F = eye(p);

Q = 10^(-7);
R = 10;

[y_pred, sigma_pred, a_pred, P_pred] = kalman_ar(y-mean_tr, a_init, P_init, F, Q, R, n_tr, size(zz,1));
plot(x, y, 'b.');
hold on
plot(xx, yy, 'k.');
plot(zz, y_pred(n_tr+1:n_tr+size(zz,1))+mean_tr, 'r.');
fill([zz' flip(zz)'], [(y_pred(n_tr+1:n_tr+size(zz,1))+mean_tr-1.96*sigma_pred(n_tr+1:n_tr+size(zz,1)))' (flipud(y_pred(n_tr+1:n_tr+size(zz,1))+mean_tr+1.96*sigma_pred(n_tr+1:n_tr+size(zz,1))))'],'r','linestyle','none');
alpha(0.25);
lgnd = legend('train', 'test', 'prediction', '±1.96SD', 'Location','southoutside');
set(lgnd, 'color', 'none');
xlabel('year')
ylabel('CO2 average (ppm)')

rms_err = rms(y_pred(n_tr+1:n_tr+n_te)+mean_tr - yy);

%% CO2 -- AR
z = mauna(:,2) ~= -99.99;
year = mauna(z,1);
co2 = mauna(z,2);

x = year(year<2005); y = co2(year<2005);
xx = year(year>2005); yy = co2(year>2005);
n_tr = size(x,1); n_te = size(xx,1);
zz = (2005+1/24:1/12:2050-1/24)';
n_pred = size(zz,1);

p = 100;
M_temp = lagmatrix(y(1:n_tr), -p+1:1:0);
M = M_temp(1:end-p, :);
X = y(p+1:n_tr);
a_emb = pinv(M)* X;

y_init = y(n_tr-p+1:n_tr);
[y_future_emb] = predict_future_AR(a_emb, p , y_init, n_pred);

plot(x, y, 'b.');
hold on;
plot(xx, yy, 'k.');
plot(zz, y_future_emb, 'r.');

lgnd = legend('train', 'test', 'prediction', 'Location','southoutside');
set(lgnd, 'color', 'none');
xlabel('year')
ylabel('CO2 average (ppm)')

rms_err = rms(y_future_emb(1:n_te) - yy);


%% 4.
% finPredProb.mat
% finance

% Common part
n = size(ttr,1);
n_tr = n * 0.8;
n_te = n-n_tr;
x = (1:n_tr)'; xx = (n_tr+1:n)';
y = ttr(x); yy = ttr(xx);
mean_tr = mean(y);

%% Finance -- Kalman filter
p = 100;
a_init = zeros(p,1);
P_init = eye(p);
F = eye(p);
Q = 0.01;
R = 200;

[y_pred, sigma_pred, a_pred, P_pred] = kalman_ar(y-mean_tr, a_init, P_init, F, Q, R, n_tr, n_te);
plot(x, y, 'b');
hold on
plot(xx, yy, 'k');
plot(xx, y_pred(n_tr+1:n_tr+n_te)+mean_tr, 'r');
fill([xx' flip(xx)'], [(y_pred(n_tr+1:n_tr+n_te)+mean_tr-1.96*sigma_pred(n_tr+1:n_tr+n_te))' (flipud(y_pred(n_tr+1:n_tr+n_te)+mean_tr+1.96*sigma_pred(n_tr+1:n_tr+n_te)))'],'r','linestyle','none');
alpha(0.25);
lgnd = legend('train', 'test', 'prediction', '±1.96SD', 'Location','southoutside');
set(lgnd, 'color', 'none');
rms_error = rms(y_pred(n_tr+1:n_tr+n_te)+mean_tr - yy);

%% Finance -- Gaussian Process (Out of memory)
k1 = @covSEiso;
k4 = @covNoise;
covfunc = {@covSum, {k1, k4}};

hyp.cov = [4 4 1]; hyp.lik = -2;

[hyp, fX, i] = minimize(hyp, @gp, -10, @infKL, [], covfunc, @likGauss, x, y-mean_tr);

[mu, s2] = gp(hyp, @infKL, [], covfunc, @likGauss, x, y-mean_tr, xx);

hold on;
plot(x,y,'b.');
plot(xx,yy,'k.');
plot(xx,mu,'r.');
f = [mu+2*sqrt(s2); flip(mu-2*sqrt(s2),1)] + mean_tr;
fill([xx; flip(xx,1)], f, 'r', 'linestyle','none');
alpha(0.25)
lgnd = legend('train', 'test', 'prediction', '±2SD', 'Location','southoutside');
set(lgnd, 'color', 'none');

n_te = size(yy,1);
yy_pred = mu(1:n_te)+mean_tr;
rms_err = rms(yy_pred - yy);

%% Finance -- AR
p = 10000;
M_temp = lagmatrix(y(1:n_tr), -p+1:1:0);
M = M_temp(1:end-p, :);
X = y(p+1:n_tr);
a_emb = pinv(M)* X;

y_init = y(n_tr-p+1:n_tr);
[y_future_emb] = predict_future_AR(a_emb, p , y_init, n_te);

plot(x, y, 'b');
hold on;
plot(xx, yy, 'k');
plot(xx, y_future_emb, 'r');

lgnd = legend('train', 'test', 'prediction', 'Location','southoutside');
set(lgnd, 'color', 'none');
xlabel('year')

rms_err = rms(y_future_emb(1:n_te) - yy);
