% Создание трехмерной траектории
t = 0:0.02:10;
x_0 = sin(t);
y_0 = cos(t);
z_0 = t;
trajectory=[x_0; y_0; z_0]; % массив с тремя координатами
SNR_dB=55;
%SNR_dB = randi([50, 100]); % случайное целое число от 50 до 100
SNR = 10^(SNR_dB/10);
noise = sqrt(1/SNR)*randn(size(trajectory)); % 
noisy_signal = trajectory + noise; %наложение белого шума на траекторию
figure;
subplot(2,2,1);
plot3(noisy_signal(1,:), noisy_signal(2,:), noisy_signal(3,:), 'r'); % сигнал с шумом
title('Trajectory with applied noise');
data_coord_won = [x_0', y_0', z_0']; %массив координат без шума
x_1=noisy_signal(1,:);
y_1=noisy_signal(2,:);
z_1=noisy_signal(3,:);
data_coord_wn = [x_1',y_1',z_1']; %массив координат с шумом
%plot3(x_0,y_0,z_0);
% нахождение скоростей
vx_won = diff(x_0) ./ diff(t); % скорость по x
vy_won = diff(y_0) ./ diff(t); % скорость по y
vz_won = diff(z_0) ./ diff(t); % скорость по z

vx_wn = diff(x_1) ./ diff(t); % скорость по x
vy_wn = diff(y_1) ./ diff(t); % скорость по y
vz_wn = diff(z_1) ./ diff(t); % скорость по z

vx_won=[zeros(1),vx_won];
vy_won=[zeros(1),vy_won];
vz_won=[zeros(1),vz_won];

vx_wn=[zeros(1),vx_wn];
vy_wn=[zeros(1),vy_wn];
vz_wn=[zeros(1),vz_wn];

% нахождение ускорений
% идеальные ускорения
ax_won = diff(vx_won) ./ diff(t); 
ay_won = diff(vy_won) ./ diff(t); 
az_won = diff(vz_won) ./ diff(t); 

ax_won=[zeros(1),ax_won];
ay_won=[zeros(1),ay_won];
az_won=[zeros(1),az_won];

%зашумленные ускорения
ax_wn = diff(vx_wn) ./ diff(t); % ускорение по x
ay_wn = diff(vy_wn) ./ diff(t); % ускорение по y
az_wn = diff(vz_wn) ./ diff(t); % ускорение по z

ax_wn=[zeros(1),ax_wn];
ay_wn=[zeros(1),ay_wn];
az_wn=[zeros(1),az_wn];

% нахождение угловых скоростей
% идеальные угловые скорости
wx_won = (vy_won(2:end).*vz_won(1:end-1) - vz_won(2:end).*vy_won(1:end-1)) ./ (vx_won(2:end).^2 + vy_won(2:end).^2 + vz_won(2:end).^2);
wy_won = (vz_won(2:end).*vx_won(1:end-1) - vx_won(2:end).*vz_won(1:end-1)) ./ (vx_won(2:end).^2 + vy_won(2:end).^2 + vz_won(2:end).^2);
wz_won = (vx_won(2:end).*vy_won(1:end-1) - vy_won(2:end).*vx_won(1:end-1)) ./ (vx_won(2:end).^2 + vy_won(2:end).^2 + vz_won(2:end).^2);

data_acc_won = [ax_won', ay_won', az_won'];
% зашумленные угловые скорости
wx_wn = (vy_wn(2:end).*vz_wn(1:end-1) - vz_wn(2:end).*vy_wn(1:end-1)) ./ (vx_wn(2:end).^2 + vy_wn(2:end).^2 + vz_wn(2:end).^2);
wy_wn = (vz_wn(2:end).*vx_wn(1:end-1) - vx_wn(2:end).*vz_wn(1:end-1)) ./ (vx_wn(2:end).^2 + vy_wn(2:end).^2 + vz_wn(2:end).^2);
wz_wn = (vx_wn(2:end).*vy_wn(1:end-1) - vy_wn(2:end).*vx_wn(1:end-1)) ./ (vx_wn(2:end).^2 + vy_wn(2:end).^2 + vz_wn(2:end).^2);

dataset_acc=[ax_wn',ay_wn',az_wn',ax_won',ay_won',az_won'];% создание датасета, содержащего данные об ускорениях
%удаление выбросов из датасета
dataEdu_acc=dataset_acc(1:round(size(dataset_acc,1)/2),:);
q1 = prctile(dataset_acc,25);
q3 = prctile(dataset_acc,75);
iqr = q3 - q1;
lower = q1 - 1.5*iqr;
upper = q3 + 1.5*iqr;
% удаление строк с выбросами
id_outlies_angular = any(dataset_acc < lower | dataset_acc > upper, 2);
dataset_acc = dataset_acc(~id_outlies_angular, :);
%создание данных для обучения и тестирования (для ускорений)
dataTest_acc=dataset_acc(round(size(dataset_acc,1)/2)+1:end,:);
dataEdu_ax=dataEdu_acc(:,1:4);
dataTest_ax=dataTest_acc(:,1:4);
dataEdu_ay=dataEdu_acc(:,[1:3,5]);
dataTest_ay=dataTest_acc(:,[1:3,5]);
dataEdu_az=dataEdu_acc(:,[1:3,6]);
dataTest_az=dataTest_acc(:,[1:3,6]);
dataOutput_acc=dataset_acc(:,1:3);

%загрузка и тестирование ANFIS
fis_ax=readfis('fis_ax.fis');
fis_ax.input(1).range = [min(dataOutput_acc(:,1)) max(dataOutput_acc(:,1))]; 
fis_ax.input(2).range = [min(dataOutput_acc(:,2)) max(dataOutput_acc(:,2))]; 
fis_ax.input(3).range = [min(dataOutput_acc(:,3)) max(dataOutput_acc(:,3))]; 
fisOutput_ax=evalfis(fis_ax,dataOutput_acc);
subplot(2,2,2);
plot(t,ax_wn,'r');
hold on;
plot(t(1:length(fisOutput_ax)),fisOutput_ax,'b');
hold on;
plot(t,ax_won);
title('a_x Testing');

fis_ay=readfis('fis_ay.fis');
fis_ay.input(1).range = [min(dataOutput_acc(:,1)) max(dataOutput_acc(:,1))]; 
fis_ay.input(2).range = [min(dataOutput_acc(:,2)) max(dataOutput_acc(:,2))]; 
fis_ay.input(3).range = [min(dataOutput_acc(:,3)) max(dataOutput_acc(:,3))]; 
fisOutput_ay=evalfis(fis_ay,dataOutput_acc);
subplot(2,2,3);
plot(t,ay_wn,'r');
hold on;
plot(t,ay_won);
hold on;
plot(t(1:length(fisOutput_ay)),fisOutput_ay,'b');
title('a_y Testing');

fis_az=readfis('fis_az.fis');
fis_az.input(1).range = [min(dataOutput_acc(:,1)) max(dataOutput_acc(:,1))]; 
fis_az.input(2).range = [min(dataOutput_acc(:,2)) max(dataOutput_acc(:,2))]; 
fis_az.input(3).range = [min(dataOutput_acc(:,3)) max(dataOutput_acc(:,3))]; 
fisOutput_az=evalfis(fis_az,dataOutput_acc);
subplot(2,2,4);
plot(t,az_wn,'r');
hold on;
plot(t,az_won);
hold on;
plot(t(1:length(fisOutput_az)),fisOutput_az,'b');
title('a_z Testing');

dataset_angular=[wx_wn',wy_wn',wz_wn',wx_won',wy_won',wz_won'];% создание датасета, содержащего данные об угловых скоростях
dataEdu_angular=dataset_angular(1:round(size(dataset_angular,1)/2),:);
%удаление выбросов
q1 = prctile(dataset_angular,25);
q3 = prctile(dataset_angular,75);
iqr = q3 - q1;
lower = q1 - 1.5*iqr;
upper = q3 + 1.5*iqr;
% удаление строк с выбросами
id_outlies_angular = any(dataset_acc < lower | dataset_acc > upper, 2);
dataset_acc = dataset_acc(~id_outlies_angular, :);
%создание данных для обучения и тестирования (для угловых скоростей)
dataTest_angular=dataset_angular(round(size(dataset_angular,1)/2)+1:end,:);
dataEdu_wx=dataEdu_angular(:,1:4);
dataTest_wx=dataTest_angular(:,1:4);
dataEdu_wy=dataEdu_angular(:,[1:3,5]);
dataTest_wy=dataTest_angular(:,[1:3,5]);
dataEdu_wz=dataEdu_angular(:,[1:3,6]);
dataTest_wz=dataTest_angular(:,[1:3,6]);
dataOutput_angular=dataset_angular(:,1:3);

%загрузка и тестирование ANFIS
figure;
fis_wx=readfis('fis_wx.fis');
fis_wx.input(1).range = [min(dataOutput_angular(:,1)) max(dataOutput_angular(:,1))]; 
fis_wx.input(2).range = [min(dataOutput_angular(:,2)) max(dataOutput_angular(:,2))]; 
fis_wx.input(3).range = [min(dataOutput_angular(:,3)) max(dataOutput_angular(:,3))]; 
fisOutput_wx=evalfis(fis_wx,dataOutput_angular);
subplot(2,2,1)
plot(t(1:length(wx_wn)),wx_wn,'r');
hold on;
plot(t(1:length(fisOutput_wx)),fisOutput_wx,'b');
hold on;
plot(t(1:length(wx_won)),wx_won);
title('w_x Testing');

fis_wy=readfis('fis_wy.fis');
fis_wy.input(1).range = [min(dataOutput_angular(:,1)) max(dataOutput_angular(:,1))]; 
fis_wy.input(2).range = [min(dataOutput_angular(:,2)) max(dataOutput_angular(:,2))]; 
fis_wy.input(3).range = [min(dataOutput_angular(:,3)) max(dataOutput_angular(:,3))]; 
fisOutput_wy=evalfis(fis_wy,dataOutput_angular);
subplot(2,2,2)
plot(t(1:length(wy_wn)),wy_wn,'r');
hold on;
plot(t(1:length(fisOutput_wy)),fisOutput_wy,'b');
hold on;
plot(t(1:length(wy_won)),wy_won);
title('w_y Testing');

fis_wz=readfis('fis_wz.fis');
fis_wz.input(1).range = [min(dataOutput_angular(:,1)) max(dataOutput_angular(:,1))]; 
fis_wz.input(2).range = [min(dataOutput_angular(:,2)) max(dataOutput_angular(:,2))]; 
fis_wz.input(3).range = [min(dataOutput_angular(:,3)) max(dataOutput_angular(:,3))]; 
fisOutput_wz=evalfis(fis_wz,dataOutput_angular);
subplot(2,2,3)
plot(t(1:length(wz_wn)),wz_wn,'r');
hold on;
plot(t(1:length(fisOutput_wz)),fisOutput_wz,'b');
hold on;
plot(t(1:length(wz_won)),wz_won);
title('w_z Testing');

data_acc_posttrain=[fisOutput_ax,fisOutput_ay,fisOutput_az];
data_angular_posttrain=[fisOutput_wx,fisOutput_wy,fisOutput_wz];
data_posttrain=[data_acc_posttrain,data_angular_posttrain(1:length(data_acc_posttrain),:)];
% восстановление трёхмерной траектории

























