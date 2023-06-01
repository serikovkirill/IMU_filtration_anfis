% Создание трехмерной траектории
t = 0:0.02:10;
x_0 = sin(t);
y_0 = cos(t);
z_0 = t;
trajectory=[x_0; y_0; z_0]; % массив с тремя координатами   
SNR_dB = randi([50, 100]); % случайное целое число от 50 до 100
SNR = 10^(SNR_dB/10);
noise = sqrt(1/SNR)*randn(size(trajectory));
noisy_signal = trajectory + noise; %наложение белого шума на траекторию
%plot3(noisy_signal(1,:), noisy_signal(2,:), noisy_signal(3,:), 'r'); % сигнал с шумом
%title('Trajectory with applied noise');
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
%расчет идеальных угловых скоростей
roll_angles_won = zeros(length(x_0), 1); % крен
pitch_angles_won = zeros(length(x_0), 1); % тангаж
yaw_angles_won = zeros(length(x_0), 1); % рыскание
wx_won = zeros(length(x_0)-1, 1); % угловая скорость по x
wy_won = zeros(length(x_0)-1, 1); % угловая скорость по y
wz_won = zeros(length(x_0)-1, 1); % угловая скорость по z

% расчет углов 
for i = 1:length(x_0)
    % расчет углов по x, y, z на основе координат
    roll_angles_won(i) = atan2(y_0(i), z_0(i));
    pitch_angles_won(i) = atan2(-x_0(i), z_0(i)*sin(roll_angles_won(i)) + y_0(i)*cos(roll_angles_won(i)));
    yaw_angles_won(i) = atan2(z_0(i)*cos(roll_angles_won(i)) - y_0(i)*sin(roll_angles_won(i)), x_0(i)*cos(pitch_angles_won(i)) + y_0(i)*sin(pitch_angles_won(i))*sin(roll_angles_won(i)) + z_0(i)*sin(pitch_angles_won(i))*cos(roll_angles_won(i)));
end

% расчет угловых скоростей 
for i = 2:(length(x_0))-1
    wx_won(i-1) = (roll_angles_won(i+1) - roll_angles_won(i-1)) / (t(i+1) - t(i-1));
    wy_won(i-1) = (pitch_angles_won(i+1) - pitch_angles_won(i-1)) / (t(i+1) - t(i-1));
    wz_won(i-1) = (yaw_angles_won(i+1) - yaw_angles_won(i-1)) / (t(i+1) - t(i-1));
end

%расчет зашумленных угловых скоростей
roll_angles_wn = zeros(length(x_0), 1); % крен
pitch_angles_wn = zeros(length(x_0), 1); % тангаж
yaw_angles_wn = zeros(length(x_0), 1); % рыскание
wx_wn = zeros(length(x_0)-1, 1); % угловая скорость по x
wy_wn = zeros(length(x_0)-1, 1); % угловая скорость по y
wz_wn = zeros(length(x_0)-1, 1); % угловая скорость по z

% расчет углов 
for i = 1:length(x_0)
    % расчет углов по x, y, z на основе координат
    roll_angles_wn(i) = atan2(y_1(i), z_1(i));
    pitch_angles_wn(i) = atan2(-x_1(i), z_1(i)*sin(roll_angles_wn(i)) + y_0(i)*cos(roll_angles_wn(i)));
    yaw_angles_wn(i) = atan2(z_1(i)*cos(roll_angles_wn(i)) - y_0(i)*sin(roll_angles_wn(i)), x_0(i)*cos(pitch_angles_wn(i)) + y_0(i)*sin(pitch_angles_wn(i))*sin(roll_angles_wn(i)) + z_0(i)*sin(pitch_angles_wn(i))*cos(roll_angles_wn(i)));
end

% расчет угловых скоростей 
for i = 2:length(x_0)-1
    wx_wn(i-1) = (roll_angles_wn(i+1) - roll_angles_wn(i-1)) / (t(i+1) - t(i-1));
    wy_wn(i-1) = (pitch_angles_wn(i+1) - pitch_angles_wn(i-1)) / (t(i+1) - t(i-1));
    wz_wn(i-1) = (yaw_angles_wn(i+1) - yaw_angles_wn(i-1)) / (t(i+1) - t(i-1));
end

dataset_accelerations=[ax_wn',ay_wn',az_wn',ax_won',ay_won',az_won'];
%удаление выбросов из датасета
dataEdu_acc=dataset_accelerations(1:round(size(dataset_accelerations,1)/2),:);
q1 = prctile(dataset_accelerations,25);
q3 = prctile(dataset_accelerations,75);
iqr = q3 - q1;
lower = q1 - 1.5*iqr;
upper = q3 + 1.5*iqr;
% удаление строк с выбросами
id_outlies_angular = any(dataset_accelerations < lower | dataset_accelerations > upper, 2);
dataset_accelerations_clear = dataset_accelerations(~id_outlies_angular, :);
%создание данных для обучения и тестирования (для ускорений)
dataTest_acc=dataset_accelerations_clear(round(size(dataset_accelerations_clear,1)/2)+1:end,:);
dataEdu_ax=dataEdu_acc(:,1:4);
dataTest_ax=dataTest_acc(:,1:4);
dataEdu_ay=dataEdu_acc(:,[1:3,5]);
dataTest_ay=dataTest_acc(:,[1:3,5]);
dataEdu_az=dataEdu_acc(:,[1:3,6]);
dataTest_az=dataTest_acc(:,[1:3,6]);
dataOutput_acc=dataset_accelerations_clear(:,1:3);

%загрузка и тестирование ANFIS
fis_ax=readfis('fis_ax.fis');
fis_ax.input(1).range = [min(dataOutput_acc(:,1)) max(dataOutput_acc(:,1))]; 
fis_ax.input(2).range = [min(dataOutput_acc(:,2)) max(dataOutput_acc(:,2))]; 
fis_ax.input(3).range = [min(dataOutput_acc(:,3)) max(dataOutput_acc(:,3))]; 
fisOutput_ax=evalfis(fis_ax,dataOutput_acc);
figure;
subplot(3,1,1);
plot(t,ax_wn,'r');
hold on;
plot(t(1:length(fisOutput_ax)),fisOutput_ax,'b');
hold on;
plot(t,ax_won);
title('a_x Testing');
legend({'a_x before training','a_x after training', 'a_x ideal'},'Location','southwest');
xlabel('t');
ylabel('m/s^2');

fis_ay=readfis('fis_ay.fis');
fis_ay.input(1).range = [min(dataOutput_acc(:,1)) max(dataOutput_acc(:,1))]; 
fis_ay.input(2).range = [min(dataOutput_acc(:,2)) max(dataOutput_acc(:,2))]; 
fis_ay.input(3).range = [min(dataOutput_acc(:,3)) max(dataOutput_acc(:,3))]; 
fisOutput_ay=evalfis(fis_ay,dataOutput_acc);
subplot(3,1,2);
plot(t,ay_wn,'r');
hold on;
plot(t,ay_won);
hold on;
plot(t(1:length(fisOutput_ay)),fisOutput_ay,'b');
title('a_y Testing');
legend({'a_y before training','a_y after training', 'a_y ideal'},'Location','southwest')
xlabel('t');
ylabel('m/s^2');

fis_az=readfis('fis_az.fis');
fis_az.input(1).range = [min(dataOutput_acc(:,1)) max(dataOutput_acc(:,1))]; 
fis_az.input(2).range = [min(dataOutput_acc(:,2)) max(dataOutput_acc(:,2))]; 
fis_az.input(3).range = [min(dataOutput_acc(:,3)) max(dataOutput_acc(:,3))]; 
fisOutput_az=evalfis(fis_az,dataOutput_acc);
subplot(3,1,3);
plot(t,az_wn,'r');
hold on;
plot(t,az_won);
hold on;
plot(t(1:length(fisOutput_az)),fisOutput_az,'b');
title('a_z Testing');

dataset_angular=[wx_wn,wy_wn,wz_wn,wx_won,wy_won,wz_won];% создание датасета, содержащего данные об угловых скоростях
dataEdu_angular=dataset_angular(1:round(size(dataset_angular,1)/2),:);
%удаление выбросов
q1_0 = prctile(dataset_angular,25);
q3_0 = prctile(dataset_angular,75);
iqr_0 = q3_0 - q1_0;
lower = q1_0 - 1.5*iqr_0;
upper = q3_0 + 1.5*iqr_0;
% удаление строк с выбросами
id_outlies_angular = any(dataset_angular < lower | dataset_angular > upper, 2);
dataset_angular_clear = dataset_angular(~id_outlies_angular, :);
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
fis_wx=readfis('fis_wx.fis');
fis_wx.input(1).range = [min(dataOutput_angular(:,1)) max(dataOutput_angular(:,1))]; 
fis_wx.input(2).range = [min(dataOutput_angular(:,2)) max(dataOutput_angular(:,2))]; 
fis_wx.input(3).range = [min(dataOutput_angular(:,3)) max(dataOutput_angular(:,3))]; 
fisOutput_wx=evalfis(fis_wx,dataOutput_angular);
figure
subplot(3,1,1)
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
subplot(3,1,2)
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
subplot(3,1,3)
plot(t(1:length(wz_wn)),wz_wn,'r');
hold on;
plot(t(1:length(fisOutput_wz)),fisOutput_wz,'b');
hold on;
plot(t(1:length(wz_won)),wz_won);
title('w_z Testing');

data_acc_posttrain=[fisOutput_ax,fisOutput_ay,fisOutput_az];
data_angular_posttrain=[fisOutput_wx,fisOutput_wy,fisOutput_wz];
data_posttrain=[data_acc_posttrain,data_angular_posttrain(1:length(data_acc_posttrain),:)];

% Получение PVA
N = 500; 
deltat = 0.02; 

acc = dataset_accelerations(1:500,1:3);
gyr = dataset_angular(:,1:3);

q = zeros(N, 4); 
q(1,:) = [1 0 0 0]; 

pos = zeros(N, 3); 
vel = zeros(N, 3); 

Cbn = zeros(3); 
Cbn(1,1) = 1; Cbn(2,2) = 1; Cbn(3,3) = 1;


for i=2:N    

    ang_deltat = gyr(i-1,:)*deltat;
    ang_axis = ang_deltat;
    ang_angle = norm(ang_deltat); 
    dq = [cos(ang_angle/2) sin(ang_angle/2)*ang_axis]; 
    q(i,:) = quatmultiply(q(i-1,:), dq); 

    
    Cnb = quat2dcm(q(i,:));
    Cbn = Cnb';
    
    
    acc_b = acc(i,:); 
    f_b = Cbn*acc_b'; 
    f_n = f_b - [0 0 9.78032677]'; 

   
    vel(i,:) = vel(i-1,:) + f_n'*deltat;  
    pos(i,:) = pos(i-1,:) + vel(i,:)*deltat; 
end

 pos_expected = ones(N,3);
 pos_experimental = pos;
 pos_diff = pos_experimental-pos_expected;
 pos_rms = rms(pos_diff);

 vel_expected = ones(N,3);
 vel_experimental = vel;
 vel_diff = vel_experimental-vel_expected;
 vel_rms = rms(vel_diff);


N0 = 460; 
 
acc_ml = data_acc_posttrain(1:N0,1:3);
gyr_ml = data_angular_posttrain(1:N0,1:3);

q0 = zeros(N, 4); 
q0(1,:) = [1 0 0 0]; 

pos_ml = zeros(N0, 3); 
vel_ml = zeros(N0, 3); 

Cbn0 = zeros(3); 
Cbn0(1,1) = 1; Cbn0(2,2) = 1; Cbn0(3,3) = 1;


for i=2:N0    

    ang_deltat0 = gyr_ml(i-1,:)*deltat;
    ang_axis0 = ang_deltat0;
    ang_angle0 = norm(ang_deltat0); 
    dq0 = [cos(ang_angle0/2) sin(ang_angle0/2)*ang_axis0]; 
    q0(i,:) = quatmultiply(q0(i-1,:), dq0); 

    
    Cnb0 = quat2dcm(q0(i,:));
    Cbn0 = Cnb0';
    
    
    acc_b0 = acc_ml(i,:); 
    f_b0 = Cbn0*acc_b0'; 
    f_n0 = f_b0 - [0 0 9.78032677]'; 

   
    vel_ml(i,:) = vel_ml(i-1,:) + f_n0'*deltat;  
    pos_ml(i,:) = pos_ml(i-1,:) + vel_ml(i,:)*deltat; 
end

 pos_expected_ml = ones(N0,3);
 pos_experimental_ml = pos_ml;
 pos_diff_ml = pos_experimental_ml-pos_expected_ml;
 pos_rms_ml = rms(pos_diff_ml);

 vel_expected_ml = ones(N0,3);
 vel_experimental_ml = vel_ml;
 vel_diff_ml = vel_experimental_ml-vel_expected_ml;
 vel_rms_ml = rms(vel_diff_ml);























