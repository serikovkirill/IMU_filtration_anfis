sampleRate = 100; 
duration = 5;    % длительность симуляции
numSamples = sampleRate * duration;

% вектор, содержащий временные метки
time = (0:numSamples-1) / sampleRate;

amplitude = 2;  
noise = 0.1;    % наложение шума
accelX = amplitude * sin(2 * pi * 1 * time) + noise * randn(1, numSamples);
accelY = amplitude * sin(2 * pi * 0.5 * time) + noise * randn(1, numSamples);
accelZ = amplitude * sin(2 * pi * 0.2 * time) + noise * randn(1, numSamples);


figure;
subplot(3, 1, 1);
plot(time, accelX);
title('Acceleration X');
xlabel('Time (s)');
ylabel('Acceleration (m/s^2)');

subplot(3, 1, 2);
plot(time, accelY);
title('Acceleration Y');
xlabel('Time (s)');
ylabel('Acceleration (m/s^2)');

subplot(3, 1, 3);
plot(time, accelZ);
title('Acceleration Z');
xlabel('Time (s)');
ylabel('Acceleration (m/s^2)');

% вычисление скоростей по осям x, y и z
velocityX = cumtrapz(time, accelX);
velocityY = cumtrapz(time, accelY);
velocityZ = cumtrapz(time, accelZ);

% вычисление координат x, y и z
positionX = cumtrapz(time, velocityX);
positionY = cumtrapz(time, velocityY);
positionZ = cumtrapz(time, velocityZ);

% создание датасета
ImuSimulationDataset = [time', positionX', positionY', positionZ'];


figure;
subplot(3, 1, 1);
plot(time, positionX);
title('Position X');
xlabel('Time (s)');
ylabel('Position (m)');

subplot(3, 1, 2);
plot(time, positionY);
title('Position Y');
xlabel('Time (s)');
ylabel('Position (m)');

subplot(3, 1, 3);
plot(time, positionZ);
title('Position Z');
xlabel('Time (s)');
ylabel('Position (m)');

FlightDataset = readmatrix('Dataset.csv');

xgr=FlightDataset(:,1);
ygr=FlightDataset(:,2);
zgr=FlightDataset(:,3);
figure;
subplot(1, 2, 1);
plot(xgr,ygr);
title('Drone Flight Trajectory');
xlabel('localPosition.x');
ylabel('localPosition.y');

subplot(1, 2, 2);
plot3(xgr,ygr,abs(zgr));

RealDataset = readmatrix('RealData.xlsx');

vx_real = diff(RealDataset(:,2)) ./ diff(RealDataset(:,1)); % скорость по x
vy_real = diff(RealDataset(:,3)) ./ diff(RealDataset(:,1)); % скорость по y
vz_real = diff(RealDataset(:,4)) ./ diff(RealDataset(:,1)); % скорость по z

ax_real = diff(RealDataset(:,2)) ./ diff(RealDataset(:,1)); 
ay_real = diff(RealDataset(:,3)) ./ diff(RealDataset(:,1)); 
az_real = diff(RealDataset(:,4)) ./ diff(RealDataset(:,1));

dataset_acc_real=[RealDataset(1:size(RealDataset)-1,1),ax_real,ay_real,az_real];
fis_tr_data_ax_real=dataset_acc_real(1:size(dataset_acc_real)/2,1:2);
fis_tr_data_ay_real=dataset_acc_real(1:size(dataset_acc_real)/2,1:3);
fis_tr_data_az_real=dataset_acc_real(1:size(dataset_acc_real)/2,1:4);











