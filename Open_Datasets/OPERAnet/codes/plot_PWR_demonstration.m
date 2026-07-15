%Author: Wenda Li
%Script to plot PWR spectrograms from surveillance channels "rx2", "rx3"
%and "rx4".
%These correspond to PWR Channel 1, PWR Channel 2 and PWR Channel 3, respectively, in this example.
%(see Fig. 1 for layout of PWR surveillance channels and reference
%channel). WiFi CSI transmitter (NUC3) was used as the source.
%%
clc
clear all

load ('../pwr/PWR_exp_018.mat')

start_time = '18:04:44.553';
end_time   = '18:08:11.553';

start_time = datetime(start_time,'InputFormat','HH:mm:ss.SSS', 'Format', 'HH:mm:ss.SSS');
end_time = datetime(end_time,'InputFormat','HH:mm:ss.SSS', 'Format', 'HH:mm:ss.SSS');

for i = 2:1:size(PWR,1)
    temp = string(char(PWR{i,2}));
    PWR_timestamp(i-1,1) = datetime(temp,'InputFormat','HH:mm:ss.SSS', 'Format', 'HH:mm:ss.SSS');
    PWR_ch1_all(:,i-1) = PWR{i,6};
    PWR_ch2_all(:,i-1) = PWR{i,7};
    PWR_ch3_all(:,i-1) = PWR{i,8};
end

[~, start_index] = min(abs(PWR_timestamp - start_time));

[~, end_index] = min(abs(PWR_timestamp - end_time));

PWR_ch1 = PWR_ch1_all(:, start_index:end_index);
PWR_ch2 = PWR_ch2_all(:, start_index:end_index);
PWR_ch3 = PWR_ch3_all(:, start_index:end_index);

PWR_ch1 = 20*log10(PWR_ch1);
PWR_ch2 = 20*log10(PWR_ch2);
PWR_ch3 = 20*log10(PWR_ch3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = 1:1:size(PWR_ch1,1);
x = x - size(PWR_ch1,1)/2;
y = 1:1:size(PWR_ch1,2);
y = y / 10;

figure(1)
clf(figure(1))
set(gcf,'rend','painters','pos',[100 100 900 600]);

subplot 311
imagesc(y,x,PWR_ch1)
colormap('jet(256)')
caxis([-55 -20])
% colorbar
axis([0 max(y) -50 50])
xlabel('Time')
ylabel('Doppler (Hz)')
title('(a) Doppler spectrogram from rx 1')

subplot 312
imagesc(y,x,PWR_ch2)
colormap('jet(256)')
caxis([-55 -20])
% colorbar
axis([0 max(y) -50 50])
xlabel('Time')
ylabel('Doppler (Hz)')
title('(b) Doppler spectrogram from rx 2')

subplot 313
imagesc(y,x,PWR_ch3)
colormap('jet(256)')
caxis([-40 -20])
% colorbar
axis([0 max(y) -50 50])
xlabel('Time') 
ylabel('Doppler (Hz)')
title('(c) Doppler spectrogram from rx 3')





















































