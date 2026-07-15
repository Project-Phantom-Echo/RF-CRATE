%Author: Mohammud J. Bocus
% Script to visualize WiFi CSI ampltiude data from the two NUC systems
% between a given pair of TX and RX antenna and subcarrier index
% considering both raw and filtered data)
%(see Fig. 1 for layout of the two CSI NUC systems in the two experiment rooms)
%%
clear
clc

nuc1 = importdata('../wificsi1/wificsi1_exp018.mat');
nuc2 = importdata('../wificsi2/wificsi2_exp018.mat');

%plot for a given duration
start_time = datetime('18:04:44.553', 'Format', 'HH:mm:ss.SSS');
end_time   = datetime('18:08:11.553', 'Format', 'HH:mm:ss.SSS');

date1 = datetime(nuc1.timestamp, 'Format', 'HH:mm:ss.SSS');

[~,idx1]    = min(abs(datenum(start_time)-datenum(date1 )));
[~,idx11]   = min(abs(datenum(end_time)  -datenum(date1 )));


samples =  idx1:1:idx11;

t_csi1  =  date1(samples)  ;
out = seconds(diff(t_csi1));
actualtime1=[0;out];     
time_duration=cumsum(actualtime1); 

t_csi2  =   date1(samples)  ;
out2 = seconds(diff(t_csi2));
actualtime2=[0;out2];      
time_duration2=cumsum(actualtime2);  


%% Raw un-filtered data
figure()
plot(time_duration,  abs(nuc1.tx1rx1_sub10(samples))) %plot CSI amplitude data for TX1, RX1, subcarrier 10
hold on
plot(time_duration2, abs(nuc2.tx1rx1_sub10(samples)))
legend('NUC1', 'NUC2')
xlabel('Duration (s)')
ylabel('CSI Amplitude')

%% 1D wavelet denoising

%tx1rx1_sub10  = transmit antenna 1, receive antenna 1, subcarrier 10

scal = 'sln'; 
dwt_denoised_sig_nuc1 = wden(abs( nuc1.tx1rx1_sub10 ),'sqtwolog','s',scal,4,'sym3'); 
dwt_denoised_sig_nuc2 = wden(abs( nuc2.tx1rx1_sub10 ),'sqtwolog','s',scal,4,'sym3');

figure()
plot(time_duration, abs(dwt_denoised_sig_nuc1(samples)))
hold on
plot(time_duration2,abs(dwt_denoised_sig_nuc2(samples)))
legend('NUC1', 'NUC2')
xlabel('Duration (s)')
ylabel('CSI Amplitude')