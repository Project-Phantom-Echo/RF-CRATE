%Author: Mohammud J. Bocus
%Script to visualize UWB CIR and CFR data from the two UWB systems (1 and 2)
%(see Fig. 1 for layout of the two UWB systems in the two experiment rooms)
%% plot CFR data versus time
clear
clc

uwb1 = readtable('../uwb1/uwb1_exp018.csv');
uwb2 = readtable('../uwb2/uwb2_exp018.csv');

tx_id_uwb1 = uwb1.tx_id ;
rx_id_uwb1 = uwb1.rx_id ; 

tx_id_uwb2 = uwb2.tx_id ;
rx_id_uwb2 = uwb2.rx_id ; 

%choose bidrectional data between nodes 0 and 3 (UWB1)
idx_1               = find(  (tx_id_uwb1 ==0 &  rx_id_uwb1  ==3 ) |  (tx_id_uwb1==3 &  rx_id_uwb1 ==0)) ; 
Filtered_data1      = uwb1(idx_1,:);

%choose bidrectional data between nodes 1 and 2 (UWB2)
idx_2               = find(  (tx_id_uwb2 ==1 &  rx_id_uwb2  ==2 ) |  (tx_id_uwb2==2 &  rx_id_uwb2 ==1)) ; 
Filtered_data2      = uwb2(idx_2,:);

datee1 = datestr(Filtered_data1.timestamp,'HH:MM:SS.FFF');  
datee2 = datestr(Filtered_data2.timestamp,'HH:MM:SS.FFF');

date1 = datetime(datee1, 'Format', 'HH:mm:ss.SSS');
date2 = datetime(datee2, 'Format', 'HH:mm:ss.SSS');

CFR_1 = fft(Filtered_data1{:,30:64},[],2); %convert 35 samples of CIR data into CFR (UWB system 1)
CFR_2 = fft(Filtered_data2{:,30:79},[],2); %convert 50 samples of CIR data into CFR (UWB system 2)

%plot for a given duration
start_time = datetime('18:04:44.553', 'Format', 'HH:mm:ss.SSS');
end_time   = datetime('18:08:11.553', 'Format', 'HH:mm:ss.SSS');

[~,idx1]    = min(abs(datenum(start_time)-datenum(date1 )));
[~,idx11]   = min(abs(datenum(end_time)-datenum(date1 )));
[~,idx2]    = min(abs(datenum(start_time)-datenum(date2 )));
[~,idx22]   = min(abs(datenum(end_time)-datenum(date2 )));
 
t_uwb1 =   date1(idx1:idx11)    ;
out = seconds(diff(t_uwb1));
actualtime1=[0;out];      
time_duration=cumsum(actualtime1);  
 
t_uwb2  =   date2(idx2:idx22)   ;
out2 = seconds(diff(t_uwb2 ));
actualtime2=[0;out2];      
time_duration2=cumsum(actualtime2);  
 
figure()
plot(time_duration , abs(CFR_1(idx1:idx11,10))) %10th CFR sample
hold on
plot(time_duration2 , abs(CFR_2(idx2:idx22,10)))%10th CFR sample
legend('UWB system 1', 'UWB system 2')
xlabel('Duration (s)')
ylabel('CFR Amplitude')

%% Plot aligned CIR measurements 
% The following script has been adapted from the original python code available from 
% "Example Python Script" at:
% https://www.research-collection.ethz.ch/handle/20.500.11850/397625
% Dataset accompanying paper "A multi-static radar network with ultra-wideband radio-equipped devices"
% Author: Ledergerber, Anton

% Part of the code for aligning and plotting the CIR measurements 
% has been rewritten in Matlab here and adapted for the current UWB datasets  

len1                 = 35;   %cir samples length for UWB system 1
len2                 = 50;   %cir samples length for UWB system 2
first_path_offset    = -2;   %offset of logged samples from first path sample as determined 
%by Decawave's leading edge detection algorithm. CIR samples were read from
%the DW1000 UWB chipset starting 3 samples below detected first path index
CIR_samp_period      = 1/(2*(499.2e6)) ;% sampling period og CIR measurement (approx. 1 ns)
SpeedofLight         = 299792458 ;% (m/s) speed of light
CIR_prop_time_axis1  = first_path_offset: (CIR_samp_period*1e9):first_path_offset+(len1-0.5)*CIR_samp_period*1e9;
CIR_prop_time_axis2  = first_path_offset: (CIR_samp_period*1e9):first_path_offset+(len2-0.5)*CIR_samp_period*1e9;

CIR1               = Filtered_data1{:,size(Filtered_data1,2)-len1+1:end};   %UWB system 1
CIR2               = Filtered_data2{:,size(Filtered_data2,2)-len2+1:end};   %UWB system 2
FP_idx1            = Filtered_data1.fp_index ; %first path index
FP_idx2            = Filtered_data2.fp_index ; %first path index
fp1                = mod(abs(FP_idx1),1); %fractional part of FP index
fp2                = mod(abs(FP_idx2),1); %fractional part of FP index
time_Offset1       = (fp1)*CIR_samp_period*1e9 ; % convert fractional part of FP index into ns  
time_tOffset2      = (fp2)*CIR_samp_period*1e9 ; % convert fractional part of FP index into ns  
CIR_mag1           = abs(CIR1)   ;  %already normalised by preamble symbol count value (i.e. "rx_pream_count" header) when raw UWB log files were converted into human readable format.
CIR_mag2           = abs(CIR2)   ;  %already normalised by preamble symbol count value (i.e. "rx_pream_count" header) when raw UWB log files were converted into human readable format.

% plotting bidirectional CIR data between a given pair of nodes
start_idx= 1;
end_idx  = 1000  ;     %number of desired CIR measurements
figure()
plot((CIR_prop_time_axis1-time_Offset1(start_idx:end_idx)  ).',  (CIR_mag1(start_idx:end_idx,:)).', '.');  
xlabel('\tau - \tau_{FP} (ns)' )
ylabel('CIR magnitude')

figure()
plot((CIR_prop_time_axis2-time_tOffset2(start_idx:end_idx)  ).',  (CIR_mag2(start_idx:end_idx,:)).', '.');  
xlabel('\tau - \tau_{FP} (ns)' )
ylabel('CIR magnitude')
