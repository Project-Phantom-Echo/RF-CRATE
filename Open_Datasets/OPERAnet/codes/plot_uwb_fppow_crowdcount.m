%Author: Mohammud J. Bocus
%Script to visualize UWB First Path power level (dBm) from the 2 systems (between a given pair of nodes) 
%for crowd counting experiment (exp028)

%See DW1000 USER MANUAL (Page 46 - Section 4.7.1 Estimating the signal power in the
%first path) to understand how the estimate of the first path power level was computed.

%%
clear
clc

uwb1 = readtable('../uwb1/uwb1_exp028.csv');
uwb2 = readtable('../uwb2/uwb2_exp028.csv');

datee1  = datestr(uwb1.timestamp,'HH:MM:SS.FFF');  
datee   = datetime(datee1 , 'Format', 'HH:mm:ss.SSS');

datee2  = datestr(uwb2.timestamp,'HH:MM:SS.FFF');  
datee22 = datetime(datee2 , 'Format', 'HH:mm:ss.SSS');

tx_id_uwb1 = uwb1.tx_id ;
rx_id_uwb1 = uwb1.rx_id ; 

tx_id_uwb2 = uwb2.tx_id ;
rx_id_uwb2 = uwb2.rx_id ; 

%choose bidrectional data between nodes 0 and 3 (UWB1)
idx_1               = find(  (tx_id_uwb1 ==0 &  rx_id_uwb1  ==3 ) |  (tx_id_uwb1==3 &  rx_id_uwb1 ==0)) ; 
Filtered_data1      = uwb1(idx_1,:);
%choose bidrectional data between nodes 3 and 4 (UWB2)
idx_2               = find(  (tx_id_uwb2 ==3 &  rx_id_uwb2  ==4 ) |  (tx_id_uwb2==4 &  rx_id_uwb2 ==3)) ; 
Filtered_data2      = uwb2(idx_2,:);


t_uwb1  =   Filtered_data1.timestamp   ;
out = seconds(diff(t_uwb1));
actualtime1=[0;out];     
time_duration=cumsum(actualtime1); 
 
t_uwb2  =  Filtered_data2.timestamp  ;
out2 = seconds(diff(t_uwb2));
actualtime2=[0;out2];   
time_duration2=cumsum(actualtime2); 

figure()
plot(time_duration ,Filtered_data1.fp_pow_dbm)
xlabel('Duration (s)')
ylabel('FP Power Level (dBm)')

figure()
plot(time_duration2 ,Filtered_data2.fp_pow_dbm)
xlabel('Duration (s)')
ylabel('FP Power Level (dBm)')
