%Author: Shelly Vishwakarma
%Script to plot motion capture data (velocity versus time) from the two Kinect systems.
% Users can also visualize the stickman (skeletal) representation of the
% motion capture data from the kinect systems.
%(see Fig. 1 for layout of the two Kinect systems in the two experiment rooms)
%%
clc
close all
clear all

load('../kinect/Kinect_exp_018.mat')

%start_time = Kinect{2,2};
%end_time   = Kinect{end,2};

start_time = '18:04:44.553';
end_time   = '18:08:11.553';

start_time = datetime(start_time,'InputFormat','HH:mm:ss.SSS', 'Format', 'HH:mm:ss.SSS');
end_time   = datetime(end_time,'InputFormat','HH:mm:ss.SSS', 'Format', 'HH:mm:ss.SSS');

for i = 2:1:size(Kinect,1)
    temp = string(char(Kinect{i,2}));
    Kinect_timestamp(i-1,1) = datetime(temp,'InputFormat','HH:mm:ss.SSS', 'Format', 'HH:mm:ss.SSS');
    Kinect_temp(i-1,:,:) = Kinect{i,6};
end

[~, start_index] = min(abs(Kinect_timestamp - start_time));

[~, end_index] = min(abs(Kinect_timestamp - end_time));

Kinect_new = Kinect_temp( start_index:end_index,:,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualise Kinect Data as if the target is moving in front of monostatic radar
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Kinect Position
kinect_pos=[0,0,0.8];

% % Number of time domain samples
Num_frames = size(Kinect_new,1);

% Mocap Rate
Mocap_rate=1/10;

% aspect angle of target
aspect_angle=0;

% Target placement in space
tar_pos=[0,4,0];

% generate human skeleton model and gather bone center locations
[end_points,num_bones]=get_end_point_data_Func(Kinect_new,kinect_pos,aspect_angle,tar_pos);

% start time of human motion
T1=0;
% stope time of human motion
T2=Num_frames*Mocap_rate-Mocap_rate;

% get target returns from motion capture data
[velocity_tar,time_axis_interpolated] = get_targetReturns_FUNC(end_points,kinect_pos,num_bones,Mocap_rate,T1,T2);

figure()
plot(time_axis_interpolated,velocity_tar);
% title('Real Data')
xlabel('time (s)','fontweight','bold','fontsize',12);
ylabel('velocity (m/s)','fontweight','bold','fontsize',12);
set(gca,'fontweight','bold','fontsize',12);
set(gcf,'rend','painters','pos',[100 100 1500 500]);
ylim([-2 2])
xlim([T1 T2])

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHOOSE WETHER TO View Skeleton as a movie 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
myfilename=strcat('Movie',num2str(randi([1 1000],1,1)));
% 0 = movie is not saved
% 1 = movie is saved
mov_in_stick =0;
% Want to view animation of primitive shaped model ?
view_stick_figure='y';
view_bone_marker_num='n';

if (strcmp(view_stick_figure,'y'))
% view stick figure animation model or not?
% show_primitive_data_Func_Trail
show_stick_figure_Func(Kinect_new,aspect_angle,kinect_pos,tar_pos,mov_in_stick,myfilename,view_bone_marker_num);
end

