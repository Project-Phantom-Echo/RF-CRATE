function [v,t] = get_targetReturns_FUNC(end_points,Kinect_pos,num_bones,Mocap_rate,T1,T2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOAD INPUT PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
format long e;
num_frames=size(end_points,1)/num_bones;
range=zeros(num_frames,num_bones*3);
for n=1:num_frames
    nmark = (n-1)*num_bones;
    end_pts_temp = end_points(nmark+1:nmark+num_bones,:);
    % Assign RCS to each body part
    [tar_range]=get_mid_points_FUNC(end_pts_temp,num_bones);
    % Collect time domain radar data while incorporating the effect of 
    % radar parameters such as Tx power etc. on the time domain radar data
    range(n,:)=tar_range;
end
    ts=Mocap_rate;
    % load target range
    tar_pos=range;
    time_axis=0:ts:num_frames*ts-ts;
    % extract desired points
    [~, ic1] = min(abs(time_axis-T1));
    [~, ic2] = min(abs(time_axis-T2));
    % define a new time axis based on selected points
    time_mocap=time_axis(ic1:ic2);
    v =  zeros(length(time_mocap),size(tar_pos,2)/3);
    for t = 1:length(time_mocap)-1
        for j = 1:3:size(v,2)*3-2
            rTx_1 = Kinect_pos(1,:)-tar_pos(t+ic1,j:j+2);
            rTx_2 = Kinect_pos(1,:)-tar_pos(t-1+ic1,j:j+2);
            rRx = Kinect_pos(1,:)-tar_pos(t-1+ic1,j:j+2);
            mag_rTx= norm(rTx_2);
            mag_rRx= norm(rRx);
            bisector_angle=dot(rTx_2,rRx)/(norm(rTx_2)*norm(rRx));
            if bisector_angle>1
            bisector_angle=1;
            elseif bisector_angle<-1
            bisector_angle=-1;
            end
            BetaInDegrees =acosd(bisector_angle)/2;
            
            tempPhi=dot(rTx_2-rTx_1,rTx_2)/(norm(rTx_2)*norm(rTx_2-rTx_1));
            if tempPhi>1
            tempPhi=1;
            elseif tempPhi<-1
            tempPhi=-1;
            end
            PhiInDegrees=acosd(tempPhi)+BetaInDegrees;
            vel = norm(rTx_2-rTx_1)/ts;
            v(t,floor(j/3)+1) = vel.*cosd(BetaInDegrees).*cosd(PhiInDegrees);
        end
    end  
   	t = time_mocap;
end
