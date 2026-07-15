function [end_points,num_bones]=get_end_point_data_Func(Markers,kinect_pos,aspect_angle,tar_pos)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this funtion computes the bone end points
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% number of desired scatterers
num_scat=21;
% number of bones
num_bones=num_scat-1;

% make new scattering centers from existing scatterers
cloud_scatterers=zeros(num_scat,3);

% make endpoints of bones
end_point_data=zeros(num_bones,6);

end_points=zeros(size(Markers,1)*num_bones,6);

for j=1:size(Markers,1)
current_frame=squeeze(Markers(j,:,:));
zmin = min(current_frame([4,8],3))-0.055;
cloud_scatterers(1:19,:)=current_frame;
% computing center 20
% for computing a normal vector for leg
A = cloud_scatterers(13,:);
B = cloud_scatterers(14,:);
% Method2: Obtaining unit vector by calculating cross product of the vectors at panel's corner point
UnitVector_left_hand=(A-B)/norm(A-B);
cloud_scatterers(20,:)=cloud_scatterers(14,:)+0.03*UnitVector_left_hand;
% computing center 21
% for computing a normal vector for leg
C = cloud_scatterers(16,:);
D = cloud_scatterers(17,:);
% Method2: Obtaining unit vector by calculating cross product of the vectors at panel's corner point
UnitVector_right_hand=(C-D)/norm(C-D);
cloud_scatterers(21,:)=cloud_scatterers(17,:)+0.03*UnitVector_right_hand;
% computing center 20
% for computing a normal vector for leg
A = cloud_scatterers(12,:);%4
B = cloud_scatterers(15,:);%5
% C = cloud_scatterers(2,:);
D =cloud_scatterers(1,:);
% Method2: Obtaining unit vector by calculating cross product of the vectors at panel's corner point
UnitVector2=cross((D-A),(B-A));
% normal vector in the direction of torso
UnitVector2=UnitVector2/norm(UnitVector2);
cloud_scatterers(5,:)=cloud_scatterers(4,:)+0.12*UnitVector2;
% computing center 21
cloud_scatterers(9,:)=cloud_scatterers(8,:)+0.12*UnitVector2;

cloud_scatterers=cloud_scatterers-[0 0 zmin];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  PLace the target at a desired location in space and keep translational
%  motion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if j==1
mid=cloud_scatterers(11,1:2);
end
s = cloud_scatterers(:,1:2); 
cloud_scatterers = [keep_translatation_on_pts(s,tar_pos(1:2),mid) cloud_scatterers(:,3)];

if j==1
% % for computing a normal vector for leg
% for all other motions
A = cloud_scatterers(12,:);
B = cloud_scatterers(15,:);
D =cloud_scatterers(1,:);
% Method2: Obtaining unit vector by calculating cross product of the vectors at panel's corner point
UnitVector3=cross((D-A),(B-A));
% normal vector in the direction of torso
UnitVector3=UnitVector3/norm(UnitVector3);
% compute vector in the direction of radar
UnitVector1=kinect_pos(1,:)-cloud_scatterers(1,:);
UnitVector1=UnitVector1/norm(UnitVector1);

va=[UnitVector3(1:2) 0];
vb=[UnitVector1(1:2) 0];
% chck angle
theta=acosd(dot(UnitVector1(1,1:2),UnitVector3(1,1:2)));
% check direction of rotation
cross_prod=cross(va,vb);
% center point for rotation
old_center_point=[tar_pos(1:2) 0];%cloud_scatterers(13,:);;
end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % This code is for intially making aspect angle zero 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if sign(cross_prod(3))==1 
    cloud_scatterers=rotate_tr(cloud_scatterers,[0,0,1],theta,old_center_point);
else 
    cloud_scatterers=rotate_tr(cloud_scatterers,[0,0,-1],theta,old_center_point);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code is for intially making aspect angle vary according to user
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cloud_scatterers=rotate_tr(cloud_scatterers,[0,0,-1],aspect_angle,old_center_point);

% make skeleton using these point scatterers
x = cloud_scatterers(:,1);
y = cloud_scatterers(:,2);
z = cloud_scatterers(:,3);

% bone 1 (left hip)
end_point_data(1,:)=[x(1) x(2) y(1) y(2) z(1) z(2)];
% bone 2 (left thigh)
end_point_data(2,:)=[x(2) x(3) y(2) y(3) z(2) z(3)];
 % bone3 left lower leg
end_point_data(3,:)=[x(3) x(4) y(3) y(4) z(3) z(4)];
 % bone4 left foot
end_point_data(4,:)=[x(4) x(5) y(4) y(5) z(4) z(5)];
 % bone5 right hip
end_point_data(5,:)=[x(1) x(6) y(1) y(6) z(1) z(6)];
 % bone6 right thigh
end_point_data(6,:)=[x(6) x(7) y(6) y(7) z(6) z(7)];
 % bone7 right lower leg
end_point_data(7,:)=[x(7) x(8) y(7) y(8) z(7) z(8)];
 % bone8 right foot
end_point_data(8,:)=[x(8) x(9) y(8) y(9) z(8) z(9)];
 % bone9 torso1
end_point_data(9,:)=[x(1) x(10) y(1) y(10) z(1) z(10)];
 % bone10 torso2
end_point_data(10,:)=[x(10) x(11) y(10) y(11) z(10) z(11)];
 % bone11 right shoulder
end_point_data(11,:)=[x(11) x(12) y(11) y(12) z(11) z(12)];
 % bone12 right upper arm
end_point_data(12,:)=[x(12) x(13) y(12) y(13) z(12) z(13)];
 % bone13 right lower arm
end_point_data(13,:)=[x(13) x(14) y(13) y(14) z(13) z(14)];
 % bone14 left shoulder
end_point_data(14,:)=[x(11) x(15) y(11) y(15) z(11) z(15)];
 % bone15 left upper arm
end_point_data(15,:)=[x(15) x(16) y(15) y(16) z(15) z(16)];
 % bone16 left lower arm
end_point_data(16,:)=[x(16) x(17) y(16) y(17) z(16) z(17)];
 % bone17 neck
end_point_data(17,:)=[x(11) x(18) y(11) y(18) z(11) z(18)];
 % bone18 head
end_point_data(18,:)=[x(18) x(19) y(18) y(19)+norm(y(19)-y(18)) z(18) z(19)];
% bone19 left hand 
end_point_data(19,:)=[x(14) x(20)+0.03 y(14) y(20) z(14) z(20)+0.03];
% bone20 right hand
end_point_data(20,:)=[x(17) x(21) y(17) y(21) z(17) z(21)+0.03];

% combined end points of all the bones
end_point_data=end_point_data(:,[1 5 3 2 6 4]);
end_points((j-1)*num_bones+1:j*num_bones,:)=end_point_data;
end
end