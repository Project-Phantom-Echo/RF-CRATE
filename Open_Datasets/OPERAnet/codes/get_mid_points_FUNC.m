function [tar_range]=get_mid_points_FUNC(end_pts,num_bones)
tar_range=zeros(1,num_bones*3);
for n=1:num_bones  
   % Get the two coordinates of the end point of the bone
   R1=end_pts(n,1:3);
   R2=end_pts(n,4:6);
   % Find the 3D mid-point of the bone. 
   rmid=0.5*(R1+R2);
   tar_range(1,3*n-2:3*n)=rmid;   
end
end


