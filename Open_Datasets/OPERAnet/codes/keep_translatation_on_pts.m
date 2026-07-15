function [cloud_scatterers] = keep_translatation_on_pts(cloud_scatterers,tar_pos,mid)
tar_loc=tar_pos-mid;
%translate N vertices v (Nx3) by vector d (1x3)
Nv = size(cloud_scatterers,1);
temp1 = ones(Nv,1);
temp2 = [tar_loc(1)*temp1,tar_loc(2)*temp1];
cloud_scatterers = cloud_scatterers + temp2;
end