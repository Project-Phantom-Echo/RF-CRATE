function [v] = translate_obj(v,d)
%translate N vertices v (Nx3) by vector d (1x3)
Nv = size(v,1);
temp1 = ones(Nv,1);
temp2 = [d(1)*temp1,d(2)*temp1,d(3)*temp1];
v = v + temp2;
end
