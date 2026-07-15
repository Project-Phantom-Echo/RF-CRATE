function [v] = rotate_tr(v,a,theta,o)
%rotates N points v (Nx3) with axis a (1x3) by angle theta(degrees) by
%bringing object origin o (1x3) to world origin ([0,0,0]) and then
%translating the points back to to o (1x3)

%translate to world origin
temp = translate_obj(v,-o);

q = [cosd(theta/2), a(1)*sind(theta/2),a(2)*sind(theta/2),a(3)*sind(theta/2)];
%normalise quaternion
q = q./(sum(q.^2).^0.5);
%conjugate of the quaternion
q_conj = [q(1),-q(2),-q(3),-q(4)];
Nv = size(v,1);
for i = 1:Nv
    p = [0,temp(i,:)];
    rp = qmult(qmult(q,p),q_conj);
    v(i,:) = rp(2:4);
end
v = translate_obj(v,o);