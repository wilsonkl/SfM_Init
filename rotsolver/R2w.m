function [w]=R2w(R)

%R2W	Rotation matrix to vector
%
%	Usage : [w]=R2w(R)
%
%	R is the 3x3 rotation matrix
%       w : 3 dimensional matrix such that R=expm(crossmat(w))

w=[R(3,2)-R(2,3),R(1,3)-R(3,1),R(2,1)-R(1,2)]/2;
s=norm(w);
if(s)
    w=w/s*atan2(s,(trace(R)-1)/2);
end
end

