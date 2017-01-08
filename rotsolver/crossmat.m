function M=crossmat(x)

%CROSSMAT	Cross-product matrix (anti-symmetric)
%
%	Usage : M=crossmat(x)
%	Given vector x, finds M such that cross(x,y)=My


M=[0 -x(3) x(2);x(3) 0 -x(1);-x(2) x(1) 0];