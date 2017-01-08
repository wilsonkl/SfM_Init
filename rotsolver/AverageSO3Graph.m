%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This implementation is based on the paper 
% "Efficient and Robust Large-Scale Rotation Averaging." by
% Avishek Chatterjee, Venu Madhav Govindu.
%
% This code robustly performs relative rotation averaging
%
% function [R] = AverageSO3Graph(RR,I,varargin)
% INPUT:        RR = 'm' number of 3 X 3 Relative Rotation Matrices (R_ij) 
%                    stacked as a 3 X 3 X m Matrix
%                    OR
%                    'm' number of 4 X 1 Relative Quaternions (R_ij) 
%                    stacked as a 4 X m  Matrix
%                I = Index matrix (ij) of size (2 X m) such that RR(:,:,p)
%                    (OR RR(:,p) for quaternion representation)  is
%                    the relative rotation from R(:,:,I(1,p)) to R(:,:,I(2,p))
%                    (OR R(:,I(1,p)) and  R(:,I(2,p)) for quaternion representation)
% OPTIONALS: Rinit = Optional initial guess. 
%    MaxIterations = MaxIterations(1) is Maximum number of L1 iterations. Default 10
%                    MaxIterations(2) is Maximum number of IRLS iterations. Default 100
%            SIGMA = Sigma value for M-Estimation in degree (5 degree is preferred)
%                    Default is 5 degree. Put [] for default.
%
% OUTPUT:       R  = 'n' number of 3 X 3 Absolute Rotation matrices stacked as
%                     a  3 X 3 X n Matrix 
%                     OR
%                     'n' number of 4 X 1 Relative Quaternions (R_ij) 
%                     stacked as a 4 X n  Matrix
%
% IMPORTANT NOTES:
% The underlying model or equation is assumed to be: 
% X'=R*X; Rij=Rj*inv(Ri) i.e. camera centered coordinate system is used
% and NOT the geocentered coordinate for which the underlying equations are
% X'=inv(R)*X; Rij=inv(Ri)*Rj. 
% To use geocentered coordinate please transpose the rotations or change
% the sign of the scalar term of the quaternions before feeding into the
% code and also after getting the result.
%
% This code is able to handle inputs in both Rotation matrix as well as
% quaternion format. The Format of output is same as that of the input.
%
% Programmer: AVISHEK CHATTERJEE
%             PhD Student (S. R. No. 04-03-05-10-12-11-1-08692)
%             Learning System and Multimedia Lab
%             Dept. of Electrical Engineering
%             INDIAN INSTITUTE OF SCIENCE
%
% Dated:  April 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [R]=AverageSO3Graph(RR,I,varargin)
inp=inputParser;
inp.addRequired('RR', @(x)isreal(x) && ((ndims(x)==3&&size(x,1)==3&&size(x,2)==3)||(ismatrix(x)&&size(x,1)==4)));
inp.addRequired('I', @(x)isreal(x) && ismatrix(x)&&size(x,1)==2);
inp.addParamValue('Rinit', [], @(x)isreal(x) && ((ndims(x)==3&&size(x,1)==3&&size(x,2)==3)||(ismatrix(x)&&size(x,1)==4)));
inp.addParamValue('SIGMA', 5, @(x)isreal(x) && isscalar(x));
inp.addParamValue('MaxIterations', [10 100], @(x)isreal(x) && numel(x)==2);
inp.parse(RR,I,varargin{:});
arg=inp.Results;
% Find the largest connected component in the view graph and reindex
% Nodes not in the largest connected component will have NaN
N=max(I(:));
[s,c]=graphconncomp(sparse([I(1,:),I(2,:)],[I(2,:),I(1,:)],ones(1,2*size(I,2)),N,N));
m=0;j=0;for(i=1:s);k=sum(c==i);if(k>m);m=k;j=i;end;end;
i=find(c==j);  [~,I]=ismember(I,i);  j=all(I);
if(size(RR,1)==4)%Quaternion form
    for ii=1:size(RR,2)
        if(abs(norm(RR(:,ii),2)-1)>.1)
            error('norm(RR(:,%d))=%f\n',ii,norm(RR(:,ii),2));
        elseif(abs(norm(RR(:,ii),2)-1)>.01)
            warning('norm(RR(:,%d))=%f\nNormalizing\n',ii,norm(RR(:,ii),2)')
        end
        RR(:,ii)=RR(:,ii)/norm(RR(:,ii),2);
    end
    R=NaN(4,N);
    R(:,i)=BoxMedianSO3Graph(RR(:,j),I(:,j),arg.Rinit,arg.MaxIterations(1));
    R(:,i)=RobustMeanSO3Graph(RR(:,j),I(:,j),arg.SIGMA,R(:,i),arg.MaxIterations(2));
else % Matrix form
    % Check whether the rotation matrices are proper.
    for ii=1:size(RR,3)
        if(det(RR(:,:,ii))<=0)
            error('det(RR(:,:,%d))=%f\n',ii,det(RR(:,:,ii)))
        end
        [U,S,V]=svd(RR(:,:,ii));
        if(abs(diag(S)-1)>=.1)
            error('svd(RR(:,:,%d))=[%f %f %f]\n',ii,S(1,1),S(2,2),S(3,3));
        elseif(abs(diag(S)-1)>=.01)
            warning('svd(RR(:,:,%d))=[%f %f %f]\nProjecting to 1 s\n',ii,S(1,1),S(2,2),S(3,3));
        end
        RR(:,:,ii)=U*round(S)*V';
    end
    R=NaN(3,3,N);
    R(:,:,i)=BoxMedianSO3Graph(RR(:,:,j),I(:,j),arg.Rinit,arg.MaxIterations(1));
    R(:,:,i)=RobustMeanSO3Graph(RR(:,:,j),I(:,j),arg.SIGMA,R(:,:,i),arg.MaxIterations(2));
end
end