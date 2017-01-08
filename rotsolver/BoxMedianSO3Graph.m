%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This implementation is based on the paper 
% "Efficient and Robust Large-Scale Rotation Averaging." by
% Avishek Chatterjee, Venu Madhav Govindu.
%
% This code performs relative rotation averaging with L1 cost function
%
% function [R] = BoxMedianSO3Graph(RR,I,[Rinit],[maxIters])
% INPUT:        RR = 'm' number of 3 X 3 Relative Rotation Matrices (R_ij) 
%                    stacked as a 3 X 3 X m Matrix
%                    OR
%                    'm' number of 4 X 1 Relative Quaternions (R_ij) 
%                    stacked as a 4 X m  Matrix
%                I = Index matrix (ij) of size (2 X m) such that RR(:,:,p)
%                    is the relative rotation from R(:,:,I(1,p)) to R(:,:,I(2,p))
%            Rinit = Optional initial guess. 
%                    Put [] to automatically compute Rinit from Spanning tree
%         maxIters = Maximum number of iterations. Default 100
%
% OUTPUT:        R  = 'n' number of 3 X 3 Absolute Rotation matrices stacked as
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
% Feeding of not connected graph is not recomended.
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
function  [R]=BoxMedianSO3Graph(RR,I,Rinit,maxIters)

tic;

if(nargin<4 || isempty(maxIters));maxIters=100;end
changeThreshold=.001;

N=max(max(I));%Number of cameras or images or nodes in view graph

QuaternionIP=(size(RR,1)==4);
if(~QuaternionIP)
    %Convert Rij to Quaternion form without function call
    QQ=[RR(1,1,:)+RR(2,2,:)+RR(3,3,:)-1, RR(3,2,:)-RR(2,3,:),RR(1,3,:)-RR(3,1,:),RR(2,1,:)-RR(1,2,:)]/2;
    QQ=reshape(QQ,4,size(QQ,3),1)';
    QQ(:,1)=sqrt((QQ(:,1)+1)/2);
    QQ(:,2:4)=(QQ(:,2:4)./repmat(QQ(:,1),[1,3]))/2;
else
    QQ=RR';
end

if(nargin>2 && (~isempty(Rinit)))
    if(size(Rinit,1)==3)
        Q=[Rinit(1,1,:)+Rinit(2,2,:)+Rinit(3,3,:)-1, Rinit(3,2,:)-Rinit(2,3,:),Rinit(1,3,:)-Rinit(3,1,:),Rinit(2,1,:)-Rinit(1,2,:)]/2;
        Q=reshape(Q,4,size(Q,3),1)';
        Q(:,1)=sqrt((Q(:,1)+1)/2);
        Q(:,2:4)=(Q(:,2:4)./repmat(Q(:,1),[1,3]))/2;
    else
        Q=Rinit';
    end
else
    Q=repmat([1,0,0,0],[N,1]);
    %Compute initial Q from a Spanning Tree
    i=zeros(N,1);
    %[~,a]=max(hist(sort(I(:)),[1:5530]));
    a=1;
    i(a)=1;
    while(sum(i)<N)
       SpanFlag=0;
        for j=1:size(I,2)
            if(i(I(1,j))==1&&i(I(2,j))==0)
                %Rinit(:,:,I(2,j))=RR(:,:,j)*Rinit(:,:,I(1,j));
                Q(I(2,j),:)=[ (QQ(j,1).*Q(I(1,j),1)-sum(QQ(j,2:4).*Q(I(1,j),2:4),2)),...  %scalar terms
                    repmat(QQ(j,1),[1,3]).*Q(I(1,j),2:4) + repmat(Q(I(1,j),1),[1,3]).*QQ(j,2:4) + ...   %vector terms
                    [QQ(j,3).*Q(I(1,j),4)-QQ(j,4).*Q(I(1,j),3),QQ(j,4).*Q(I(1,j),2)-QQ(j,2).*Q(I(1,j),4),QQ(j,2).*Q(I(1,j),3)-QQ(j,3).*Q(I(1,j),2)] ];   %cross product terms
                i(I(2,j))=1;
                SpanFlag=1;
            end
            if(i(I(1,j))==0&&i(I(2,j))==1)
                %Rinit(:,:,I(1,j))=RR(:,:,j)'*Rinit(:,:,I(2,j));
                Q(I(1,j),:)=[ (-QQ(j,1).*Q(I(2,j),1)-sum(QQ(j,2:4).*Q(I(2,j),2:4),2)),...  %scalar terms
                    repmat(-QQ(j,1),[1,3]).*Q(I(2,j),2:4) + repmat(Q(I(2,j),1),[1,3]).*QQ(j,2:4) + ...   %vector terms
                    [QQ(j,3).*Q(I(2,j),4)-QQ(j,4).*Q(I(2,j),3),QQ(j,4).*Q(I(2,j),2)-QQ(j,2).*Q(I(2,j),4),QQ(j,2).*Q(I(2,j),3)-QQ(j,3).*Q(I(2,j),2)] ];   %cross product terms
                i(I(1,j))=1;
                SpanFlag=1;
            end
        end
        if(SpanFlag==0&&sum(i)<N)
            fprintf('Relative rotations DO NOT SPAN all the nodes in the VIEW GRAPH');
            fprintf('Number of nodes in Spanning Tree = %d\n',sum(i));
            fprintf('Connected Nodes are given as output\n');
            fprintf('Remove extra nodes and retry\n');
            R=i;
            return;
        end
    end    
end

m=size(I,2);
i=[(1:m);(1:m)];i=i(:);
j=I(:);
s=repmat([-1;1],[m,1]);
k=(j~=1);
Amatrix=sparse(i(k),j(k)-1,s(k),m,N-1);

n=N-1;    i=I(1,:)-1;    j=I(2,:)-1;
r=[n*(i-1)+i,n*(j-1)+j,n*(i-1)+j,n*(j-1)+i]';
c=[1:m,1:m,1:m,1:m]';
s=[ones(2*m,1);-ones(2*m,1)];
k=[i&i,j&j,i&j,j&i];
r=r(k,1);  c=c(k,1);  s=s(k,1);
AtA=sparse(r,c,s,n*n,m);

w=zeros(size(QQ,1),4);W=zeros(N,4);

score=inf;    Iteration=0;    L1Step=2;

fprintf('Itr\tMaxChange\tTime\n');
disp(num2str([0 NaN toc]));

while(((score>=changeThreshold)||(L1Step<2))&&(Iteration<maxIters))
    if(score<changeThreshold);L1Step=L1Step*4;changeThreshold=changeThreshold/100;end;
        
    i=I(1,:);j=I(2,:);

    w(:,:)=[ (QQ(:,1).*Q(i,1)-sum(QQ(:,2:4).*Q(i,2:4),2)),...  %scalar terms
        repmat(QQ(:,1),[1,3]).*Q(i,2:4) + repmat(Q(i,1),[1,3]).*QQ(:,2:4) + ...   %vector terms
        [QQ(:,3).*Q(i,4)-QQ(:,4).*Q(i,3),QQ(:,4).*Q(i,2)-QQ(:,2).*Q(i,4),QQ(:,2).*Q(i,3)-QQ(:,3).*Q(i,2)] ];   %cross product terms

    w(:,:)=[ (-Q(j,1).*w(:,1)-sum(Q(j,2:4).*w(:,2:4),2)),...  %scalar terms
        repmat(-Q(j,1),[1,3]).*w(:,2:4) + repmat(w(:,1),[1,3]).*Q(j,2:4) + ...   %vector terms
        [Q(j,3).*w(:,4)-Q(j,4).*w(:,3),Q(j,4).*w(:,2)-Q(j,2).*w(:,4),Q(j,2).*w(:,3)-Q(j,3).*w(:,2)] ];   %cross product terms

%     i=w(:,1)<0;w(i,:)=-w(i,:);
%     theta2=acos(w(:,1));
%     B=((w(:,2:4).*repmat((2*theta2./sin(theta2)),[1,3])));
    s2=sqrt(sum(w(:,2:4).*w(:,2:4),2));
    w(:,1)=2*atan2(s2,w(:,1));
    i=w(:,1)<-pi;  w(i,1)=w(i,1)+2*pi;  i=w(:,1)>=pi;  w(i,1)=w(i,1)-2*pi;
    B=w(:,2:4).*repmat(w(:,1)./s2,[1,3]);
    
    
    
    B(isnan(B))=0;
  
    W(1,:)=[1 0 0 0];
    
    %W(2:end,2:4)=Amatrix\B;
    %if(Iteration==0&&nargin<3);W(2:end,2:4)=Amatrix\B;else 
                    W(2:end,2:4)=zeros(N-1,3);%end;
    W(2:end,2)=l1decode_pd(W(2:end,2),Amatrix, [],B(:,1),eps,L1Step,AtA);
    W(2:end,3)=l1decode_pd(W(2:end,3),Amatrix, [],B(:,2),eps,L1Step,AtA);
    W(2:end,4)=l1decode_pd(W(2:end,4),Amatrix, [],B(:,3),eps,L1Step,AtA);
    
    %score=norm(W(2:end,2:4),'fro')/N;
    score=max(sqrt(sum(W(2:end,2:4).*W(2:end,2:4),2)));
    
    theta=sqrt(sum(W(:,2:4).*W(:,2:4),2));
    W(:,1)=cos(theta/2);
    W(:,2:4)=W(:,2:4).*repmat(sin(theta/2)./theta,[1,3]);
    
    W(isnan(W))=0;
    
    Q=[ (Q(:,1).*W(:,1)-sum(Q(:,2:4).*W(:,2:4),2)),...  %scalar terms
        repmat(Q(:,1),[1,3]).*W(:,2:4) + repmat(W(:,1),[1,3]).*Q(:,2:4) + ...   %vector terms
        [Q(:,3).*W(:,4)-Q(:,4).*W(:,3),Q(:,4).*W(:,2)-Q(:,2).*W(:,4),Q(:,2).*W(:,3)-Q(:,3).*W(:,2)] ];   %cross product terms

    Iteration=Iteration+1;
    disp(num2str([Iteration score toc]));

end;

if(~QuaternionIP)
    R=zeros(3,3,N);
    for i=1:size(Q,1)
        R(:,:,i)=real(q2R(Q(i,:)));
    end
else
    R=Q';
end

if(Iteration>=maxIters);disp('Max iterations reached');end;

toc
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function H=AtDiagA(AtA,D)
n=sqrt(size(AtA,1));
H=reshape(full(AtA*D),n,n);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% l1decode_pd.m
%
% Decoding via linear programming.
% Solve
% min_x  ||b-Ax||_1 .
%
% Recast as the linear program
% min_{x,u} sum(u)  s.t.  -Ax - u + y <= 0
%                          Ax - u - y <= 0
% and solve using primal-dual interior point method.
%
% Usage: xp = l1decode_pd(x0, A, At, y, pdtol, pdmaxiter, cgtol, cgmaxiter)
%
% x0 - Nx1 vector, initial point.
%
% A - Either a handle to a function that takes a N vector and returns a M 
%     vector, or a MxN matrix.  If A is a function handle, the algorithm
%     operates in "largescale" mode, solving the Newton systems via the
%     Conjugate Gradients algorithm.
%
% At - Handle to a function that takes an M vector and returns an N vector.
%      If A is a matrix, At is ignored.
%
% y - Mx1 observed code (M > N).
%
% pdtol - Tolerance for primal-dual algorithm (algorithm terminates if
%     the duality gap is less than pdtol).  
%     Default = 1e-3.
%
% pdmaxiter - Maximum number of primal-dual iterations.  
%     Default = 50.

function xp = l1decode_pd(x0, A, At, y, pdtol, pdmaxiter,AtA)  

if (nargin < 5), pdtol = 1e-3; end
if (nargin < 6), pdmaxiter = 50; end

N = length(x0);
M = length(y);

alpha = 0.01;
beta = 0.5;
mu = 10;

gradf0 = [zeros(N,1); ones(M,1)];

x = x0;
Ax = A*x;
u = (0.95)*abs(y-Ax) + (0.10)*max(abs(y-Ax));

fu1 = Ax - y - u;
fu2 = -Ax + y - u;

lamu1 = -1./fu1;
lamu2 = -1./fu2;

Atv = A'*(lamu1-lamu2);

sdg = -(fu1'*lamu1 + fu2'*lamu2);
tau = mu*2*M/sdg;

rcent = [-lamu1.*fu1; -lamu2.*fu2] - (1/tau);
rdual = gradf0 + [Atv; -lamu1-lamu2];
resnorm = norm([rdual; rcent]);

pditer = 0;
done = (sdg < pdtol)| (pditer >= pdmaxiter);
while (~done)
  
  pditer = pditer + 1;
  
  w2 = -1 - 1/tau*(1./fu1 + 1./fu2);
  
  sig1 = -lamu1./fu1 - lamu2./fu2;
  sig2 = lamu1./fu1 - lamu2./fu2;
  sigx = sig1 - sig2.^2./sig1;
  
  
    w1 = -1/tau*(A'*(-1./fu1 + 1./fu2));
    w1p = w1 - A'*((sig2./sig1).*w2);
    %H11p = A'*(sparse(diag(sigx))*A);
    H11p = AtDiagA(AtA,sigx);
    %opts.POSDEF = true; opts.SYM = true;
    %[dx, hcond] = linsolve(H11p, w1p,opts);
    [dx, hcond] = linsolve(H11p, w1p);
    
    if (hcond < 1e-14)
      disp('Matrix ill-conditioned.  Returning previous iterate.  (See Section 4 of notes for more information.)');
      xp = x;
      return
    end
    Adx = A*dx;

  
  du = (w2 - sig2.*Adx)./sig1;
  
  dlamu1 = -(lamu1./fu1).*(Adx-du) - lamu1 - (1/tau)*1./fu1;
  dlamu2 = (lamu2./fu2).*(Adx + du) -lamu2 - (1/tau)*1./fu2;
  Atdv = A'*(dlamu1-dlamu2);
  
  % make sure that the step is feasible: keeps lamu1,lamu2 > 0, fu1,fu2 < 0
  indl = find(dlamu1 < 0);  indu = find(dlamu2 < 0);
  s = min([1; -lamu1(indl)./dlamu1(indl); -lamu2(indu)./dlamu2(indu)]);
  indl = find((Adx-du) > 0);  indu = find((-Adx-du) > 0);
  s = (0.99)*min([s; -fu1(indl)./(Adx(indl)-du(indl)); -fu2(indu)./(-Adx(indu)-du(indu))]);
  
  % backtrack 
  suffdec = 0;
  backiter = 0;
  while(~suffdec)
    xp = x + s*dx;  up = u + s*du;
    Axp = Ax + s*Adx;  Atvp = Atv + s*Atdv;
    lamu1p = lamu1 + s*dlamu1;  lamu2p = lamu2 + s*dlamu2;
    fu1p = Axp - y - up;  fu2p = -Axp + y - up;
    rdp = gradf0 + [Atvp; -lamu1p-lamu2p];
    rcp = [-lamu1p.*fu1p; -lamu2p.*fu2p] - (1/tau);
    suffdec = (norm([rdp; rcp]) <= (1-alpha*s)*resnorm);
    s = beta*s;
    backiter = backiter + 1;
    if (backiter > 32)
      disp('Stuck backtracking, returning last iterate.  (See Section 4 of notes for more information.)')
      xp = x;
      return
    end
  end
  
  % next iteration
  x = xp;  u = up;
  Ax = Axp;  Atv = Atvp;
  lamu1 = lamu1p;  lamu2 = lamu2p;
  fu1 = fu1p;  fu2 = fu2p;
  
  % surrogate duality gap
  sdg = -(fu1'*lamu1 + fu2'*lamu2);
  tau = mu*2*M/sdg;
  rcent = [-lamu1.*fu1; -lamu2.*fu2] - (1/tau);
  rdual = rdp;
  resnorm = norm([rdual; rcent]);
  
  done = (sdg < pdtol) | (pditer >= pdmaxiter);
  
  %disp(sprintf('Iteration = %d, tau = %8.3e, Primal = %8.3e, PDGap = %8.3e, Dual res = %8.3e',...
  %  pditer, tau, sum(u), sdg, norm(rdual)));

  %disp(sprintf('                  H11p condition number = %8.3e', hcond));
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
