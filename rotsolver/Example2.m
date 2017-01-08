% Quaterion input/output
[R,RR,I]=RandomSO3Graph(500,.5,.1,250);
fprintf('Validating Input data w.r.t Ground truth\n');
ValidateSO3Graph(R,RR,I);close all;
for i=1:size(RR,3);QQ(:,i)=R2q(RR(:,:,i));end
fprintf('Performing Robust SO3 Graph Estimation\n');
Qest=AverageSO3Graph(QQ,I);
for i=1:size(Qest,2);Rest(:,:,i)=q2R(Qest(:,i));end
fprintf('Comparing Estimated Rotations to Ground truth\n');
CompareRotationGraph(R,Rest);