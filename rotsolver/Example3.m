% Not fully connected
[R1,RR1,I1]=RandomSO3Graph(100,.5,.1,250);
[R2,RR2,I2]=RandomSO3Graph(500,.5,.1,100);
R=cat(3,R1,R2); RR=cat(3,RR1,RR2); I=cat(2,I1,I2+size(R1,3));
fprintf('Validating Input data w.r.t Ground truth\n');
ValidateSO3Graph(R,RR,I);close all;
fprintf('Performing Robust SO3 Graph Estimation\n');
Rest=AverageSO3Graph(RR,I);
fprintf('Comparing Estimated Rotations to Ground truth\n');
CompareRotationGraph(R,Rest);