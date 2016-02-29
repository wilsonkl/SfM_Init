[R,RR,I]=RandomSO3Graph(500,.5,.1,250);
fprintf('Validating Input data w.r.t Ground truth\n');
ValidateSO3Graph(R,RR,I);close all;
fprintf('Performing Robust SO3 Graph Estimation\n');
Rest=AverageSO3Graph(RR,I);
fprintf('Comparing Estimated Rotations to Ground truth\n');
CompareRotationGraph(R,Rest);