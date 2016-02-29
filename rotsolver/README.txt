In this package we provide our implementation of the methods described in 
Avishek Chatterjee and Venu Madhav Govindu, "Efficient and Robust Large-Scale Rotation Averaging", ICCV 2013.
If you use this package, please cite the above listed paper. The corresponding bibtex entry is :

@InProceedings{Chatterjee_2013_ICCV,
author = {Avishek Chatterjee and Venu Madhav Govindu},
title = {Efficient and Robust Large-Scale Rotation Averaging},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {December},
year = {2013}
}

ACKNOWLEDGMENT : We thank Justin Romberg for permission to use a modified version of l1magic from http://users.ece.gatech.edu/~justin/l1magic/ in our implementation.

CONTACT : For comments/help please email eeavishekchatterjee@gmail.com

USAGE : We provide below a brief guide the functions available in this package. Please note that the functions in this package will be subsumed in the future into a toolbox for estimation on SO(3).

Conversion Functions:
==============
R2w.m: Convert rotation matrix to angle-axis form
w2R.m: Convert angle-axis form to rotation matrix
R2q.m: Convert rotation matrix to quaternion
q2R.m: Convert quaternion to rotation matrix

Main Functions:
==============

RandomSO3Graph.m: Simulating a random graph of SO(3) relationships
ValidateSO3Graph.m: Check the level of error in the simulated graph or relative rotations

AverageSO3Graph.m: Find a robust SO(3) averaging estimate [THIS IS THE MAIN FUNCTION HERE]
CompareRotationGraph.m: Align and compare two rotation estimates

Brief Function Descriptions:
===========================

[R,RR, I] = RandomSO3Graph(N,Completeness,Sigma,Noutlier)
is a simple program for generating simulated rotations

Usage :
N: number of nodes
Completeness: (0 to 1) fraction of connecting edges (relative rotations) observed.
Sigma: Noise Level. For example, Sigma=0.1 corresponds to .1*sqrt(3)*180/pi = 10 degrees standard deviation (approximately)
NOTE : The noise distribution is Gaussian in the Lie-algebra so(3) and not on the group SO(3)
Noutliers: Number of outliers.
R: ground truth rotations. 
RR: Contaminated Relative Rotations.
I: edge indices.
Example:[R, RR, I]=RandomSO3Graph(100,.5,.05,25);

R = BoxMedianSO3Graph(RR,I,{Rinit})
is an implementation of the L1RA algorithm

RobustMeanSO3Graph(RR,I,SIGMA,Rinit)
is an implementation of the IRLS algorithm.
SIGMA is in degrees.

Example: 
Rb=BoxMedianSO3Graph(RR,I); 
Rm=RobustMeanSO3Graph(RR,I,5,Rb);

CompareRotationGraph(R1,R2)
is for comparing the estimate with the ground truth
Example: CompareRotationGraph(R,Rm);

The user may also consult the example functions provided.
