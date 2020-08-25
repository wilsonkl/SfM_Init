SfM Init  User's Manual
=======================
copyright 2012-2014 Kyle Wilson (wilsonkl@cs.cornell.edu)

based on the global structure from motion work with Noah Snavely


Introduction:
-------------
SfM Init is a toolkit for solving some parts of a global Stucture from Motion
pipeline. Such a pipeline would typically reconstruct the 3D (sparse) geometry
of some scene given many photos with the following steps:

1. Feature Detection
2. Feature Matching
3. Two View Model Estimation
4. Solve for Globally Consistant Camera rotations (wrappers included)
5. Solve for Globally Consistant Camera positions (included)
6. Refine the Model through Bundle Adjustment

SfM Init goes from pairwise geometry to a good guess of global geometry, which
is then the initialization to bundle adjustment. SfM Init does not ship with
a system for computing pairwise models or with a bundle adjustor.

SfM Init uses the excellent [Rotations Graph Averaging](http://www.ee.iisc.ac.in/labs/cvl/research/rotaveraging/)
package by Chatterjee and Govindu [2]. This is available from their project webpage.
SfM Init only provides wrappers to call this code.

Conditions of Use
-----------------
SfM Init is distributed under a Simplified BSD / 2-clause license. If you use
SfM Init for a publication, please cite the following paper:

Kyle Wilson and Noah Snavely. Robust Global Translations with 1DSfM. ECCV 2014.

What's Included
---------------
SfM Init is distributed as a python package. Some of the key computations are
written in C++ and are wrapped in via cython. This should be transparent to
the user.

This toolkit includes a wrapper to Chatterjee and Govindu's rotations averaging
code, as well 1DSfM translations problem outlier detection and a chordal
distance based translations solver.

Examples of how to use SfM Init are in the scripts directory. In particular,
`scripts/eccv_demo.py` shows how to run all the steps of the pipeline described in
[1] on the datasets provided at the [1DSfM project page](https://research.cs.cornell.edu/1dsfm/).
Look in `scripts/coords_demo.py` for some examples of how the coordinate systems
in 1DSfM are used.

Before You Begin
----------------
SfM Init is python based, and depends on the following standard python packages
--- python 2.7, numpy, and scipy. Additionally, to compile the C++ numeric
routines, it requires cython.

The translations solver requires the [Ceres Solver](http://ceres-solver.org)
nonlinear least squares package.

Chatterjee and Govindu's rotations averaging code can be found at their [project
page](http://www.ee.iisc.ac.in/labs/cvl/research/rotaveraging/).
Unzip the contents of this tar file into the rotsolver directory.

Finally, to compile the numerics routines, run the following from SfM Init's
root directory:

    > python setup.py build_ext --inplace

If this fails, check the compile and link paths in setup.py to be sure that
cython can see the Ceres include and lib files, as well as Eigen and the SuiteSparse
libs that Ceres depends on.

The most common causes of build failure reported to the authors come down to
very old versions of python/cython or libceres not being visible during linking.

File Formats
------------
Running the SfM Init pipeline in `scripts/eccv_demo.py` requires several files
describing reconstructed two view models. Our datasets are available on our [project
page](https://research.cs.cornell.edu/1dsfm/). Note that the photos and data files are
distributed separately. The dataset files describe a single connected component, but
all images related to each Landmark are given, in case these are useful in another
context. If the following descriptions are insufficient, please also look at
`scripts/coords_demo.py` for some further examples.

Input files:

*   cc.txt: This is a list of camera indices, one per line, specifying which
    images to reconstruct. These form a single connected component of EGs.
*   EGs.txt: Two-image models are listed in this file, one per line. The format
    is: `<i> <j> <Rij> <tij>` where i and j are camera indices, Rij is a row-major
    pairwise rotation matrix, and tij is the position of camera j in camera i's
    coordinate system. If Ri and Rj are the rotation matrices of cameras i and
    j (world-to-camera maps) then in the absence of noise Rij = Ri * Rj', ie
    Rij is the pose of camera j in camera i's coordinate system (where a pose
    is the transpose of a rotation matrix, a camera-to-world map). All of these
    EGs are within the connected component.
*   coords.txt: This is a summary of the local image features found in each
    image in the connected component. Each image starts with a header, followed
    by a row for each key in that image. The header contains the number of keys
    in the image, the focal length in pixels, and the principal points (half the
    width and height). Keys are given as
    `<key number> <x> <y> <ignore0> <ignore1> <R> <G> <B>` where R,G,B are a
    sampled rgb color. The keys are numbered sequentially. The coordinates `<x>`
    and `<y>` are in a coordinate system from SIFT: the center of the upper
    left pixel is (1,1), +x points right, and +y points down.
*   tracks.txt: This describes which features in the images in the connected
    component have been matched into a track. The first line is the number of tracks, and then each
    following line is a single track with format: `N <img1> <feature1> ... <imgN>
    <featureN>`

Output formats:

*   prob.txt: SfM Init reads and writes translations problems as edge lists. A
    translations problem file has the format: `<i> <j> <tij>` where tij is a unit
    vector pointing from node i to node j.
*   soln.txt: SfM Init reads and writes solutions to translations problems as a
    vertex list. Each line is `<i> <Xi>` where Xi is a 3-vector.
*   rots.txt: SfM Init reads and writes global rotations solutions as a vertex
    list: `<i> <Ri>`, where Ri is a 3-by-3 rotation matrix written row major.

Other included files:

*   list.txt: a list of all of the images in a dataset, as well as image
    focal lengths in pixels. The format per line is `<image name> 0 <focal
    length>`, although when the focal length is unknown the latter two fields are
    omitted. SfM Init ignores photos with unknown focal length. The line number
    of an image in this file is its identifying index in the rest of the
    toolkit. (Note that this list typically includes many more images than are in
    connected component supplied above.)
*   bundle.out: This is a reconstruction of approximately the same component of
    the dataset which is described by the other files. Do to differences in the
    reconstruction method, it may have a few extra images, or fail to reconstruct
    some images in the connected component. This reconstruction is made with [3], and
    is provided for comparison purposes. See the bundler manual for details
    about the file format: http://www.cs.cornell.edu/~snavely/bundler

Contact
-------
Please email Kyle Wilson (wilsonkl@cs.cornell.edu) with any questions, comments,
or bug reports.

References
----------
[1] Kyle Wilson and Noah Snavely. Robust Global Translations with 1DSfM. ECCV
2014.

[2] Avishek Chatterjee and Venu Madhav Govindu. Efficient and Robust Large-Scale
Rotation Averaging. ICCV 2013.

[3] Noah Snavely, Steven M. Seitz, and Richard Szeliski. Photo Tourism:
Exploring Photo Collections in 3D. SIGGRAPH Conf. Proc., 2006.
