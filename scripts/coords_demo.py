
# Use this to import a package (sfminit) in a sibling directory
# Guido says we shouldn't do this, so there's no tidy syntax.
import sys; import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import sfminit

import argparse
import numpy as np
import copy
import sys


"""
This script manipulates input files to demonstrate the coordinate systems in the
1DSfM datasets. All calculations are written out longhand, rather than being
buried inside methods. Hopefully this will help clear up questions about
coordinate systems.

[1] K. Wilson and N. Snavely. Robust Global Translations with 1DSfM. ECCV 2014.
"""

def make_cli():
    parser = argparse.ArgumentParser(
        description="Run 1DSfM paper on a supplied dataset")
    parser.add_argument("dataset_dir", help="problem description")
    return parser.parse_args()

def run_pipeline():
    args = make_cli()
    data_dir = args.dataset_dir
    input = {'EGs'     : os.path.join(data_dir, 'EGs.txt'),
         'ccs'     : os.path.join(data_dir, 'cc.txt'),
         'tracks'  : os.path.join(data_dir, 'tracks.txt'),
         'coords'  : os.path.join(data_dir, 'coords.txt'),
         'gt_soln' : os.path.join(data_dir, 'gt_bundle.out')   }

    if not os.path.exists(input['gt_soln']):
        print '[coords_demo] Error: gt_bundle does not exist! Aborting.'
        sys.exit(1)

    # Load all data files from the dataset
    print '[coords_demo] Reading input files from ', data_dir
    EGs = sfminit.read_EGs_file(input['EGs'])
    cc = np.loadtxt(input['ccs'])
    tracks = sfminit.Tracks.from_file(input['tracks'])
    coords = sfminit.Coords.from_file(input['coords'])
    gt_bundle = sfminit.Bundle.from_file(input['gt_soln'])

    ##################################
    ### DEMO 1: RELATIVE ROTATIONS ###
    ##################################

    # Plan: Compare the two-view relative rotations provided in EGs.txt with
    # the relative rotations induced by the 'ground truth' solution in
    # gt_bundle.out.
    print '[coords_demo] Comparing Rij in EGs.txt to Ri in gt_bundle.out:'

    residuals = []
    for EG in EGs:
        i,j = EG.i, EG.j # camera indices of this two-view geometry
        Rij = EG.R # two-view model's relative rotation

        if not (gt_bundle.cameras[i].initialized() and
                gt_bundle.cameras[j].initialized()):
            continue

        Ri_gt = gt_bundle.cameras[i].R # rotation matrix of cam i in gt solution
        Rj_gt = gt_bundle.cameras[j].R # rotation matrix of cam j in gt solution
        Rij_gt = np.dot(Ri_gt, Rj_gt.T) # Rij = Ri * Rj.T, by definition

        residuals.append(sfminit.SO3_geodesic_metric(Rij, Rij_gt))
    print '[coords_demo] Relative rotation residual quantiles (in degrees):'
    percentiles = [25, 50, 75]
    print np.percentile(np.degrees(residuals), percentiles)

    #####################################
    ### DEMO 2: RELATIVE TRANSLATIONS ###
    #####################################

    # Plan: Compare the two-view relative translations provided in EGs.txt with
    # the relative translations induced by the 'ground truth' solution in
    # gt_bundle.out.
    print '[coords_demo] Comparing tij in EGs.txt to Xi in gt_bundle.out:'

    residuals = []
    for EG in EGs:
        i,j = EG.i, EG.j # camera indices of this two-view geometry
        tij = EG.t # two-view model's relative translation
                   # = direction to cam j in cam i's coords
        tij /= np.linalg.norm(tij) # just in case

        if not (gt_bundle.cameras[i].initialized() and
                gt_bundle.cameras[j].initialized()):
            continue

        Ri_gt = gt_bundle.cameras[i].R # rotation matrix of cam i in gt solution
        Rj_gt = gt_bundle.cameras[j].R # rotation matrix of cam j in gt solution
        ti_gt = gt_bundle.cameras[i].t # location of world 0 in cam i coords
        tj_gt = gt_bundle.cameras[j].t # location of world 0 in cam j coords

        Xi_gt = np.dot(Ri_gt.T, -ti_gt) # location of cam i in world coords
        Xj_gt = np.dot(Rj_gt.T, -tj_gt) # location of cam j in world coords
        tij_gt_world = (Xj_gt - Xi_gt) / np.linalg.norm(Xj_gt - Xi_gt)
                                # unit vector from i to j in world coords

        tij_gt = np.dot(Ri_gt, tij_gt_world)
                                # unit vector from i to j in cam i's coords


        residuals.append(np.arccos(np.clip(np.dot(tij_gt.T, tij), -1.0, 1.0)))

    print '[coords_demo] Relative translation residual quantiles (in degrees):'
    percentiles = [25, 50, 75]
    print np.percentile(np.degrees(residuals), percentiles)

    ###################################
    ### DEMO 3: BUNDLE REPROJECTION ###
    ###################################

    # Plan: This isn't comparisoning anything. This is just a demo of how to
    # reproject points into cameras. Note that this is identical to the
    # equations in the bundler manual.

    print '[coords_demo] Computing reprojections in gt_bundle.out:'

    residuals = []
    for pt in gt_bundle.points:
        pt_X = pt.X
        for (img_num, feature_num, pt_x, pt_y) in pt.observations:
            # In the 1DSfM datasets, pt_x and pt_y aren't set. In a general
            # bundle file, they probably would be. TODO: why are they missing?

            cam = gt_bundle.cameras[img_num]
            if not cam.initialized():
                continue
            P = np.dot(cam.R, pt_X) + cam.t
            p = -1*P[0:2]/P[2]
            norm_p2 = np.sum(np.square(p))
            r = 1 + cam.k1 * norm_p2 + cam.k2 * norm_p2 * norm_p2
            pt_reprojected = cam.f * r * p



if __name__ == '__main__':
    sys.exit(run_pipeline())
