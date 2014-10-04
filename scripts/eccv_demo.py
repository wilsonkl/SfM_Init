
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
This is a demo driver for the ECCV 2014 1DSfM / trans solver paper [1]. It demonstrates
how to call the methods and solvers in the included sfminit package, producing a robust
solution to the SfM translations problem.

[1] K. Wilson and N. Snavely. Robust Global Translations with 1DSfM. ECCV 2014.
"""

def make_cli():
    parser = argparse.ArgumentParser(
        description="Run 1DSfM paper on a supplied dataset")
    parser.add_argument("dataset_dir", help="problem description")
    parser.add_argument("output_dir", help="where to write the results")
    return parser.parse_args()

def run_pipeline():
    args = make_cli()
    data_dir = args.dataset_dir
    out_dir = args.output_dir
    input = {'EGs'     : os.path.join(data_dir, 'EGs.txt'),
         'ccs'     : os.path.join(data_dir, 'cc.txt'),
         'tracks'  : os.path.join(data_dir, 'tracks.txt'),
         'coords'  : os.path.join(data_dir, 'coords.txt'),
         'gt_soln' : os.path.join(data_dir, 'gt_bundle.out')   }

    if not os.path.isdir(out_dir):
        print '[eccv_demo] Error: output dir does not exist! Aborting.'
        sys.exit(1)

    ################
    ### PIPELINE ###
    ################

    print '[eccv_demo] Reading input files'
    cc = np.loadtxt(input['ccs'])
    tracks = sfminit.Tracks.from_file(input['tracks'])
    coords = sfminit.Coords.from_file(input['coords'])
    if os.path.exists(input['gt_soln']): # skip comparisons if the gt isn't there
        gt_bundle = sfminit.Bundle.from_file(input['gt_soln'])

    print '[eccv_demo] Running global rotations solver'
    models = sfminit.ModelList.from_EG_file(input['EGs'])
    edges, pairwise_rotations = models.get_rotations_problem()
    global_rot_ids, global_rots = sfminit.solve_global_rotations(edges, pairwise_rotations, cc=cc)

    print '[eccv_demo] Making a translations problem'
    models.apply_rotations_solution(global_rot_ids, global_rots)
    trans_problem = models.get_translations_problem()
    trans_problem.add_track_edges(tracks, coords, global_rot_ids, global_rots)

    print '[eccv_demo] Running 1DSfM to remove outliers'
    clean_trans_problem = copy.deepcopy(trans_problem)
    clean_trans_problem.run_1DSfM(verbose=True)

    print '[eccv_demo] Solving translations problem'
    X_ids, X = clean_trans_problem.solve(loss=None, function_tol=1e-14, parameter_tol=1e-14, max_iterations=1000)

    ###########################
    ### EVALUATION / OUTPUT ###
    ###########################

    print '[eccv_demo] Writing output files'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    sfminit.write_rot_file(os.path.join(out_dir, 'rot_solution.txt'), global_rot_ids, global_rots)
    trans_problem.write(os.path.join(out_dir, 'orig_trans_problem.txt'))
    clean_trans_problem.write(os.path.join(out_dir, '1DSfM_trans_problem.txt'))
    sfminit.write_trans_soln_file(os.path.join(out_dir, 'trans_problem_solution.txt'), X_ids, X)

    if os.path.exists(input['gt_soln']):
        print '[eccv_demo] Running comparisons to ground truth'
        # get the common camera locations
        X_gt = []
        X_soln = []
        for i,x in zip(X_ids, X):
            if i < len(gt_bundle.cameras) and gt_bundle.cameras[i].initialized():
                X_soln.append(x)
                X_gt.append(gt_bundle.cameras[i].location())
        X_gt = np.array(X_gt)
        X_soln = np.array(X_soln)
        # robustly align our solution to ground truth9
        (R, t, s, _) = sfminit.robust_horn(X_gt, X_soln)
        residuals = np.sqrt(np.sum(np.square(X_gt - s*np.dot(R, X_soln.T).T - t), axis=1))
        np.savetxt(os.path.join(out_dir, 'trans_solution_gt_error.txt'), residuals)

    print '[eccv_demo] Done!'

if __name__ == '__main__':
    sys.exit(run_pipeline())
