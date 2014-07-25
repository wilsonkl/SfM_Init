import os
import shutil
import numpy as np
import tempfile
import subprocess
from .sfminittypes import EG, write_EGs_file, read_rot_file

def solve_global_rotations(indices, pairwise_rotations, cc=None):
    """
    Solve the multiple rotations averaging problem using Chatterjee and Govindu's 
    L1_IRLS method. The implementation is in Matlab, so this is a wrapper that writes
    the problem to temporary text files and calls Matlab through the command line.

    Input:
        indices:    a list of pairs (i,j)
        pairwise_rotations: a list of rotation matrices Rij
        cc: a list of integers {i}. Only compute global rotations on indices in cc

    Returns:
        ind: a list of indices (vertex numbers)
        R: rotation matrices corresponding to ind, which are approximately 
            consistent with the pairwise rotations, ie Rij ~ Ri * Rj'
    """
    ############
    # Path to rot solver Matlab source dir
    ROT_SOLVER_DIR = os.path.abspath(os.path.join(
        os.path.dirname(__file__),'..','rotsolver'))
    ############

    # get a working temp directory
    tmpdir = tempfile.mkdtemp()

    # write the input files
    ccfile = os.path.join(tmpdir, 'cc.txt')
    np.savetxt(ccfile, cc, fmt='%d')
    egfile = os.path.join(tmpdir, 'eg.txt')
    EGs = []
    for (i,j), Rij in zip(indices, pairwise_rotations):
        # tij isn't used, so just spoof it to 0's
        EGs.append(EG(i, j, Rij, np.zeros((3,1))))
    write_EGs_file(egfile, EGs)

    # call the rot solver
    log = os.path.join(tmpdir, 'log.txt')
    rotfile = os.path.join(tmpdir, 'rots.txt')

    with open(os.devnull, 'wb') as devnul:
        subprocess.call(['matlab', '-nodisplay', '-nojvm', '-logfile', log, '-r', 
            "try; " +
                "rot_driver('{}', '{}', '{}');".format(egfile, rotfile, ccfile) +
            "catch err; " +
                "disp(getReport(err)); " +
            "end; " +
            "exit;" ],
            cwd=ROT_SOLVER_DIR, stdout=devnul, stderr=devnul)

    # if the rot solver wasn't successful (ie, a bug), give a message and an error
    if not os.path.exists(rotfile):
        print '[rotsolver:solve_global_rotations] Error! The Matlab rotsolver did not complete.'
        print 'LOG:'
        with open(log, 'r') as fin:
            print fin.read()
        shutil.rmtree(tmpdir)
        raise Exception("Global Rotations solver was not successful!")

    # read the result and clean up
    ind, R = read_rot_file(rotfile)
    shutil.rmtree(tmpdir)
    return ind, R
