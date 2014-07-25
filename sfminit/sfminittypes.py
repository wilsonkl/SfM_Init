import numpy as np
from .utils import normalize_rows

class EG(object):
    """
    Container class: an EG (epipolar geometry) has a first index i,
    a second index j, a camera-to-world rotation matrix, and a world
    coords translation vector
    """
    def __init__(self, i=-1, j=-1, R=None, t=None):
        self.i = i
        self.j = j
        self.R = R
        self.t = t

    def __str__(self):
        return ''.join(['{} {} '.format(self.i, self.j),
                '{} {} {} '.format(self.R[0,0], self.R[0,1], self.R[0,2]),
                '{} {} {} '.format(self.R[1,0], self.R[1,1], self.R[1,2]),
                '{} {} {} '.format(self.R[2,0], self.R[2,1], self.R[2,2]),
                '{} {} {}'.format(self.t[0,0], self.t[1,0], self.t[2,0])])

def read_rot_file(fname):
    """
    Format: one rotation matrix per line: [ind] [R]
    R is printed row-major.

    Returns:
        ind: nlines long np array of ints
        R: nlines-by-3-by-3 np array
    """
    arr = np.loadtxt(fname)
    ind = arr[:,0].astype(int)
    R = arr[:,1:10].reshape((-1,3,3))
    return ind, R

def write_rot_file(fname, ind, R):
    """
    Format: one rotation matrix per line: [ind] [R]

    ind: length n iterable in ints
    R: length n iterable of rotmats
    """
    if len(ind) != len(R):
        raise ValueError('ind and R must be the same length')
    with open(fname, 'w') as f:
        for i,r in zip(ind,R):
            f.write(str(i) + ' ' + ' '.join(map(str,r.flatten())) + '\n')

def read_trans_soln_file(fname):
    """
    Format: [ind] [x] [y] [z]

    Returns:
        ind: nlines long np array of ints
        X: nlines-by-3 np array of coordinates
    """
    arr = np.loadtxt(fname)
    ind = arr[:,0].astype(int)
    X = arr[:,1:4]
    return ind, X

def write_trans_soln_file(fname, ind, X):
    """
    Format: [ind] [x] [y] [z]

    Params:
        ind: length n iteratable of ints
        X: length n iterable of 3-vectors
    """
    if len(ind) != len(X):
        raise ValueError('ind and X must be the same length')
    with open(fname, 'w') as f:
        for i,x in zip(ind,X):
            f.write(str(i) + ' ' + ' '.join(map(str,x.flatten())) + '\n')

def read_trans_prob_file(fname):
    """
    Format: [i] [j] [tx] [ty] [tz]

    Returns:
        edges: nlines-by-2 np array of ints
        poses: nlines-by-3 np array, rows have unit length
    """
    arr = np.loadtxt(fname)
    edges = arr[:,0:2].astype(int)
    poses = normalize_rows(arr[:,2:5])
    return edges, poses

def write_trans_prob_file(fname, edges, poses):
    """
    Format: [i] [j] [tx] [ty] [tz]

    Params:
        edges: length n iterable of ints (i,j)
        poses: length n iterable of unit 3-vectors
    """
    if len(edges) != len(poses):
        raise ValueError('edges and poses must be the same length')
    with open(fname, 'w') as f:
        for e,t in zip(edges,poses):
            f.write(str(e[0]) + ' ' + str(e[1]) + ' ' + ' '.join(map(str,t.flatten())) + '\n')

def read_edge_weight_file(fname):
    """
    Format: [i] [j] [w]

    Returns:
        edges: nlines-by-2 np array of ints
        weights: nlines long np array
    """
    arr = np.loadtxt(fname)
    edges = arr[:,0:2].astype(int)
    weights = arr[:,2]
    return edges, weights

def write_edge_weight_file(fname, edges, weights):
    """
    Format: [i] [j] [w]

    Params:
        edges: length n iterable of ints (i,j)
        weights: length n iterable
    """
    if len(edges) != len(weights):
        raise ValueError('edges and weights must be the same length')
    with open(fname, 'w') as f:
        for e,w in zip(edges,weights):
            f.write('{0:d} {1:d} {2:f}\n'.format(e[0], e[1], w))

def read_EGs_file(fname):
    """
    Format: [i] [j] [R] [t]
    One line per EG.
    R is a 3-by-3 matrix written row major, and t is a unit 3-vector.

    Returns:
        EGs: a list of bundle_types.EG objects, one per line of the input file
    """
    arr = np.loadtxt(fname)
    EGs = []
    for row in arr:
        EGs.append(EG(int(row[0]), int(row[1]),
            row[2:11].reshape((3,3)), row[11:]))
    return EGs

def write_EGs_file(fname, EGs):
    """
    Format: [i] [j] [R] [t]
    One line per EG.
    R is a 3-by-3 matrix written row major, and t is a unit 3-vector.

    Params:
        EGs: a list of bundle_types.EG objects
    """
    with open(fname, 'w') as f:
        for EG in EGs:
            f.write(str(EG) + '\n')

