import numpy as np
import scipy.stats
import random
import multiprocessing as mp

from ._wrapper_mfas import cmfas, creindex, cflip_neg_edges, cbroken_weight
"""
This file provides an implementation of the 1DSfM algorithm described in the 2014 ECCV paper [1]. 1DSfM
preprocesses instances of the SfM Translations problem to detect outlier measurements. This makes
the problem easier to solve reliably.

Please refer to [1] for a full description of the principles behind the method. Briefly, a SfM
Translations problem is projected onto a single dimension, and an ordering problem is solved there.
Then results from many such ordering problems are aggregated in a voting procedure.

References:
[1] K. Wilson, N. Snavely, "Robust Global Translations with 1DSfM", ECCV 2014.
"""

def oneDSfM(edges, poses, num_samples=48, weights_file=None,
            threshold=0.10, num_workers=None, verbose=False):
    """
    Remove outliers from a SfM translations problem with 1DSfM.

    Inputs:
        edges         n-by-2 numpy array of ints 
        poses         n-by-3 numpy array of unit vectors, corresponding to the edges
        num_samples   number of 1D subproblems to aggregate 
        weights_file  optionally write the aggregate of broken edge weight here
        threshold     lower rejects more edges, >= 0
        num_workers   number of parallel workers (default: num available threads)
        verbose       print timing and debugging output

    Returns:
        ind_good      list of indices into 1:n of edges to keep; reject all other edges
    """

    edges, _ = creindex(np.copy(edges))

    if verbose:
        print '[1DSfM] input problem with {} edges'.format(len(edges))
        print '[1DSfM] using threshold = {}'.format(threshold)
        print '[1DSfM] using num samples = {}'.format(num_samples)

    projection_dirs = oneDSfM_sample_directions(poses, num_samples)

    # prepare a description of each subproblem to pass to a multiprocessing pool 
    pool_args = []
    for dir in projection_dirs:
        pool_args.append((poses, edges, dir))

    # distribute the work and block. This will return 1D orderings and weights.
    results = mp.Pool(num_workers).map(pool_worker, pool_args)

    # accumulate which edges were broken by each ordering
    broken_weight = np.sum(np.vstack(results), axis=0) / num_samples

    # make the decisions
    ind_good = np.where(broken_weight <= threshold)[0]

    if verbose:
        print '[1DSfM] removed {} edges'.format(len(edges) - len(ind_good))

    if weights_file:
        np.savetxt(weights_file, broken_weight)

    return ind_good


def pool_worker((poses, edges, dir)):
    """
    Run an edgelist based version of the mfas solver written in c++ and tied in through a cython
    interface. This version assumes non-negative edge weights.
    """
    weights = np.dot(poses, dir)
    (edges, weights) = cflip_neg_edges(edges, weights)
    order = cmfas(edges, weights)
    return cbroken_weight(edges, weights, order)

def broken_weight(edges, weights, order):
    cost = np.zeros(len(weights)) 
    inv_order = inverse_permutation(order)
    for j, e in enumerate(edges):
        x0 = inv_order[e[0]]
        x1 = inv_order[e[1]]
        if np.sign(x1 - x0) != np.sign(weights[j]):
            cost[j] += np.absolute(weights[j])
    return cost

def inverse_permutation(perm):
    """
    Given an order (a permutation of 1..n) return an inverse lookup.
    That is, if per,m[i] = j, then invPerm[j] = i.
    """
    nVertices = np.max(perm)+1
    invPerm = [-1]*nVertices
    for i in xrange(nVertices):
        invPerm[perm[i]] = i
    return invPerm

def oneDSfM_sample_directions(poses, num_samples):
    """
    Given an array of unit vectors, sample new unit vectors according to the density of the input on
    the sphere.

    poses: n-by-3 np array of unit vectors
    num_samples: number of sampled directions to return

    samples: num_samples-by-3 np array of new sampled directions
    """
    # sample a set of directions to project this graph onto
    (zen, azm) = euclideanToSphere(poses)
    sampleDirs = kdeSample(azm, zen, num_samples)
    return sphereToEuclidean(sampleDirs[:,0], sampleDirs[:,1])

def kdeSample(x, y, samples):
    """
    Take 2D vectors x and y, fit a kernel density estimate to them, and
    resample new points.
    """
    # downsample if we have ton of data- we don't need to be that precise
    inputDownsamples = 2000
    if len(x) > inputDownsamples:
        x = x[random.sample(xrange(len(x)), inputDownsamples)]
        y = y[random.sample(xrange(len(y)), inputDownsamples)]
    rvs = np.column_stack( (x, y) )

    # fit a kde to the data (this can take a little bit)
    kde = scipy.stats.kde.gaussian_kde(rvs.T)
    return kde.resample(size=samples).T

def euclideanToSphere(poses):
    """
    Convert (x,y,z) OpenGL coord system points on the unit sphere S2 to
    zenith/azimuth pairs. Angles are given in a compass frame:
    zenith=0 is always straight up (y-axis). Azimuth=0 points down -z, and 
    Azimuth=pi/2 points down +x.
    """
    zen = np.arccos(poses[:,1])
    azm = np.arctan2(poses[:,0], -poses[:,2])
    return (zen, azm)

def sphereToEuclidean(azm, zen):
    y = np.cos(zen)
    x = np.sin(azm) * np.sin(zen)
    z = -np.cos(azm) * np.sin(zen)
    return np.column_stack( ( x, y, z) )
