"""
Routines for rigidly aligning point clouds
"""
import numpy as np


def robust_horn(X0, X1, ransac_rounds=4096, ransac_threshold=10.0):
    """
    Use Horn's method to align corresponding points 
    clouds X0 and X1. Wrap this inside a RANSAC loop
    for robustness.
    """
    hypothesis_size = 3
    n = X0.shape[0]

    hypothesis = None
    max_inliers = -1

    # main ransac loop
    for i in xrange(ransac_rounds):
        # solve a random minimal problem
        ind = np.random.choice(n, size=hypothesis_size, replace=False)
        (R, r0, s, err) = horns_method(X0[ind,:], X1[ind,:])
        
        # compute residuals on the full problem
        residuals_sq = np.sum(np.square(X0 - s*np.dot(R, X1.T).T - r0), axis=1)
        num_inliers = np.count_nonzero(residuals_sq < ransac_threshold*ransac_threshold)

        # update our hypothesis if this one is better
        if num_inliers > max_inliers:
            hypothesis = (R, r0, s)
            max_inliers = num_inliers

    # put a final polish on this:
    (R, r0, s) = hypothesis
    residuals_sq = np.sum(np.square(X0 - s*np.dot(R, X1.T).T - r0), axis=1)
    inliers = np.flatnonzero(residuals_sq < ransac_threshold*ransac_threshold)
    return horns_method(X0[inliers,:], X1[inliers,:])
    

def align_unit_vectors(X0, X1):
    """
    Align n-by-d points clouds X0 and X1. Return the
    rotation matrix R that best symmetrically aligns
    them: X0 = R*X1. d is either 2 or 3.
    """
    (R, r0, s, err) = horns_method(X0, X1)
    return R, err

def horns_method(X0, X1):
    """
    Align n-by-d points clouds X0 and X1. Return the
    transformation (R,t,s) that best symmetrically aligns
    them. X0 = s*R*X1 + t. d is either 2 or 3.

    Use Horn's method, a quaternion based closed for least
    squares solution. I'll try to stay close to his notation.

    B Horn. Closed-Form Solution of Absolute Orientation Using
    Unit Quaternions. JOSA A 4.4. 1984. 629-642.
    """
    dims = X0.shape[1]
    rr = X0
    rl = X1
    n = X0.shape[0]
    # look for: rr = s*R*rl + r0

    # find left and right centroids
    l_centroid = np.mean(rl, axis=0)
    r_centroid = np.mean(rr, axis=0)

    # subtract them from all measurements
    rl_p = rl - l_centroid
    rr_p = rr - r_centroid

    if dims == 3:
        # for each point, accumulate into Sxx, Sxy, Sxz, ...
        Sxx = Sxy = Sxz = 0
        Syx = Syy = Syz = 0
        Szx = Szy = Szz = 0
        for i in xrange(n):
            Sxx += rl_p[i,0] * rr_p[i,0]
            Sxy += rl_p[i,0] * rr_p[i,1]
            Sxz += rl_p[i,0] * rr_p[i,2]

            Syx += rl_p[i,1] * rr_p[i,0]
            Syy += rl_p[i,1] * rr_p[i,1]
            Syz += rl_p[i,1] * rr_p[i,2]

            Szx += rl_p[i,2] * rr_p[i,0]
            Szy += rl_p[i,2] * rr_p[i,1]
            Szz += rl_p[i,2] * rr_p[i,2]

        # form matrix N
        N = np.array([[Sxx+Syy+Szz, Syz-Szy,     Szx-Sxz,      Sxy-Syx     ],
                      [Syz-Szy,     Sxx-Syy-Szz, Sxy+Syx,      Szx+Sxz     ],
                      [Szx-Sxz,     Sxy+Syx,     -Sxx+Syy-Szz, Syz+Szy     ],
                      [Sxy-Syx,     Szx+Sxz,     Syz+Szy,      -Sxx-Syy+Szz]])

        # find most positive eigenvalue - eigenvector
        # this is the solution as a quaternion
        w, v = np.linalg.eig(N)
        ind = np.argmax(w)
        q = v[:,ind]
        q /= np.linalg.norm(q)

        # construct an orthonormal matrix R from q
        q0 = q[0]
        qx = q[1]
        qy = q[2]
        qz = q[3]
        R = np.array([[q0*q0 + qx*qx - qy*qy - qz*qz,
                       2*(qx*qy - q0*qz),
                       2*(qx*qz + q0*qy)],
                      [2*(qy*qx + q0*qz),
                       q0*q0 - qx*qx + qy*qy - qz*qz,
                       2*(qy*qz - q0*qx)],
                      [2*(qz*qx - q0*qy),
                       2*(qz*qy + q0*qx),
                       q0*q0 - qx*qx - qy*qy + qz*qz]])
    elif dims == 2:
        # form horn's accumulation matrix
        Sxx = Sxy = 0
        Syx = Syy = 0
        for i in xrange(n):
            Sxx += rl_p[i,0] * rr_p[i,0]
            Sxy += rl_p[i,0] * rr_p[i,1]
            Syx += rl_p[i,1] * rr_p[i,0]
            Syy += rl_p[i,1] * rr_p[i,1]
        N = np.array([[Sxx+Syy,  Sxy-Syx],
                      [Sxy-Syx, -Sxx-Syy]])

        # the principle eigenvector is the rotation quaternion
        w, v = np.linalg.eig(N)
        ind = np.argmax(w)
        q = np.real(v[:,ind])
        q = q * np.sign(q[1] + (q[1]>=0)) # sign ambiguity
        q /= np.linalg.norm(q)

        # construct an orthonormal matrix R from q
        R11 = q[0]**2 - q[1]**2
        R21 = 2 * q[0] * q[1]
        R = np.array([[R11, -R21],[R21, R11]])

    else:
        raise ValueError('Input must be 2 or 3 dimensional')

    # compute scale via sum of squares
    rl_ss = np.sum(rl_p * rl_p, axis=None)
    rr_ss = np.sum(rr_p * rr_p, axis=None)
    s = np.sqrt(rr_ss / rl_ss)

    # compute translation from new centroid coords
    r0 = r_centroid - s * np.dot(R, l_centroid)

    # check my work: is the residual believable?
    # err = np.sum(np.square(X0 - s*np.dot(R, X1.T).T - r0), axis=None)
    err = np.sqrt(np.median(np.sum(np.square(X0 - s*np.dot(R, X1.T).T - r0), axis=1)))

    # debug: histogram the residuals
    if False:
        res = np.sum(np.square(X0 - s*np.dot(R, X1.T).T - r0), axis=1)
        print res

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        n, bins, patches = ax.hist(np.log(res), 50, normed=1)
        plt.show()

    # transformation that takes X1 to X0:
    return (R, r0, s, err)
