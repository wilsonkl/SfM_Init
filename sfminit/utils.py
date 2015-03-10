import numpy as np

def wc_length(filename):
    """Use a subprocess call to wc to count the lines in a file"""
    return int(subprocess.Popen(['wc', '-l', filename],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].partition(b' ')[0])

def normalize_rows(arr):
    """
    rescale every row of array arr so that each row has unit 2-norm.
    """
    row_length = np.sqrt(np.sum(np.square(arr),axis=1))
    return arr / row_length[:,np.newaxis]

def indices_to_direct(ind, data):
    """
    Return a new array where rows of data are accessed by index: direct[ ind[i],:] = data[ i,:]
    Rows of direct not in ind are undefined. Also return a boolean lookup: member[i] = true iff
    i is in ind.
    """
    n = np.max(ind)+1
    sh = list(data.shape)
    sh[0] = n
    direct = np.zeros(sh)
    direct[ind] = data
    member = np.zeros(n, dtype=bool)
    member[ind] = True
    return direct, member

def txSift2Bundler(ptSift, imageDims):
    """
    Convert from David Lowe's (Matlab) image coordinates to Noah
    Snavely's image coordinates. Given (X,Y) in Sift coords and
    image dims (width, height), compute (x,y) in bundler coords.
    SIFT: the center of the upper-left pixel is (1,1), +X points
        right and +Y points down
    Bundler: the center of the image is (0,0), +X points right
        and +Y points up
    """
    sx,sy = ptSift
    w,h = imageDims
    bx = sx - 0.5 * (w+1);
    by = 0.5 * (h+1) - sy;
    return (bx, by)

def rand_S2():
    """
    Return a 3x1 vector chosen uniformily at random in the sphere S(2).
    Use hypersphere point picking:
    http://mathworld.wolfram.com/HyperspherePointPicking.html
    """
    t = np.random.uniform(0, 2*np.pi)
    u = np.random.uniform(-1,1)
    y = np.zeros(3)
    y[0] = np.sqrt(1-u*u) * np.cos(t)
    y[1] = np.sqrt(1-u*u) * np.sin(t)
    y[2] = u
    return y

def SO3_geodesic_norm(R):
    """
    Return the angle of the rotation R.
    R is a 3-by-3 numpy array.
    """
    assert R.shape == (3,3)
    return np.arccos(np.clip((np.trace(R) - 1.0)/2.0, -1.0, 1.0))

def SO3_geodesic_metric(R0, R1):
    """
    Return the angle of the rotation from R0 to R1.
    R0 and R1 are 3-by-3 numpy arrays.
    """
    assert R0.shape == (3,3) and R1.shape == (3,3)
    R_delta = np.dot(R0, R1.T)
    return SO3_geodesic_norm(R_delta)

def rand_noise_rotmat(sigma):
    v = np.random.rand(3,1)
    length = np.sqrt(np.sum(np.square(v)))
    v *= np.absolute(np.random.normal(0.0, sigma)) / length
    return axis_angle_to_rotmat(v)

def axis_angle_to_rotmat(v):
    t = v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
    if t == 0:
        return np.identity(3)
    c = np.cos(t)
    s = np.sin(t)
    x = v[0] / t
    y = v[1] / t
    z = v[2] / t
    R = np.zeros((3,3))

    R[0,0] = x*x*(1-c) + c
    R[0,1] = x*y*(1-c) - z*s
    R[0,2] = x*z*(1-c) + y*s

    R[1,0] = y*x*(1-c) + z*s
    R[1,1] = y*y*(1-c) + c
    R[1,2] = y*z*(1-c) - x*s

    R[2,0] = z*x*(1-c) - y*s
    R[2,1] = z*y*(1-c) + x*s
    R[2,2] = z*z*(1-c) + c
    return R
