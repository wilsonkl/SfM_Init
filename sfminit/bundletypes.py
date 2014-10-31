import tempfile
import sys
import os
import subprocess
import shutil
import numpy as np
from PIL import Image
import scipy.sparse
from .hornsmethod import robust_horn

def read_sift_key(fname):
    """
    Read a SIFT .key file produced by David Lowe's SIFT feature detector. This function ignores
    the SIFT descriptors and returns only the feature locations.

    Given:
        fname: a filename to read
    Returns:
        keys: n_keys-by-2 numpy array of SIFT key locations.

    Notes:
    Sift keys are in the following coordinate system: (x,y) in [0:width-1, 0:height-1].
    """
    # get a working directory
    tmpdir = tempfile.mkdtemp()

    # start by unzipping the key if necessary
    if fname.endswith('.gz'):
        fname_orig = fname
        fname = os.path.join(tmpdir, os.path.basename(fname_orig).replace('.key.gz', '.key'))
        cmd = "gunzip -c " + fname_orig + " > " + fname
        subprocess.call(cmd, shell=True)

    # it's fastest to read this into python if we've already stripped out the descriptors
    fname_stripped = os.path.join(tmpdir, os.path.basename(fname).replace('.key','.stripped.key'))
    cmd = r"awk 'NR % 8 == 2' " + fname + r' > ' + fname_stripped
    subprocess.call(cmd, shell=True)

    # load the data, clean up, and return
    data = np.loadtxt(fname_stripped)
    shutil.rmtree(tmpdir)
    return data[:,0:2]

def read_cc_file(fname):
    ccs = []
    with open(fname) as f:
        f.readline()
        for i, cc in enumerate(f.readlines()):
            cc = [int(x) for x in cc.split()]
            ccs.append(cc[1:])
    return ccs

def read_image_sizes(listdata, image_path=None, subset=None):
    n = len(listdata.images)
    img_sizes = np.zeros((n,2))
    for i in subset:
        img_name = listdata.images[i]
        if image_path:
            img_name = os.path.join(image_path, img_name)
        if os.path.exists(img_name):
            im = Image.open(img_name)
            img_sizes[i,:] = im.size
    return img_sizes

class CoordsEntry(object):
    def __init__(self, name='', dims=None, fl=0.0, keys=None, rgb=None):
        if keys is None:
            keys = []
        if rgb is None:
            rgb = []
        if dims is None:
            dims = (0.0, 0.0)
        self.name = name
        self.dims = dims
        self.fl = fl
        self.keys = keys
        self.rgb = rgb

class Coords(object):
    def __init__(self, indices=None, data=None):
        if indices is None:
            indices = []
        if data is None:
            data = []
        self.indices = indices
        self.data = data
        self.reverse_index = []
        self.update_lookup()

    def update_lookup(self):
        n = max(self.indices)
        self.reverse_index = [ -1 for i in xrange(n+1)]
        for i, x in enumerate(self.indices):
            self.reverse_index[x] = i

    def __getitem__(self, key):
        if self.reverse_index[key] < 0:
            return None
        return self.data[self.reverse_index[key]]

    @classmethod
    def from_file(cls, fname, skip_names=False, skip_colors=False, skip_dims=False):
        # regex expressions for float, int, and string
        import re
        re_d = r'([-+]?\d+)'
        re_f = r'([-+]?(\d+(\.\d*))?)'
        re_s = r'([^\s,]*)'

        indices = []
        data = []
        with open(fname) as f:
            line = f.readline()
            while line != '':
                # read a header line
                index = int(re.search('index = ' + re_d, line).group(1))
                num_keys = int(re.search('keys = ' + re_d, line).group(1))
                focal_search = re.search('focal = ' + re_f, line)
                if focal_search:
                    fl = float(focal_search.group(1))
                else:
                    fl = 0.0


                if skip_names:
                    image_name = ''
                else:
                    image_name = re.search('name = ' + re_s, line).group(1)

                if skip_dims:
                    dims = None
                else:
                    px = float(re.search('px = ' + re_f, line).group(1))
                    py = float(re.search('py = ' + re_f, line).group(1))
                    dims = (2*px, 2*py)


                # read the body lines to this block
                keys = []
                rgb = []
                for i in xrange(num_keys):
                    tokens = f.readline().split()
                    feature_index = int(tokens[0])
                    x = float(tokens[1])
                    y = float(tokens[2])
                    keys.append((x,y))
                    if skip_colors:
                        pass
                    else:
                        r = int(tokens[5])
                        g = int(tokens[6])
                        b = int(tokens[7])
                        rgb.append((r,g,b))

                indices.append(index)
                data.append(CoordsEntry(image_name, dims, fl, keys, rgb))
                line = f.readline()
        return cls(indices, data)

class Tracks(object):
    def __init__(self, tracks=None):
        if tracks is None:
            tracks = []
        self.tracks = tracks

    def num_tracks(self):
        return len(self.tracks)

    def __getitem__(self, i):
        return self.tracks[i]

    def __setitem__(self, i, track):
        self.tracks[i] = track

    @classmethod
    def from_models(cls, models, min_track_length=2, max_track_length=None):
        """
        return a Tracks object describing the tracks in 
        a given list of models.
        """
        if max_track_length is None:
            max_track_length = sys.maxint

        # Tracks are lists of pairs (image_num, feature_num)
        # compute the largest image num so we can hash these tuples to one int
        max_image_num = np.max(np.array([(m.i, m.j) for m in models]), axis=None)
        hash = lambda x: x[0] + x[1]*max_image_num
        unhash = lambda x: (x % max_image_num, int(x/max_image_num))

        # Make a (hashed) list of feature matches
        edges = np.array([(hash((m.i, pt.fid_i)), hash((m.j, pt.fid_j))) for pt in m.points for m in models])

        # Use scipy.sparse to run a connected component calculation
        max_hash = np.max(edges, axis=None)
        graph = scipy.sparse.csr_matrix((np.ones((len(edges),)), edges.T), shape=(max_hash+1,max_hash+1))
        N_components, labels = scipy.sparse.csgraph.connected_components(graph, directed=False)

        # Parse the labels into a list of tracks
        # --- I'm a little worried that this will eat up too much memory! ---
        # --- This is time efficient, but might blow up in space.         ---
        tracks = [[] for i in xrange(N_components)]
        for i, l in enumerate(labels):
            tracks[l].append(unhash(i))

        tracks = [t for t in tracks if len(t) >= min_track_length]
        return cls(tracks)

    @classmethod
    def from_file(cls, fname):
        tracks = []
        with open(fname) as f:
            num_tracks = int(f.readline())
            for i in xrange(num_tracks):
                data = np.fromstring(f.readline(), sep=' ', dtype=int)
                tracks.append(np.reshape(data[1:],(-1,2)))
        return cls(tracks)

    def write(self, fname):
        with open(fname, 'w') as f:
            f.write('{}\n'.format(self.num_tracks()))
            for track in self.tracks:
                data = track.flatten()
                data_str = ' '.join([str(x) for x in data])
                f.write('{} {}\n'.format(data.size/2, data_str))

class Listfile(object):

    def __init__(self, images=None, focal_lengths=None):
        if images is None:
            images = []
        if focal_lengths is None:
            focal_lengths = []
        self.images = images
        self.focal_lengths = focal_lengths

    @classmethod
    def from_file(cls, fname):
        images = []
        focal_lengths = []
        with open(fname) as f:
            for line in f:
                tokens = line.split(' ')
                images.append(tokens[0])
                if len(tokens) < 3:
                    focal_lengths.append(0)
                else:
                    focal_lengths.append(float(tokens[2]))
        return cls(images, focal_lengths)

    def write(self, fname):
        assert len(self.images) == len(self.focal_lengths)

        with open(fname, 'w') as f:
            for image, fl in zip(self.images, self.focal_lengths):
                if fl == 0:
                    f.write('{}\n'.format(image))
                else:
                    f.write('{} 0 {}\n'.format(image, fl))

class Rotfile(object):

    def __init__(self, rotations=None, indices=None):
        if rotations is None:
            rotations = []
        if indices is None:
            indices = []
        self.rotations = rotations
        self.indices = indices

    @classmethod
    def from_file(cls, fname):
        rotations = []
        indices = []

        with open(fname) as f:
            for line in f:
                data = np.fromstring(line, sep=' ')
                if np.any(data[1:]):
                    indices.append(int(data[0]))
                    R = np.reshape(data[1:],(3,3))
                                        # bug: rotfiles are col major
                    rotations.append(R.T)
        return cls(rotations, indices)

    def write(self, fname):
        with open(fname, 'w') as f:
            for index, R in zip(self.indices, self.rotations):
                f.write('{} {}\n'.format(
                    index,
                    ' '.join([str(x) for x in R.flatten()])))

class Camera(object):
    """
    Represents a bundler image. f is the pixel focal length, k1 and k2 are
    first and second order radial distortion coeffs, R is camera rotation, and
    t is camera translation. For more information, see:
    http://www.cs.cornell.edu/~snavely/bundler/bundler-v0.3-manual.html#S6
    """

    def __init__(self, f=0, k1=0, k2=0, R=None, t=None):
        if R is None:
            R=np.zeros((3,3))
        if t is None:
            t=np.array([0,0,0])
        self.f = f
        self.k1 = k1
        self.k2 = k2
        self.R = R
        self.t = t


    def project(self, X):
        """
        Project ndarray 3-vector X into this camera. Return ndarray
        2-vector x. See the bundler manual for details.
        http://www.cs.cornell.edu/~snavely/bundler/bundler-v0.3-manual.html

        TODO: vectorize this?
        """
        P = np.dot(self.R, X) + self.t
        p = -1*P[0:2]/P[2]
        norm_p = np.linalg.norm(p)
        r = 1 + self.k1 * norm_p + self.k2 * norm_p * norm_p
        return self.f * r * p

    def viewing_direction(self):
        return np.dot(self.R.transpose(), np.array([[0],[0],[-1]]))

    def set_X(self, X):
        self.t = np.dot(self.R, -1*X)

    def location(self):
        return np.dot(self.R.transpose(), -self.t)

    def initialized(self):
        return np.any(self.R)

    def P_matrix(self):
        """
        return P = K * [R | t]
        """
        K = np.diag([self.f,self.f,1])
        return np.dot(K, np.hstack((self.R, self.t[:,np.newaxis])))

    def __str__(self):
        return ''.join(['{} {} {}\n'.format(self.f     , self.k1    , self.k2)    ,
                        '{} {} {}\n'.format(self.R[0,0], self.R[0,1], self.R[0,2]),
                        '{} {} {}\n'.format(self.R[1,0], self.R[1,1], self.R[1,2]),
                        '{} {} {}\n'.format(self.R[2,0], self.R[2,1], self.R[2,2]),
                        '{} {} {}'.format(  self.t[0]  , self.t[1]  , self.t[2])  ])

class Point(object):
    """
    Represents a 3D point. X is a coordinate in R3, color is an RGB triple, and observations
    is a list of tuples (camera, feature_id, x_pxl, y_pxl)
    """

    def __init__(self, X=None, color=None, observations=None):
        if X is None:
            X=np.array([0,0,0])
        if color is None:
            color=np.array([255,255,255],dtype=np.uint8)
        if observations is None:
            observations = []

        self.X = X
        self.color = color
        self.observations = observations

    def __str__(self):
        myStr = []
        myStr.append(' '.join([str(x) for x in self.X]))
        myStr.append('\n')
        myStr.append(' '.join([str(x) for x in self.color]))
        myStr.append('\n')
        myStr.append('{} '.format(len(self.observations)))
        myStr.append(''.join(['{0:d} {1:d} {2:6.5f} {3:6.5f} '.format(*obs) for obs in self.observations]))
        return ''.join(myStr)

class BundleException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class Bundle(object):
    """
    Represents the cameras and points in a structure from motion reconstruction.
    """

    def __init__(self, cameras=None, points=None, num_cameras=0):
        """
        Make a blank bundle, possibly with given lists of cameras and points.
        If specified, reserve space for nCams uninitialized cameras
        """
        if cameras is None:
            cameras = [ Camera() for i in xrange(num_cameras)]
        if points is None:
            points = []
        self.cameras = cameras
        self.points = points

    def num_points(self):
        """Get the number of points in this bundle"""
        return len(self.points)

    def num_cameras(self):
        """Get the number of cameras in this bundle, whether they are initialized or not"""
        return len(self.cameras)

    def num_active_cameras(self):
        """Get the number of cameras which are initialized"""
        return len([x for x in self.cameras if x.initialized()])

    def num_observations(self):
        """Get the number of camera-point observations in this bundle"""
        return sum([len(x.observations) for x in self.points])

    def all_rots(self):
        """
        Get (ind, R) the indices and rotation matrices of all initialized cameras.
        Returns:
            ind: length n numpy int array of camera indices
            R: n-by-3-by-3 numpy array of camera rotation matrices
        """
        n = self.num_active_cameras()
        ind = np.zeros(n)
        R = np.zeros((n,3,3))
        ptr = 0
        for i,cam in enumerate(self.cameras):
            if cam.initialized():
                ind[ptr] = i
                R[ptr,:,:] = cam.R
                ptr += 1
        return ind, R

    def all_positions(self):
        """
        Get (ind, X) the indices and locations of all initialized cameras.
        Returns:
            ind: length n numpy int array of camera indices
            X: n-by-3 numpy array of camera locations (world xyz coords)
        """
        n = self.num_active_cameras()
        ind = np.zeros(n)
        X = np.zeros((n,3))
        ptr = 0
        for i,cam in enumerate(self.cameras):
            if cam.initialized():
                ind[ptr] = i
                X[ptr,:] = cam.location()
                ptr += 1
        return ind, X

    def compare_to(self, other_bundle):
        """
        Return a vector of the distances between corresponding cameras in two
        bundles.
        """
        residuals = []
        for camA, camB in zip(self.cameras, other_bundle.cameras):
            if camA.initialized() and camB.initialized():
                XA = camA.location()
                XB = camB.location()
                residuals.append(np.sqrt(np.sum(np.square(XB - XA))))
        return np.array(residuals)

    def align_to(self, other_bundle, ransac_rounds=4096, ransac_threshold=10.0):
        """
        Robustly compute and then apply a similarity transformation that attempts
        to align corresponding cameras in two bundles.
        """
        XA = []
        XB = []
        for camA, camB in zip(self.cameras, other_bundle.cameras):
            if camA.initialized() and camB.initialized():
                XA.append(camA.location().flatten())
                XB.append(camB.location().flatten())
        XA = np.array(XA)
        XB = np.array(XB)
        (R, t, s, inlier_err) = robust_horn(XB, XA, ransac_rounds, ransac_threshold)
        self.transform(R, t, s)
        return (R,t,s)

    def transform(self, R=None, t=None, s=1):
        """
        Apply a change of coordinates to a bundle file.
        T(x) = s * R * x + t
        """
        if R is None:
            R = np.identity(3)
        if t is None:
            t = np.array([0,0,0])

        # compute the inverse transform
        R_inv = R.T
        t_inv = -1 * np.dot(R_inv, t) / s
        s_inv = 1 / s

        for cam in self.cameras:
            R_new = np.dot(cam.R, R_inv)
            t_new = (np.dot(cam.R, t_inv).flatten() + cam.t) / s_inv
            cam.R = R_new
            cam.t = t_new
        for pt in self.points:
            pt.X = np.dot(R, pt.X) * s + t
            pt.X = pt.X.flatten()

    def reprojection_error(self):
        error = 0.0
        for point in self.points:
            for obs in point.observations:
                cam = self.cameras[obs[0]]
                pixel = np.array(obs[2:])
                res = pixel - cam.project(point.X)
                error += np.sum(res * res)
        return error

    @classmethod
    def from_file(cls, fname):
        """
        Read in a Bundle file.
        """
        with open(fname) as f:

            # parse the header
            line = f.readline()
            bundle_version = line.split()[-1]
            if bundle_version != 'v0.3':
                raise BundleException('Bundle version {} not supported.'.format(bundle_version))
            tokens = f.readline().split()
            nCams = int(tokens[0])
            nPts = int(tokens[1])
            my_bundle = cls(num_cameras=nCams)

            # parse all of the cameras
            for i in xrange(nCams):
                try:
                    data = np.fromstring(f.readline(), sep=' ')
                    fl = data[0]
                    k1 = data[1]
                    k2 = data[2]
                    R0 = np.fromstring(f.readline(), sep=' ')
                    R1 = np.fromstring(f.readline(), sep=' ')
                    R2 = np.fromstring(f.readline(), sep=' ')
                    R = np.vstack((R0, R1, R2))
                    t = np.fromstring(f.readline(), sep=' ')
                    my_bundle.cameras[i] = Camera(fl, k1, k2, R, t)
                except:
                    raise BundleException('Bundle file format error in camera {} block'.format(i))

            # parse all of the points
            for i in xrange(nPts):
                try:
                    X = np.fromstring(f.readline(), sep=' ')
                    color = np.fromstring(f.readline(), sep=' ', dtype=np.uint8)
                    data = np.fromstring(f.readline(), sep=' ')
                    data = np.reshape(data[1:],(-1,4))
                    observations = []
                    for i in xrange(data.shape[0]):
                        img_num = int(data[i,0])
                        feature_num = int(data[i,1])
                        x_pxl = data[i,2]
                        y_pxl = data[i,3]
                        observations.append((img_num, feature_num, x_pxl, y_pxl))
                    my_bundle.points.append(Point(X, color, observations))
                except:
                    raise BundleException('Bundle file format error in point {} block'.format(i))

            return my_bundle

    def write(self, fname):
        """
        Write this bundle to a given file in Bundler v0.3 format.
        """
        with open(fname, 'w') as f:
            f.write('# Bundle file v0.3\n')
            f.write('{} {}'.format(self.num_cameras(), self.num_points()))
            for camera in self.cameras:
                f.write('\n' + str(camera))
            for point in self.points:
                f.write('\n' + str(point))
