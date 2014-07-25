import os
import numpy as np
from warnings import warn
from .utils import indices_to_direct
from .sfminittypes import read_EGs_file


class ModelList(object):

    def __init__(self, models=None):
        self.models = [] if models is None else models

    def __len__(self):
        return self.models.__len__()

    def __getitem__(self, key):
        return self.models.__getitem__(key)

    def __setitem__(self, key, value):
        self.models__setitem__(key, value)

    def __delitem__(self, key):
        self.models.__delitem__(key)

    def __iter__(self):
        return self.models.__iter__()

    def apply_rotations_solution(self, R_index, R):
        """
        Rotate a list of TwoViewModels to put them all in the common rotational
        coordinate system given by (R_index, R). Since a center of rotation is
        arbitrary, the scene is moved so that camera i is at (0,0,0).

        The new list of models may be shorter than the original, since models where
        at least one camera is not in R_index will be omitted.

        This method does not alter its input.

        Inputs:
            R_index: lookup for which camera the i'th rotation matrix goes to
            R: a list of rotation matrices

        """
        rots, rot_bool = indices_to_direct(R_index, R)
        models_new = []
        for m in self.models:
            if m.i < len(rot_bool) and m.j < len(rot_bool) and rot_bool[m.i] and rot_bool[m.j]:
                for pt in m.points:
                    pt.X = np.dot(rots[m.i], pt.X - m.Xi)
                m.Xi = np.zeros((3,1))
                m.Xj = np.dot(m.Ri.T, m.tij())
                m.Ri = rots[m.i]
                m.Rj = rots[m.j]

                models_new.append(m)
        self.models = models_new

    def get_rotations_problem(self):
        """
        Extract a global rotations problem---a graph with edges (i,j) annotated by
        relative rotations Rij---from a list of TwoViewModels.

        Returns:
            edges: a list of tuples (i,j)
            pairwise_rotations: a list of rotation matrices Rij
        """
        edges = []
        pairwise_rotations = []
        for m in self.models:
            edges.append((m.i, m.j))
            pairwise_rotations.append(m.Rij())
        return (edges, pairwise_rotations)

    def get_translations_problem(self):
        """
        Build a translations problem from a list of TwoViewModel objects.
        This will make one edge per model.
        """
        from .transproblem import TransProblem
        edges = []
        poses = []
        for m in self.models:
            dX = m.Xj - m.Xi
            dX /= np.sqrt(np.sum(np.square(dX)))
            edges.append((m.i, m.j))
            poses.append(dX.flatten())
        return TransProblem(np.array(edges), np.array(poses))


    @classmethod
    def from_bin_file(cls, fname):
        import marshall
        with open(fname, 'rb') as f:
            return marshall.load(f)

    def write_to_bin(self, fname):
        import marshall
        with open(fname, 'wb') as f:
            marshall.dump(self, f)

    @classmethod
    def from_EG_file(cls, fname):
        EGs = read_EGs_file(fname)
        models = []
        for EG in EGs:
            models.append(TwoViewModel(i=EG.i, j=EG.j, Rj=EG.R.T, Xj=EG.t.reshape((3,1))))
        return cls(models)

class ModelPoint(object):
    """
    Represents a point seen in a two view Structure from Motion reconstruction.
    """

    def __init__(self, fid_i=None, fid_j=None,
                 pi=None, pj=None,
                 rgb_i=None, rgb_j=None, X=None):

        # feature ids in each image
        self.fid_i = fid_i
        self.fid_j = fid_j

        # feature location in each image
        self.pi = pi
        self.pj = pj

        # pixel color in each location
        self.rgb_i = rgb_i
        self.rgb_j = rgb_j

        # reconstructed position
        self.X = X if X is not None else np.zeros((3,1))

    def __str__(self):
        return '{0:d} {1:d} {2:f} {3:f} {4:f}'.format(self.fid_i, self.fid_j, *self.X)

class TwoViewModel(object):
    """
    Represents a two-camera Structure from Motion reconstruction. Includes methods for
    representation, reconstruction, and I/O.

    Fields:
    i,j     :   camera indices (metadata)
    fi, fj  :   focal lengths, in pixel units, of first and second camera
    Ri, Rj  :   camera rotation matrices: 3-by-3 numpy arrays, representing
                a world-to-camera coordinates map
    Xi, Xj  :   camera locations in world coordinates, (3,1) numpy arrays
    points  :   a list of modelPoint objects
    """

    def __init__(self, name_i=None, name_j=None, i=None, j=None,
                 fi=None, fj=None,
                 size_i=None, size_j=None,
                 ki=None, kj=None,
                 Ri=None, Rj=None, Xi=None, Xj=None,
                 points=None):
        # camera index numbers
        self.name_i = name_i
        self.name_j = name_j
        self.i = i
        self.j = j

        # camera intrinsics
        self.fi = fi
        self.fj = fj
        self.size_i = size_i
        self.size_j = size_j
        self.ki = ki if ki is not None else np.zeros(2)
        self.kj = kj if kj is not None else np.zeros(2)


        # camera extrinsics
        self.Ri = Ri if Ri is not None else np.eye(3)
        self.Rj = Rj if Rj is not None else np.eye(3)
        self.Xi = Xi if Xi is not None else np.zeros((3,1))
        self.Xj = Xj if Xj is not None else np.zeros((3,1))

        # points in this twoViewModel
        self.points = points if points is not None else []


    def Rij(self):
        """
        Rij: the pose of camera j in camera i's coordinate
        system. Rij = Ri * Rj'
        """
        return np.dot(self.Ri, self.Rj.T)

    def tij(self):
        tij = np.dot(self.Ri, self.Xj - self.Xi)
        #print tij.size, self.Xi.size, self.Xj.size, self.Ri.size
        return tij / np.sqrt(np.sum(np.square(tij)))

    def center_and_scale(self, scale=1.0):
        # center camera i to the origin
        self.Xj -= self.Xi
        for pt in self.points:
            pt.X -= self.Xi
        self.Xi = np.zeros((3,1))

        # scale the model to have || Xj - Xi || = scale
        s = np.sqrt(np.sum(np.square(self.Xj))) / scale
        self.Xj /= s
        for pt in self.points:
            pt.X /= s

    def view_angle(self):
        """
        Return the angle (radians) between the viewing directions of the two cameras in this model.
        """
        view_i = -self.Ri[2,:].T
        view_j = -self.Rj[2,:].T
        return np.arccos(np.dot(view_i.T, view_j))

    def __str__(self):
        return ''.join([
            'Model {},{} with {} points'.format(self.i, self.j, len(self.points)), '\n',
            'Data: ',
            'fi: {}, fj: {}'.format(self.fi, self.fj), '\n',
            'Ri: ', str(self.Ri), '\n',
            'Xi: ', str(self.Xi), '\n',
            'Rj: ', str(self.Rj), '\n',
            'Xj: ', str(self.Xj), '\n'])




