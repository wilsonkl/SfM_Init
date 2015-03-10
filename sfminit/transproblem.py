import numpy as np
from .bundletypes import Bundle
from .sfminittypes import (read_trans_prob_file, read_edge_weight_file,
    write_trans_prob_file, write_edge_weight_file)
from .hornsmethod import robust_horn
from .utils import normalize_rows, indices_to_direct, txSift2Bundler
from ._wrapper_transsolver import solve_trans_problem

class TransProblem(object):
    """
    A Structure from Motion translations problem: a graph given by directed
    edges, 3D unit vector poses for each edge, and a solver weight for each edge.
    """

    def __init__(self, edges, poses, weights=None):
        """
        edges: n-by-2 numpy array of nonnegative ints. Each row is an edge of the
            problem (i,j)
        poses: n-by-3 numpy array of unit vectors---the 3D pose of each edge above
        weights: cost function weight of each edge when solving this problem.
        """
        self.edges = edges
        self.poses = poses
        if weights is None:
            self.weights = np.ones(len(poses))
        else:
            self.weights = weights

        if len(self.edges) != len(self.poses) or len(self.edges) != len(self.weights):
            error('Mismatched dimensions!')

    @property
    def num_edges(self):
        return len(self.edges)

    @property
    def num_nodes(self):
        return len(np.unique(self.edges))

    @property
    def nodes(self):
        return np.unique(edges)

    def run_1DSfM(self, **kwargs):
        """
        Run 1DSfM to remove outliers. See the wrapped function for keyword explanations.
        """
        from .onedsfm import oneDSfM
        mask = oneDSfM(self.edges, self.poses, **kwargs)
        self.edges = self.edges[mask,:]
        self.poses = self.poses[mask,:]
        self.weights = self.weights[mask]

    def solve(self, **kwargs):
        """
        Solve this problem by running the trans solver in the wrapped method
        """
        return solve_trans_problem(self, **kwargs)

    def get_matched_subproblems(self, other):

        # flip and sort all edges and permutations
        edgesA = np.array([e if e[0]<e[1] else (e[1],e[0]) for e in self.edges ])
        posesA = np.array([p if e[0]<e[1] else -p for (p,e) in zip(self.poses,  self.edges) ])
        perm = np.argsort(edgesA.view('i8,i8'), order=['f0','f1'], axis=0)
        edgesA = edgesA[perm,:]
        posesA = posesA[perm,:]

        edgesB = np.array([e if e[0]<e[1] else (e[1],e[0]) for e in other.edges ])
        posesB = np.array([p if e[0]<e[1] else -p for (p,e) in zip(other.poses, other.edges) ])
        perm = np.argsort(edgesB.view('i8,i8'), order=['f0','f1'], axis=0)
        edgesB = edgesB[perm,:]
        posesB = posesB[perm,:]

        # walk through the sorted lists together
        nA = len(posesA)
        nB = len(posesB)
        ptrA = 0
        ptrB = 0
        ptrB_reset = 0
        edges_new = []
        posesA_new = []
        posesB_new = []

        while ptrA < nA and ptrB < nB:
            eA = edgesA[ptrA,:].flatten()
            eB = edgesB[ptrB,:].flatten()
            pA = posesA[ptrA,:].flatten()
            pB = posesB[ptrB,:].flatten()
            # *ptrB > *ptrA
            if eA[0] < eB[0] or (eA[0] == eB[0] and eA[1]<eB[1]):
                ptrB = ptrB_reset
                ptrA += 1
            # *ptrA > *ptrB
            elif eA[0] > eB[0] or (eA[0] == eB[0] and eA[1]>eB[1]):
                ptrB += 1
                ptrB_reset = ptrB
            # *ptrA = *ptrB
            else:
                edges_new.append(eA)
                posesA_new.append(pA)
                posesB_new.append(pB)
                ptrB += 1

                if ptrB == nB:
                    ptrB = ptrB_reset
                    ptrA += 1

        edges_new = np.array(edges_new)
        posesA_new = np.array(posesA_new)
        posesB_new = np.array(posesB_new)

        return TransProblem(edges_new, posesA_new), TransProblem(edges_new, posesB_new)

    def match_to(self, other):
        """
        Match edges in one trans problem to edges in another.
        Return a list of (possibly empty) lists:
        self.edges[i] = other.edges[j] for every j in lookup[i]
        I think this will work correctly for repeated edges, but
        I haven't had a reason to test that yet ...
        """
        lookup = [[] for i in xrange(self.num_edges)]
        n = max(np.max(self.edges, axis=None), np.max(other.edges, axis=None))
        hashA = np.array([min(e)*n+max(e) for e in self.edges])
        hashB = np.array([min(e)*n+max(e) for e in other.edges])
        permB = np.argsort(hashB)

        ptrsB = np.searchsorted(hashB[permB], hashA)
        for i, (hA, ptrB) in enumerate(zip(hashA, ptrsB)):
            while ptrB < len(permB) and hA == hashB[permB][ptrB]:
                lookup[i].append(permB[ptrB])
                ptrB += 1
        return lookup

    def make_gt_problem(self, bundle):
        edges_new = []
        poses_new = []
        weights_new = []
        for w, (i,j) in zip(self.weights, self.edges):
            if (i < len(bundle.cameras) and bundle.cameras[i].initialized() and
                j < len(bundle.cameras) and bundle.cameras[j].initialized() ):
                edges_new.append((i,j))
                poses_new.append(bundle.cameras[j].location() - bundle.cameras[i].location())
                weights_new.append(w)
        edges_new = np.array(edges_new)
        poses_new = normalize_rows(np.array(poses_new))
        weights_new = np.array(weights_new)
        return TransProblem(edges_new, poses_new, weights=weights_new)

    def align_to(self, other):
        probA, probB = self.get_matched_subproblems(other)
        (R, _, _, _) = robust_horn(probA.poses, probB.poses)
        self.poses = np.dot(self.poses, R)


    def compare_to(self, other, distance='geodesic'):
        """
        match corresponding edges between two problems, and return the distance
        between them.

        distance = {'geodesic', 'chordal'}
        """
        probA, probB = self.get_matched_subproblems(other)
        posesA = probA.poses
        posesB = probB.poses

        if distance == 'geodesic':
            return np.arccos(np.clip(np.sum(posesA*posesB, axis=1), -1.0, 1.0))
        elif distance == 'chordal':
            return np.sqrt(np.sum(np.square(posesA - posesB), axis=1))

    def add_track_edges(self, tracks, coords, global_rot_ids, global_rots,
            k=6, min_point_size=2, cp_edge_relative_weight=0.5):
        subset = pick_track_subset(tracks, k)
        max_image = np.max(self.edges)
        num_images = max_image + 1

        R, R_mask = indices_to_direct(global_rot_ids, global_rots)

        # make cp edges
        poses = []
        edges = []
        for i in subset:
            t = tracks.tracks[i]

            # add an edge to the problem for every time a camera sees a
            # track in the subset. Watch out for missing necessary data.
            new_edges = []
            new_poses = []
            for img, feature_num in t:

                if (img < num_images and img<len(R_mask) and R_mask[img]
                    and coords[img] and coords[img].fl > 0.0):

                    # coordinate conversion: coords.txt files use SIFT
                    # conventions, but SfM_Init uses bundler conventions
                    ptSift = coords[img].keys[feature_num]
                    imageDims = coords[img].dims
                    (x,y) = txSift2Bundler(ptSift, imageDims)

                    t_local = np.array([x,y,-coords[img].fl])
                    t_local /= np.sqrt(np.sum(np.square(t_local)))
                    new_poses.append(np.dot(R[img].T, t_local))
                    new_edges.append((img, i+num_images))
            if len(new_edges) >= min_point_size:
                edges.extend(new_edges)
                poses.extend(new_poses)

        weights = np.empty((len(poses),)) # k - times_already_covered
        weights.fill(cp_edge_relative_weight*len(self.edges)/len(edges))

        self.poses = np.vstack((self.poses, np.array(poses)))
        self.edges = np.vstack((self.edges, np.array(edges)))
        self.weights = np.concatenate((self.weights, weights))



    @classmethod
    def from_file(cls, problem_file, weights_file=None):
        edges, poses = read_trans_prob_file(problem_file)

        if weights_file is None:
            return cls(edges, poses)
        else:
            _, weights = read_edge_weight_file(weights_file)
            return cls(edges, poses, weights)

    def write(self, problem_file=None, weights_file=None):
        if problem_file is not None:
            write_trans_prob_file(problem_file, self.edges, self.poses)
        if weights_file is not None:
            write_edge_weight_file(weights_file, self.edges, self.weights)

def pick_track_subset(tracks, k):
    """
    Pick a subset of tracks which cover all the images in the full tracks at least
    k times. Use a greedy algorithm to pick the subset.
    """
    max_image = np.max([np.max(t[:,0]) for t in tracks.tracks])
    num_images = max_image + 1
    subset = set()

    remaining_cover = np.empty(num_images) # k - times_already_covered
    remaining_cover.fill(k)
    improvement = [len(t) for t in tracks.tracks] # how much cover each track would add

    # preparation: make a lookup from images to tracks that see them
    image_vis_lookup = [[] for x in xrange(num_images)]
    for j,t in enumerate(tracks.tracks):
        for img in t[:,0]:
            image_vis_lookup[img].append(j)

    while True:
        # choose track i that maximizes the heuristic
        i = np.argmax(improvement)
        if improvement[i] <= 0:
            break
        subset.add(i)

        # update the state variables: remaining_cover and improvement
        improvement[i] = 0
        for img in tracks.tracks[i][:,0]:
            # if this image just got covered the k'th time, it's done.
            # lower the improvement for all tracks that see it
            if remaining_cover[img] == 1:
                for t in image_vis_lookup[img]:
                    improvement[t] = improvement[t]-1 if improvement[t]>0 else 0
            remaining_cover[img] = remaining_cover[img]-1 if remaining_cover[img]>0 else 0

    return subset
