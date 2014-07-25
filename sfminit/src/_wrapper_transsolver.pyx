import numpy as np
cimport numpy as np


cdef extern from "trans_solver.h":

    void solve_translations_problem(
        const int* edges,
        const double* poses,
        const double* weights,
        int num_edges,
        double loss_width,
        double* X,
        double function_tolerance,
        double parameter_tolerance,
        int max_iterations
    )

def solve_trans_problem(problem, loss=None, function_tol=1e-7, parameter_tol=1e-8, max_iterations=500):
    
    # deal with the loss function option
    cdef double loss_width = 0 if loss is None else loss

    # put the return data here
    max_node = np.max(problem.edges)
    X = np.zeros((max_node+1, 3))

    # make pointers to the data via numpy array views
    # http://docs.cython.org/src/userguide/memoryviews.html
    cdef np.ndarray[int, ndim=2] edges_c= problem.edges.astype(np.intc, order='C', casting='same_kind')
    cdef int [:, :] edge_view = edges_c
    cdef double  [:, :] pose_view = problem.poses
    cdef double  [:] weight_view = problem.weights
    cdef double  [:, :] soln_view = X

    solve_translations_problem(
        &edge_view[0, 0],
        &pose_view[0, 0],
        &weight_view[0],
        problem.num_edges,
        loss_width,
        &soln_view[0, 0],
        function_tol,
        parameter_tol,
        max_iterations
    )

    X_ids = np.unique(problem.edges)
    X = X[X_ids,:]
    return (X_ids, X)

