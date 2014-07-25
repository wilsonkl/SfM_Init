#include <cstdlib>
#include <vector>
#include <ctime>
#include <set>
#include <map>
#include "ceres/ceres.h"
#include "trans_solver.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using ceres::HuberLoss;



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
    ) {
    // seed the random number generator
    std::srand( std::time( NULL ) );

    // reindex the edges to be a sequential set
    int *_edges = new int[2*num_edges];
    memcpy(_edges, edges, 2*num_edges*sizeof(int));
    std::vector<int> reindex_lookup;
    reindex_problem(_edges, num_edges, reindex_lookup);
    int num_nodes = reindex_lookup.size();

	// Make a random guess solution
    double *x = new double[3*num_nodes];
    for (int i=0; i<3*num_nodes; ++i)
        x[i] = (double)rand() / RAND_MAX;

	// add the parameter blocks (a 3-vector for each node)
    Problem problem;
    for (int i=0; i<num_nodes; ++i)
        problem.AddParameterBlock(x+3*i, 3);

	// set the residual function (chordal distance for each edge)
    for (int i=0; i<num_edges; ++i) {
        CostFunction* cost_function =
                new AutoDiffCostFunction<ChordFunctor, 3, 3, 3>(
                new ChordFunctor(poses+3*i,weights[i]));

        if (loss_width == 0.0) {
            // No robust loss function 
            problem.AddResidualBlock(cost_function, NULL, x+3*_edges[2*i+0], x+3*_edges[2*i+1]);
        } else {
            problem.AddResidualBlock(cost_function, new ceres::HuberLoss(loss_width), x+3*_edges[2*i+0], x+3*_edges[2*i+1]);
        }
    }

	// solve
    Solver::Options options;
    options.num_threads = 16;
    options.num_linear_solver_threads = 16;
    //options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = max_iterations;
    options.function_tolerance = function_tolerance;
    options.parameter_tolerance = parameter_tolerance;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;                   
    options.preconditioner_type = ceres::SCHUR_JACOBI; 

    Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";

    // undo the reindexing
    for (int i=0; i<num_nodes; ++i) {
        int j = reindex_lookup[i];
        X[3*j+0] = x[3*i+0];
        X[3*j+1] = x[3*i+1];
        X[3*j+2] = x[3*i+2];
    }

    delete[] _edges;
}

template <typename T>
bool ChordFunctor::operator()(
        const T* const x0,
        const T* const x1,
        T* residual) const {

    // compute ||x1 - x0||_2
    T norm = sqrt((x1[0]-x0[0])*(x1[0]-x0[0]) +
                  (x1[1]-x0[1])*(x1[1]-x0[1]) +
                  (x1[2]-x0[2])*(x1[2]-x0[2]));
    residual[0] = w_*((x1[0]-x0[0]) / norm - T(u_[0]));
    residual[1] = w_*((x1[1]-x0[1]) / norm - T(u_[1]));
    residual[2] = w_*((x1[2]-x0[2]) / norm - T(u_[2]));
    return true;
}

void 
reindex_problem(int* edges, int num_edges, std::vector<int> &reindex_lookup) {

    reindex_lookup.clear();

    // get the unique set of notes
    std::set<int> nodes;
    for (int i=0; i<2*num_edges; ++i)
        nodes.insert(edges[i]);

    std::map<int, int> reindexing_key;

    // iterator through them and assign a new name to each vertex
    std::set<int>::const_iterator it;
    int n=0;
    for (it = nodes.begin(); it != nodes.end(); ++it) {
        reindex_lookup.push_back(*it);
        reindexing_key[*it] = n;
        ++n;
    }

    // now renumber the edges
    for (int i=0; i<2*num_edges; ++i)
        edges[i]  = reindexing_key[edges[i]];

}