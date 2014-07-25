#ifndef __TRANS_SOLVER_H__
#define __TRANS_SOLVER_H__

#include <vector>

void 
solve_translations_problem(
        const int* edges,
        const double* poses,
        const double* weights,
        int num_edges,
        double loss_width,
        double* X,
        double function_tolerance,
        double parameter_tolerance,
        int max_iterations
    );

void 
reindex_problem(int* edges, int num_edges, std::vector<int> &reindex_lookup);

struct ChordFunctor {
  ChordFunctor(const double *direction, double weight)
    : u_(direction), w_(weight){}

  const double *u_;
  const double w_;

  template <typename T>
  bool operator()(const T* const x0, const T* const x1, T* residual) const;
};

#endif /* __TRANS_SOLVER_H__ */