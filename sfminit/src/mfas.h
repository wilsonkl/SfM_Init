#ifndef __MFAS_H__
#define __MFAS_H__

#include <vector>
#include <map>
typedef std::pair<int, int> Edge;


void mfas_ratio(const std::vector<Edge>   &edges,
                const std::vector<double> &weight,
                      std::vector<int>    &order);

void reindex_problem(std::vector<Edge> &edges,
                     std::map<int,int> &reindexing_key);

void flip_neg_edges(std::vector<Edge> &edges,
                    std::vector<double> &weights);

void broken_weight(const std::vector<Edge>   &edges,
                   const std::vector<double> &weight,
                   const std::vector<int>    &order,
                         std::vector<double> &broken);


#endif // __MFAS_H__

