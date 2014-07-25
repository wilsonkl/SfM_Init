from cython.operator cimport dereference as deref
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.map cimport map

ctypedef pair[int, int] Edge

cdef extern from "mfas.h":
    # Prototypes:
    void mfas_ratio(
        const vector[Edge] edges,
        const vector[double] weight,
        vector[int] order
    )

    void reindex_problem(
        vector[Edge] edges,
        map[int, int] reindexing_key
    )

    void flip_neg_edges(
        vector[Edge] edges,
        vector[double] weights,
    )

    void broken_weight(
        vector[Edge] edges,
        vector[double] weight,
        vector[int] order,
        vector[double] broken
    )


def creindex(
        vector[Edge] edges,
    ):

    cdef map[int,int] reindexing_key
    reindex_problem(edges, reindexing_key)
    return (edges, reindexing_key)

def cflip_neg_edges(
        vector[Edge] edges,
        vector[double] weights
    ):

    flip_neg_edges(edges, weights)
    return (edges, weights)

def cmfas(
        vector[Edge] edges,
        vector[double] weights,
    ):

    cdef vector[int] order
    mfas_ratio(edges, weights, order)

    return order

def cbroken_weight(
        vector[Edge] edges,
        vector[double] weights,
        vector[int] order
    ):

    cdef vector[double] broken
    broken_weight(edges, weights, order, broken)

    return broken
