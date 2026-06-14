#pragma once

#include "common/types.h"
#include <omp.h>

namespace anti_icing {
namespace parallel {

inline void setNumThreads(Index numThreads) {
    omp_set_num_threads(static_cast<int>(numThreads));
}

inline Index getNumThreads() {
    return static_cast<Index>(omp_get_max_threads());
}

inline Index getThreadId() {
    return static_cast<Index>(omp_get_thread_num());
}

inline bool inParallel() {
    return omp_in_parallel() != 0;
}

template <typename Func>
inline void parallelFor(Index start, Index end, Func&& func, bool useParallel = true) {
    if (useParallel) {
        #pragma omp parallel for
        for (Index i = start; i < end; ++i) {
            func(i);
        }
    } else {
        for (Index i = start; i < end; ++i) {
            func(i);
        }
    }
}

template <typename Func>
inline void parallelFor(Index end, Func&& func, bool useParallel = true) {
    parallelFor(0, end, std::forward<Func>(func), useParallel);
}

template <typename T, typename Func>
inline T parallelReduce(Index start, Index end, T init, Func&& func, bool useParallel = true) {
    T result = init;
    if (useParallel) {
        #pragma omp parallel for reduction(+:result)
        for (Index i = start; i < end; ++i) {
            result += func(i);
        }
    } else {
        for (Index i = start; i < end; ++i) {
            result += func(i);
        }
    }
    return result;
}

inline Scalar wallTime() {
    return static_cast<Scalar>(omp_get_wtime());
}

}
}
