#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <memory>
#include <cmath>

namespace anti_icing {

using Scalar = double;
using Index  = Eigen::Index;

using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;

using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;
using SparseVector = Eigen::SparseVector<Scalar, Eigen::RowMajor>;
using Triplet      = Eigen::Triplet<Scalar>;

using VectorXArray = std::vector<VectorX, Eigen::aligned_allocator<VectorX>>;
using Vector3Array = std::vector<Vector3, Eigen::aligned_allocator<Vector3>>;
using Matrix3Array = std::vector<Matrix3, Eigen::aligned_allocator<Matrix3>>;

static constexpr Scalar R_AIR      = 287.0;
static constexpr Scalar GAMMA_AIR  = 1.4;
static constexpr Scalar CP_AIR     = 1005.0;
static constexpr Scalar MU_AIR_REF = 1.789e-5;
static constexpr Scalar T_AIR_REF  = 288.15;
static constexpr Scalar SUTHERLAND = 110.4;

static constexpr Scalar LATENT_HEAT_ICE = 334000.0;
static constexpr Scalar ICE_DENSITY     = 917.0;
static constexpr Scalar WATER_DENSITY   = 1000.0;

static constexpr Scalar PI = 3.14159265358979323846;

struct MaterialProperties {
    Matrix3 conductivity;
    Scalar  density;
    Scalar  specificHeat;
};

struct SolverConfig {
    Scalar timeStep        = 1.0e-6;
    Scalar totalTime       = 1.0;
    Index  maxIterations   = 10000;
    Scalar tolerance       = 1.0e-8;
    Index  numThreads      = 8;
    bool   useParallel     = true;
    Scalar couplingRelax   = 0.7;
    Index  couplingMaxIter = 20;
};

inline Scalar computeAirViscosity(Scalar T) {
    Scalar ratio = T / T_AIR_REF;
    return MU_AIR_REF * ratio * std::sqrt(ratio) * (T_AIR_REF + SUTHERLAND) / (T + SUTHERLAND);
}

inline Scalar computeAirDensity(Scalar P, Scalar T) {
    return P / (R_AIR * T);
}

inline Scalar computeSpeedOfSound(Scalar T) {
    return std::sqrt(GAMMA_AIR * R_AIR * T);
}

inline bool isNaN(Scalar x) {
    return std::isnan(x) || std::isinf(x);
}

inline Scalar clamp(Scalar x, Scalar lo, Scalar hi) {
    return std::max(lo, std::min(hi, x));
}

inline Scalar smoothClamp(Scalar x, Scalar lo, Scalar hi, Scalar transitionWidth = 1.0) {
    if (x <= lo - transitionWidth) return lo;
    if (x >= hi + transitionWidth) return hi;
    if (x >= lo && x <= hi) return x;
    if (x < lo) {
        Scalar t = (x - (lo - transitionWidth)) / transitionWidth;
        return lo + transitionWidth * (t * t * (3.0 - 2.0 * t)) * 0.0;
    } else {
        Scalar t = (x - hi) / transitionWidth;
        return hi + transitionWidth * (1.0 - t * t * (3.0 - 2.0 * t)) * 0.0;
    }
    return clamp(x, lo, hi);
}

inline bool isVectorValid(const VectorX& v) {
    for (Index i = 0; i < v.size(); ++i) {
        if (isNaN(v(i))) return false;
    }
    return true;
}

inline VectorX clampVector(const VectorX& v, Scalar lo, Scalar hi) {
    VectorX result(v.size());
    for (Index i = 0; i < v.size(); ++i) {
        result(i) = clamp(v(i), lo, hi);
    }
    return result;
}

inline Scalar sigmoid(Scalar x, Scalar x0 = 0.0, Scalar width = 1.0) {
    Scalar z = (x - x0) / width;
    z = clamp(z, -50.0, 50.0);
    return 1.0 / (1.0 + std::exp(-z));
}

}
