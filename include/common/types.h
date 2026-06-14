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

}
