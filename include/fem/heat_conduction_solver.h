#pragma once

#include "common/types.h"
#include "common/grid.h"
#include "common/parallel_utils.h"
#include <memory>
#include <vector>

namespace anti_icing {
namespace fem {

struct FEMConfig {
    Scalar theta;
    Scalar tolerance;
    Index  maxIter;
    bool   useLumpedMass;
    bool   useParallel;

    Scalar numericalDampingCoeff;
    bool   enableNumericalDamping;

    Scalar tempMinPhysical;
    Scalar tempMaxPhysical;

    Scalar freezeTransitionWidth;
    Scalar freezingPointTemp;

    Index  maxNewtonIter;
    Scalar newtonTolerance;
    bool   enableLineSearch;
};

struct FEMState {
    VectorX temperature;
    VectorX temperaturePrev;
    VectorX heatFlux;
    VectorX iceThickness;
    VectorX iceGrowthRate;

    void resize(Index nNodes) {
        temperature.resize(nNodes);
        temperaturePrev.resize(nNodes);
        heatFlux.resize(nNodes);
        iceThickness.resize(nNodes);
        iceGrowthRate.resize(nNodes);
    }
};

struct FEMBoundaryCondition {
    enum class Type {
        DIRICHLET,
        NEUMANN,
        CONVECTION,
        ICE_LATENT_HEAT
    };

    Type type;
    Index nodeId;
    Scalar value;
    Scalar convectiveCoeff;
    Scalar ambientTemp;
};

class HeatConductionSolver {
public:
    HeatConductionSolver(std::shared_ptr<Grid3D> grid,
                         const std::vector<MaterialProperties>& materials,
                         const FEMConfig& config);

    void initialize(Scalar initialTemp = 288.15);
    void assembleSystem(Scalar dt);
    void applyBoundaryConditions(const std::vector<FEMBoundaryCondition>& bcs);
    void solve();
    void advance(Scalar dt, const std::vector<FEMBoundaryCondition>& bcs);

    const VectorX& temperature() const { return state_.temperature; }
    const VectorX& heatFlux() const { return state_.heatFlux; }
    const FEMState& state() const { return state_; }
    FEMState& state() { return state_; }

    void setTemperature(const VectorX& T) { state_.temperature = T; }
    void setBoundaryHeatFlux(Index boundaryId, const VectorX& flux);
    void setBoundaryTemperature(Index boundaryId, const VectorX& temp);

    VectorX getBoundaryHeatFlux(Index boundaryId) const;
    VectorX getBoundaryTemperature(Index boundaryId) const;

    Index numNodes() const { return grid_->nNodes; }

private:
    void computeTetStiffnessMatrix(const TetrahedronCell& cell,
                                   const MaterialProperties& mat,
                                   Eigen::Matrix<Scalar, 4, 4>& K) const;

    void computeTetMassMatrix(const TetrahedronCell& cell,
                              const MaterialProperties& mat,
                              Eigen::Matrix<Scalar, 4, 4>& M) const;

    void computeTetLoadVector(const TetrahedronCell& cell,
                              Scalar dt,
                              Eigen::Matrix<Scalar, 4, 1>& F) const;

    void computeNodalGradient(const VectorX& field, Vector3Array& grad) const;

    void computeNodalHeatFlux();

    std::shared_ptr<Grid3D> grid_;
    std::vector<MaterialProperties> materials_;
    FEMConfig config_;
    FEMState state_;

    SparseMatrix K_global_;
    SparseMatrix M_global_;
    SparseMatrix A_global_;
    VectorX F_global_;
    VectorX R_global_;

    std::vector<std::vector<Triplet>> tripletBuffer_;
};

std::shared_ptr<HeatConductionSolver> createHeatConductionSolver(
    std::shared_ptr<Grid3D> grid,
    const std::vector<MaterialProperties>& materials,
    const FEMConfig& config);

}
}
