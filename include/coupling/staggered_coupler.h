#pragma once

#include "common/types.h"
#include "fvm/pipe_flow_solver.h"
#include "fem/heat_conduction_solver.h"
#include "bc/boundary_conditions.h"
#include <memory>
#include <deque>

namespace anti_icing {
namespace coupling {

struct CouplingInterface {
    Index pipeBoundaryId;
    Index externalBoundaryId;
    Scalar internalConvCoeff;
    Scalar pipeDiameter;

    VectorX pipeWallTemp;
    VectorX pipeWallHeatFlux;
    VectorX surfaceTemp;
    VectorX surfaceHeatFlux;

    VectorX mappedPipeToSurfaceTemp;
    VectorX mappedPipeToSurfaceFlux;
    VectorX mappedSurfaceToPipeTemp;
    VectorX mappedSurfaceToPipeFlux;
};

enum class CouplingStatus {
    CONVERGED,
    STABLE,
    RESIDUAL_GROWING,
    DIVERGING,
    INVALID,
    BACKTRACKED
};

struct CouplingConfig {
    Scalar relaxationFactor;
    Index  maxIterations;
    Scalar tolerance;
    bool   useAitken;
    bool   enforceEnergyConservation;

    Scalar minRelaxation;
    Scalar maxRelaxation;
    Scalar residualGrowthThreshold;
    Scalar divergenceThreshold;
    Scalar relaxationCutFactor;
    Scalar relaxationRecoveryFactor;
    Index  residualHistorySize;

    Scalar tempMinPhysical;
    Scalar tempMaxPhysical;
    Scalar heatFluxMaxPhysical;

    bool   enableBacktracking;
    Index  maxBacktrackSteps;
    Scalar backtrackFactor;

    Scalar newtonDampingCoeff;
    bool   enableNewtonDamping;
};

struct CouplingIterationRecord {
    Scalar residual;
    Scalar relaxationFactor;
    Scalar energyImbalance;
    CouplingStatus status;
};

class StaggeredCoupler {
public:
    StaggeredCoupler(std::shared_ptr<fvm::PipeFlowSolver> pipeSolver,
                     std::shared_ptr<fem::HeatConductionSolver> solidSolver,
                     const CouplingInterface& interface,
                     const CouplingConfig& config);

    void initialize();
    bool iterate(Scalar dt);
    void couple(Scalar dt);

    Scalar computeResidual() const;
    bool isConverged() const { return residual_ < config_.tolerance; }

    void enforceEnergyBalance();

    void mapPipeToSolid();
    void mapSolidToPipe();

    Scalar computeEnergyImbalance() const;

    const CouplingInterface& interface() const { return interface_; }
    Scalar residual() const { return residual_; }
    Index iterations() const { return iterations_; }
    Scalar currentRelaxation() const { return currentRelaxation_; }
    CouplingStatus status() const { return lastStatus_; }

    const std::deque<CouplingIterationRecord>& history() const { return history_; }

private:
    CouplingStatus assessConvergenceStatus(Scalar currentResidual, Scalar prevResidual) const;

    Scalar adaptiveRelaxationFactor(Scalar currentRelax, CouplingStatus status) const;

    void aitkenRelaxation(VectorX& field, const VectorX& newField, const VectorX& oldField,
                          Scalar relaxationOverride = -1.0);

    bool validateAndClampFields();

    void saveState();
    bool restoreState();

    void addNumericalDampingToStiffness(SparseMatrix& A, Scalar dampingCoeff);

    void recordIteration(Scalar res, Scalar relax, Scalar imbalance, CouplingStatus s);

    std::shared_ptr<fvm::PipeFlowSolver> pipeSolver_;
    std::shared_ptr<fem::HeatConductionSolver> solidSolver_;
    CouplingInterface interface_;
    CouplingConfig config_;

    Scalar residual_;
    Index iterations_;
    Scalar aitkenFactor_;
    Scalar currentRelaxation_;
    CouplingStatus lastStatus_;

    VectorX prevSurfaceTemp_;
    VectorX prevPipeWallTemp_;
    VectorX prevSurfaceFlux_;
    VectorX prevPipeWallFlux_;

    fvm::PipeFlowState savedPipeState_;
    fem::FEMState savedSolidState_;
    VectorX savedSurfaceTemp_;
    VectorX savedPipeWallTemp_;

    std::deque<CouplingIterationRecord> history_;
    Index backtrackCount_;
};

std::shared_ptr<StaggeredCoupler> createStaggeredCoupler(
    std::shared_ptr<fvm::PipeFlowSolver> pipeSolver,
    std::shared_ptr<fem::HeatConductionSolver> solidSolver,
    const CouplingInterface& interface,
    const CouplingConfig& config);

}
}
