#pragma once

#include "common/types.h"
#include "fvm/pipe_flow_solver.h"
#include "fem/heat_conduction_solver.h"
#include "bc/boundary_conditions.h"
#include <memory>

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

struct CouplingConfig {
    Scalar relaxationFactor;
    Index  maxIterations;
    Scalar tolerance;
    bool   useAitken;
    bool   enforceEnergyConservation;
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

private:
    void aitkenRelaxation(VectorX& field, const VectorX& newField, const VectorX& oldField);

    std::shared_ptr<fvm::PipeFlowSolver> pipeSolver_;
    std::shared_ptr<fem::HeatConductionSolver> solidSolver_;
    CouplingInterface interface_;
    CouplingConfig config_;

    Scalar residual_;
    Index iterations_;
    Scalar aitkenFactor_;
    VectorX prevSurfaceTemp_;
    VectorX prevPipeWallTemp_;

    VectorX residualHistoryTemp_;
    VectorX residualHistoryFlux_;
};

std::shared_ptr<StaggeredCoupler> createStaggeredCoupler(
    std::shared_ptr<fvm::PipeFlowSolver> pipeSolver,
    std::shared_ptr<fem::HeatConductionSolver> solidSolver,
    const CouplingInterface& interface,
    const CouplingConfig& config);

}
}
