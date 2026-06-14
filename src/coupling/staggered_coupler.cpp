#include "coupling/staggered_coupler.h"
#include <cmath>
#include <algorithm>

namespace anti_icing {
namespace coupling {

StaggeredCoupler::StaggeredCoupler(std::shared_ptr<fvm::PipeFlowSolver> pipeSolver,
                                   std::shared_ptr<fem::HeatConductionSolver> solidSolver,
                                   const CouplingInterface& interface,
                                   const CouplingConfig& config)
    : pipeSolver_(pipeSolver), solidSolver_(solidSolver),
      interface_(interface), config_(config),
      residual_(1.0), iterations_(0), aitkenFactor_(config.relaxationFactor) {
    Index nPipe = pipeSolver_->grid().nCells;
    Index nSolid = solidSolver_->numNodes();

    interface_.pipeWallTemp.resize(nPipe);
    interface_.pipeWallHeatFlux.resize(nPipe);
    interface_.surfaceTemp.resize(nSolid);
    interface_.surfaceHeatFlux.resize(nSolid);

    prevSurfaceTemp_.resize(nSolid);
    prevPipeWallTemp_.resize(nPipe);

    residualHistoryTemp_.resize(nPipe);
    residualHistoryFlux_.resize(nPipe);
}

void StaggeredCoupler::initialize() {
    Index nPipe = pipeSolver_->grid().nCells;
    Index nSolid = solidSolver_->numNodes();

    interface_.pipeWallTemp = pipeSolver_->state().wallT;
    interface_.pipeWallHeatFlux = pipeSolver_->state().wallHeatFlux;
    interface_.surfaceTemp = solidSolver_->state().temperature;
    interface_.surfaceHeatFlux = solidSolver_->state().heatFlux;

    interface_.mappedPipeToSurfaceTemp.resize(nSolid);
    interface_.mappedPipeToSurfaceFlux.resize(nSolid);
    interface_.mappedSurfaceToPipeTemp.resize(nPipe);
    interface_.mappedSurfaceToPipeFlux.resize(nPipe);

    prevSurfaceTemp_ = interface_.surfaceTemp;
    prevPipeWallTemp_ = interface_.pipeWallTemp;

    mapPipeToSolid();
    mapSolidToPipe();
}

void StaggeredCoupler::mapPipeToSolid() {
    Index nPipe = pipeSolver_->grid().nCells;
    Index nSolid = solidSolver_->numNodes();

    Index start = solidSolver_->state().temperature.size();
    const auto& grid = solidSolver_->state();

    interface_.mappedPipeToSurfaceTemp.setZero();
    interface_.mappedPipeToSurfaceFlux.setZero();

    Index bStart = 0;
    Index bEnd = nPipe;

    for (Index i = 0; i < nSolid; ++i) {
        Index pipeIdx = std::min(nPipe - 1, static_cast<Index>(
            static_cast<Scalar>(i) / static_cast<Scalar>(nSolid) * static_cast<Scalar>(nPipe)));

        interface_.mappedPipeToSurfaceTemp(i) = interface_.pipeWallTemp(pipeIdx);
        interface_.mappedPipeToSurfaceFlux(i) = interface_.pipeWallHeatFlux(pipeIdx);
    }
}

void StaggeredCoupler::mapSolidToPipe() {
    Index nPipe = pipeSolver_->grid().nCells;
    Index nSolid = solidSolver_->numNodes();

    interface_.mappedSurfaceToPipeTemp.setZero();
    interface_.mappedSurfaceToPipeFlux.setZero();

    for (Index i = 0; i < nPipe; ++i) {
        Index solidIdx = std::min(nSolid - 1, static_cast<Index>(
            static_cast<Scalar>(i) / static_cast<Scalar>(nPipe) * static_cast<Scalar>(nSolid)));

        interface_.mappedSurfaceToPipeTemp(i) = interface_.surfaceTemp(solidIdx);
        interface_.mappedSurfaceToPipeFlux(i) = interface_.surfaceHeatFlux(solidIdx);
    }
}

bool StaggeredCoupler::iterate(Scalar dt) {
    iterations_++;

    interface_.pipeWallTemp = pipeSolver_->state().wallT;
    interface_.pipeWallHeatFlux = pipeSolver_->state().wallHeatFlux;
    interface_.surfaceTemp = solidSolver_->state().temperature;
    interface_.surfaceHeatFlux = solidSolver_->state().heatFlux;

    mapPipeToSolid();

    VectorX newPipeWallTemp = interface_.mappedSurfaceToPipeTemp;
    VectorX newSurfaceFlux = interface_.mappedPipeToSurfaceFlux;

    if (config_.useAitken) {
        aitkenRelaxation(newPipeWallTemp, newPipeWallTemp, prevPipeWallTemp_);
    } else {
        Scalar omega = config_.relaxationFactor;
        newPipeWallTemp = omega * newPipeWallTemp + (1.0 - omega) * prevPipeWallTemp_;
    }

    pipeSolver_->setWallTemperature(newPipeWallTemp);
    pipeSolver_->advance(dt);
    pipeSolver_->transferBoundaryData();

    interface_.pipeWallHeatFlux = pipeSolver_->getWallHeatFlux();
    mapPipeToSolid();

    std::vector<fem::FEMBoundaryCondition> internalBCs;
    Index nPipe = pipeSolver_->grid().nCells;
    for (Index i = 0; i < nPipe; ++i) {
        fem::FEMBoundaryCondition bc;
        bc.type = fem::FEMBoundaryCondition::Type::NEUMANN;
        bc.nodeId = std::min(solidSolver_->numNodes() - 1,
                             static_cast<Index>(static_cast<Scalar>(i) / static_cast<Scalar>(nPipe) * solidSolver_->numNodes()));
        bc.value = interface_.mappedPipeToSurfaceFlux(bc.nodeId);
        internalBCs.push_back(bc);
    }

    solidSolver_->advance(dt, internalBCs);

    interface_.surfaceTemp = solidSolver_->state().temperature;
    interface_.surfaceHeatFlux = solidSolver_->state().heatFlux;
    mapSolidToPipe();

    if (config_.enforceEnergyConservation) {
        enforceEnergyBalance();
    }

    residual_ = computeResidual();

    prevPipeWallTemp_ = newPipeWallTemp;
    prevSurfaceTemp_ = interface_.surfaceTemp;

    return isConverged();
}

void StaggeredCoupler::couple(Scalar dt) {
    initialize();
    iterations_ = 0;
    residual_ = 1.0;

    for (Index iter = 0; iter < config_.maxIterations; ++iter) {
        bool converged = iterate(dt);
        if (converged) break;
    }
}

Scalar StaggeredCoupler::computeResidual() const {
    Scalar maxResidual = 0.0;

    Index nPipe = pipeSolver_->grid().nCells;
    for (Index i = 0; i < nPipe; ++i) {
        Scalar deltaT = std::abs(interface_.pipeWallTemp(i) - prevPipeWallTemp_(i));
        Scalar normT = std::abs(prevPipeWallTemp_(i)) + 1.0e-10;
        Scalar res = deltaT / normT;
        if (res > maxResidual) maxResidual = res;
    }

    Index nSolid = solidSolver_->numNodes();
    for (Index i = 0; i < nSolid; ++i) {
        Scalar deltaT = std::abs(interface_.surfaceTemp(i) - prevSurfaceTemp_(i));
        Scalar normT = std::abs(prevSurfaceTemp_(i)) + 1.0e-10;
        Scalar res = deltaT / normT;
        if (res > maxResidual) maxResidual = res;
    }

    return maxResidual;
}

void StaggeredCoupler::enforceEnergyBalance() {
    Scalar pipeHeat = 0.0;
    Index nPipe = pipeSolver_->grid().nCells;
    Scalar perimeter = PI * interface_.pipeDiameter;
    for (Index i = 0; i < nPipe; ++i) {
        pipeHeat += interface_.pipeWallHeatFlux(i) * perimeter * pipeSolver_->grid().dx(i);
    }

    Scalar surfaceHeat = 0.0;
    Index nSolid = solidSolver_->numNodes();
    for (Index i = 0; i < nSolid; ++i) {
        surfaceHeat += interface_.surfaceHeatFlux(i);
    }

    Scalar imbalance = pipeHeat - surfaceHeat;
    if (std::abs(surfaceHeat) > 1.0e-10) {
        Scalar correction = imbalance / surfaceHeat;
        for (Index i = 0; i < nSolid; ++i) {
            interface_.surfaceHeatFlux(i) *= (1.0 + correction);
        }
    }
}

Scalar StaggeredCoupler::computeEnergyImbalance() const {
    Scalar pipeHeat = 0.0;
    Index nPipe = pipeSolver_->grid().nCells;
    Scalar perimeter = PI * interface_.pipeDiameter;
    for (Index i = 0; i < nPipe; ++i) {
        pipeHeat += interface_.pipeWallHeatFlux(i) * perimeter * pipeSolver_->grid().dx(i);
    }

    Scalar surfaceHeat = 0.0;
    Index nSolid = solidSolver_->numNodes();
    for (Index i = 0; i < nSolid; ++i) {
        surfaceHeat += interface_.surfaceHeatFlux(i);
    }

    return pipeHeat - surfaceHeat;
}

void StaggeredCoupler::aitkenRelaxation(VectorX& field, const VectorX& newField, const VectorX& oldField) {
    VectorX deltaNew = newField - oldField;
    VectorX deltaOld = oldField - prevSurfaceTemp_;

    Scalar numerator = deltaOld.dot(deltaNew - deltaOld);
    Scalar denominator = (deltaNew - deltaOld).squaredNorm();

    if (std::abs(denominator) > 1.0e-20) {
        Scalar omega_new = -numerator / denominator;
        omega_new = std::max(0.1, std::min(1.5, omega_new));
        aitkenFactor_ = omega_new;
    }

    field = oldField + aitkenFactor_ * (newField - oldField);
}

std::shared_ptr<StaggeredCoupler> createStaggeredCoupler(
    std::shared_ptr<fvm::PipeFlowSolver> pipeSolver,
    std::shared_ptr<fem::HeatConductionSolver> solidSolver,
    const CouplingInterface& interface,
    const CouplingConfig& config) {
    auto coupler = std::make_shared<StaggeredCoupler>(pipeSolver, solidSolver, interface, config);
    coupler->initialize();
    return coupler;
}

}
}
