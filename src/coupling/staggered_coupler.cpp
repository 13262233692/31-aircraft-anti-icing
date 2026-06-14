#include "coupling/staggered_coupler.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace anti_icing {
namespace coupling {

StaggeredCoupler::StaggeredCoupler(std::shared_ptr<fvm::PipeFlowSolver> pipeSolver,
                                   std::shared_ptr<fem::HeatConductionSolver> solidSolver,
                                   const CouplingInterface& interface,
                                   const CouplingConfig& config)
    : pipeSolver_(pipeSolver), solidSolver_(solidSolver),
      interface_(interface), config_(config),
      residual_(1.0), iterations_(0),
      aitkenFactor_(config.relaxationFactor),
      currentRelaxation_(config.relaxationFactor),
      lastStatus_(CouplingStatus::STABLE),
      backtrackCount_(0) {
    Index nPipe = pipeSolver_->grid().nCells;
    Index nSolid = solidSolver_->numNodes();

    interface_.pipeWallTemp.resize(nPipe);
    interface_.pipeWallHeatFlux.resize(nPipe);
    interface_.surfaceTemp.resize(nSolid);
    interface_.surfaceHeatFlux.resize(nSolid);

    prevSurfaceTemp_.resize(nSolid);
    prevPipeWallTemp_.resize(nPipe);
    prevSurfaceFlux_.resize(nSolid);
    prevPipeWallFlux_.resize(nPipe);

    savedPipeState_.resize(nPipe);
    savedSurfaceTemp_.resize(nSolid);
    savedPipeWallTemp_.resize(nPipe);

    history_.clear();
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
    prevSurfaceFlux_ = interface_.surfaceHeatFlux;
    prevPipeWallFlux_ = interface_.pipeWallHeatFlux;

    currentRelaxation_ = config_.relaxationFactor;
    aitkenFactor_ = config_.relaxationFactor;
    residual_ = 1.0;
    lastStatus_ = CouplingStatus::STABLE;
    iterations_ = 0;
    backtrackCount_ = 0;
    history_.clear();

    mapPipeToSolid();
    mapSolidToPipe();

    validateAndClampFields();
}

void StaggeredCoupler::mapPipeToSolid() {
    Index nPipe = pipeSolver_->grid().nCells;
    Index nSolid = solidSolver_->numNodes();

    interface_.mappedPipeToSurfaceTemp.setZero();
    interface_.mappedPipeToSurfaceFlux.setZero();

    for (Index i = 0; i < nSolid; ++i) {
        Scalar ratio = static_cast<Scalar>(i) / std::max(1.0, static_cast<Scalar>(nSolid - 1));
        Index pipeIdx = static_cast<Index>(ratio * static_cast<Scalar>(nPipe - 1));
        pipeIdx = clamp(pipeIdx, Index(0), nPipe - 1);

        Scalar frac = ratio * static_cast<Scalar>(nPipe - 1) - static_cast<Scalar>(pipeIdx);
        Index pipeNext = std::min(pipeIdx + 1, nPipe - 1);

        interface_.mappedPipeToSurfaceTemp(i) =
            (1.0 - frac) * interface_.pipeWallTemp(pipeIdx) +
            frac * interface_.pipeWallTemp(pipeNext);
        interface_.mappedPipeToSurfaceFlux(i) =
            (1.0 - frac) * interface_.pipeWallHeatFlux(pipeIdx) +
            frac * interface_.pipeWallHeatFlux(pipeNext);
    }
}

void StaggeredCoupler::mapSolidToPipe() {
    Index nPipe = pipeSolver_->grid().nCells;
    Index nSolid = solidSolver_->numNodes();

    interface_.mappedSurfaceToPipeTemp.setZero();
    interface_.mappedSurfaceToPipeFlux.setZero();

    for (Index i = 0; i < nPipe; ++i) {
        Scalar ratio = static_cast<Scalar>(i) / std::max(1.0, static_cast<Scalar>(nPipe - 1));
        Index solidIdx = static_cast<Index>(ratio * static_cast<Scalar>(nSolid - 1));
        solidIdx = clamp(solidIdx, Index(0), nSolid - 1);

        Scalar frac = ratio * static_cast<Scalar>(nSolid - 1) - static_cast<Scalar>(solidIdx);
        Index solidNext = std::min(solidIdx + 1, nSolid - 1);

        interface_.mappedSurfaceToPipeTemp(i) =
            (1.0 - frac) * interface_.surfaceTemp(solidIdx) +
            frac * interface_.surfaceTemp(solidNext);
        interface_.mappedSurfaceToPipeFlux(i) =
            (1.0 - frac) * interface_.surfaceHeatFlux(solidIdx) +
            frac * interface_.surfaceHeatFlux(solidNext);
    }
}

CouplingStatus StaggeredCoupler::assessConvergenceStatus(Scalar currentResidual,
                                                         Scalar prevResidual) const {
    if (currentResidual <= config_.tolerance) {
        return CouplingStatus::CONVERGED;
    }

    if (!isVectorValid(interface_.surfaceTemp) ||
        !isVectorValid(interface_.pipeWallTemp) ||
        !isVectorValid(interface_.surfaceHeatFlux) ||
        !isVectorValid(interface_.pipeWallHeatFlux)) {
        return CouplingStatus::INVALID;
    }

    Scalar ratio = currentResidual / std::max(prevResidual, 1.0e-20);

    if (currentResidual > config_.divergenceThreshold) {
        return CouplingStatus::DIVERGING;
    }

    if (ratio > config_.residualGrowthThreshold) {
        return CouplingStatus::RESIDUAL_GROWING;
    }

    return CouplingStatus::STABLE;
}

Scalar StaggeredCoupler::adaptiveRelaxationFactor(Scalar currentRelax,
                                                   CouplingStatus status) const {
    Scalar newRelax = currentRelax;

    switch (status) {
    case CouplingStatus::CONVERGED:
        newRelax = std::min(config_.maxRelaxation,
                            currentRelax * config_.relaxationRecoveryFactor);
        break;
    case CouplingStatus::STABLE:
        newRelax = std::min(config_.maxRelaxation,
                            currentRelax * (1.0 + 0.02 * (config_.relaxationRecoveryFactor - 1.0)));
        break;
    case CouplingStatus::RESIDUAL_GROWING:
        newRelax = currentRelax * config_.relaxationCutFactor;
        break;
    case CouplingStatus::DIVERGING:
        newRelax = currentRelax * config_.relaxationCutFactor * config_.relaxationCutFactor;
        break;
    case CouplingStatus::INVALID:
        newRelax = config_.minRelaxation;
        break;
    case CouplingStatus::BACKTRACKED:
        newRelax = std::max(config_.minRelaxation,
                            currentRelax * config_.backtrackFactor);
        break;
    }

    newRelax = clamp(newRelax, config_.minRelaxation, config_.maxRelaxation);
    return newRelax;
}

void StaggeredCoupler::recordIteration(Scalar res, Scalar relax,
                                        Scalar imbalance, CouplingStatus s) {
    CouplingIterationRecord rec{res, relax, imbalance, s};
    history_.push_back(rec);
    if (history_.size() > config_.residualHistorySize) {
        history_.pop_front();
    }
}

void StaggeredCoupler::aitkenRelaxation(VectorX& field, const VectorX& newField,
                                         const VectorX& oldField,
                                         Scalar relaxationOverride) {
    Scalar omega;
    if (relaxationOverride > 0.0) {
        omega = relaxationOverride;
    } else {
        VectorX deltaNew = newField - oldField;
        VectorX deltaOld = oldField - prevPipeWallTemp_;

        Scalar numerator = deltaOld.dot(deltaNew - deltaOld);
        Scalar denominator = (deltaNew - deltaOld).squaredNorm();

        if (std::abs(denominator) > 1.0e-20 && isNaN(denominator) == false) {
            Scalar omega_aitken = -numerator / denominator;
            if (!isNaN(omega_aitken)) {
                omega_aitken = clamp(omega_aitken, config_.minRelaxation,
                                     config_.maxRelaxation);
                omega = 0.5 * omega_aitken + 0.5 * currentRelaxation_;
            } else {
                omega = currentRelaxation_;
            }
        } else {
            omega = currentRelaxation_;
        }
        aitkenFactor_ = omega;
    }

    omega = clamp(omega, config_.minRelaxation, config_.maxRelaxation);

    field = oldField + omega * (newField - oldField);

    field = clampVector(field, config_.tempMinPhysical, config_.tempMaxPhysical);
}

bool StaggeredCoupler::validateAndClampFields() {
    bool wasInvalid = false;

    if (!isVectorValid(interface_.surfaceTemp)) {
        wasInvalid = true;
        interface_.surfaceTemp = prevSurfaceTemp_;
    }
    interface_.surfaceTemp = clampVector(interface_.surfaceTemp,
                                          config_.tempMinPhysical,
                                          config_.tempMaxPhysical);

    if (!isVectorValid(interface_.pipeWallTemp)) {
        wasInvalid = true;
        interface_.pipeWallTemp = prevPipeWallTemp_;
    }
    interface_.pipeWallTemp = clampVector(interface_.pipeWallTemp,
                                           config_.tempMinPhysical,
                                           config_.tempMaxPhysical);

    if (!isVectorValid(interface_.surfaceHeatFlux)) {
        wasInvalid = true;
        interface_.surfaceHeatFlux = prevSurfaceFlux_;
    }
    for (Index i = 0; i < interface_.surfaceHeatFlux.size(); ++i) {
        interface_.surfaceHeatFlux(i) = clamp(interface_.surfaceHeatFlux(i),
                                               -config_.heatFluxMaxPhysical,
                                               config_.heatFluxMaxPhysical);
    }

    if (!isVectorValid(interface_.pipeWallHeatFlux)) {
        wasInvalid = true;
        interface_.pipeWallHeatFlux = prevPipeWallFlux_;
    }
    for (Index i = 0; i < interface_.pipeWallHeatFlux.size(); ++i) {
        interface_.pipeWallHeatFlux(i) = clamp(interface_.pipeWallHeatFlux(i),
                                                -config_.heatFluxMaxPhysical,
                                                config_.heatFluxMaxPhysical);
    }

    return wasInvalid;
}

void StaggeredCoupler::saveState() {
    savedPipeState_ = pipeSolver_->state();
    savedSolidState_ = solidSolver_->state();
    savedSurfaceTemp_ = interface_.surfaceTemp;
    savedPipeWallTemp_ = interface_.pipeWallTemp;
}

bool StaggeredCoupler::restoreState() {
    if (backtrackCount_ >= config_.maxBacktrackSteps) {
        return false;
    }
    backtrackCount_++;

    pipeSolver_->state() = savedPipeState_;
    solidSolver_->state() = savedSolidState_;
    interface_.surfaceTemp = savedSurfaceTemp_;
    interface_.pipeWallTemp = savedPipeWallTemp_;

    prevSurfaceTemp_ = savedSurfaceTemp_;
    prevPipeWallTemp_ = savedPipeWallTemp_;

    currentRelaxation_ = std::max(config_.minRelaxation,
                                   currentRelaxation_ * config_.backtrackFactor);

    lastStatus_ = CouplingStatus::BACKTRACKED;
    return true;
}

bool StaggeredCoupler::iterate(Scalar dt) {
    iterations_++;
    Scalar prevResidual = residual_;

    if (iterations_ == 1 || backtrackCount_ == 0) {
        saveState();
    }

    interface_.pipeWallTemp = pipeSolver_->state().wallT;
    interface_.pipeWallHeatFlux = pipeSolver_->state().wallHeatFlux;
    interface_.surfaceTemp = solidSolver_->state().temperature;
    interface_.surfaceHeatFlux = solidSolver_->state().heatFlux;

    mapPipeToSolid();

    VectorX newPipeWallTemp = interface_.mappedSurfaceToPipeTemp;
    CouplingStatus preStatus = assessConvergenceStatus(prevResidual, prevResidual);
    currentRelaxation_ = adaptiveRelaxationFactor(currentRelaxation_, preStatus);

    if (config_.useAitken) {
        aitkenRelaxation(newPipeWallTemp, newPipeWallTemp, prevPipeWallTemp_);
    } else {
        aitkenRelaxation(newPipeWallTemp, newPipeWallTemp, prevPipeWallTemp_,
                         currentRelaxation_);
    }

    pipeSolver_->setWallTemperature(newPipeWallTemp);
    pipeSolver_->advance(dt);
    pipeSolver_->transferBoundaryData();

    interface_.pipeWallHeatFlux = pipeSolver_->getWallHeatFlux();
    mapPipeToSolid();

    bool fieldsInvalid = validateAndClampFields();

    std::vector<fem::FEMBoundaryCondition> internalBCs;
    Index nPipe = pipeSolver_->grid().nCells;
    Index nSolid = solidSolver_->numNodes();
    for (Index i = 0; i < nPipe; ++i) {
        fem::FEMBoundaryCondition bc;
        bc.type = fem::FEMBoundaryCondition::Type::NEUMANN;
        Scalar ratio = static_cast<Scalar>(i) / std::max(1.0, static_cast<Scalar>(nPipe - 1));
        bc.nodeId = static_cast<Index>(ratio * static_cast<Scalar>(nSolid - 1));
        bc.nodeId = clamp(bc.nodeId, Index(0), nSolid - 1);
        bc.value = interface_.mappedPipeToSurfaceFlux(bc.nodeId);
        bc.value = clamp(bc.value, -config_.heatFluxMaxPhysical, config_.heatFluxMaxPhysical);
        internalBCs.push_back(bc);
    }

    try {
        solidSolver_->advance(dt, internalBCs);
    } catch (...) {
        if (config_.enableBacktracking && restoreState()) {
            Scalar imbalance = computeEnergyImbalance();
            recordIteration(residual_, currentRelaxation_, imbalance, CouplingStatus::BACKTRACKED);
            return false;
        }
        throw;
    }

    interface_.surfaceTemp = solidSolver_->state().temperature;
    interface_.surfaceHeatFlux = solidSolver_->state().heatFlux;
    mapSolidToPipe();

    if (config_.enforceEnergyConservation) {
        enforceEnergyBalance();
    }

    fieldsInvalid = fieldsInvalid || validateAndClampFields();

    residual_ = computeResidual();
    CouplingStatus status = assessConvergenceStatus(residual_, prevResidual);

    if (fieldsInvalid) {
        status = CouplingStatus::INVALID;
    }

    if ((status == CouplingStatus::DIVERGING || status == CouplingStatus::INVALID) &&
        config_.enableBacktracking) {
        if (restoreState()) {
            residual_ = prevResidual;
            Scalar imbalance = computeEnergyImbalance();
            recordIteration(residual_, currentRelaxation_, imbalance, status);
            lastStatus_ = status;
            return false;
        }
    }

    currentRelaxation_ = adaptiveRelaxationFactor(currentRelaxation_, status);
    lastStatus_ = status;

    prevPipeWallTemp_ = newPipeWallTemp;
    prevSurfaceTemp_ = interface_.surfaceTemp;
    prevPipeWallFlux_ = interface_.pipeWallHeatFlux;
    prevSurfaceFlux_ = interface_.surfaceHeatFlux;

    Scalar imbalance = computeEnergyImbalance();
    recordIteration(residual_, currentRelaxation_, imbalance, status);

    return isConverged();
}

void StaggeredCoupler::couple(Scalar dt) {
    initialize();
    backtrackCount_ = 0;

    for (Index iter = 0; iter < config_.maxIterations; ++iter) {
        bool converged = iterate(dt);

        if (lastStatus_ == CouplingStatus::INVALID ||
            lastStatus_ == CouplingStatus::DIVERGING) {
            backtrackCount_ = 0;
            continue;
        }

        if (converged) break;

        if (backtrackCount_ > 0 && iter > 5) {
            backtrackCount_ = 0;
        }
    }
}

Scalar StaggeredCoupler::computeResidual() const {
    Scalar maxResidual = 0.0;

    Index nPipe = pipeSolver_->grid().nCells;
    for (Index i = 0; i < nPipe; ++i) {
        Scalar deltaT = std::abs(interface_.pipeWallTemp(i) - prevPipeWallTemp_(i));
        Scalar normT = std::abs(prevPipeWallTemp_(i)) + 1.0e-10;
        Scalar res = deltaT / normT;
        if (!isNaN(res)) {
            maxResidual = std::max(maxResidual, res);
        }
    }

    Index nSolid = solidSolver_->numNodes();
    for (Index i = 0; i < nSolid; ++i) {
        Scalar deltaT = std::abs(interface_.surfaceTemp(i) - prevSurfaceTemp_(i));
        Scalar normT = std::abs(prevSurfaceTemp_(i)) + 1.0e-10;
        Scalar res = deltaT / normT;
        if (!isNaN(res)) {
            maxResidual = std::max(maxResidual, res);
        }
    }

    return isNaN(maxResidual) ? 1.0 : maxResidual;
}

void StaggeredCoupler::enforceEnergyBalance() {
    Scalar pipeHeat = 0.0;
    Index nPipe = pipeSolver_->grid().nCells;
    Scalar perimeter = PI * interface_.pipeDiameter;
    for (Index i = 0; i < nPipe; ++i) {
        Scalar q = interface_.pipeWallHeatFlux(i);
        if (!isNaN(q)) {
            pipeHeat += q * perimeter * pipeSolver_->grid().dx(i);
        }
    }

    Scalar surfaceHeat = 0.0;
    Index nSolid = solidSolver_->numNodes();
    for (Index i = 0; i < nSolid; ++i) {
        Scalar q = interface_.surfaceHeatFlux(i);
        if (!isNaN(q)) {
            surfaceHeat += q;
        }
    }

    Scalar imbalance = pipeHeat - surfaceHeat;
    Scalar correction = 0.0;
    if (std::abs(surfaceHeat) > 1.0e-10) {
        correction = imbalance / surfaceHeat;
        correction = clamp(correction, -0.5, 0.5);
    }

    if (std::abs(correction) > 0.0) {
        for (Index i = 0; i < nSolid; ++i) {
            interface_.surfaceHeatFlux(i) *= (1.0 + 0.5 * correction);
        }
    }

    validateAndClampFields();
}

Scalar StaggeredCoupler::computeEnergyImbalance() const {
    Scalar pipeHeat = 0.0;
    Index nPipe = pipeSolver_->grid().nCells;
    Scalar perimeter = PI * interface_.pipeDiameter;
    for (Index i = 0; i < nPipe; ++i) {
        Scalar q = interface_.pipeWallHeatFlux(i);
        if (!isNaN(q)) pipeHeat += q * perimeter * pipeSolver_->grid().dx(i);
    }

    Scalar surfaceHeat = 0.0;
    Index nSolid = solidSolver_->numNodes();
    for (Index i = 0; i < nSolid; ++i) {
        Scalar q = interface_.surfaceHeatFlux(i);
        if (!isNaN(q)) surfaceHeat += q;
    }

    return pipeHeat - surfaceHeat;
}

void StaggeredCoupler::addNumericalDampingToStiffness(SparseMatrix& A, Scalar dampingCoeff) {
    for (Index k = 0; k < A.outerSize(); ++k) {
        for (SparseMatrix::InnerIterator it(A, k); it; ++it) {
            if (it.row() == it.col()) {
                Scalar damping = dampingCoeff * std::abs(it.value());
                it.valueRef() += damping;
            }
        }
    }
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
