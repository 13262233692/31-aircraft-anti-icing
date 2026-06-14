#include "solver/anti_icing_solver.h"
#include "common/parallel_utils.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace anti_icing {

AntiIcingSolver::AntiIcingSolver(const AntiIcingSystemConfig& config)
    : config_(config), currentTime_(0.0), currentDt_(0.0),
      currentStep_(0), totalSubSteps_(0),
      dtHistoryPtr_(0), savedTime_(0.0),
      levelSetSolver_(nullptr) {
    diag_ = {};
    for (auto& d : dtHistory_) d = 0.0;
}

void AntiIcingSolver::initialize() {
    parallel::setNumThreads(config_.solverConfig.numThreads);

    std::cout << "========================================" << std::endl;
    std::cout << "  Aircraft Anti-Icing Solver v2.5" << std::endl;
    std::cout << "  Robust Adaptive Coupling + Level-Set" << std::endl;
    std::cout << "  Dynamic Ice Shape Evolution" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Initializing pipe flow solver (FVM)..." << std::endl;

    pipeSolver_ = fvm::createPipeFlowSolver(config_.pipeConfig, config_.pipeBC);

    std::cout << "  Pipe cells: " << config_.pipeConfig.nCells << std::endl;
    std::cout << "  Pipe diameter: " << config_.pipeConfig.diameter << " m" << std::endl;
    std::cout << "  Pipe length: " << config_.pipeConfig.length << " m" << std::endl;

    std::cout << "Initializing solid heat conduction solver (FEM)..." << std::endl;

    Index nx = 10, ny = 10, nz = 4;
    Scalar Lx = 0.5, Ly = 0.3, Lz = 0.02;
    solidGrid_ = createStructuredTetGrid(Lx, Ly, Lz, nx, ny, nz);

    MaterialProperties composite;
    composite.conductivity <<
        5.0, 0.5, 0.5,
        0.5, 1.5, 0.3,
        0.5, 0.3, 1.5;
    composite.density = 1580.0;
    composite.specificHeat = 1200.0;
    materials_.push_back(composite);

    solidSolver_ = fem::createHeatConductionSolver(solidGrid_, materials_, config_.femConfig);

    std::cout << "  Solid nodes: " << solidGrid_->nNodes << std::endl;
    std::cout << "  Solid cells: " << solidGrid_->nCells << std::endl;

    std::cout << "Initializing staggered coupler..." << std::endl;

    config_.couplingInterface.pipeDiameter = config_.pipeConfig.diameter;
    coupler_ = coupling::createStaggeredCoupler(
        pipeSolver_, solidSolver_,
        config_.couplingInterface, config_.couplingConfig);

    std::cout << "  Adaptive relaxation range: ["
              << config_.couplingConfig.minRelaxation << ", "
              << config_.couplingConfig.maxRelaxation << "]" << std::endl;
    std::cout << "  Backtracking: "
              << (config_.couplingConfig.enableBacktracking ? "ON" : "OFF") << std::endl;
    std::cout << "  Energy conservation: "
              << (config_.couplingConfig.enforceEnergyConservation ? "ON" : "OFF") << std::endl;

    std::cout << "  Freeze transition width: "
              << config_.femConfig.freezeTransitionWidth << " K" << std::endl;
    std::cout << "  Numerical damping: "
              << (config_.femConfig.enableNumericalDamping ? "ON" : "OFF")
              << " (coeff=" << config_.femConfig.numericalDampingCoeff << ")" << std::endl;
    std::cout << "  Line search Newton: "
              << (config_.femConfig.enableLineSearch ? "ON" : "OFF") << std::endl;

    if (config_.enableDynamicIceShape) {
        std::cout << "Initializing Level-Set dynamic ice shape solver..." << std::endl;
        levelSetSolver_ = levelset::createLevelSetSolver(
            config_.levelSetConfig, config_.iceShapeConfig);
        std::cout << "  Level-Set grid: " << config_.levelSetConfig.gridSizeX
                  << " x " << config_.levelSetConfig.gridSizeY << std::endl;
        std::cout << "  WENO order: " << config_.levelSetConfig.wenoOrder << std::endl;
        std::cout << "  Reinit steps: " << config_.levelSetConfig.reinitializationSteps << std::endl;
        std::cout << "  Update interval: every "
                  << config_.iceShapeUpdateInterval << " macro steps" << std::endl;
        std::cout << "  Remesh threshold: " << config_.iceRemeshThreshold << " m" << std::endl;
        Index nPts = levelSetSolver_->numSurfacePoints();
        baseConvectionFactors_ = VectorX::Ones(nPts);
        savedIceThickness_.resize(nPts);
        savedIceThickness_.setZero();
    } else {
        std::cout << "  Dynamic ice shape evolution: DISABLED" << std::endl;
    }

    std::cout << "Solver initialization complete." << std::endl;
    std::cout << "  Base time step: " << config_.solverConfig.timeStep << " s" << std::endl;
    std::cout << "  Total time: " << config_.solverConfig.totalTime << " s" << std::endl;
    std::cout << "  Threads: " << config_.solverConfig.numThreads << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    if (config_.enableDynamicIceShape) {
        std::cout << "Step |   time   |  sub   |     dt     | T_max/K  | T_min/K  | CoupRes | Relax  | Mach  | IceTh/m | LE/m   | h_mod | Status" << std::endl;
        std::cout << "-----+----------+--------+------------+----------+----------+---------+--------+-------+---------+--------+-------+-------" << std::endl;
    } else {
        std::cout << "Step |   time   |  sub   |     dt     | T_max/K  | T_min/K  | CoupRes | Relax  | Mach  | Status" << std::endl;
        std::cout << "-----+----------+--------+------------+----------+----------+---------+--------+-------+-------" << std::endl;
    }

    savedPipeState_.resize(pipeSolver_->grid().nCells);
    savedSolidState_.resize(solidSolver_->numNodes());
    saveState();
}

Scalar AntiIcingSolver::getStabilityCriterion() {
    Scalar criterion = 1.0;

    const auto& T = solidSolver_->state().temperature;
    Scalar T_freeze = config_.femConfig.freezingPointTemp;
    Scalar transW = config_.femConfig.freezeTransitionWidth;

    Scalar minDistToFreeze = 1.0e10;
    for (Index i = 0; i < T.size(); ++i) {
        Scalar dist = std::abs(T(i) - T_freeze);
        if (dist < minDistToFreeze) {
            minDistToFreeze = dist;
        }
    }

    if (minDistToFreeze < 5.0 * transW) {
        criterion *= 0.3;
    } else if (minDistToFreeze < 10.0 * transW) {
        criterion *= 0.6;
    }

    const auto& Mach = pipeSolver_->state().Mach;
    Scalar maxMach = Mach.maxCoeff();
    if (maxMach > 0.8) {
        criterion *= 0.5;
    } else if (maxMach > 0.5) {
        criterion *= 0.8;
    }

    Scalar LWC = config_.flightCondition.LWC;
    if (LWC > 2.0) {
        criterion *= 0.4;
    } else if (LWC > 0.5) {
        criterion *= 0.7;
    }

    Scalar coupRes = coupler_->residual();
    if (coupRes > 1.0e-1) {
        criterion *= 0.2;
    } else if (coupRes > 1.0e-2) {
        criterion *= 0.5;
    }

    if (levelSetSolver_) {
        Scalar iceTh = levelSetSolver_->maxIceThickness();
        Scalar iceRate = 0.0;
        if (config_.iceRemeshThreshold > 0) {
            if (iceTh > 0.5 * config_.iceRemeshThreshold) criterion *= 0.5;
            if (iceTh > 0.8 * config_.iceRemeshThreshold) criterion *= 0.3;
        }

        Scalar LE = levelSetSolver_->leadingEdgeRadius();
        Scalar LE0 = config_.iceShapeConfig.leadingEdgeRadius;
        if (std::abs(LE - LE0) / std::max(LE0, 1.0e-10) > 0.2) {
            criterion *= 0.6;
        }
    }

    return clamp(criterion, 0.01, 1.0);
}

Scalar AntiIcingSolver::adaptiveTimeStep() {
    Scalar pipeDt = pipeSolver_->computeStableTimeStep();
    Scalar baseDt = clamp(pipeDt, 1.0e-20, config_.solverConfig.timeStep);
    baseDt = clamp(baseDt, 1.0e-12, 1.0);

    Scalar stability = getStabilityCriterion();
    Scalar adaptiveDt = baseDt * stability;

    Scalar prevDt = currentDt_ > 0 ? currentDt_ : adaptiveDt;
    Scalar maxIncrease = 1.2;
    Scalar maxDecrease = 0.3;

    Scalar ratio = adaptiveDt / std::max(prevDt, 1.0e-20);
    if (ratio > maxIncrease) {
        adaptiveDt = prevDt * maxIncrease;
    } else if (ratio < maxDecrease) {
        adaptiveDt = prevDt * maxDecrease;
    }

    adaptiveDt = clamp(adaptiveDt, 1.0e-14, config_.solverConfig.timeStep);

    dtHistory_[dtHistoryPtr_] = adaptiveDt;
    dtHistoryPtr_ = (dtHistoryPtr_ + 1) % 5;

    return adaptiveDt;
}

void AntiIcingSolver::saveState() {
    savedPipeState_ = pipeSolver_->state();
    savedSolidState_ = solidSolver_->state();
    savedTime_ = currentTime_;
}

void AntiIcingSolver::restoreState() {
    pipeSolver_->state() = savedPipeState_;
    solidSolver_->state() = savedSolidState_;
    currentTime_ = savedTime_;
}

bool AntiIcingSolver::validateSolution() {
    if (!isVectorValid(solidSolver_->state().temperature)) return false;
    if (!isVectorValid(pipeSolver_->state().T)) return false;
    if (!isVectorValid(pipeSolver_->state().rho)) return false;
    if (!isVectorValid(pipeSolver_->state().P)) return false;
    if (!isVectorValid(coupler_->interface().surfaceTemp)) return false;
    if (!isVectorValid(coupler_->interface().pipeWallTemp)) return false;

    Scalar Tmin = solidSolver_->state().temperature.minCoeff();
    Scalar Tmax = solidSolver_->state().temperature.maxCoeff();

    Scalar physMin = config_.femConfig.tempMinPhysical;
    Scalar physMax = config_.femConfig.tempMaxPhysical;

    if (Tmin < physMin - 10.0 || Tmax > physMax + 50.0) {
        return false;
    }

    Scalar coupRes = coupler_->residual();
    if (coupRes > 1.0e5) {
        return false;
    }

    return true;
}

void AntiIcingSolver::step() {
    saveState();
    Scalar targetDt = adaptiveTimeStep();
    currentDt_ = targetDt;

    Scalar remainingDt = targetDt;
    Index subStep = 0;
    Index maxSubSteps = 100;
    diag_.subSteps = 0;
    diag_.backtrackCount = 0;

    while (remainingDt > 1.0e-16 && subStep < maxSubSteps) {
        Scalar subDt = std::min(remainingDt, targetDt);
        if (subDt < 1.0e-16) break;

        bool subStepSuccess = false;
        Index localBacktrack = 0;
        Scalar currentSubDt = subDt;

        while (localBacktrack < 5 && !subStepSuccess) {
            try {
                auto externalBCs = bc::buildExternalBoundaryConditions(
                    solidGrid_, config_.flightCondition,
                    solidSolver_->state().temperature,
                    config_.couplingInterface.externalBoundaryId,
                    config_.convectionModel);

                if (levelSetSolver_) {
                    VectorX factors = levelSetSolver_->getBoundaryConvectionFactors(
                        static_cast<Index>(externalBCs.size()));
                    for (Index k = 0; k < static_cast<Index>(externalBCs.size()); ++k) {
                        if (externalBCs[k].type == fem::FEMBoundaryCondition::Type::CONVECTION ||
                            externalBCs[k].type == fem::FEMBoundaryCondition::Type::ICE_LATENT_HEAT) {
                            Scalar f = (k < factors.size()) ? factors(k) : 1.0;
                            f = clamp(f, 0.1, 10.0);
                            externalBCs[k].convectiveCoeff *= f;
                        }
                    }
                }

                auto internalBCs = bc::buildInternalBoundaryConditions(
                    solidGrid_,
                    coupler_->interface().pipeWallTemp,
                    coupler_->interface().pipeWallHeatFlux,
                    config_.couplingInterface.pipeBoundaryId,
                    config_.couplingInterface.pipeDiameter);

                std::vector<fem::FEMBoundaryCondition> allBCs;
                allBCs.reserve(externalBCs.size() + internalBCs.size());
                allBCs.insert(allBCs.end(), externalBCs.begin(), externalBCs.end());
                allBCs.insert(allBCs.end(), internalBCs.begin(), internalBCs.end());

                coupler_->couple(currentSubDt);

                if (!validateSolution()) {
                    throw std::runtime_error("Invalid solution after coupling");
                }

                subStepSuccess = true;

            } catch (...) {
                localBacktrack++;
                diag_.backtrackCount++;
                restoreState();
                currentSubDt *= 0.25;

                if (currentSubDt < 1.0e-16) break;
            }
        }

        if (!subStepSuccess) {
            restoreState();
            diag_.stepStatus = -1.0;
            break;
        }

        remainingDt -= currentSubDt;
        currentTime_ += currentSubDt;
        subStep++;
        totalSubSteps_++;
        saveState();
    }

    diag_.subSteps = subStep;
    diag_.currentDt = targetDt;
    diag_.stepStatus = validateSolution() ? 1.0 : 0.0;

    currentStep_++;

    if (levelSetSolver_ && (currentStep_ % config_.iceShapeUpdateInterval == 0 || currentStep_ == 1)) {
        try {
            auto accretion = collectAccretionData();
            Index nSurfPts = levelSetSolver_->numSurfacePoints();
            if (accretion.iceGrowthRate.size() > 0) {
                levelSetSolver_->updateIceShape(targetDt, accretion);
            }

            if (levelSetSolver_->needsRemeshing() ||
                levelSetSolver_->hasExceededThreshold(config_.iceRemeshThreshold)) {
                updateConvectionCoefficientsFromShape();
            }
        } catch (...) {
        }
    }

    computeDiagnostics();

    bool shouldOutput = (currentStep_ % 20 == 0) ||
                        (currentTime_ >= config_.solverConfig.totalTime) ||
                        (diag_.subSteps > 3) ||
                        (diag_.backtrackCount > 0);

    if (config_.enableDynamicIceShape) {
        shouldOutput = shouldOutput || (currentStep_ % config_.iceShapeUpdateInterval == 0);
    }

    if (shouldOutput) {
        outputResults(currentStep_, currentTime_);
    }
}

void AntiIcingSolver::run() {
    Scalar startTime = parallel::wallTime();

    while (currentTime_ < config_.solverConfig.totalTime &&
           currentStep_ < config_.solverConfig.maxIterations) {
        step();

        if (diagCallback_) {
            diagCallback_(diag_);
        }

        if (currentStep_ % 50 == 0) {
            saveState();
        }
    }

    Scalar elapsed = parallel::wallTime() - startTime;
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Simulation complete." << std::endl;
    std::cout << "  Total macro steps: " << currentStep_ << std::endl;
    std::cout << "  Total sub steps: " << totalSubSteps_ << std::endl;
    std::cout << "  Wall clock time: " << std::fixed << std::setprecision(3) << elapsed << " s" << std::endl;
    std::cout << "  Avg time per macro step: "
              << (elapsed / std::max(1.0, static_cast<Scalar>(currentStep_))) * 1.0e6
              << " us" << std::endl;
    std::cout << "========================================" << std::endl;
}

void AntiIcingSolver::computeDiagnostics() {
    diag_.currentTime = currentTime_;
    diag_.currentDt = currentDt_;
    diag_.timeStep = currentStep_;

    const auto& pipeState = pipeSolver_->state();
    const auto& solidState = solidSolver_->state();

    diag_.maxPipeMach = isVectorValid(pipeState.Mach) ? pipeState.Mach.maxCoeff() : 0.0;
    diag_.avgPipeTemperature = isVectorValid(pipeState.T) ? pipeState.T.mean() : 0.0;

    if (isVectorValid(solidState.temperature)) {
        diag_.maxTemperature = solidState.temperature.maxCoeff();
        diag_.minTemperature = solidState.temperature.minCoeff();
    } else {
        diag_.maxTemperature = 0.0;
        diag_.minTemperature = 0.0;
    }

    diag_.couplingResidual = coupler_->residual();
    diag_.couplingIterations = coupler_->iterations();
    diag_.couplingRelaxation = coupler_->currentRelaxation();
    diag_.energyImbalance = coupler_->computeEnergyImbalance();
    diag_.wallClockTime = parallel::wallTime();

    diag_.maxIceThickness = 0.0;
    diag_.totalIceVolume = 0.0;
    diag_.leadingEdgeRadius = config_.iceShapeConfig.leadingEdgeRadius;
    diag_.convectionModifier = 1.0;
    diag_.icedAreaFraction = 0.0;
    diag_.surfacePointCount = 0;

    if (levelSetSolver_) {
        diag_.maxIceThickness = levelSetSolver_->maxIceThickness();
        diag_.totalIceVolume = levelSetSolver_->totalIceVolume();
        diag_.leadingEdgeRadius = levelSetSolver_->leadingEdgeRadius();
        diag_.convectionModifier = levelSetSolver_->computeConvectiveCoeffModifier();
        diag_.surfacePointCount = levelSetSolver_->numSurfacePoints();

        Scalar iced = 0.0;
        Index n = levelSetSolver_->numSurfacePoints();
        Scalar r0 = config_.iceShapeConfig.leadingEdgeRadius;
        Scalar cx = config_.levelSetConfig.domainWidth * 0.35;
        Scalar cy = config_.levelSetConfig.domainHeight * 0.5;
        for (Index k = 0; k < n; ++k) {
            Vector2 p = levelSetSolver_->getSurfacePoint(k);
            Scalar dist = (p - Vector2(cx, cy)).norm();
            Scalar thickness = std::max(0.0, dist - r0);
            if (thickness > 1.0e-5) iced += 1.0;
        }
        diag_.icedAreaFraction = n > 0 ? iced / static_cast<Scalar>(n) : 0.0;
    }
}

void AntiIcingSolver::outputResults(Index step, Scalar time) {
    const char* statusStr = "OK";
    if (diag_.backtrackCount > 0) statusStr = "BT";
    if (diag_.subSteps > 2) statusStr = "SP";
    if (diag_.stepStatus < 0.0) statusStr = "FL";

    std::cout << std::scientific << std::setprecision(3);
    std::cout << std::setw(4) << step << " | "
              << std::setw(8) << time << " | "
              << std::setw(6) << diag_.subSteps << " | "
              << std::setw(10) << diag_.currentDt << " | "
              << std::fixed << std::setprecision(2)
              << std::setw(8) << diag_.maxTemperature << " | "
              << std::setw(8) << diag_.minTemperature << " | "
              << std::scientific << std::setprecision(2)
              << std::setw(7) << diag_.couplingResidual << " | "
              << std::fixed << std::setprecision(2)
              << std::setw(6) << diag_.couplingRelaxation << " | "
              << std::setw(5) << std::setprecision(3) << diag_.maxPipeMach << " | ";

    if (config_.enableDynamicIceShape && levelSetSolver_) {
        std::cout << std::scientific << std::setprecision(2)
                  << std::setw(7) << diag_.maxIceThickness << " | "
                  << std::scientific << std::setprecision(2)
                  << std::setw(6) << diag_.leadingEdgeRadius << " | "
                  << std::fixed << std::setprecision(2)
                  << std::setw(5) << diag_.convectionModifier << " | ";
    }

    std::cout << statusStr << std::endl;
}

levelset::IceAccretionData AntiIcingSolver::collectAccretionData() const {
    levelset::IceAccretionData data;

    Index nSurf = levelSetSolver_ ? levelSetSolver_->numSurfacePoints() : 0;

    data.iceGrowthRate.resize(nSurf);
    data.collectionEfficiency.resize(nSurf);
    data.surfaceHeatFlux.resize(nSurf);
    data.surfaceTemperature.resize(nSurf);
    data.surfaceCurvature.resize(nSurf);
    data.iceGrowthRate.setZero();
    data.collectionEfficiency.setZero();
    data.surfaceHeatFlux.setZero();
    data.surfaceTemperature.setZero();
    data.surfaceCurvature.setZero();

    if (!levelSetSolver_) return data;

    Index boundaryId = config_.couplingInterface.externalBoundaryId;
    VectorX boundaryFlux = solidSolver_->getBoundaryHeatFlux(boundaryId);
    VectorX boundaryTemp = solidSolver_->getBoundaryTemperature(boundaryId);

    Scalar LWC = config_.flightCondition.LWC;
    Scalar Vinf = config_.flightCondition.V_inf;
    Scalar Tfreeze = config_.femConfig.freezingPointTemp;

    Scalar rhoIce = 917.0;
    Scalar Lf = 334000.0;

    Scalar stagnationIdx = static_cast<Scalar>(levelSetSolver_->findStagnationPoint());

    for (Index k = 0; k < nSurf; ++k) {
        Scalar distFromStag = static_cast<Scalar>(k) - stagnationIdx;
        Scalar distAbs = std::abs(distFromStag) * 2.0 / std::max(Index(1), nSurf);
        Scalar beta = config_.iceShapeConfig.collectionEfficiencyPeak *
                      std::exp(-distAbs * distAbs * 3.0);
        beta = clamp(beta, 0.0, 1.0);
        data.collectionEfficiency(k) = beta;

        Scalar tempK = 270.0;
        if (boundaryTemp.size() > 0) {
            Scalar r = static_cast<Scalar>(k) / std::max(Index(1), nSurf - 1);
            Index idx = static_cast<Index>(clamp(r * static_cast<Scalar>(boundaryTemp.size() - 1),
                                                  0.0, static_cast<Scalar>(boundaryTemp.size() - 1)));
            tempK = boundaryTemp(idx);
        }
        data.surfaceTemperature(k) = tempK;

        Scalar q_wall = 0.0;
        if (boundaryFlux.size() > 0) {
            Scalar r = static_cast<Scalar>(k) / std::max(Index(1), nSurf - 1);
            Index idx = static_cast<Index>(clamp(r * static_cast<Scalar>(boundaryFlux.size() - 1),
                                                  0.0, static_cast<Scalar>(boundaryFlux.size() - 1)));
            q_wall = boundaryFlux(idx);
        }
        data.surfaceHeatFlux(k) = q_wall;

        Scalar kappa = levelSetSolver_->getSurfaceCurvature(k);
        data.surfaceCurvature(k) = kappa;

        Scalar LWC_kg = LWC * 1.0e-3;
        Scalar waterMassFlux = LWC_kg * Vinf * beta;

        Scalar deltaT_sub = std::max(0.0, Tfreeze - tempK);
        Scalar q_sens = 4217.0 * waterMassFlux * deltaT_sub;

        Scalar q_available = std::max(0.0, waterMassFlux * Lf - q_sens + q_wall * 0.1);

        Scalar dt_eff = config_.solverConfig.timeStep *
                        static_cast<Scalar>(config_.iceShapeUpdateInterval);
        Scalar growthRate = 0.0;

        if (tempK < Tfreeze - 0.1 && q_available > 0) {
            growthRate = q_available / (rhoIce * Lf + 1.0e-10);
            growthRate = clamp(growthRate, 0.0, 5.0e-3 / std::max(1.0e-12, dt_eff));
        }

        if (isNaN(growthRate)) growthRate = 0.0;
        data.iceGrowthRate(k) = growthRate;
    }

    Scalar totalRate = 0.0;
    for (Index k = 0; k < nSurf; ++k) totalRate += data.iceGrowthRate(k);
    data.totalIceVolume = totalRate * 0.01 * 0.02;

    Scalar maxThick = 0.0;
    Scalar icedFrac = 0.0;
    Scalar r0 = config_.iceShapeConfig.leadingEdgeRadius;
    Scalar cx = config_.levelSetConfig.domainWidth * 0.35;
    Scalar cy = config_.levelSetConfig.domainHeight * 0.5;
    for (Index k = 0; k < nSurf; ++k) {
        Vector2 p = levelSetSolver_->getSurfacePoint(k);
        Scalar dist = (p - Vector2(cx, cy)).norm();
        Scalar thickness = std::max(0.0, dist - r0);
        if (thickness > maxThick) maxThick = thickness;
        if (thickness > 1.0e-5) icedFrac += 1.0;
    }
    data.maxIceThickness = maxThick;
    data.icedAreaFraction = nSurf > 0 ? icedFrac / static_cast<Scalar>(nSurf) : 0.0;

    return data;
}

void AntiIcingSolver::updateIceShapeDynamics(Scalar dt) {
    if (!levelSetSolver_) return;
    auto accretion = collectAccretionData();
    levelSetSolver_->updateIceShape(dt, accretion);
}

void AntiIcingSolver::updateConvectionCoefficientsFromShape() {
    if (!levelSetSolver_) return;

    Scalar newR = levelSetSolver_->leadingEdgeRadius();
    Scalar r0 = config_.iceShapeConfig.leadingEdgeRadius;
    Scalar rFactor = newR / std::max(r0, 1.0e-10);
    rFactor = clamp(rFactor, 0.5, 3.0);

    config_.convectionModel.characteristicLength = rFactor * r0 * 2.0;
    config_.convectionModel.referenceRe *= std::sqrt(rFactor);
}

AntiIcingSystemConfig createDefaultConfig() {
    AntiIcingSystemConfig config;

    config.pipeConfig.diameter = 0.025;
    config.pipeConfig.length = 2.0;
    config.pipeConfig.wallRoughness = 0.0001;
    config.pipeConfig.nCells = 200;
    config.pipeConfig.CFL = 0.5;
    config.pipeConfig.relaxationFactor = 0.8;
    config.pipeConfig.maxInnerIter = 5000;
    config.pipeConfig.innerTol = 1.0e-6;

    config.pipeBC.inletTotalPressure = 350000.0;
    config.pipeBC.inletTotalTemperature = 473.15;
    config.pipeBC.outletStaticPressure = 101325.0;
    config.pipeBC.inletMassFlowRate = 0.05;
    config.pipeBC.useInletMassFlow = true;

    config.femConfig.theta = 0.667;
    config.femConfig.tolerance = 1.0e-8;
    config.femConfig.maxIter = 1000;
    config.femConfig.useLumpedMass = true;
    config.femConfig.useParallel = true;

    config.femConfig.numericalDampingCoeff = 1.0e-5;
    config.femConfig.enableNumericalDamping = true;
    config.femConfig.tempMinPhysical = 150.0;
    config.femConfig.tempMaxPhysical = 800.0;
    config.femConfig.freezeTransitionWidth = 3.0;
    config.femConfig.freezingPointTemp = 273.15;
    config.femConfig.maxNewtonIter = 10;
    config.femConfig.newtonTolerance = 1.0e-6;
    config.femConfig.enableLineSearch = true;

    config.flightCondition.V_inf = 130.0;
    config.flightCondition.T_inf = 248.15;
    config.flightCondition.P_inf = 30000.0;
    config.flightCondition.LWC = 3.0;
    config.flightCondition.MVD = 40.0e-6;
    config.flightCondition.AoA = 5.0;
    config.flightCondition.altitude = 9000.0;
    config.flightCondition.mach = 0.78;

    config.convectionModel.type = bc::ConvectionModel::Type::CYLINDER;
    config.convectionModel.characteristicLength = 0.3;
    config.convectionModel.referenceRe = 1.0e6;

    config.couplingConfig.relaxationFactor = 0.3;
    config.couplingConfig.maxIterations = 50;
    config.couplingConfig.tolerance = 1.0e-4;
    config.couplingConfig.useAitken = true;
    config.couplingConfig.enforceEnergyConservation = true;

    config.couplingConfig.minRelaxation = 0.01;
    config.couplingConfig.maxRelaxation = 0.7;
    config.couplingConfig.residualGrowthThreshold = 1.5;
    config.couplingConfig.divergenceThreshold = 1.0e2;
    config.couplingConfig.relaxationCutFactor = 0.3;
    config.couplingConfig.relaxationRecoveryFactor = 1.1;
    config.couplingConfig.residualHistorySize = 30;

    config.couplingConfig.tempMinPhysical = 150.0;
    config.couplingConfig.tempMaxPhysical = 800.0;
    config.couplingConfig.heatFluxMaxPhysical = 1.0e8;

    config.couplingConfig.enableBacktracking = true;
    config.couplingConfig.maxBacktrackSteps = 8;
    config.couplingConfig.backtrackFactor = 0.25;

    config.couplingConfig.newtonDampingCoeff = 1.0e-6;
    config.couplingConfig.enableNewtonDamping = true;

    config.couplingInterface.pipeBoundaryId = 4;
    config.couplingInterface.externalBoundaryId = 5;
    config.couplingInterface.internalConvCoeff = 100.0;
    config.couplingInterface.pipeDiameter = config.pipeConfig.diameter;

    config.levelSetConfig = levelset::createDefaultLevelSetConfig();
    config.iceShapeConfig = levelset::createDefaultIceShapeConfig();

    config.enableDynamicIceShape = false;
    config.iceShapeUpdateInterval = 10;
    config.iceRemeshThreshold = 5.0e-3;

    config.solverConfig.timeStep = 1.0e-6;
    config.solverConfig.totalTime = 0.002;
    config.solverConfig.maxIterations = 2000;
    config.solverConfig.tolerance = 1.0e-8;
    config.solverConfig.numThreads = 8;
    config.solverConfig.useParallel = true;
    config.solverConfig.couplingRelax = 0.3;
    config.solverConfig.couplingMaxIter = 50;

    return config;
}

std::shared_ptr<AntiIcingSolver> createAntiIcingSolver(const AntiIcingSystemConfig& config) {
    auto solver = std::make_shared<AntiIcingSolver>(config);
    solver->initialize();
    return solver;
}

}
