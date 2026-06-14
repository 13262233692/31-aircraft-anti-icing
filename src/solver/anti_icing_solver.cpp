#include "solver/anti_icing_solver.h"
#include "common/parallel_utils.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace anti_icing {

AntiIcingSolver::AntiIcingSolver(const AntiIcingSystemConfig& config)
    : config_(config), currentTime_(0.0), currentStep_(0) {
    diag_ = {};
}

void AntiIcingSolver::initialize() {
    parallel::setNumThreads(config_.solverConfig.numThreads);

    std::cout << "========================================" << std::endl;
    std::cout << "  Aircraft Anti-Icing Solver" << std::endl;
    std::cout << "  Multi-Physics Nonlinear Coupling Kernel" << std::endl;
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
    std::cout << "  Anisotropic conductivity:" << std::endl;
    std::cout << "    k_xx=" << composite.conductivity(0, 0)
              << " k_yy=" << composite.conductivity(1, 1)
              << " k_zz=" << composite.conductivity(2, 2) << std::endl;

    std::cout << "Initializing staggered coupler..." << std::endl;

    config_.couplingInterface.pipeDiameter = config_.pipeConfig.diameter;
    coupler_ = coupling::createStaggeredCoupler(
        pipeSolver_, solidSolver_,
        config_.couplingInterface, config_.couplingConfig);

    std::cout << "  Coupling relaxation: " << config_.couplingConfig.relaxationFactor << std::endl;
    std::cout << "  Max coupling iterations: " << config_.couplingConfig.maxIterations << std::endl;
    std::cout << "  Aitken acceleration: " << (config_.couplingConfig.useAitken ? "ON" : "OFF") << std::endl;

    std::cout << "Solver initialization complete." << std::endl;
    std::cout << "  Time step: " << config_.solverConfig.timeStep << " s" << std::endl;
    std::cout << "  Total time: " << config_.solverConfig.totalTime << " s" << std::endl;
    std::cout << "  Threads: " << config_.solverConfig.numThreads << std::endl;
    std::cout << "========================================" << std::endl;
}

Scalar AntiIcingSolver::adaptiveTimeStep() {
    Scalar pipeDt = pipeSolver_->computeStableTimeStep();
    Scalar dt = std::min(pipeDt, config_.solverConfig.timeStep);
    return std::max(dt, 1.0e-10);
}

void AntiIcingSolver::step() {
    Scalar dt = adaptiveTimeStep();

    auto externalBCs = bc::buildExternalBoundaryConditions(
        solidGrid_, config_.flightCondition,
        solidSolver_->state().temperature,
        config_.couplingInterface.externalBoundaryId,
        config_.convectionModel);

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

    coupler_->couple(dt);

    currentTime_ += dt;
    currentStep_++;

    computeDiagnostics();
}

void AntiIcingSolver::run() {
    Scalar startTime = parallel::wallTime();

    while (currentTime_ < config_.solverConfig.totalTime &&
           currentStep_ < config_.solverConfig.maxIterations) {
        step();

        if (currentStep_ % 100 == 0 || currentTime_ >= config_.solverConfig.totalTime) {
            outputResults(currentStep_, currentTime_);
        }

        if (diagCallback_) {
            diagCallback_(diag_);
        }
    }

    Scalar elapsed = parallel::wallTime() - startTime;
    std::cout << "\nSimulation complete." << std::endl;
    std::cout << "  Total steps: " << currentStep_ << std::endl;
    std::cout << "  Wall clock time: " << std::fixed << std::setprecision(3) << elapsed << " s" << std::endl;
    std::cout << "  Time per step: " << (elapsed / std::max(1.0, static_cast<Scalar>(currentStep_))) * 1.0e6
              << " us" << std::endl;
}

void AntiIcingSolver::computeDiagnostics() {
    diag_.currentTime = currentTime_;
    diag_.timeStep = currentStep_;

    const auto& pipeState = pipeSolver_->state();
    const auto& solidState = solidSolver_->state();

    diag_.maxPipeMach = pipeState.Mach.maxCoeff();
    diag_.avgPipeTemperature = pipeState.T.mean();

    diag_.maxTemperature = solidState.temperature.maxCoeff();
    diag_.minTemperature = solidState.temperature.minCoeff();

    diag_.couplingResidual = coupler_->residual();
    diag_.energyImbalance = coupler_->computeEnergyImbalance();
    diag_.wallClockTime = parallel::wallTime();
}

void AntiIcingSolver::outputResults(Index step, Scalar time) {
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "Step=" << std::setw(6) << step
              << " t=" << std::setw(10) << time
              << " T_max=" << std::setw(10) << diag_.maxTemperature
              << " T_min=" << std::setw(10) << diag_.minTemperature
              << " Mach_max=" << std::setw(10) << diag_.maxPipeMach
              << " coupling_res=" << std::setw(10) << diag_.couplingResidual
              << std::endl;
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

    config.flightCondition.V_inf = 130.0;
    config.flightCondition.T_inf = 248.15;
    config.flightCondition.P_inf = 30000.0;
    config.flightCondition.LWC = 0.45;
    config.flightCondition.MVD = 20.0e-6;
    config.flightCondition.AoA = 3.0;
    config.flightCondition.altitude = 9000.0;
    config.flightCondition.mach = 0.78;

    config.convectionModel.type = bc::ConvectionModel::Type::CYLINDER;
    config.convectionModel.characteristicLength = 0.3;
    config.convectionModel.referenceRe = 1.0e6;

    config.couplingConfig.relaxationFactor = 0.5;
    config.couplingConfig.maxIterations = 20;
    config.couplingConfig.tolerance = 1.0e-4;
    config.couplingConfig.useAitken = true;
    config.couplingConfig.enforceEnergyConservation = true;

    config.couplingInterface.pipeBoundaryId = 4;
    config.couplingInterface.externalBoundaryId = 5;
    config.couplingInterface.internalConvCoeff = 100.0;
    config.couplingInterface.pipeDiameter = config.pipeConfig.diameter;

    config.solverConfig.timeStep = 1.0e-6;
    config.solverConfig.totalTime = 0.001;
    config.solverConfig.maxIterations = 10000;
    config.solverConfig.tolerance = 1.0e-8;
    config.solverConfig.numThreads = 8;
    config.solverConfig.useParallel = true;
    config.solverConfig.couplingRelax = 0.5;
    config.solverConfig.couplingMaxIter = 20;

    return config;
}

std::shared_ptr<AntiIcingSolver> createAntiIcingSolver(const AntiIcingSystemConfig& config) {
    auto solver = std::make_shared<AntiIcingSolver>(config);
    solver->initialize();
    return solver;
}

}
