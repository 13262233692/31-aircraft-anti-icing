#pragma once

#include "common/types.h"
#include "fvm/pipe_flow_solver.h"
#include "fem/heat_conduction_solver.h"
#include "bc/boundary_conditions.h"
#include "coupling/staggered_coupler.h"
#include "levelset/level_set_solver.h"
#include <memory>
#include <functional>
#include <string>

namespace anti_icing {

struct AntiIcingSystemConfig {
    fvm::PipeConfig pipeConfig;
    fvm::PipeBoundaryConditions pipeBC;
    fem::FEMConfig femConfig;
    bc::FlightCondition flightCondition;
    bc::ConvectionModel convectionModel;
    coupling::CouplingConfig couplingConfig;
    coupling::CouplingInterface couplingInterface;

    levelset::LevelSetConfig levelSetConfig;
    levelset::IceShapeConfig iceShapeConfig;

    bool enableDynamicIceShape;
    Index iceShapeUpdateInterval;
    Scalar iceRemeshThreshold;

    SolverConfig solverConfig;
};

struct SolverDiagnostics {
    Scalar currentTime;
    Scalar currentDt;
    Index timeStep;
    Index subSteps;
    Scalar pipeResidual;
    Scalar solidResidual;
    Scalar couplingResidual;
    Scalar maxTemperature;
    Scalar minTemperature;
    Scalar avgPipeTemperature;
    Scalar maxPipeMach;
    Scalar energyImbalance;
    Scalar wallClockTime;
    Index couplingIterations;
    Scalar couplingRelaxation;
    Index backtrackCount;
    Scalar stepStatus;

    Scalar maxIceThickness;
    Scalar totalIceVolume;
    Scalar leadingEdgeRadius;
    Scalar convectionModifier;
    Scalar icedAreaFraction;
    Index surfacePointCount;
};

class AntiIcingSolver {
public:
    explicit AntiIcingSolver(const AntiIcingSystemConfig& config);

    void initialize();
    void run();
    void step();

    void setTimeStep(Scalar dt) { config_.solverConfig.timeStep = dt; }
    void setTotalTime(Scalar t) { config_.solverConfig.totalTime = t; }

    const SolverDiagnostics& diagnostics() const { return diag_; }
    const fvm::PipeFlowSolver& pipeSolver() const { return *pipeSolver_; }
    const fem::HeatConductionSolver& solidSolver() const { return *solidSolver_; }
    const coupling::StaggeredCoupler& coupler() const { return *coupler_; }
    const levelset::LevelSetSolver& levelSetSolver() const { return *levelSetSolver_; }
    bool hasLevelSet() const { return levelSetSolver_ != nullptr; }

    using OutputCallback = std::function<void(Index, Scalar, const SolverDiagnostics&)>;
    void setOutputCallback(OutputCallback cb) { outputCallback_ = cb; }

    using DiagnosticCallback = std::function<void(const SolverDiagnostics&)>;
    void setDiagnosticCallback(DiagnosticCallback cb) { diagCallback_ = cb; }

private:
    void computeDiagnostics();
    void outputResults(Index step, Scalar time);
    Scalar adaptiveTimeStep();
    bool validateSolution();
    void restoreState();
    void saveState();
    Scalar getStabilityCriterion();

    void updateIceShapeDynamics(Scalar dt);
    void updateConvectionCoefficientsFromShape();
    levelset::IceAccretionData collectAccretionData() const;

    AntiIcingSystemConfig config_;
    std::shared_ptr<fvm::PipeFlowSolver> pipeSolver_;
    std::shared_ptr<fem::HeatConductionSolver> solidSolver_;
    std::shared_ptr<coupling::StaggeredCoupler> coupler_;
    std::shared_ptr<levelset::LevelSetSolver> levelSetSolver_;
    std::shared_ptr<Grid3D> solidGrid_;
    std::vector<MaterialProperties> materials_;
    SolverDiagnostics diag_;

    OutputCallback outputCallback_;
    DiagnosticCallback diagCallback_;

    Scalar currentTime_;
    Scalar currentDt_;
    Index currentStep_;
    Index totalSubSteps_;
    Scalar dtHistory_[5];
    Index dtHistoryPtr_;

    fvm::PipeFlowState savedPipeState_;
    fem::FEMState savedSolidState_;
    Scalar savedTime_;

    VectorX savedIceThickness_;
    VectorX baseConvectionFactors_;
};

std::shared_ptr<AntiIcingSolver> createAntiIcingSolver(const AntiIcingSystemConfig& config);

AntiIcingSystemConfig createDefaultConfig();

}
