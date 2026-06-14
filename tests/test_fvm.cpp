#include <gtest/gtest.h>
#include "fvm/pipe_flow_solver.h"
#include "common/types.h"

using namespace anti_icing;
using namespace anti_icing::fvm;

TEST(FVMTest, PipeFlowInitialization) {
    PipeConfig config;
    config.diameter = 0.025;
    config.length = 2.0;
    config.wallRoughness = 0.0001;
    config.nCells = 100;
    config.CFL = 0.5;
    config.relaxationFactor = 0.8;
    config.maxInnerIter = 1000;
    config.innerTol = 1.0e-6;

    PipeBoundaryConditions bc;
    bc.inletTotalPressure = 350000.0;
    bc.inletTotalTemperature = 473.15;
    bc.outletStaticPressure = 101325.0;
    bc.inletMassFlowRate = 0.05;
    bc.useInletMassFlow = true;

    auto solver = createPipeFlowSolver(config, bc);

    EXPECT_GT(solver->state().rho.size(), 0);
    EXPECT_GT(solver->state().P(0), 0.0);
    EXPECT_GT(solver->state().T(0), 0.0);
}

TEST(FVMTest, StableTimeStep) {
    PipeConfig config;
    config.diameter = 0.025;
    config.length = 2.0;
    config.nCells = 100;
    config.CFL = 0.5;
    config.relaxationFactor = 0.8;
    config.maxInnerIter = 100;
    config.innerTol = 1.0e-6;

    PipeBoundaryConditions bc;
    bc.inletTotalPressure = 350000.0;
    bc.inletTotalTemperature = 473.15;
    bc.outletStaticPressure = 101325.0;
    bc.inletMassFlowRate = 0.05;
    bc.useInletMassFlow = true;

    auto solver = createPipeFlowSolver(config, bc);
    Scalar dt = solver->computeStableTimeStep();

    EXPECT_GT(dt, 0.0);
    EXPECT_LT(dt, 1.0);
}

TEST(FVMTest, SingleStep) {
    PipeConfig config;
    config.diameter = 0.025;
    config.length = 2.0;
    config.nCells = 50;
    config.CFL = 0.5;
    config.relaxationFactor = 0.5;
    config.maxInnerIter = 100;
    config.innerTol = 1.0e-6;

    PipeBoundaryConditions bc;
    bc.inletTotalPressure = 350000.0;
    bc.inletTotalTemperature = 473.15;
    bc.outletStaticPressure = 101325.0;
    bc.inletMassFlowRate = 0.05;
    bc.useInletMassFlow = true;

    auto solver = createPipeFlowSolver(config, bc);
    Scalar dt = solver->computeStableTimeStep();
    solver->advance(dt);

    EXPECT_GT(solver->state().rho.minCoeff(), 0.0);
}
