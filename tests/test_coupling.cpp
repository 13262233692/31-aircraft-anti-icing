#include <gtest/gtest.h>
#include "coupling/staggered_coupler.h"
#include "fvm/pipe_flow_solver.h"
#include "fem/heat_conduction_solver.h"
#include "common/grid.h"

using namespace anti_icing;
using namespace anti_icing::coupling;
using namespace anti_icing::fvm;
using namespace anti_icing::fem;

TEST(CouplingTest, CouplerInitialization) {
    PipeConfig pipeConfig;
    pipeConfig.diameter = 0.025;
    pipeConfig.length = 2.0;
    pipeConfig.nCells = 50;
    pipeConfig.CFL = 0.5;
    pipeConfig.relaxationFactor = 0.5;
    pipeConfig.maxInnerIter = 100;
    pipeConfig.innerTol = 1.0e-6;

    PipeBoundaryConditions pipeBC;
    pipeBC.inletTotalPressure = 350000.0;
    pipeBC.inletTotalTemperature = 473.15;
    pipeBC.outletStaticPressure = 101325.0;
    pipeBC.inletMassFlowRate = 0.05;
    pipeBC.useInletMassFlow = true;

    auto pipeSolver = createPipeFlowSolver(pipeConfig, pipeBC);

    auto grid = createStructuredTetGrid(0.1, 0.1, 0.01, 4, 4, 2);

    MaterialProperties mat;
    mat.conductivity = Matrix3::Identity() * 5.0;
    mat.density = 1580.0;
    mat.specificHeat = 1200.0;

    FEMConfig femConfig;
    femConfig.theta = 1.0;
    femConfig.tolerance = 1.0e-8;
    femConfig.maxIter = 100;
    femConfig.useLumpedMass = true;
    femConfig.useParallel = false;

    auto solidSolver = createHeatConductionSolver(grid, {mat}, femConfig);

    CouplingInterface interface;
    interface.pipeBoundaryId = 4;
    interface.externalBoundaryId = 5;
    interface.internalConvCoeff = 100.0;
    interface.pipeDiameter = 0.025;

    CouplingConfig couplingConfig;
    couplingConfig.relaxationFactor = 0.5;
    couplingConfig.maxIterations = 10;
    couplingConfig.tolerance = 1.0e-3;
    couplingConfig.useAitken = true;
    couplingConfig.enforceEnergyConservation = true;

    auto coupler = createStaggeredCoupler(pipeSolver, solidSolver, interface, couplingConfig);

    EXPECT_GT(coupler->interface().pipeWallTemp.size(), 0);
    EXPECT_GT(coupler->interface().surfaceTemp.size(), 0);
}
