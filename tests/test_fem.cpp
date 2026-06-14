#include <gtest/gtest.h>
#include "fem/heat_conduction_solver.h"
#include "common/grid.h"

using namespace anti_icing;
using namespace anti_icing::fem;

TEST(FEMTest, HeatConductionInitialization) {
    auto grid = createStructuredTetGrid(0.1, 0.1, 0.01, 5, 5, 2);

    MaterialProperties mat;
    mat.conductivity = Matrix3::Identity() * 5.0;
    mat.density = 1580.0;
    mat.specificHeat = 1200.0;

    std::vector<MaterialProperties> materials = {mat};

    FEMConfig config;
    config.theta = 1.0;
    config.tolerance = 1.0e-8;
    config.maxIter = 100;
    config.useLumpedMass = true;
    config.useParallel = false;

    auto solver = createHeatConductionSolver(grid, materials, config);

    EXPECT_EQ(solver->temperature().size(), grid->nNodes);
    EXPECT_NEAR(solver->temperature()(0), 288.15, 1.0e-10);
}

TEST(FEMTest, SteadyStateHeatConduction) {
    auto grid = createStructuredTetGrid(0.1, 0.1, 0.01, 4, 4, 2);

    MaterialProperties mat;
    mat.conductivity = Matrix3::Identity() * 10.0;
    mat.density = 1000.0;
    mat.specificHeat = 500.0;

    std::vector<MaterialProperties> materials = {mat};

    FEMConfig config;
    config.theta = 1.0;
    config.tolerance = 1.0e-10;
    config.maxIter = 100;
    config.useLumpedMass = true;
    config.useParallel = false;

    auto solver = createHeatConductionSolver(grid, materials, config);

    std::vector<FEMBoundaryCondition> bcs;
    for (Index i = 0; i < grid->nNodes; ++i) {
        if (grid->nodes[i](0) < 1.0e-10) {
            FEMBoundaryCondition bc;
            bc.type = FEMBoundaryCondition::Type::DIRICHLET;
            bc.nodeId = i;
            bc.value = 400.0;
            bcs.push_back(bc);
        }
        if (grid->nodes[i](0) > 0.1 - 1.0e-10) {
            FEMBoundaryCondition bc;
            bc.type = FEMBoundaryCondition::Type::DIRICHLET;
            bc.nodeId = i;
            bc.value = 300.0;
            bcs.push_back(bc);
        }
    }

    solver->advance(1.0, bcs);

    EXPECT_GT(solver->temperature().maxCoeff(), 300.0);
    EXPECT_LT(solver->temperature().minCoeff(), 400.0);
}
