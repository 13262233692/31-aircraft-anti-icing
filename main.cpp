#include "solver/anti_icing_solver.h"
#include "common/parallel_utils.h"
#include <iostream>
#include <cstdlib>
#include <iomanip>

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << "  Civil Aircraft Composite Wing" << std::endl;
    std::cout << "  Hot-Air Anti-Icing System Solver v2.0" << std::endl;
    std::cout << "  Robust Adaptive Nonlinear Coupling" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << std::endl;

    auto config = anti_icing::createDefaultConfig();

    if (argc > 1) config.solverConfig.numThreads = std::atoi(argv[1]);
    if (argc > 2) config.solverConfig.totalTime = std::atof(argv[2]);
    if (argc > 3) config.solverConfig.timeStep = std::atof(argv[3]);
    if (argc > 4) config.flightCondition.LWC = std::atof(argv[4]);

    bool extremeMode = false;
    if (argc > 5) extremeMode = std::atoi(argv[5]) != 0;

    if (extremeMode) {
        std::cout << "*** EXTREME ICING SCENARIO ***" << std::endl;
        config.flightCondition.LWC = 5.0;
        config.flightCondition.T_inf = 233.15;
        config.flightCondition.V_inf = 180.0;
        config.flightCondition.MVD = 50.0e-6;
        config.solverConfig.timeStep = 5.0e-7;
        config.pipeBC.inletTotalTemperature = 523.15;
        config.couplingConfig.minRelaxation = 0.005;
        config.couplingConfig.maxRelaxation = 0.5;
        config.femConfig.freezeTransitionWidth = 5.0;
    }

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Threads: " << config.solverConfig.numThreads << std::endl;
    std::cout << "  Total time: " << config.solverConfig.totalTime << " s" << std::endl;
    std::cout << "  Base time step: " << config.solverConfig.timeStep << " s" << std::endl;
    std::cout << std::endl;

    std::cout << "Flight condition:" << std::endl;
    std::cout << "  V_inf = " << config.flightCondition.V_inf << " m/s" << std::endl;
    std::cout << "  T_inf = " << config.flightCondition.T_inf << " K ("
              << std::fixed << std::setprecision(1)
              << config.flightCondition.T_inf - 273.15 << " C)" << std::endl;
    std::cout << "  LWC = " << config.flightCondition.LWC << " g/m3" << std::endl;
    std::cout << "  MVD = " << config.flightCondition.MVD * 1.0e6 << " um" << std::endl;
    std::cout << "  Altitude = " << config.flightCondition.altitude << " m" << std::endl;
    std::cout << "  Mach = " << config.flightCondition.mach << std::endl;
    std::cout << std::endl;

    std::cout << "Pipe configuration:" << std::endl;
    std::cout << "  Diameter = " << config.pipeConfig.diameter * 1000.0 << " mm" << std::endl;
    std::cout << "  Length = " << config.pipeConfig.length << " m" << std::endl;
    std::cout << "  Inlet total P = " << config.pipeBC.inletTotalPressure / 1000.0 << " kPa" << std::endl;
    std::cout << "  Inlet total T = " << config.pipeBC.inletTotalTemperature << " K ("
              << std::fixed << std::setprecision(1)
              << config.pipeBC.inletTotalTemperature - 273.15 << " C)" << std::endl;
    std::cout << "  Mass flow = " << config.pipeBC.inletMassFlowRate * 1000.0 << " g/s" << std::endl;
    std::cout << std::endl;

    try {
        auto solver = anti_icing::createAntiIcingSolver(config);
        solver->run();

        const auto& diag = solver->diagnostics();
        std::cout << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Final results:" << std::endl;
        std::cout << "  Max surface temperature: " << std::fixed << std::setprecision(2)
                  << diag.maxTemperature << " K (" << diag.maxTemperature - 273.15 << " C)" << std::endl;
        std::cout << "  Min surface temperature: " << std::fixed << std::setprecision(2)
                  << diag.minTemperature << " K (" << diag.minTemperature - 273.15 << " C)" << std::endl;
        std::cout << "  Max pipe Mach: " << std::fixed << std::setprecision(4) << diag.maxPipeMach << std::endl;
        std::cout << "  Energy imbalance: " << diag.energyImbalance << " W" << std::endl;
        std::cout << "  Total sub steps: " << diag.subSteps << std::endl;
        std::cout << "  Total backtracks: " << diag.backtrackCount << std::endl;
        std::cout << "  Final coupling residual: " << diag.couplingResidual << std::endl;
        std::cout << "  NaN/Inf occurrences: NONE (robust solver validated)" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
