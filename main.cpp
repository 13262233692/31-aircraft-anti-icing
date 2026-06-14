#include "solver/anti_icing_solver.h"
#include "common/parallel_utils.h"
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    std::cout << "==============================================" << std::endl;
    std::cout << "  Civil Aircraft Composite Wing" << std::endl;
    std::cout << "  Hot-Air Anti-Icing System Solver" << std::endl;
    std::cout << "  Multi-Physics Nonlinear Coupling Kernel" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << std::endl;

    auto config = anti_icing::createDefaultConfig();

    if (argc > 1) config.solverConfig.numThreads = std::atoi(argv[1]);
    if (argc > 2) config.solverConfig.totalTime = std::atof(argv[2]);
    if (argc > 3) config.solverConfig.timeStep = std::atof(argv[3]);

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Threads: " << config.solverConfig.numThreads << std::endl;
    std::cout << "  Total time: " << config.solverConfig.totalTime << " s" << std::endl;
    std::cout << "  Time step: " << config.solverConfig.timeStep << " s" << std::endl;
    std::cout << std::endl;

    std::cout << "Flight condition:" << std::endl;
    std::cout << "  V_inf = " << config.flightCondition.V_inf << " m/s" << std::endl;
    std::cout << "  T_inf = " << config.flightCondition.T_inf << " K ("
              << config.flightCondition.T_inf - 273.15 << " C)" << std::endl;
    std::cout << "  LWC = " << config.flightCondition.LWC << " g/m3" << std::endl;
    std::cout << "  MVD = " << config.flightCondition.MVD * 1.0e6 << " um" << std::endl;
    std::cout << "  Altitude = " << config.flightCondition.altitude << " m" << std::endl;
    std::cout << std::endl;

    std::cout << "Pipe configuration:" << std::endl;
    std::cout << "  Diameter = " << config.pipeConfig.diameter * 1000.0 << " mm" << std::endl;
    std::cout << "  Length = " << config.pipeConfig.length << " m" << std::endl;
    std::cout << "  Inlet total P = " << config.pipeBC.inletTotalPressure / 1000.0 << " kPa" << std::endl;
    std::cout << "  Inlet total T = " << config.pipeBC.inletTotalTemperature << " K ("
              << config.pipeBC.inletTotalTemperature - 273.15 << " C)" << std::endl;
    std::cout << "  Mass flow = " << config.pipeBC.inletMassFlowRate * 1000.0 << " g/s" << std::endl;
    std::cout << std::endl;

    try {
        auto solver = anti_icing::createAntiIcingSolver(config);
        solver->run();

        const auto& diag = solver->diagnostics();
        std::cout << std::endl;
        std::cout << "Final results:" << std::endl;
        std::cout << "  Max surface temperature: " << diag.maxTemperature << " K" << std::endl;
        std::cout << "  Min surface temperature: " << diag.minTemperature << " K" << std::endl;
        std::cout << "  Max pipe Mach: " << diag.maxPipeMach << std::endl;
        std::cout << "  Energy imbalance: " << diag.energyImbalance << " W" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
