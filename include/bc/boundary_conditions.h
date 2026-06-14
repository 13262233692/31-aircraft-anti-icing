#pragma once

#include "common/types.h"
#include "common/grid.h"
#include "fem/heat_conduction_solver.h"
#include <vector>

namespace anti_icing {
namespace bc {

struct FlightCondition {
    Scalar V_inf;
    Scalar T_inf;
    Scalar P_inf;
    Scalar LWC;
    Scalar MVD;
    Scalar AoA;
    Scalar altitude;
    Scalar mach;
};

struct ConvectionModel {
    enum class Type {
        FLAT_PLATE,
        CYLINDER,
        NUSSELT_CORRELATION
    };

    Type type;
    Scalar characteristicLength;
    Scalar referenceRe;
};

Scalar computeReynoldsNumber(Scalar V, Scalar L, Scalar T, Scalar P);

Scalar computePrandtlNumber(Scalar T);

Scalar computeFlatPlateNusselt(Scalar Re, Scalar Pr, Scalar x, bool turbulent = true);

Scalar computeCylinderNusselt(Scalar Re, Scalar Pr, Scalar D);

Scalar computeConvectiveHeatTransferCoeff(Scalar Re, Scalar Pr, Scalar k, Scalar L,
                                          ConvectionModel::Type type);

Scalar computeCollectionEfficiency(Scalar V, Scalar D, Scalar LWC, Scalar MVD,
                                   Scalar theta, Scalar beta0);

Scalar computeIceGrowthRate(Scalar LWC, Scalar V, Scalar beta, Scalar T_surface,
                            Scalar T_freeze = 273.15);

Scalar computeLatentHeatFlux(Scalar m_impinge, Scalar T_surface, Scalar T_freeze = 273.15);

std::vector<fem::FEMBoundaryCondition> buildExternalBoundaryConditions(
    std::shared_ptr<Grid3D> grid,
    const FlightCondition& flight,
    const VectorX& surfaceTemperature,
    Index boundaryId,
    const ConvectionModel& convModel);

std::vector<fem::FEMBoundaryCondition> buildInternalBoundaryConditions(
    std::shared_ptr<Grid3D> grid,
    const VectorX& pipeWallTemperature,
    const VectorX& pipeHeatFlux,
    Index boundaryId,
    Scalar pipeDiameter);

}
}
