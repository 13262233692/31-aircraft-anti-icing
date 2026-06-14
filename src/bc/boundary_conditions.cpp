#include "bc/boundary_conditions.h"
#include <cmath>
#include <algorithm>
#include <set>

namespace anti_icing {
namespace bc {

Scalar computeReynoldsNumber(Scalar V, Scalar L, Scalar T, Scalar P) {
    Scalar rho = computeAirDensity(P, T);
    Scalar mu = computeAirViscosity(T);
    return rho * V * L / mu;
}

Scalar computePrandtlNumber(Scalar T) {
    Scalar mu = computeAirViscosity(T);
    Scalar k = 0.024 * std::pow(T / T_AIR_REF, 0.8);
    return CP_AIR * mu / k;
}

Scalar computeFlatPlateNusselt(Scalar Re, Scalar Pr, Scalar x, bool turbulent) {
    if (turbulent && Re > 5.0e5) {
        return 0.0296 * std::pow(Re, 4.0 / 5.0) * std::pow(Pr, 1.0 / 3.0);
    } else {
        return 0.332 * std::sqrt(Re) * std::pow(Pr, 1.0 / 3.0);
    }
}

Scalar computeCylinderNusselt(Scalar Re, Scalar Pr, Scalar D) {
    if (Re < 4.0) {
        return 0.989 * std::pow(Re, 0.330) * std::pow(Pr, 1.0 / 3.0);
    } else if (Re < 40.0) {
        return 0.911 * std::pow(Re, 0.385) * std::pow(Pr, 1.0 / 3.0);
    } else if (Re < 4000.0) {
        return 0.683 * std::pow(Re, 0.466) * std::pow(Pr, 1.0 / 3.0);
    } else if (Re < 40000.0) {
        return 0.193 * std::pow(Re, 0.618) * std::pow(Pr, 1.0 / 3.0);
    } else {
        return 0.0266 * std::pow(Re, 0.805) * std::pow(Pr, 1.0 / 3.0);
    }
}

Scalar computeConvectiveHeatTransferCoeff(Scalar Re, Scalar Pr, Scalar k, Scalar L,
                                          ConvectionModel::Type type) {
    Scalar Nu;
    switch (type) {
    case ConvectionModel::Type::FLAT_PLATE:
        Nu = computeFlatPlateNusselt(Re, Pr, L, Re > 5.0e5);
        break;
    case ConvectionModel::Type::CYLINDER:
        Nu = computeCylinderNusselt(Re, Pr, L);
        break;
    case ConvectionModel::Type::NUSSELT_CORRELATION:
        Nu = 0.0296 * std::pow(Re, 0.8) * std::pow(Pr, 1.0 / 3.0);
        break;
    default:
        Nu = computeFlatPlateNusselt(Re, Pr, L, true);
        break;
    }
    return Nu * k / L;
}

Scalar computeCollectionEfficiency(Scalar V, Scalar D, Scalar LWC, Scalar MVD,
                                   Scalar theta, Scalar beta0) {
    Scalar K = WATER_DENSITY * MVD * MVD * V / (18.0 * computeAirViscosity(255.0) * D);
    Scalar modified_K = K / (K + 0.5);

    Scalar phi = 0.5 * modified_K;
    Scalar beta = beta0 * (1.0 - 0.2 * std::abs(theta));

    return std::max(0.0, std::min(1.0, beta));
}

Scalar computeIceGrowthRate(Scalar LWC, Scalar V, Scalar beta, Scalar T_surface,
                            Scalar T_freeze) {
    if (T_surface >= T_freeze) {
        return 0.0;
    }

    Scalar m_impinge = LWC * V * beta;
    Scalar freezing_fraction = std::min(1.0, (T_freeze - T_surface) / 10.0);

    return m_impinge * freezing_fraction / ICE_DENSITY;
}

Scalar computeLatentHeatFlux(Scalar m_impinge, Scalar T_surface, Scalar T_freeze) {
    if (T_surface >= T_freeze) {
        return 0.0;
    }

    Scalar freezing_fraction = std::min(1.0, (T_freeze - T_surface) / 10.0);
    return m_impinge * LATENT_HEAT_ICE * freezing_fraction;
}

std::vector<fem::FEMBoundaryCondition> buildExternalBoundaryConditions(
    std::shared_ptr<Grid3D> grid,
    const FlightCondition& flight,
    const VectorX& surfaceTemperature,
    Index boundaryId,
    const ConvectionModel& convModel) {
    std::vector<fem::FEMBoundaryCondition> bcs;

    Index start = grid->boundaryFaceStart[boundaryId];
    Index end = grid->boundaryFaceStart[boundaryId + 1];

    Scalar Re = computeReynoldsNumber(flight.V_inf, convModel.characteristicLength,
                                      flight.T_inf, flight.P_inf);
    Scalar Pr = computePrandtlNumber(flight.T_inf);
    Scalar k_air = 0.024 * std::pow(flight.T_inf / T_AIR_REF, 0.8);
    Scalar h_conv = computeConvectiveHeatTransferCoeff(Re, Pr, k_air,
                                                       convModel.characteristicLength,
                                                       convModel.type);

    std::set<Index> processedNodes;

    for (Index f = start; f < end; ++f) {
        const auto& face = grid->boundaryFaces[f];
        for (int n = 0; n < 3; ++n) {
            Index nid = face.nodes[n];
            if (processedNodes.find(nid) != processedNodes.end()) continue;
            processedNodes.insert(nid);

            fem::FEMBoundaryCondition conv_bc;
            conv_bc.type = fem::FEMBoundaryCondition::Type::CONVECTION;
            conv_bc.nodeId = nid;
            conv_bc.value = 0.0;
            conv_bc.convectiveCoeff = h_conv;
            conv_bc.ambientTemp = flight.T_inf;
            bcs.push_back(conv_bc);

            if (flight.LWC > 0.0) {
                Scalar beta = computeCollectionEfficiency(
                    flight.V_inf, convModel.characteristicLength,
                    flight.LWC, flight.MVD, 0.0, 0.8);

                fem::FEMBoundaryCondition ice_bc;
                ice_bc.type = fem::FEMBoundaryCondition::Type::ICE_LATENT_HEAT;
                ice_bc.nodeId = nid;
                ice_bc.value = flight.LWC;
                ice_bc.convectiveCoeff = flight.V_inf;
                ice_bc.ambientTemp = beta;
                bcs.push_back(ice_bc);
            }
        }
    }

    return bcs;
}

std::vector<fem::FEMBoundaryCondition> buildInternalBoundaryConditions(
    std::shared_ptr<Grid3D> grid,
    const VectorX& pipeWallTemperature,
    const VectorX& pipeHeatFlux,
    Index boundaryId,
    Scalar pipeDiameter) {
    std::vector<fem::FEMBoundaryCondition> bcs;

    Index start = grid->boundaryFaceStart[boundaryId];
    Index end = grid->boundaryFaceStart[boundaryId + 1];

    std::set<Index> processedNodes;

    for (Index f = start; f < end; ++f) {
        const auto& face = grid->boundaryFaces[f];
        for (int n = 0; n < 3; ++n) {
            Index nid = face.nodes[n];
            if (processedNodes.find(nid) != processedNodes.end()) continue;
            processedNodes.insert(nid);

            fem::FEMBoundaryCondition flux_bc;
            flux_bc.type = fem::FEMBoundaryCondition::Type::NEUMANN;
            flux_bc.nodeId = nid;

            Index pipeIdx = std::min(static_cast<Index>(pipeHeatFlux.size() - 1),
                                     static_cast<Index>((f - start) * pipeHeatFlux.size() / (end - start)));
            flux_bc.value = pipeHeatFlux(pipeIdx);
            bcs.push_back(flux_bc);
        }
    }

    return bcs;
}

}
}
