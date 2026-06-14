#pragma once

#include "common/types.h"
#include "common/grid.h"
#include "common/parallel_utils.h"
#include <memory>

namespace anti_icing {
namespace fvm {

struct PipeFlowState {
    VectorX rho;
    VectorX rhoU;
    VectorX rhoE;
    VectorX P;
    VectorX T;
    VectorX U;
    VectorX Mach;
    VectorX wallT;
    VectorX wallHeatFlux;

    void resize(Index n) {
        rho.resize(n);
        rhoU.resize(n);
        rhoE.resize(n);
        P.resize(n);
        T.resize(n);
        U.resize(n);
        Mach.resize(n);
        wallT.resize(n);
        wallHeatFlux.resize(n);
    }
};

struct PipeBoundaryConditions {
    Scalar inletTotalPressure;
    Scalar inletTotalTemperature;
    Scalar outletStaticPressure;
    Scalar inletMassFlowRate;
    bool useInletMassFlow;
};

struct PipeConfig {
    Scalar diameter;
    Scalar length;
    Scalar wallRoughness;
    Index  nCells;
    Scalar CFL;
    Scalar relaxationFactor;
    Index  maxInnerIter;
    Scalar innerTol;
};

class PipeFlowSolver {
public:
    PipeFlowSolver(const PipeConfig& config,
                   const PipeBoundaryConditions& bc);

    void initialize();
    Scalar computeStableTimeStep() const;
    void advance(Scalar dt);
    void solveSteadyState();
    void transferBoundaryData();

    const Grid1D& grid() const { return grid_; }
    const PipeFlowState& state() const { return state_; }
    PipeFlowState& state() { return state_; }

    void setWallTemperature(const VectorX& wallT);
    void setWallHeatFlux(const VectorX& heatFlux);

    VectorX getWallHeatFlux() const { return state_.wallHeatFlux; }
    VectorX getWallTemperature() const { return state_.wallT; }

private:
    void computePrimitiveVariables();
    void computeFluxes();
    void computeSourceTerms();
    void updateConservativeVariables(Scalar dt);
    void applyBoundaryConditions();

    Scalar computeFrictionFactor(Scalar Re, Scalar roughness, Scalar diameter) const;
    Scalar computeHeatTransferCoeff(Scalar Re, Scalar Pr, Scalar k, Scalar diameter) const;

    void reconstructAtFaces(const VectorX& phi,
                            VectorX& phiLeft,
                            VectorX& phiRight,
                            Scalar limiterK = 1.5) const;

    Scalar roeFlux(Scalar rL, Scalar rR, Scalar uL, Scalar uR,
                   Scalar pL, Scalar pR, Scalar hL, Scalar hR,
                   Scalar& massFlux, Scalar& momFlux, Scalar& engFlux) const;

    void computeWallHeatTransfer();

    PipeConfig config_;
    PipeBoundaryConditions bc_;
    Grid1D grid_;
    PipeFlowState state_;
    PipeFlowState statePrev_;

    VectorX massFlux_;
    VectorX momFlux_;
    VectorX engFlux_;
    VectorX sourceMass_;
    VectorX sourceMom_;
    VectorX sourceEng_;

    VectorX rhoLeft_, rhoRight_;
    VectorX uLeft_, uRight_;
    VectorX pLeft_, pRight_;
    VectorX hLeft_, hRight_;
};

std::shared_ptr<PipeFlowSolver> createPipeFlowSolver(
    const PipeConfig& config,
    const PipeBoundaryConditions& bc);

}
}
