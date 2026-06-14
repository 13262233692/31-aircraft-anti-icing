#include "fvm/pipe_flow_solver.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace anti_icing {
namespace fvm {

PipeFlowSolver::PipeFlowSolver(const PipeConfig& config,
                               const PipeBoundaryConditions& bc)
    : config_(config), bc_(bc) {
    grid_.initialize(config_.length, config_.nCells);
    state_.resize(config_.nCells);
    statePrev_.resize(config_.nCells);

    massFlux_.resize(grid_.nFaces);
    momFlux_.resize(grid_.nFaces);
    engFlux_.resize(grid_.nFaces);
    sourceMass_.resize(config_.nCells);
    sourceMom_.resize(config_.nCells);
    sourceEng_.resize(config_.nCells);

    rhoLeft_.resize(grid_.nFaces);
    rhoRight_.resize(grid_.nFaces);
    uLeft_.resize(grid_.nFaces);
    uRight_.resize(grid_.nFaces);
    pLeft_.resize(grid_.nFaces);
    pRight_.resize(grid_.nFaces);
    hLeft_.resize(grid_.nFaces);
    hRight_.resize(grid_.nFaces);
}

void PipeFlowSolver::initialize() {
    Scalar P0 = bc_.inletTotalPressure;
    Scalar T0 = bc_.inletTotalTemperature;
    Scalar P_out = bc_.outletStaticPressure;

    Scalar P_ratio = P_out / P0;
    Scalar T_ratio = std::pow(P_ratio, (GAMMA_AIR - 1.0) / GAMMA_AIR);

    parallel::parallelFor(config_.nCells, [&](Index i) {
        Scalar x = grid_.x(i) / config_.length;
        Scalar P = P0 * (1.0 - x) + P_out * x;
        Scalar T = T0 * std::pow(P / P0, (GAMMA_AIR - 1.0) / GAMMA_AIR);
        Scalar rho = computeAirDensity(P, T);
        Scalar U = 50.0;

        state_.rho(i) = rho;
        state_.rhoU(i) = rho * U;
        state_.P(i) = P;
        state_.T(i) = T;
        state_.U(i) = U;
        state_.Mach(i) = U / computeSpeedOfSound(T);

        Scalar e = CP_AIR * (T - T_AIR_REF) + 0.5 * U * U;
        state_.rhoE(i) = rho * e;

        state_.wallT(i) = T;
        state_.wallHeatFlux(i) = 0.0;
    });

    statePrev_ = state_;
}

Scalar PipeFlowSolver::computeStableTimeStep() const {
    Scalar minDt = std::numeric_limits<Scalar>::max();

    parallel::parallelFor(config_.nCells, [&](Index i) {
        Scalar c = computeSpeedOfSound(state_.T(i));
        Scalar U = std::abs(state_.U(i));
        Scalar maxSpeed = U + c;
        Scalar dt = config_.CFL * grid_.dx(i) / maxSpeed;
        #pragma omp critical
        {
            if (dt < minDt) minDt = dt;
        }
    });

    return minDt;
}

void PipeFlowSolver::advance(Scalar dt) {
    computePrimitiveVariables();
    reconstructAtFaces(state_.rho, rhoLeft_, rhoRight_, 1.5);
    reconstructAtFaces(state_.U, uLeft_, uRight_, 1.5);
    reconstructAtFaces(state_.P, pLeft_, pRight_, 1.5);

    parallel::parallelFor(config_.nCells, [&](Index i) {
        Scalar hL = CP_AIR * (state_.T(i) - T_AIR_REF) + 0.5 * state_.U(i) * state_.U(i);
        if (i > 0) {
            hL = CP_AIR * (pLeft_(i) / (R_AIR * rhoLeft_(i)) - T_AIR_REF) + 0.5 * uLeft_(i) * uLeft_(i);
        }
        hLeft_(i) = hL;
        hRight_(i) = CP_AIR * (pRight_(i) / (R_AIR * rhoRight_(i)) - T_AIR_REF) + 0.5 * uRight_(i) * uRight_(i);
    });

    hLeft_(config_.nCells) = CP_AIR * (state_.T(config_.nCells - 1) - T_AIR_REF) + 0.5 * state_.U(config_.nCells - 1) * state_.U(config_.nCells - 1);
    hRight_(config_.nCells) = hLeft_(config_.nCells);

    computeFluxes();
    computeSourceTerms();
    computeWallHeatTransfer();
    updateConservativeVariables(dt);
    applyBoundaryConditions();
    computePrimitiveVariables();
}

void PipeFlowSolver::computePrimitiveVariables() {
    parallel::parallelFor(config_.nCells, [&](Index i) {
        Scalar rho = state_.rho(i);
        Scalar rhoU = state_.rhoU(i);
        Scalar rhoE = state_.rhoE(i);

        Scalar U = rhoU / rho;
        Scalar e = rhoE / rho;
        Scalar T = (e - 0.5 * U * U) / CP_AIR + T_AIR_REF;
        Scalar P = rho * R_AIR * T;

        state_.U(i) = U;
        state_.T(i) = std::max(200.0, T);
        state_.P(i) = std::max(10000.0, P);
        state_.Mach(i) = U / computeSpeedOfSound(state_.T(i));
    });
}

void PipeFlowSolver::reconstructAtFaces(const VectorX& phi,
                                        VectorX& phiLeft,
                                        VectorX& phiRight,
                                        Scalar limiterK) const {
    Index n = config_.nCells;

    phiLeft(0) = phi(0);
    phiRight(0) = phi(0);

    parallel::parallelFor(Index(1), n, [&](Index i) {
        Scalar phi_im1 = (i > 1) ? phi(i - 2) : phi(0);
        Scalar phi_i = phi(i - 1);
        Scalar phi_ip1 = phi(i);
        Scalar phi_ip2 = (i < n - 1) ? phi(i + 1) : phi(n - 1);

        Scalar grad_L = phi_i - phi_im1;
        Scalar grad_R = phi_ip1 - phi_i;
        Scalar grad_C = 0.5 * (phi_ip1 - phi_im1);

        Scalar limited_grad_L = 0.0;
        if (grad_L * grad_R > 0.0) {
            limited_grad_L = std::copysign(1.0, grad_C) *
                std::min(std::abs(grad_C), std::min(limiterK * std::abs(grad_L), limiterK * std::abs(grad_R)));
        }

        grad_L = grad_R;
        grad_R = phi_ip2 - phi_ip1;
        grad_C = 0.5 * (phi_ip2 - phi_i);

        Scalar limited_grad_R = 0.0;
        if (grad_L * grad_R > 0.0) {
            limited_grad_R = std::copysign(1.0, grad_C) *
                std::min(std::abs(grad_C), std::min(limiterK * std::abs(grad_L), limiterK * std::abs(grad_R)));
        }

        phiLeft(i) = phi_i + 0.5 * limited_grad_L;
        phiRight(i) = phi_ip1 - 0.5 * limited_grad_R;
    });

    phiLeft(n) = phi(n - 1);
    phiRight(n) = phi(n - 1);
}

Scalar PipeFlowSolver::roeFlux(Scalar rL, Scalar rR, Scalar uL, Scalar uR,
                               Scalar pL, Scalar pR, Scalar hL, Scalar hR,
                               Scalar& massFlux, Scalar& momFlux, Scalar& engFlux) const {
    Scalar rRL = std::sqrt(rR / rL);
    Scalar denom = 1.0 + rRL;

    Scalar rhoTilde = std::sqrt(rL * rR);
    Scalar uTilde = (uL + rRL * uR) / denom;
    Scalar hTilde = (hL + rRL * hR) / denom;
    Scalar cTilde = std::sqrt((GAMMA_AIR - 1.0) * (hTilde - 0.5 * uTilde * uTilde));

    Scalar drho = rR - rL;
    Scalar du = uR - uL;
    Scalar dp = pR - pL;

    Scalar alpha1 = (dp - rhoTilde * cTilde * du) / (2.0 * cTilde * cTilde);
    Scalar alpha2 = drho - dp / (cTilde * cTilde);
    Scalar alpha3 = (dp + rhoTilde * cTilde * du) / (2.0 * cTilde * cTilde);

    Scalar lambda1 = std::abs(uTilde - cTilde);
    Scalar lambda2 = std::abs(uTilde);
    Scalar lambda3 = std::abs(uTilde + cTilde);

    Scalar eps = 0.1 * cTilde;
    lambda1 = (lambda1 < eps) ? (lambda1 * lambda1 + eps * eps) / (2.0 * eps) : lambda1;
    lambda2 = (lambda2 < eps) ? (lambda2 * lambda2 + eps * eps) / (2.0 * eps) : lambda2;
    lambda3 = (lambda3 < eps) ? (lambda3 * lambda3 + eps * eps) / (2.0 * eps) : lambda3;

    Scalar fluxRho_L = rL * uL;
    Scalar fluxRho_R = rR * uR;
    Scalar fluxMom_L = rL * uL * uL + pL;
    Scalar fluxMom_R = rR * uR * uR + pR;
    Scalar fluxEng_L = rL * uL * hL;
    Scalar fluxEng_R = rR * uR * hR;

    massFlux = 0.5 * (fluxRho_L + fluxRho_R) -
               0.5 * (lambda1 * alpha1 + lambda2 * alpha2 + lambda3 * alpha3);
    momFlux = 0.5 * (fluxMom_L + fluxMom_R) -
              0.5 * (lambda1 * alpha1 * (uTilde - cTilde) +
                     lambda2 * alpha2 * uTilde +
                     lambda3 * alpha3 * (uTilde + cTilde));
    engFlux = 0.5 * (fluxEng_L + fluxEng_R) -
              0.5 * (lambda1 * alpha1 * (hTilde - uTilde * cTilde) +
                     lambda2 * alpha2 * 0.5 * uTilde * uTilde +
                     lambda3 * alpha3 * (hTilde + uTilde * cTilde));

    return std::max(std::abs(uTilde) + cTilde, std::max(lambda1, lambda3));
}

void PipeFlowSolver::computeFluxes() {
    Index n = config_.nCells;

    Scalar inlet_rho, inlet_u, inlet_p, inlet_h;
    if (bc_.useInletMassFlow) {
        Scalar mdot = bc_.inletMassFlowRate;
        Scalar A = 0.25 * PI * config_.diameter * config_.diameter;
        Scalar T0 = bc_.inletTotalTemperature;
        Scalar P0 = bc_.inletTotalPressure;

        Scalar P = state_.P(0);
        Scalar T = T0 * std::pow(P / P0, (GAMMA_AIR - 1.0) / GAMMA_AIR);
        Scalar rho = computeAirDensity(P, T);
        Scalar U = mdot / (rho * A);

        inlet_rho = rho;
        inlet_u = U;
        inlet_p = P;
        inlet_h = CP_AIR * (T - T_AIR_REF) + 0.5 * U * U;
    } else {
        Scalar T0 = bc_.inletTotalTemperature;
        Scalar P0 = bc_.inletTotalPressure;
        Scalar P = state_.P(0);
        Scalar T = T0 * std::pow(P / P0, (GAMMA_AIR - 1.0) / GAMMA_AIR);
        Scalar rho = computeAirDensity(P, T);
        Scalar M = std::sqrt(2.0 / (GAMMA_AIR - 1.0) * (T0 / T - 1.0));
        Scalar U = M * computeSpeedOfSound(T);

        inlet_rho = rho;
        inlet_u = U;
        inlet_p = P;
        inlet_h = CP_AIR * (T - T_AIR_REF) + 0.5 * U * U;
    }

    Scalar outlet_rho = state_.rho(n - 1);
    Scalar outlet_u = state_.U(n - 1);
    Scalar outlet_p = bc_.outletStaticPressure;
    Scalar outlet_T = outlet_p / (R_AIR * outlet_rho);
    Scalar outlet_h = CP_AIR * (outlet_T - T_AIR_REF) + 0.5 * outlet_u * outlet_u;

    Scalar dummy;
    roeFlux(inlet_rho, rhoRight_(0), inlet_u, uRight_(0),
            inlet_p, pRight_(0), inlet_h, hRight_(0),
            massFlux_(0), momFlux_(0), engFlux_(0));

    parallel::parallelFor(Index(1), n, [&](Index f) {
        roeFlux(rhoLeft_(f), rhoRight_(f), uLeft_(f), uRight_(f),
                pLeft_(f), pRight_(f), hLeft_(f), hRight_(f),
                massFlux_(f), momFlux_(f), engFlux_(f));
    });

    roeFlux(rhoLeft_(n), outlet_rho, uLeft_(n), outlet_u,
            pLeft_(n), outlet_p, hLeft_(n), outlet_h,
            massFlux_(n), momFlux_(n), engFlux_(n));
}

Scalar PipeFlowSolver::computeFrictionFactor(Scalar Re, Scalar roughness, Scalar diameter) const {
    if (Re < 2300.0) {
        return 64.0 / std::max(Re, 1.0);
    } else {
        Scalar relRough = roughness / diameter;
        Scalar f = 0.02;

        for (int iter = 0; iter < 20; ++iter) {
            Scalar rhs = -2.0 * std::log10(relRough / 3.7 + 2.51 / (Re * std::sqrt(std::max(f, 1.0e-12))));
            Scalar f_new = 1.0 / (rhs * rhs);
            if (std::abs(f_new - f) < 1.0e-8) {
                f = f_new;
                break;
            }
            f = f_new;
        }
        return f;
    }
}

Scalar PipeFlowSolver::computeHeatTransferCoeff(Scalar Re, Scalar Pr, Scalar k, Scalar diameter) const {
    Scalar Nu;
    if (Re < 2300.0) {
        Nu = 4.36;
    } else {
        Scalar f = computeFrictionFactor(Re, config_.wallRoughness, diameter);
        Nu = (f / 8.0) * (Re - 1000.0) * Pr / (1.0 + 12.7 * std::sqrt(f / 8.0) * (std::pow(Pr, 2.0 / 3.0) - 1.0));
    }
    return Nu * k / diameter;
}

void PipeFlowSolver::computeSourceTerms() {
    Scalar D = config_.diameter;
    Scalar A = 0.25 * PI * D * D;
    Scalar perimeter = PI * D;

    parallel::parallelFor(config_.nCells, [&](Index i) {
        Scalar rho = state_.rho(i);
        Scalar U = state_.U(i);
        Scalar T = state_.T(i);
        Scalar mu = computeAirViscosity(T);
        Scalar Re = std::max(1.0, rho * U * D / mu);
        Scalar f = computeFrictionFactor(Re, config_.wallRoughness, D);

        Scalar tau_w = 0.25 * f * rho * U * std::abs(U);
        Scalar pressureLoss = tau_w * perimeter / A;

        sourceMass_(i) = 0.0;
        sourceMom_(i) = -pressureLoss * grid_.cellVolume(i);
        sourceEng_(i) = -tau_w * U * perimeter * grid_.dx(i);
    });
}

void PipeFlowSolver::computeWallHeatTransfer() {
    Scalar D = config_.diameter;
    Scalar perimeter = PI * D;

    parallel::parallelFor(config_.nCells, [&](Index i) {
        Scalar rho = state_.rho(i);
        Scalar U = state_.U(i);
        Scalar T = state_.T(i);
        Scalar mu = computeAirViscosity(T);
        Scalar k = 0.024 * std::pow(T / T_AIR_REF, 0.8);
        Scalar Pr = CP_AIR * mu / k;
        Scalar Re = std::max(1.0, rho * U * D / mu);

        Scalar h = computeHeatTransferCoeff(Re, Pr, k, D);
        Scalar T_wall = state_.wallT(i);

        Scalar q_wall = h * (T - T_wall);
        state_.wallHeatFlux(i) = q_wall;

        sourceEng_(i) += -q_wall * perimeter * grid_.dx(i);
    });
}

void PipeFlowSolver::updateConservativeVariables(Scalar dt) {
    parallel::parallelFor(config_.nCells, [&](Index i) {
        Scalar dMdt = (massFlux_(i) - massFlux_(i + 1)) * grid_.faceArea(0) + sourceMass_(i);
        Scalar dMudt = (momFlux_(i) - momFlux_(i + 1)) * grid_.faceArea(0) + sourceMom_(i);
        Scalar dMedt = (engFlux_(i) - engFlux_(i + 1)) * grid_.faceArea(0) + sourceEng_(i);

        Scalar relax = config_.relaxationFactor;
        state_.rho(i) = state_.rho(i) + relax * (dt * dMdt / grid_.cellVolume(i));
        state_.rhoU(i) = state_.rhoU(i) + relax * (dt * dMudt / grid_.cellVolume(i));
        state_.rhoE(i) = state_.rhoE(i) + relax * (dt * dMedt / grid_.cellVolume(i));

        state_.rho(i) = std::max(0.1, state_.rho(i));
    });
}

void PipeFlowSolver::applyBoundaryConditions() {
    if (bc_.useInletMassFlow) {
        Scalar mdot = bc_.inletMassFlowRate;
        Scalar A = 0.25 * PI * config_.diameter * config_.diameter;
        Scalar T0 = bc_.inletTotalTemperature;
        Scalar P0 = bc_.inletTotalPressure;

        Scalar P = state_.P(0);
        Scalar T = T0 * std::pow(P / P0, (GAMMA_AIR - 1.0) / GAMMA_AIR);
        Scalar rho = computeAirDensity(P, T);
        Scalar U = mdot / (rho * A);

        state_.rho(0) = rho;
        state_.rhoU(0) = rho * U;
        state_.rhoE(0) = rho * (CP_AIR * (T - T_AIR_REF) + 0.5 * U * U);
    } else {
        Scalar T0 = bc_.inletTotalTemperature;
        Scalar P0 = bc_.inletTotalPressure;
        Scalar P = state_.P(0);
        Scalar T = T0 * std::pow(P / P0, (GAMMA_AIR - 1.0) / GAMMA_AIR);
        Scalar rho = computeAirDensity(P, T);
        Scalar M = std::sqrt(2.0 / (GAMMA_AIR - 1.0) * (T0 / T - 1.0));
        Scalar U = M * computeSpeedOfSound(T);

        state_.rho(0) = rho;
        state_.rhoU(0) = rho * U;
        state_.rhoE(0) = rho * (CP_AIR * (T - T_AIR_REF) + 0.5 * U * U);
    }

    Index n = config_.nCells - 1;
    Scalar rho_out = state_.rho(n);
    Scalar U_out = state_.U(n);
    Scalar P_out = bc_.outletStaticPressure;
    Scalar T_out = P_out / (R_AIR * rho_out);

    state_.rho(n) = rho_out;
    state_.rhoU(n) = rho_out * U_out;
    state_.rhoE(n) = rho_out * (CP_AIR * (T_out - T_AIR_REF) + 0.5 * U_out * U_out);
    state_.P(n) = P_out;
}

void PipeFlowSolver::solveSteadyState() {
    Scalar dt = computeStableTimeStep();
    Scalar residual = 1.0;
    Index iter = 0;

    while (residual > config_.innerTol && iter < config_.maxInnerIter) {
        statePrev_ = state_;
        advance(dt);

        residual = 0.0;
        parallel::parallelFor(config_.nCells, [&](Index i) {
            Scalar drho = std::abs(state_.rho(i) - statePrev_.rho(i)) / std::abs(statePrev_.rho(i) + 1.0e-10);
            #pragma omp critical
            {
                residual = std::max(residual, drho);
            }
        });

        iter++;
        if (iter % 100 == 0) {
            dt = computeStableTimeStep();
        }
    }
}

void PipeFlowSolver::transferBoundaryData() {
    computeWallHeatTransfer();
}

void PipeFlowSolver::setWallTemperature(const VectorX& wallT) {
    state_.wallT = wallT;
}

void PipeFlowSolver::setWallHeatFlux(const VectorX& heatFlux) {
    state_.wallHeatFlux = heatFlux;
}

std::shared_ptr<PipeFlowSolver> createPipeFlowSolver(
    const PipeConfig& config,
    const PipeBoundaryConditions& bc) {
    auto solver = std::make_shared<PipeFlowSolver>(config, bc);
    solver->initialize();
    return solver;
}

}
}
