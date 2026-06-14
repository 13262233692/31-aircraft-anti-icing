#include "fem/heat_conduction_solver.h"
#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace anti_icing {
namespace fem {

HeatConductionSolver::HeatConductionSolver(std::shared_ptr<Grid3D> grid,
                                           const std::vector<MaterialProperties>& materials,
                                           const FEMConfig& config)
    : grid_(grid), materials_(materials), config_(config) {
    Index n = grid_->nNodes;
    state_.resize(n);

    K_global_.resize(n, n);
    M_global_.resize(n, n);
    A_global_.resize(n, n);
    F_global_.resize(n);
    R_global_.resize(n);

    F_global_.setZero();
    R_global_.setZero();

    Index nThreads = config_.useParallel ? parallel::getNumThreads() : 1;
    tripletBuffer_.resize(nThreads);
}

void HeatConductionSolver::initialize(Scalar initialTemp) {
    Scalar safeTemp = clamp(initialTemp, config_.tempMinPhysical, config_.tempMaxPhysical);
    state_.temperature.setConstant(safeTemp);
    state_.temperaturePrev.setConstant(safeTemp);
    state_.heatFlux.setZero();
    state_.iceThickness.setZero();
    state_.iceGrowthRate.setZero();
}

void HeatConductionSolver::computeTetStiffnessMatrix(const TetrahedronCell& cell,
                                                     const MaterialProperties& mat,
                                                     Eigen::Matrix<Scalar, 4, 4>& K) const {
    const auto& n = cell.nodes;
    Vector3 v0 = grid_->nodes[n[0]];
    Vector3 v1 = grid_->nodes[n[1]] - v0;
    Vector3 v2 = grid_->nodes[n[2]] - v0;
    Vector3 v3 = grid_->nodes[n[3]] - v0;

    Matrix3 J;
    J.col(0) = v1;
    J.col(1) = v2;
    J.col(2) = v3;

    Scalar detJ = J.determinant();
    Scalar absDetJ = std::abs(detJ);

    Matrix3 invJ;
    Scalar detJ_safe = std::max(detJ, 1.0e-20);
    if (std::abs(detJ_safe) > 1.0e-20) {
        invJ = J.inverse();
    } else {
        invJ = Matrix3::Identity();
    }

    Eigen::Matrix<Scalar, 3, 4> dN;
    dN.col(0) = -invJ.row(0).transpose() - invJ.row(1).transpose() - invJ.row(2).transpose();
    dN.col(1) = invJ.row(0).transpose();
    dN.col(2) = invJ.row(1).transpose();
    dN.col(3) = invJ.row(2).transpose();

    Matrix3 kT = mat.conductivity;
    K = dN.transpose() * kT * dN * absDetJ / 6.0;
}

void HeatConductionSolver::computeTetMassMatrix(const TetrahedronCell& cell,
                                                const MaterialProperties& mat,
                                                Eigen::Matrix<Scalar, 4, 4>& M) const {
    Scalar rhoCp = mat.density * mat.specificHeat;
    Scalar vol = std::max(std::abs(cell.volume), 1.0e-20);

    if (config_.useLumpedMass) {
        M.setZero();
        Scalar lumped = rhoCp * vol / 4.0;
        M.diagonal().setConstant(lumped);
    } else {
        Scalar coeff = rhoCp * vol / 20.0;
        M.setConstant(2.0 * coeff);
        M.diagonal().setConstant(4.0 * coeff);
    }
}

void HeatConductionSolver::computeTetLoadVector(const TetrahedronCell& cell,
                                                Scalar dt,
                                                Eigen::Matrix<Scalar, 4, 1>& F) const {
    F.setZero();
}

void HeatConductionSolver::assembleSystem(Scalar dt) {
    Index n = grid_->nNodes;
    Index nThreads = config_.useParallel ? parallel::getNumThreads() : 1;

    for (auto& buf : tripletBuffer_) {
        buf.clear();
        buf.reserve(grid_->nCells * 16 / nThreads + 64);
    }

    std::vector<std::vector<Triplet>> kTriplets(nThreads);
    std::vector<std::vector<Triplet>> mTriplets(nThreads);

    for (auto& buf : kTriplets) {
        buf.reserve(grid_->nCells * 16 / nThreads + 64);
    }
    for (auto& buf : mTriplets) {
        buf.reserve(grid_->nCells * 16 / nThreads + 64);
    }

    parallel::parallelFor(grid_->nCells, [&](Index e) {
        Index tid = config_.useParallel ? parallel::getThreadId() : 0;
        const auto& cell = grid_->cells[e];
        const auto& mat = materials_[cell.materialId];

        Eigen::Matrix<Scalar, 4, 4> K_e, M_e;
        computeTetStiffnessMatrix(cell, mat, K_e);
        computeTetMassMatrix(cell, mat, M_e);

        for (int i = 0; i < 4; ++i) {
            Index ni = cell.nodes[i];
            for (int j = 0; j < 4; ++j) {
                Index nj = cell.nodes[j];
                Scalar kij = K_e(i, j);
                Scalar mij = M_e(i, j);
                if (isNaN(kij)) kij = 0.0;
                if (isNaN(mij)) mij = 0.0;
                kTriplets[tid].emplace_back(ni, nj, kij);
                mTriplets[tid].emplace_back(ni, nj, mij);
            }
        }
    }, config_.useParallel);

    std::vector<Triplet> allK, allM;
    allK.reserve(grid_->nCells * 16);
    allM.reserve(grid_->nCells * 16);
    for (auto& buf : kTriplets) allK.insert(allK.end(), buf.begin(), buf.end());
    for (auto& buf : mTriplets) allM.insert(allM.end(), buf.begin(), buf.end());

    K_global_.setZero();
    K_global_.setFromTriplets(allK.begin(), allK.end());

    M_global_.setZero();
    M_global_.setFromTriplets(allM.begin(), allM.end());

    Scalar theta = config_.theta;
    Scalar dt_safe = std::max(dt, 1.0e-15);
    A_global_ = M_global_ / dt_safe + theta * K_global_;

    if (config_.enableNumericalDamping) {
        Scalar diagMax = 0.0;
        for (Index k = 0; k < A_global_.outerSize(); ++k) {
            for (SparseMatrix::InnerIterator it(A_global_, k); it; ++it) {
                if (it.row() == it.col()) {
                    diagMax = std::max(diagMax, std::abs(it.value()));
                }
            }
        }
        Scalar damping = config_.numericalDampingCoeff * std::max(diagMax, 1.0e-10);
        for (Index k = 0; k < A_global_.outerSize(); ++k) {
            for (SparseMatrix::InnerIterator it(A_global_, k); it; ++it) {
                if (it.row() == it.col()) {
                    it.valueRef() += damping;
                }
            }
        }
    }

    VectorX T_prev = state_.temperaturePrev;
    for (Index i = 0; i < T_prev.size(); ++i) {
        if (isNaN(T_prev(i))) {
            T_prev(i) = (config_.tempMinPhysical + config_.tempMaxPhysical) * 0.5;
        }
    }

    R_global_ = (M_global_ / dt_safe - (1.0 - theta) * K_global_) * T_prev;

    for (Index i = 0; i < R_global_.size(); ++i) {
        if (isNaN(R_global_(i))) {
            R_global_(i) = 0.0;
        }
    }

    F_global_ = R_global_;
}

void HeatConductionSolver::applyBoundaryConditions(const std::vector<FEMBoundaryCondition>& bcs) {
    Scalar th = config_.theta;
    Scalar T_freeze = config_.freezingPointTemp;
    Scalar transW = config_.freezeTransitionWidth;

    for (const auto& bc : bcs) {
        Index nid = bc.nodeId;
        if (nid >= F_global_.size()) continue;

        switch (bc.type) {
        case FEMBoundaryCondition::Type::DIRICHLET: {
            Scalar val = clamp(bc.value, config_.tempMinPhysical, config_.tempMaxPhysical);
            for (SparseMatrix::InnerIterator it(A_global_, nid); it; ++it) {
                if (it.row() == nid) {
                    it.valueRef() = 1.0;
                } else {
                    F_global_(it.row()) -= it.valueRef() * val;
                    it.valueRef() = 0.0;
                }
            }
            for (SparseMatrix::InnerIterator it(A_global_, nid); it; ++it) {
                if (it.row() != nid) {
                    it.valueRef() = 0.0;
                }
            }
            F_global_(nid) = val;
            break;
        }
        case FEMBoundaryCondition::Type::NEUMANN: {
            Scalar val = bc.value;
            Scalar maxVal = 1.0e8;
            if (isNaN(val)) val = 0.0;
            val = clamp(val, -maxVal, maxVal);
            F_global_(nid) += val;
            break;
        }
        case FEMBoundaryCondition::Type::CONVECTION: {
            Scalar h_conv = bc.convectiveCoeff;
            Scalar T_inf = clamp(bc.ambientTemp, config_.tempMinPhysical, config_.tempMaxPhysical);

            if (isNaN(h_conv)) h_conv = 0.0;
            h_conv = std::max(0.0, std::min(h_conv, 1.0e6));

            for (SparseMatrix::InnerIterator it(K_global_, nid); it; ++it) {
                if (it.row() == nid && it.col() == nid) {
                    A_global_.coeffRef(it.row(), it.col()) += th * h_conv;
                }
            }
            A_global_.coeffRef(nid, nid) += th * h_conv;

            Scalar h_neumann = h_conv * T_inf;
            if (isNaN(h_neumann)) h_neumann = 0.0;
            F_global_(nid) += h_neumann;

            Scalar T_prev_n = state_.temperaturePrev(nid);
            if (isNaN(T_prev_n)) T_prev_n = T_inf;
            Scalar prev_contrib = (1.0 - th) * h_conv * (T_inf - T_prev_n);
            if (isNaN(prev_contrib)) prev_contrib = 0.0;
            F_global_(nid) += prev_contrib;
            break;
        }
        case FEMBoundaryCondition::Type::ICE_LATENT_HEAT: {
            Scalar T_node = state_.temperature(nid);
            Scalar LWC = bc.value;
            Scalar V_inf = bc.convectiveCoeff;
            Scalar beta = bc.ambientTemp;

            if (isNaN(T_node)) T_node = T_freeze;
            if (isNaN(LWC)) LWC = 0.0;
            if (isNaN(V_inf)) V_inf = 0.0;
            if (isNaN(beta)) beta = 0.0;

            LWC = std::max(0.0, std::min(LWC, 10.0));
            V_inf = std::max(0.0, std::min(V_inf, 500.0));
            beta = clamp(beta, 0.0, 1.0);

            Scalar m_impinge = LWC * V_inf * beta;

            Scalar deltaT = T_freeze - T_node;
            Scalar x = deltaT / transW;
            Scalar freezing_fraction = sigmoid(x, 0.0, 1.0);

            Scalar q_latent = m_impinge * LATENT_HEAT_ICE * freezing_fraction;

            if (isNaN(q_latent)) q_latent = 0.0;
            Scalar max_q = 1.0e8;
            q_latent = clamp(q_latent, 0.0, max_q);

            Scalar ice_rate = m_impinge * freezing_fraction / ICE_DENSITY;
            if (isNaN(ice_rate)) ice_rate = 0.0;
            ice_rate = std::max(0.0, std::min(ice_rate, 1.0e-2));
            state_.iceGrowthRate(nid) = ice_rate;

            F_global_(nid) -= q_latent;
            break;
        }
        }
    }

    for (Index i = 0; i < F_global_.size(); ++i) {
        if (isNaN(F_global_(i))) {
            F_global_(i) = 0.0;
        }
        Scalar Tavg = (config_.tempMinPhysical + config_.tempMaxPhysical) * 0.5;
        Scalar maxF = 1.0e12;
        if (std::abs(F_global_(i)) > maxF && std::abs(F_global_(i)) > 1.0) {
            F_global_(i) = Tavg * M_global_.coeff(i, i);
        }
    }
}

void HeatConductionSolver::solve() {
    VectorX solution;
    bool solveSuccess = false;

    Eigen::SparseLU<SparseMatrix> solver;
    solver.analyzePattern(A_global_);
    solver.factorize(A_global_);

    if (solver.info() == Eigen::Success) {
        solution = solver.solve(F_global_);
        if (solver.info() == Eigen::Success && isVectorValid(solution)) {
            solveSuccess = true;
        }
    }

    if (!solveSuccess) {
        for (Index k = 0; k < A_global_.outerSize(); ++k) {
            for (SparseMatrix::InnerIterator it(A_global_, k); it; ++it) {
                if (it.row() == it.col()) {
                    Scalar scale = 1.0e-6 * std::abs(it.value());
                    it.valueRef() += std::max(scale, 1.0e-8);
                }
            }
        }

        solver.analyzePattern(A_global_);
        solver.factorize(A_global_);

        if (solver.info() == Eigen::Success) {
            solution = solver.solve(F_global_);
            if (solver.info() == Eigen::Success && isVectorValid(solution)) {
                solveSuccess = true;
            }
        }
    }

    if (!solveSuccess) {
        Eigen::BiCGSTAB<SparseMatrix, Eigen::IncompleteLUT<Scalar>> iterSolver;
        iterSolver.setMaxIterations(config_.maxIter * 2);
        iterSolver.setTolerance(std::max(config_.tolerance, 1.0e-6));
        iterSolver.compute(A_global_);
        solution = iterSolver.solveWithGuess(F_global_, state_.temperaturePrev);
        solveSuccess = isVectorValid(solution);
    }

    if (!solveSuccess) {
        solution = state_.temperaturePrev;
        for (Index i = 0; i < solution.size(); ++i) {
            if (isNaN(solution(i))) {
                solution(i) = (config_.tempMinPhysical + config_.tempMaxPhysical) * 0.5;
            }
        }
    }

    if (config_.enableLineSearch) {
        Scalar alpha = 1.0;
        VectorX candidate = solution;
        VectorX base = state_.temperaturePrev;
        VectorX diff = candidate - base;

        for (Index iter = 0; iter < 6; ++iter) {
            VectorX trial = base + alpha * diff;
            trial = clampVector(trial, config_.tempMinPhysical, config_.tempMaxPhysical);

            Scalar normTrial = (A_global_ * trial - F_global_).norm();
            Scalar normBase = (A_global_ * base - F_global_).norm();

            if (isNaN(normTrial) || normTrial > normBase * 1.5) {
                alpha *= 0.5;
            } else {
                solution = trial;
                break;
            }

            if (iter == 5) {
                solution = clampVector(base + 0.1 * diff,
                                        config_.tempMinPhysical,
                                        config_.tempMaxPhysical);
            }
        }
    }

    state_.temperature = clampVector(solution,
                                      config_.tempMinPhysical,
                                      config_.tempMaxPhysical);

    for (Index i = 0; i < state_.temperature.size(); ++i) {
        if (isNaN(state_.temperature(i))) {
            state_.temperature(i) = state_.temperaturePrev(i);
            if (isNaN(state_.temperature(i))) {
                state_.temperature(i) = (config_.tempMinPhysical + config_.tempMaxPhysical) * 0.5;
            }
        }
    }

    computeNodalHeatFlux();
}

void HeatConductionSolver::advance(Scalar dt, const std::vector<FEMBoundaryCondition>& bcs) {
    state_.temperaturePrev = state_.temperature;

    for (Index i = 0; i < state_.temperaturePrev.size(); ++i) {
        if (isNaN(state_.temperaturePrev(i))) {
            state_.temperaturePrev(i) = (config_.tempMinPhysical + config_.tempMaxPhysical) * 0.5;
        }
    }

    assembleSystem(dt);
    applyBoundaryConditions(bcs);
    solve();
}

void HeatConductionSolver::computeNodalGradient(const VectorX& field, Vector3Array& grad) const {
    grad.assign(grid_->nNodes, Vector3::Zero());
    VectorX count(grid_->nNodes);
    count.setZero();

    for (Index e = 0; e < grid_->nCells; ++e) {
        const auto& cell = grid_->cells[e];
        const auto& n = cell.nodes;

        Vector3 v0 = grid_->nodes[n[0]];
        Vector3 v1 = grid_->nodes[n[1]] - v0;
        Vector3 v2 = grid_->nodes[n[2]] - v0;
        Vector3 v3 = grid_->nodes[n[3]] - v0;

        Matrix3 J;
        J.col(0) = v1;
        J.col(1) = v2;
        J.col(2) = v3;

        Scalar detJ = J.determinant();
        if (std::abs(detJ) < 1.0e-20) continue;
        Matrix3 invJ = J.inverse();

        Scalar T0 = field(n[0]);
        Scalar T1 = field(n[1]);
        Scalar T2 = field(n[2]);
        Scalar T3 = field(n[3]);
        if (isNaN(T0) || isNaN(T1) || isNaN(T2) || isNaN(T3)) continue;

        Scalar dTdx = invJ(0, 0) * (T1 - T0) + invJ(0, 1) * (T2 - T0) + invJ(0, 2) * (T3 - T0);
        Scalar dTdy = invJ(1, 0) * (T1 - T0) + invJ(1, 1) * (T2 - T0) + invJ(1, 2) * (T3 - T0);
        Scalar dTdz = invJ(2, 0) * (T1 - T0) + invJ(2, 1) * (T2 - T0) + invJ(2, 2) * (T3 - T0);

        if (isNaN(dTdx)) dTdx = 0.0;
        if (isNaN(dTdy)) dTdy = 0.0;
        if (isNaN(dTdz)) dTdz = 0.0;

        Vector3 elemGrad(dTdx, dTdy, dTdz);

        for (int i = 0; i < 4; ++i) {
            grad[n[i]] += elemGrad;
            count(n[i]) += 1.0;
        }
    }

    for (Index i = 0; i < grid_->nNodes; ++i) {
        if (count(i) > 0.5) {
            grad[i] /= count(i);
        }
    }
}

void HeatConductionSolver::computeNodalHeatFlux() {
    Vector3Array gradT;
    computeNodalGradient(state_.temperature, gradT);

    state_.heatFlux.resize(grid_->nNodes);
    for (Index i = 0; i < grid_->nNodes; ++i) {
        Index matId = 0;
        for (Index e = 0; e < grid_->nCells; ++e) {
            const auto& cell = grid_->cells[e];
            bool found = false;
            for (int j = 0; j < 4; ++j) {
                if (cell.nodes[j] == i) {
                    matId = cell.materialId;
                    found = true;
                    break;
                }
            }
            if (found) break;
        }
        if (matId >= static_cast<Index>(materials_.size())) matId = 0;

        Vector3 q = -materials_[matId].conductivity * gradT[i];
        Scalar qNorm = q.norm();
        if (isNaN(qNorm)) qNorm = 0.0;
        state_.heatFlux(i) = std::min(qNorm, 1.0e8);
    }
}

void HeatConductionSolver::setBoundaryHeatFlux(Index boundaryId, const VectorX& flux) {
    Index start = grid_->boundaryFaceStart[boundaryId];
    Index end = grid_->boundaryFaceStart[boundaryId + 1];
    for (Index f = start; f < end; ++f) {
        const auto& face = grid_->boundaryFaces[f];
        for (int n = 0; n < 3; ++n) {
            Index idx = f - start;
            if (idx >= 0 && idx < flux.size()) {
                Scalar val = flux(idx);
                if (!isNaN(val)) {
                    state_.heatFlux(face.nodes[n]) = clamp(val, -1.0e8, 1.0e8);
                }
            }
        }
    }
}

void HeatConductionSolver::setBoundaryTemperature(Index boundaryId, const VectorX& temp) {
    Index start = grid_->boundaryFaceStart[boundaryId];
    Index end = grid_->boundaryFaceStart[boundaryId + 1];
    for (Index f = start; f < end; ++f) {
        const auto& face = grid_->boundaryFaces[f];
        for (int n = 0; n < 3; ++n) {
            Index idx = f - start;
            if (idx >= 0 && idx < temp.size()) {
                Scalar val = temp(idx);
                if (!isNaN(val)) {
                    state_.temperature(face.nodes[n]) = clamp(val,
                                                                config_.tempMinPhysical,
                                                                config_.tempMaxPhysical);
                }
            }
        }
    }
}

VectorX HeatConductionSolver::getBoundaryHeatFlux(Index boundaryId) const {
    Index start = grid_->boundaryFaceStart[boundaryId];
    Index end = grid_->boundaryFaceStart[boundaryId + 1];
    Index nFaces = end - start;
    VectorX flux(nFaces);
    flux.setZero();
    for (Index f = start; f < end; ++f) {
        const auto& face = grid_->boundaryFaces[f];
        Scalar avgFlux = 0.0;
        int cnt = 0;
        for (int n = 0; n < 3; ++n) {
            Scalar val = state_.heatFlux(face.nodes[n]);
            if (!isNaN(val)) {
                avgFlux += val;
                cnt++;
            }
        }
        if (cnt > 0) avgFlux /= cnt;
        flux(f - start) = avgFlux;
    }
    return flux;
}

VectorX HeatConductionSolver::getBoundaryTemperature(Index boundaryId) const {
    Index start = grid_->boundaryFaceStart[boundaryId];
    Index end = grid_->boundaryFaceStart[boundaryId + 1];
    Index nFaces = end - start;
    VectorX temp(nFaces);
    temp.setConstant((config_.tempMinPhysical + config_.tempMaxPhysical) * 0.5);
    for (Index f = start; f < end; ++f) {
        const auto& face = grid_->boundaryFaces[f];
        Scalar avgTemp = 0.0;
        int cnt = 0;
        for (int n = 0; n < 3; ++n) {
            Scalar val = state_.temperature(face.nodes[n]);
            if (!isNaN(val)) {
                avgTemp += val;
                cnt++;
            }
        }
        if (cnt > 0) avgTemp /= cnt;
        temp(f - start) = avgTemp;
    }
    return temp;
}

std::shared_ptr<HeatConductionSolver> createHeatConductionSolver(
    std::shared_ptr<Grid3D> grid,
    const std::vector<MaterialProperties>& materials,
    const FEMConfig& config) {
    auto solver = std::make_shared<HeatConductionSolver>(grid, materials, config);
    solver->initialize();
    return solver;
}

}
}
