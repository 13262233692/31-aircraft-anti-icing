#include "fem/heat_conduction_solver.h"
#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers>
#include <algorithm>
#include <cmath>

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
    state_.temperature.setConstant(initialTemp);
    state_.temperaturePrev.setConstant(initialTemp);
    state_.heatFlux.setZero();
    state_.iceThickness.setZero();
    state_.iceGrowthRate.setZero();
}

void HeatConductionSolver::computeTetStiffnessMatrix(const TetrahedronCell& cell,
                                                     const MaterialProperties& mat,
                                                     Eigen::Matrix<Scalar, 4, 4>& K) const {
    const auto& n = cell.nodes;
    Vector3 v0 = grid_->nodes[n[0]];
    Vector3 v1 = grid_->nodes[n[1]];
    Vector3 v2 = grid_->nodes[n[2]];
    Vector3 v3 = grid_->nodes[n[3]];

    Matrix3 J;
    J.col(0) = v1 - v0;
    J.col(1) = v2 - v0;
    J.col(2) = v3 - v0;

    Scalar detJ = J.determinant();
    Matrix3 invJ = J.inverse();

    Eigen::Matrix<Scalar, 3, 4> dN;
    dN.col(0) = -invJ.row(0).transpose() - invJ.row(1).transpose() - invJ.row(2).transpose();
    dN.col(1) = invJ.row(0).transpose();
    dN.col(2) = invJ.row(1).transpose();
    dN.col(3) = invJ.row(2).transpose();

    Eigen::Matrix<Scalar, 4, 4> K_iso = dN.transpose() * dN * std::abs(detJ) / 6.0;

    Matrix3 kT = mat.conductivity;
    K = dN.transpose() * kT * dN * std::abs(detJ) / 6.0;
}

void HeatConductionSolver::computeTetMassMatrix(const TetrahedronCell& cell,
                                                const MaterialProperties& mat,
                                                Eigen::Matrix<Scalar, 4, 4>& M) const {
    Scalar rhoCp = mat.density * mat.specificHeat;
    Scalar vol = std::abs(cell.volume);

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
            for (int j = 0; j < 4; ++j) {
                kTriplets[tid].emplace_back(cell.nodes[i], cell.nodes[j], K_e(i, j));
                mTriplets[tid].emplace_back(cell.nodes[i], cell.nodes[j], M_e(i, j));
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
    A_global_ = M_global_ / dt + theta * K_global_;

    VectorX T_prev = state_.temperaturePrev;
    VectorX T_curr = state_.temperature;

    R_global_ = (M_global_ / dt - (1.0 - theta) * K_global_) * T_prev;

    F_global_ = R_global_;
}

void HeatConductionSolver::applyBoundaryConditions(const std::vector<FEMBoundaryCondition>& bcs) {
    for (const auto& bc : bcs) {
        Index nid = bc.nodeId;
        switch (bc.type) {
        case FEMBoundaryCondition::Type::DIRICHLET: {
            for (SparseMatrix::InnerIterator it(A_global_, nid); it; ++it) {
                if (it.row() == nid) {
                    it.valueRef() = 1.0;
                } else {
                    F_global_(it.row()) -= it.valueRef() * bc.value;
                    it.valueRef() = 0.0;
                }
            }
            F_global_(nid) = bc.value;
            break;
        }
        case FEMBoundaryCondition::Type::NEUMANN: {
            F_global_(nid) += bc.value;
            break;
        }
        case FEMBoundaryCondition::Type::CONVECTION: {
            Scalar h_conv = bc.convectiveCoeff;
            Scalar T_inf = bc.ambientTemp;
            Scalar th = config_.theta;

            for (SparseMatrix::InnerIterator it(K_global_, nid); it; ++it) {
                if (it.row() == nid) {
                    A_global_.coeffRef(it.row(), it.col()) += th * h_conv;
                }
            }
            A_global_.coeffRef(nid, nid) += th * h_conv;

            Scalar h_neumann = h_conv * T_inf;
            F_global_(nid) += h_neumann;

            Scalar prev_contrib = (1.0 - th) * h_conv * (T_inf - state_.temperaturePrev(nid));
            F_global_(nid) += prev_contrib;
            break;
        }
        case FEMBoundaryCondition::Type::ICE_LATENT_HEAT: {
            Scalar T_freeze = 273.15;
            Scalar T_node = state_.temperature(nid);
            Scalar LWC = bc.value;
            Scalar V_inf = bc.convectiveCoeff;
            Scalar beta = bc.ambientTemp;

            Scalar m_impinge = LWC * V_inf * beta;
            Scalar q_latent = 0.0;

            if (T_node <= T_freeze) {
                Scalar freezing_fraction = std::min(1.0, (T_freeze - T_node) * 10.0);
                q_latent = m_impinge * LATENT_HEAT_ICE * freezing_fraction;

                Scalar ice_rate = m_impinge * freezing_fraction / ICE_DENSITY;
                state_.iceGrowthRate(nid) = ice_rate;
                state_.iceThickness(nid) += ice_rate * 0.0;
            } else {
                state_.iceGrowthRate(nid) = 0.0;
            }

            F_global_(nid) -= q_latent;
            break;
        }
        }
    }
}

void HeatConductionSolver::solve() {
    Eigen::SparseLU<SparseMatrix> solver;
    solver.analyzePattern(A_global_);
    solver.factorize(A_global_);

    if (solver.info() != Eigen::Success) {
        Eigen::BiCGSTAB<SparseMatrix, Eigen::IncompleteLUT<Scalar>> iterSolver;
        iterSolver.setMaxIterations(config_.maxIter);
        iterSolver.setTolerance(config_.tolerance);
        iterSolver.compute(A_global_);
        state_.temperature = iterSolver.solve(F_global_);
    } else {
        state_.temperature = solver.solve(F_global_);
    }

    computeNodalHeatFlux();
}

void HeatConductionSolver::advance(Scalar dt, const std::vector<FEMBoundaryCondition>& bcs) {
    state_.temperaturePrev = state_.temperature;
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
        Vector3 v1 = grid_->nodes[n[1]];
        Vector3 v2 = grid_->nodes[n[2]];
        Vector3 v3 = grid_->nodes[n[3]];

        Matrix3 J;
        J.col(0) = v1 - v0;
        J.col(1) = v2 - v0;
        J.col(2) = v3 - v0;
        Matrix3 invJ = J.inverse();

        Scalar T0 = field(n[0]);
        Scalar T1 = field(n[1]);
        Scalar T2 = field(n[2]);
        Scalar T3 = field(n[3]);

        Scalar dTdx = invJ(0, 0) * (T1 - T0) + invJ(0, 1) * (T2 - T0) + invJ(0, 2) * (T3 - T0);
        Scalar dTdy = invJ(1, 0) * (T1 - T0) + invJ(1, 1) * (T2 - T0) + invJ(1, 2) * (T3 - T0);
        Scalar dTdz = invJ(2, 0) * (T1 - T0) + invJ(2, 1) * (T2 - T0) + invJ(2, 2) * (T3 - T0);

        Vector3 elemGrad(dTdx, dTdy, dTdz);

        for (int i = 0; i < 4; ++i) {
            grad[n[i]] += elemGrad;
            count(n[i]) += 1.0;
        }
    }

    for (Index i = 0; i < grid_->nNodes; ++i) {
        if (count(i) > 0) {
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
        if (grid_->nCells > 0) {
            for (Index e = 0; e < grid_->nCells; ++e) {
                const auto& cell = grid_->cells[e];
                for (int j = 0; j < 4; ++j) {
                    if (cell.nodes[j] == i) {
                        matId = cell.materialId;
                        break;
                    }
                }
                if (matId > 0) break;
            }
        }
        Vector3 q = -materials_[matId].conductivity * gradT[i];
        state_.heatFlux(i) = q.norm();
    }
}

void HeatConductionSolver::setBoundaryHeatFlux(Index boundaryId, const VectorX& flux) {
    Index start = grid_->boundaryFaceStart[boundaryId];
    Index end = grid_->boundaryFaceStart[boundaryId + 1];
    for (Index f = start; f < end; ++f) {
        const auto& face = grid_->boundaryFaces[f];
        for (int n = 0; n < 3; ++n) {
            state_.heatFlux(face.nodes[n]) = flux(f - start);
        }
    }
}

void HeatConductionSolver::setBoundaryTemperature(Index boundaryId, const VectorX& temp) {
    Index start = grid_->boundaryFaceStart[boundaryId];
    Index end = grid_->boundaryFaceStart[boundaryId + 1];
    for (Index f = start; f < end; ++f) {
        const auto& face = grid_->boundaryFaces[f];
        for (int n = 0; n < 3; ++n) {
            state_.temperature(face.nodes[n]) = temp(f - start);
        }
    }
}

VectorX HeatConductionSolver::getBoundaryHeatFlux(Index boundaryId) const {
    Index start = grid_->boundaryFaceStart[boundaryId];
    Index end = grid_->boundaryFaceStart[boundaryId + 1];
    Index nFaces = end - start;
    VectorX flux(nFaces);
    for (Index f = start; f < end; ++f) {
        const auto& face = grid_->boundaryFaces[f];
        Scalar avgFlux = 0.0;
        for (int n = 0; n < 3; ++n) {
            avgFlux += state_.heatFlux(face.nodes[n]);
        }
        flux(f - start) = avgFlux / 3.0;
    }
    return flux;
}

VectorX HeatConductionSolver::getBoundaryTemperature(Index boundaryId) const {
    Index start = grid_->boundaryFaceStart[boundaryId];
    Index end = grid_->boundaryFaceStart[boundaryId + 1];
    Index nFaces = end - start;
    VectorX temp(nFaces);
    for (Index f = start; f < end; ++f) {
        const auto& face = grid_->boundaryFaces[f];
        Scalar avgTemp = 0.0;
        for (int n = 0; n < 3; ++n) {
            avgTemp += state_.temperature(face.nodes[n]);
        }
        temp(f - start) = avgTemp / 3.0;
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
