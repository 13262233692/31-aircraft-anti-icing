#include "levelset/level_set_solver.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace anti_icing {
namespace levelset {

LevelSetSolver::LevelSetSolver(const LevelSetConfig& config,
                               const IceShapeConfig& shapeConfig)
    : config_(config), shapeConfig_(shapeConfig),
      maxIceThickness_(0.0), totalIceVolume_(0.0), initialVolume_(0.0),
      stagnationIndex_(0), chordRef_(0.0), reinitCounter_(0) {
    phi_.initialize(config_.domainWidth, config_.domainHeight,
                    config_.gridSizeX, config_.gridSizeY);
    phiPrev_.initialize(config_.domainWidth, config_.domainHeight,
                        config_.gridSizeX, config_.gridSizeY);
    velocityX_.initialize(config_.domainWidth, config_.domainHeight,
                          config_.gridSizeX, config_.gridSizeY);
    velocityY_.initialize(config_.domainWidth, config_.domainHeight,
                          config_.gridSizeX, config_.gridSizeY);
    velocity_.initialize(config_.domainWidth, config_.domainHeight,
                         config_.gridSizeX, config_.gridSizeY);
    curvature_.initialize(config_.domainWidth, config_.domainHeight,
                          config_.gridSizeX, config_.gridSizeY);
}

void LevelSetSolver::initialize() {
    initializeAirfoil();
    computeSurfaceQuantities();
    initialVolume_ = totalIceVolume_;
}

Scalar LevelSetSolver::nacaAirfoil(Scalar x, Scalar y) const {
    Scalar c = shapeConfig_.chord;
    Scalar t = shapeConfig_.thickness / 100.0;
    Scalar m = (static_cast<int>(shapeConfig_.type) / 100) * 0.01;
    Scalar p = ((static_cast<int>(shapeConfig_.type) % 100) / 10) * 0.1;

    if (shapeConfig_.type == AirfoilType::CIRCULAR_CYLINDER) {
        Scalar cx = config_.domainWidth * 0.35;
        Scalar cy = config_.domainHeight * 0.5;
        Scalar r = shapeConfig_.leadingEdgeRadius;
        Scalar dx = x - cx;
        Scalar dy = y - cy;
        return std::sqrt(dx*dx + dy*dy) - r;
    }

    if (shapeConfig_.type == AirfoilType::ELLIPSE) {
        Scalar cx = config_.domainWidth * 0.35;
        Scalar cy = config_.domainHeight * 0.5;
        Scalar a = c * 0.5;
        Scalar b = shapeConfig_.leadingEdgeRadius;
        Scalar dx = (x - cx) / a;
        Scalar dy = (y - cy) / b;
        return (std::sqrt(dx*dx + dy*dy) - 1.0) * std::min(a, b);
    }

    Scalar xc = std::max(0.0, std::min(1.0, x / c));
    Scalar yt = 5.0 * t * (0.2969 * std::sqrt(xc) -
                           0.1260 * xc -
                           0.3516 * xc * xc +
                           0.2843 * xc * xc * xc -
                           0.1015 * xc * xc * xc * xc);

    Scalar yc = 0.0;
    Scalar dyc_dx = 0.0;
    if (m > 0 && p > 0) {
        if (xc <= p) {
            yc = m / (p * p) * (2.0 * p * xc - xc * xc);
            dyc_dx = 2.0 * m / (p * p) * (p - xc);
        } else {
            Scalar denom = (1.0 - p) * (1.0 - p);
            yc = m / denom * ((1.0 - 2.0 * p) + 2.0 * p * xc - xc * xc);
            dyc_dx = 2.0 * m / denom * (p - xc);
        }
    }

    Scalar theta = std::atan(dyc_dx);
    Scalar yu = yc + yt * std::cos(theta);
    Scalar yl = yc - yt * std::cos(theta);
    Scalar xu = xc - yt * std::sin(theta);
    Scalar xl = xc + yt * std::sin(theta);

    Scalar cy = config_.domainHeight * 0.5 + c * m;

    Scalar y_local = y - cy;

    Scalar d_upper = std::abs(y_local - yu * c);
    Scalar d_lower = std::abs(y_local + yl * c);

    Scalar d = std::min(d_upper, d_lower);
    Scalar sign = (y_local > yu * c || y_local < -yl * c) ? 1.0 : -1.0;

    return sign * d;
}

void LevelSetSolver::initializeAirfoil() {
    Scalar cx = config_.domainWidth * 0.35;
    Scalar cy = config_.domainHeight * 0.5;
    Scalar r = shapeConfig_.leadingEdgeRadius;

    chordRef_ = shapeConfig_.chord;

    parallel::parallelFor(config_.gridSizeY, [&](Index j) {
        for (Index i = 0; i < config_.gridSizeX; ++i) {
            Vector2 pos = phi_.coord(i, j);
            Scalar x = pos.x();
            Scalar y = pos.y();

            Scalar dx = x - cx;
            Scalar dy = y - cy;
            Scalar phi = std::sqrt(dx*dx + dy*dy) - r;

            if (shapeConfig_.type != AirfoilType::CIRCULAR_CYLINDER) {
                Scalar localPhi = nacaAirfoil(x - cx + r, y - cy + cy);
                phi = std::min(phi, localPhi);
            }

            if (shapeConfig_.initialIceAccretionFactor > 0.0) {
                Scalar theta = std::atan2(dy, dx);
                Scalar impinge = std::exp(-std::abs(theta) * 3.0 / shapeConfig_.impingementLimit);
                Scalar ice = shapeConfig_.initialIceAccretionFactor * r *
                             shapeConfig_.collectionEfficiencyPeak * impinge;
                phi -= ice;
            }

            phi_.at(i, j) = phi;
        }
    }, config_.wenoOrder > 0);

    phiPrev_ = phi_;
    reinitialize(config_.reinitializationSteps);
}

void LevelSetSolver::computeSignedDistance(const VectorX& surfaceCoords,
                                            const VectorX& surfaceNormals) {
    Index n = static_cast<Index>(surfaceCoords.size()) / 2;

    parallel::parallelFor(config_.gridSizeY, [&](Index j) {
        for (Index i = 0; i < config_.gridSizeX; ++i) {
            Vector2 pos = phi_.coord(i, j);
            Scalar minDist = 1.0e20;
            Scalar sign = 1.0;

            for (Index k = 0; k < n; ++k) {
                Vector2 sp(surfaceCoords(2*k), surfaceCoords(2*k+1));
                Scalar d = (pos - sp).norm();
                if (d < minDist) {
                    minDist = d;
                    Vector2 nrm(surfaceNormals(2*k), surfaceNormals(2*k+1));
                    sign = ((pos - sp).dot(nrm) >= 0) ? 1.0 : -1.0;
                }
            }

            phi_.at(i, j) = sign * minDist;
        }
    }, true);

    reinitialize();
}

Scalar LevelSetSolver::computeWENODerivativeX(const Grid2D& f,
                                               Index i, Index j,
                                               bool upwind) const {
    Index im2 = std::max(Index(0), i - 2);
    Index im1 = std::max(Index(0), i - 1);
    Index ip1 = std::min(config_.gridSizeX - 1, i + 1);
    Index ip2 = std::min(config_.gridSizeX - 1, i + 2);

    Scalar f_im2 = f.at(im2, j);
    Scalar f_im1 = f.at(im1, j);
    Scalar f_i   = f.at(i, j);
    Scalar f_ip1 = f.at(ip1, j);
    Scalar f_ip2 = f.at(ip2, j);

    Scalar eps = 1.0e-12;

    Scalar d1 = f_i - f_im1;
    Scalar d2 = f_im1 - f_im2;
    Scalar d3 = f_ip1 - f_i;
    Scalar d4 = f_ip2 - f_ip1;

    if (!upwind) {
        std::swap(d1, d3);
        std::swap(d2, d4);
        std::swap(f_i, f_ip1);
        std::swap(f_im1, f_i);
    }

    Scalar IS1 = (13.0/12.0) * (d1 - 2.0*d2 + d2) * (d1 - 2.0*d2 + d2) +
                 0.25 * (3.0*d1 - 4.0*d2) * (3.0*d1 - 4.0*d2);
    Scalar IS2 = (13.0/12.0) * (d3 - d1) * (d3 - d1) +
                 0.25 * (d1 + d3) * (d1 + d3);
    Scalar IS3 = (13.0/12.0) * (d4 - 2.0*d3 + d1) * (d4 - 2.0*d3 + d1) +
                 0.25 * (4.0*d3 - 3.0*d4) * (4.0*d3 - 3.0*d4);

    IS1 = 1.0 / ((eps + IS1) * (eps + IS1));
    IS2 = 1.0 / ((eps + IS2) * (eps + IS2));
    IS3 = 1.0 / ((eps + IS3) * (eps + IS3));

    Scalar w1 = IS1 / (IS1 + IS2 + IS3);
    Scalar w2 = IS2 / (IS1 + IS2 + IS3);
    Scalar w3 = IS3 / (IS1 + IS2 + IS3);

    w1 = std::max(0.0, std::min(1.0, w1));
    w2 = std::max(0.0, std::min(1.0, w2));
    w3 = std::max(0.0, std::min(1.0, w3));

    Scalar q1 = d1 / 2.0 - d2 / 2.0;
    Scalar q2 = d1 / 2.0 + d3 / 2.0;
    Scalar q3 = 3.0 * d3 / 2.0 - d4 / 2.0;

    Scalar deriv = (w1 * q1 + w2 * q2 + w3 * q3) * phi_.dxInv;

    if (config_.wenoOrder <= 2) {
        deriv = upwind ? (f_i - f_im1) * phi_.dxInv
                       : (f_ip1 - f_i) * phi_.dxInv;
    }

    return deriv;
}

Scalar LevelSetSolver::computeWENODerivativeY(const Grid2D& f,
                                               Index i, Index j,
                                               bool upwind) const {
    Index jm2 = std::max(Index(0), j - 2);
    Index jm1 = std::max(Index(0), j - 1);
    Index jp1 = std::min(config_.gridSizeY - 1, j + 1);
    Index jp2 = std::min(config_.gridSizeY - 1, j + 2);

    Scalar f_jm2 = f.at(i, jm2);
    Scalar f_jm1 = f.at(i, jm1);
    Scalar f_j   = f.at(i, j);
    Scalar f_jp1 = f.at(i, jp1);
    Scalar f_jp2 = f.at(i, jp2);

    Scalar eps = 1.0e-12;

    Scalar d1 = f_j - f_jm1;
    Scalar d2 = f_jm1 - f_jm2;
    Scalar d3 = f_jp1 - f_j;
    Scalar d4 = f_jp2 - f_jp1;

    if (!upwind) {
        std::swap(d1, d3);
        std::swap(d2, d4);
    }

    Scalar IS1 = (13.0/12.0) * (d1 - 2.0*d2) * (d1 - 2.0*d2) +
                 0.25 * (3.0*d1 - 4.0*d2) * (3.0*d1 - 4.0*d2);
    Scalar IS2 = (13.0/12.0) * (d3 - d1) * (d3 - d1) +
                 0.25 * (d1 + d3) * (d1 + d3);
    Scalar IS3 = (13.0/12.0) * (d4 - 2.0*d3) * (d4 - 2.0*d3) +
                 0.25 * (4.0*d3 - 3.0*d4) * (4.0*d3 - 3.0*d4);

    IS1 = 1.0 / ((eps + IS1) * (eps + IS1));
    IS2 = 1.0 / ((eps + IS2) * (eps + IS2));
    IS3 = 1.0 / ((eps + IS3) * (eps + IS3));

    Scalar w1 = IS1 / (IS1 + IS2 + IS3);
    Scalar w2 = IS2 / (IS1 + IS2 + IS3);
    Scalar w3 = IS3 / (IS1 + IS2 + IS3);

    Scalar q1 = d1 / 2.0 - d2 / 2.0;
    Scalar q2 = d1 / 2.0 + d3 / 2.0;
    Scalar q3 = 3.0 * d3 / 2.0 - d4 / 2.0;

    Scalar deriv = (w1 * q1 + w2 * q2 + w3 * q3) * phi_.dyInv;

    if (config_.wenoOrder <= 2) {
        deriv = upwind ? (f_j - f_jm1) * phi_.dyInv
                       : (f_jp1 - f_j) * phi_.dyInv;
    }

    return deriv;
}

Scalar LevelSetSolver::godunovFlux(Scalar phiL, Scalar phiR, Scalar u) const {
    Scalar d = (phiR - phiL) * 0.5;
    Scalar s = 0.5 * (phiL + phiR);

    if (u >= 0.0) {
        return u * phiL;
    } else {
        return u * phiR;
    }
}

void LevelSetSolver::advectLevelSet(Scalar dt, const Grid2D& velX,
                                     const Grid2D& velY) {
    phiPrev_ = phi_;

    parallel::parallelFor(config_.gridSizeY, [&](Index j) {
        for (Index i = 0; i < config_.gridSizeX; ++i) {
            Scalar u = velX.at(i, j);
            Scalar v = velY.at(i, j);

            bool uw_x = u >= 0.0;
            bool uw_y = v >= 0.0;

            Scalar dphi_dx = computeWENODerivativeX(phiPrev_, i, j, uw_x);
            Scalar dphi_dy = computeWENODerivativeY(phiPrev_, i, j, uw_y);

            if (!isNaN(dphi_dx) && !isNaN(dphi_dy)) {
                Scalar HJB = u * dphi_dx + v * dphi_dy;

                Scalar magGrad = std::sqrt(dphi_dx*dphi_dx + dphi_dy*dphi_dy);
                if (magGrad > 1.0e-10) {
                    Scalar v_n = u * (dphi_dx / magGrad) + v * (dphi_dy / magGrad);
                    HJB = v_n * magGrad;
                }

                phi_.at(i, j) = phiPrev_.at(i, j) - dt * HJB;
            }
        }
    }, true);

    for (Index j = 0; j < config_.gridSizeY; ++j) {
        phi_.at(0, j) = phi_.at(1, j);
        phi_.at(config_.gridSizeX - 1, j) = phi_.at(config_.gridSizeX - 2, j);
    }
    for (Index i = 0; i < config_.gridSizeX; ++i) {
        phi_.at(i, 0) = phi_.at(i, 1);
        phi_.at(i, config_.gridSizeY - 1) = phi_.at(i, config_.gridSizeY - 2);
    }
}

void LevelSetSolver::computeCurvature() {
    parallel::parallelFor(config_.gridSizeY, [&](Index j) {
        for (Index i = 0; i < config_.gridSizeX; ++i) {
            Index im1 = std::max(Index(0), i - 1);
            Index ip1 = std::min(config_.gridSizeX - 1, i + 1);
            Index jm1 = std::max(Index(0), j - 1);
            Index jp1 = std::min(config_.gridSizeY - 1, j + 1);

            Scalar dx = phi_.dx;
            Scalar dy = phi_.dy;

            Scalar phix = (phi_.at(ip1, j) - phi_.at(im1, j)) / (2.0 * dx);
            Scalar phiy = (phi_.at(i, jp1) - phi_.at(i, jm1)) / (2.0 * dy);
            Scalar phixx = (phi_.at(ip1, j) - 2.0 * phi_.at(i, j) + phi_.at(im1, j)) / (dx * dx);
            Scalar phiyy = (phi_.at(i, jp1) - 2.0 * phi_.at(i, j) + phi_.at(i, jm1)) / (dy * dy);
            Scalar phixy = (phi_.at(ip1, jp1) - phi_.at(ip1, jm1) - phi_.at(im1, jp1) + phi_.at(im1, jm1)) / (4.0 * dx * dy);

            Scalar magGrad = std::sqrt(phix*phix + phiy*phiy) + 1.0e-12;
            Scalar kappa = (phixx * phiy * phiy - 2.0 * phix * phiy * phixy +
                            phiyy * phix * phix) / (magGrad * magGrad * magGrad);

            curvature_.at(i, j) = kappa;
        }
    }, true);
}

void LevelSetSolver::extractSurfacePoints() {
    surfacePoints_.clear();
    surfaceNormals_.clear();
    surfaceCurvatures_.clear();

    Scalar threshold = config_.interfaceThreshold;
    std::vector<std::pair<Scalar, Vector2>> rawPoints;

    for (Index j = 0; j < config_.gridSizeY - 1; ++j) {
        for (Index i = 0; i < config_.gridSizeX - 1; ++i) {
            Scalar f00 = phi_.at(i, j);
            Scalar f10 = phi_.at(i + 1, j);
            Scalar f01 = phi_.at(i, j + 1);
            Scalar f11 = phi_.at(i + 1, j + 1);

            std::vector<Vector2> crossings;

            if (f00 * f10 < 0) {
                Scalar t = -f00 / (f10 - f00);
                crossings.emplace_back(static_cast<Scalar>(i) + t, static_cast<Scalar>(j));
            }
            if (f00 * f01 < 0) {
                Scalar t = -f00 / (f01 - f00);
                crossings.emplace_back(static_cast<Scalar>(i), static_cast<Scalar>(j) + t);
            }
            if (f10 * f11 < 0) {
                Scalar t = -f10 / (f11 - f10);
                crossings.emplace_back(static_cast<Scalar>(i + 1), static_cast<Scalar>(j) + t);
            }
            if (f01 * f11 < 0) {
                Scalar t = -f01 / (f11 - f01);
                crossings.emplace_back(static_cast<Scalar>(i) + t, static_cast<Scalar>(j + 1));
            }

            for (const auto& c : crossings) {
                Vector2 p(c.x() * phi_.dx + phi_.origin.x(),
                          c.y() * phi_.dy + phi_.origin.y());
                rawPoints.emplace_back(c.y() * config_.gridSizeX + c.x(), p);
            }
        }
    }

    std::sort(rawPoints.begin(), rawPoints.end(),
              [](const std::pair<Scalar, Vector2>& a,
                 const std::pair<Scalar, Vector2>& b) {
                  return a.first < b.first;
              });

    for (const auto& rp : rawPoints) {
        surfacePoints_.push_back(rp.second);
    }

    Index n = static_cast<Index>(surfacePoints_.size());
    surfaceNormals_.resize(n);
    surfaceCurvatures_.resize(n);

    for (Index k = 0; k < n; ++k) {
        Index km1 = (k > 0) ? k - 1 : k;
        Index kp1 = (k < n - 1) ? k + 1 : k;

        Vector2 tangent = surfacePoints_[kp1] - surfacePoints_[km1];
        tangent.normalize();

        Vector2 normal(-tangent.y(), tangent.x());
        surfaceNormals_[k] = normal;

        if (k > 0 && k < n - 1) {
            Vector2 t1 = surfacePoints_[k] - surfacePoints_[k - 1];
            Vector2 t2 = surfacePoints_[k + 1] - surfacePoints_[k];
            Scalar cross = t1.x() * t2.y() - t1.y() * t2.x();
            Scalar l1 = t1.norm();
            Scalar l2 = t2.norm();
            Scalar avgL = 0.5 * (l1 + l2) + 1.0e-20;
            surfaceCurvatures_[k] = 2.0 * cross / (l1 * l2 + avgL * avgL);
        } else {
            surfaceCurvatures_[k] = 0.0;
        }
    }
}

Scalar LevelSetSolver::interpolateAt(const Grid2D& field,
                                      Scalar x, Scalar y) const {
    Scalar fx = (x - field.origin.x()) * field.dxInv;
    Scalar fy = (y - field.origin.y()) * field.dyInv;

    Index i0 = static_cast<Index>(std::max(0.0, std::floor(fx)));
    Index j0 = static_cast<Index>(std::max(0.0, std::floor(fy)));
    Index i1 = std::min(field.nx - 1, i0 + 1);
    Index j1 = std::min(field.ny - 1, j0 + 1);

    Scalar sx = fx - static_cast<Scalar>(i0);
    Scalar sy = fy - static_cast<Scalar>(j0);

    Scalar f00 = field.at(i0, j0);
    Scalar f10 = field.at(i1, j0);
    Scalar f01 = field.at(i0, j1);
    Scalar f11 = field.at(i1, j1);

    Scalar f0 = (1.0 - sx) * f00 + sx * f10;
    Scalar f1 = (1.0 - sx) * f01 + sx * f11;
    Scalar result = (1.0 - sy) * f0 + sy * f1;

    return result;
}

void LevelSetSolver::reinitialize(Index nSteps) {
    if (nSteps == 0) nSteps = config_.reinitializationSteps;
    Scalar dt = config_.reinitializationCoeff * std::min(phi_.dx, phi_.dy);

    Grid2D phi0 = phi_;
    Grid2D& phi = phi_;

    for (Index step = 0; step < nSteps; ++step) {
        Grid2D phiOld = phi;

        parallel::parallelFor(config_.gridSizeY, [&](Index j) {
            for (Index i = 0; i < config_.gridSizeX; ++i) {
                Index im1 = std::max(Index(0), i - 1);
                Index ip1 = std::min(config_.gridSizeX - 1, i + 1);
                Index jm1 = std::max(Index(0), j - 1);
                Index jp1 = std::min(config_.gridSizeY - 1, j + 1);

                Scalar dx = phi.dx;
                Scalar dy = phi.dy;

                Scalar phi0_ij = phi0.at(i, j);
                Scalar sign = phi0_ij / std::sqrt(phi0_ij * phi0_ij + dx * dx);

                Scalar dphidx_m = (phiOld.at(i, j) - phiOld.at(im1, j)) / dx;
                Scalar dphidx_p = (phiOld.at(ip1, j) - phiOld.at(i, j)) / dx;
                Scalar dphidy_m = (phiOld.at(i, j) - phiOld.at(i, jm1)) / dy;
                Scalar dphidy_p = (phiOld.at(i, jp1) - phiOld.at(i, j)) / dy;

                Scalar a_p = std::max(std::max(dphidx_m, 0.0), -std::min(dphidx_p, 0.0));
                Scalar a_m = std::max(std::max(-dphidx_p, 0.0), -std::min(dphidx_m, 0.0));
                Scalar b_p = std::max(std::max(dphidy_m, 0.0), -std::min(dphidy_p, 0.0));
                Scalar b_m = std::max(std::max(-dphidy_p, 0.0), -std::min(dphidy_m, 0.0));

                Scalar grad_sq_pos = a_p * a_p + b_p * b_p;
                Scalar grad_sq_neg = a_m * a_m + b_m * b_m;

                Scalar sign_p = std::max(sign, 0.0);
                Scalar sign_m = std::min(sign, 0.0);

                Scalar rhs = (sign_p * std::sqrt(grad_sq_pos) + sign_m * std::sqrt(grad_sq_neg)) - sign;

                phi.at(i, j) = phiOld.at(i, j) - dt * rhs;
            }
        }, true);

        for (Index j = 0; j < config_.gridSizeY; ++j) {
            phi.at(0, j) = phi.at(1, j);
            phi.at(config_.gridSizeX - 1, j) = phi.at(config_.gridSizeX - 2, j);
        }
        for (Index i = 0; i < config_.gridSizeX; ++i) {
            phi.at(i, 0) = phi.at(i, 1);
            phi.at(i, config_.gridSizeY - 1) = phi.at(i, config_.gridSizeY - 2);
        }
    }
}

void LevelSetSolver::computeSurfaceQuantities() {
    extractSurfacePoints();
    computeCurvature();

    maxIceThickness_ = 0.0;
    totalIceVolume_ = 0.0;

    for (Index j = 0; j < config_.gridSizeY; ++j) {
        for (Index i = 0; i < config_.gridSizeX; ++i) {
            if (phi_.at(i, j) < 0) {
                totalIceVolume_ += phi_.dx * phi_.dy;
            }
        }
    }

    Index n = static_cast<Index>(surfacePoints_.size());
    stagnationIndex_ = 0;
    Scalar cx = config_.domainWidth * 0.35;
    Scalar minDist = 1.0e20;
    for (Index k = 0; k < n; ++k) {
        Scalar d = (surfacePoints_[k] - Vector2(cx, config_.domainHeight * 0.5)).norm();
        if (d < minDist) {
            minDist = d;
            stagnationIndex_ = k;
        }

        Scalar thickness = -phi_.at(
            std::min(config_.gridSizeX - 1,
                     static_cast<Index>(surfacePoints_[k].x() * phi_.dxInv)),
            std::min(config_.gridSizeY - 1,
                     static_cast<Index>(surfacePoints_[k].y() * phi_.dyInv))
        );
        if (thickness > maxIceThickness_) {
            maxIceThickness_ = thickness;
        }
    }
}

void LevelSetSolver::computeNormalVelocity(const IceAccretionData& accretion) {
    Index n = numSurfacePoints();
    velocity_.data.setZero();
    velocityX_.data.setZero();
    velocityY_.data.setZero();

    parallel::parallelFor(n, [&](Index k) {
        if (k >= static_cast<Index>(accretion.iceGrowthRate.size())) return;

        Scalar iceRate = accretion.iceGrowthRate(k);
        Scalar beta = (k < static_cast<Index>(accretion.collectionEfficiency.size()))
                      ? accretion.collectionEfficiency(k) : 0.5;

        if (isNaN(iceRate)) iceRate = 0.0;

        iceRate = clamp(iceRate, 0.0, 1.0e-2);
        beta = clamp(beta, 0.0, 1.0);

        Vector2 normal = surfaceNormals_[k];
        Vector2 point = surfacePoints_[k];

        Scalar v_n = iceRate * beta;

        Index i = static_cast<Index>(clamp(point.x() * phi_.dxInv, 0.0,
                                           static_cast<Scalar>(config_.gridSizeX - 1)));
        Index j = static_cast<Index>(clamp(point.y() * phi_.dyInv, 0.0,
                                           static_cast<Scalar>(config_.gridSizeY - 1)));

        Scalar bandWidth = 5.0 * std::max(phi_.dx, phi_.dy);
        Scalar d = std::abs(phi_.at(i, j));

        if (d < bandWidth) {
            Scalar weight = 1.0 - d / bandWidth;
            weight = weight * weight * (3.0 - 2.0 * weight);

            for (Index di = -3; di <= 3; ++di) {
                for (Index dj = -3; dj <= 3; ++dj) {
                    Index ii = i + di;
                    Index jj = j + dj;
                    if (ii >= 0 && ii < config_.gridSizeX &&
                        jj >= 0 && jj < config_.gridSizeY) {
                        Scalar dist = std::sqrt(static_cast<Scalar>(di*di) * phi_.dx * phi_.dx +
                                                static_cast<Scalar>(dj*dj) * phi_.dy * phi_.dy);
                        Scalar w = std::exp(-dist * dist / (2.0 * bandWidth * bandWidth * 0.25));
                        w *= weight;
                        velocityX_.at(ii, jj) += normal.x() * v_n * w;
                        velocityY_.at(ii, jj) += normal.y() * v_n * w;
                        velocity_.at(ii, jj) += w;
                    }
                }
            }
        }
    }, true);

    for (Index j = 0; j < config_.gridSizeY; ++j) {
        for (Index i = 0; i < config_.gridSizeX; ++i) {
            Scalar w = velocity_.at(i, j);
            if (w > 1.0e-20) {
                velocityX_.at(i, j) /= w;
                velocityY_.at(i, j) /= w;
            } else {
                velocityX_.at(i, j) = 0.0;
                velocityY_.at(i, j) = 0.0;
            }

            if (isNaN(velocityX_.at(i, j))) velocityX_.at(i, j) = 0.0;
            if (isNaN(velocityY_.at(i, j))) velocityY_.at(i, j) = 0.0;
        }
    }
}

void LevelSetSolver::advance(Scalar dt, const VectorX& normalVelocity) {
    velocity_.data.setZero();
    velocityX_.data.setZero();
    velocityY_.data.setZero();

    Index n = numSurfacePoints();
    for (Index k = 0; k < n && k < normalVelocity.size(); ++k) {
        Scalar v_n = clamp(normalVelocity(k), -1.0e-3, 1.0e-3);
        Vector2 normal = surfaceNormals_[k];
        Vector2 point = surfacePoints_[k];

        Index i = static_cast<Index>(clamp(point.x() * phi_.dxInv, 0.0,
                                           static_cast<Scalar>(config_.gridSizeX - 1)));
        Index j = static_cast<Index>(clamp(point.y() * phi_.dyInv, 0.0,
                                           static_cast<Scalar>(config_.gridSizeY - 1)));

        for (Index di = -2; di <= 2; ++di) {
            for (Index dj = -2; dj <= 2; ++dj) {
                Index ii = i + di;
                Index jj = j + dj;
                if (ii >= 0 && ii < config_.gridSizeX &&
                    jj >= 0 && jj < config_.gridSizeY) {
                    Scalar dist2 = static_cast<Scalar>(di*di) + static_cast<Scalar>(dj*dj);
                    Scalar w = std::exp(-dist2 * 0.5);
                    velocityX_.at(ii, jj) += normal.x() * v_n * w;
                    velocityY_.at(ii, jj) += normal.y() * v_n * w;
                    velocity_.at(ii, jj) += w;
                }
            }
        }
    }

    for (Index j = 0; j < config_.gridSizeY; ++j) {
        for (Index i = 0; i < config_.gridSizeX; ++i) {
            Scalar w = velocity_.at(i, j);
            if (w > 1.0e-20) {
                velocityX_.at(i, j) /= w;
                velocityY_.at(i, j) /= w;
            } else {
                velocityX_.at(i, j) = 0.0;
                velocityY_.at(i, j) = 0.0;
            }
        }
    }

    Scalar maxV = 0.0;
    for (Index i = 0; i < velocityX_.data.size(); ++i) {
        Scalar v = std::sqrt(velocityX_.data(i)*velocityX_.data(i) +
                             velocityY_.data(i)*velocityY_.data(i));
        if (v > maxV) maxV = v;
    }

    if (maxV > 1.0e-10) {
        Scalar dt_cfl = config_.CFL * std::min(phi_.dx, phi_.dy) / maxV;
        Scalar actualDt = std::min(dt, dt_cfl);

        Index subSteps = static_cast<Index>(std::ceil(dt / std::max(actualDt, 1.0e-20)));
        subSteps = std::max(Index(1), std::min(Index(50), subSteps));
        Scalar subDt = dt / static_cast<Scalar>(subSteps);

        for (Index s = 0; s < subSteps; ++s) {
            advectLevelSet(subDt, velocityX_, velocityY_);
        }
    }

    reinitCounter_++;
    if (reinitCounter_ % 5 == 0) {
        reinitialize();
        reinitCounter_ = 0;
    }

    computeSurfaceQuantities();

    if (config_.massConservation) {
        Scalar currentVol = 0.0;
        for (Index j = 0; j < config_.gridSizeY; ++j) {
            for (Index i = 0; i < config_.gridSizeX; ++i) {
                if (phi_.at(i, j) < 0) currentVol += phi_.dx * phi_.dy;
            }
        }

        Scalar targetVol = initialVolume_ + totalIceVolume_;
        Scalar volDiff = (targetVol - currentVol) * config_.massCorrectionFactor;

        if (std::abs(volDiff) > 1.0e-14) {
            for (Index j = 0; j < config_.gridSizeY; ++j) {
                for (Index i = 0; i < config_.gridSizeX; ++i) {
                    Scalar magGrad = 1.0;
                    Index im1 = std::max(Index(0), i - 1);
                    Index ip1 = std::min(config_.gridSizeX - 1, i + 1);
                    Index jm1 = std::max(Index(0), j - 1);
                    Index jp1 = std::min(config_.gridSizeY - 1, j + 1);
                    Scalar px = (phi_.at(ip1, j) - phi_.at(im1, j)) * 0.5 * phi_.dxInv;
                    Scalar py = (phi_.at(i, jp1) - phi_.at(i, jm1)) * 0.5 * phi_.dyInv;
                    magGrad = std::sqrt(px*px + py*py) + 1.0e-12;

                    Scalar bandWidth = 3.0 * std::max(phi_.dx, phi_.dy);
                    Scalar dist = std::abs(phi_.at(i, j));
                    if (dist < bandWidth) {
                        phi_.at(i, j) -= volDiff / (magGrad * config_.gridSizeX * config_.gridSizeY) * 0.01;
                    }
                }
            }
        }
    }
}

void LevelSetSolver::updateIceShape(Scalar dt, const IceAccretionData& accretion) {
    computeNormalVelocity(accretion);

    VectorX velN(numSurfacePoints());
    Index n = numSurfacePoints();
    for (Index k = 0; k < n; ++k) {
        Scalar iceRate = (k < static_cast<Index>(accretion.iceGrowthRate.size()))
                         ? accretion.iceGrowthRate(k) : 0.0;
        Scalar beta = (k < static_cast<Index>(accretion.collectionEfficiency.size()))
                      ? accretion.collectionEfficiency(k) : 0.0;
        velN(k) = iceRate * beta * dt;
    }

    advance(dt, velN);
}

bool LevelSetSolver::needsRemeshing() const {
    return maxIceThickness_ > config_.minInterfaceWidth * 2.0;
}

bool LevelSetSolver::hasExceededThreshold(Scalar threshold) const {
    return maxIceThickness_ > threshold;
}

Vector2 LevelSetSolver::getSurfacePoint(Index i) const {
    if (i >= 0 && i < static_cast<Index>(surfacePoints_.size())) {
        return surfacePoints_[i];
    }
    return Vector2::Zero();
}

Vector2 LevelSetSolver::getSurfaceNormal(Index i) const {
    if (i >= 0 && i < static_cast<Index>(surfaceNormals_.size())) {
        return surfaceNormals_[i];
    }
    return Vector2(1.0, 0.0);
}

Scalar LevelSetSolver::getSurfaceCurvature(Index i) const {
    if (i >= 0 && i < static_cast<Index>(surfaceCurvatures_.size())) {
        return surfaceCurvatures_[i];
    }
    return 0.0;
}

Index LevelSetSolver::findStagnationPoint() const {
    return stagnationIndex_;
}

Scalar LevelSetSolver::leadingEdgeRadius() const {
    if (stagnationIndex_ >= static_cast<Index>(surfaceCurvatures_.size()) ||
        surfaceCurvatures_.empty()) {
        return shapeConfig_.leadingEdgeRadius;
    }

    Scalar kappa = surfaceCurvatures_[stagnationIndex_];
    if (std::abs(kappa) < 1.0e-10) return shapeConfig_.leadingEdgeRadius;

    Scalar r = 1.0 / std::abs(kappa);
    return clamp(r, shapeConfig_.leadingEdgeRadius * 0.5,
                  shapeConfig_.leadingEdgeRadius * 5.0);
}

Scalar LevelSetSolver::computeConvectiveCoeffModifier() const {
    Scalar rNew = leadingEdgeRadius();
    Scalar r0 = shapeConfig_.leadingEdgeRadius;

    Scalar ratio = rNew / std::max(r0, 1.0e-10);
    Scalar ReRatio = std::pow(ratio, 0.5);
    Scalar NuRatio = ReRatio * 0.5;

    return 1.0 / std::max(NuRatio, 0.1);
}

VectorX LevelSetSolver::getBoundaryConvectionFactors(Index nPoints) const {
    VectorX factors(nPoints);
    Index n = numSurfacePoints();
    Scalar baseMod = computeConvectiveCoeffModifier();

    for (Index i = 0; i < nPoints; ++i) {
        Scalar ratio = static_cast<Scalar>(i) / std::max(Index(1), nPoints - 1);
        Index k = static_cast<Index>(ratio * static_cast<Scalar>(n - 1));
        k = clamp(k, Index(0), n - 1);

        Scalar kappa = (k < static_cast<Index>(surfaceCurvatures_.size()))
                       ? surfaceCurvatures_[k] : 0.0;
        Scalar curvatureFactor = 1.0 / (1.0 + 2.0 * std::abs(kappa) * shapeConfig_.leadingEdgeRadius);

        Scalar thicknessFactor = 1.0;
        if (maxIceThickness_ > 1.0e-10) {
            Scalar localThickness = -interpolateAt(
                phi_,
                surfacePoints_[k].x(),
                surfacePoints_[k].y()
            );
            thicknessFactor = 1.0 / (1.0 + 3.0 * localThickness / shapeConfig_.leadingEdgeRadius);
        }

        factors(i) = baseMod * curvatureFactor * thicknessFactor;
        factors(i) = clamp(factors(i), 0.1, 10.0);
    }

    return factors;
}

VectorX LevelSetSolver::getBoundaryTemperature(Index nPoints) const {
    VectorX temps(nPoints);
    Index n = numSurfacePoints();

    for (Index i = 0; i < nPoints; ++i) {
        Scalar ratio = static_cast<Scalar>(i) / std::max(Index(1), nPoints - 1);
        Index k = static_cast<Index>(ratio * static_cast<Scalar>(n - 1));
        k = clamp(k, Index(0), n - 1);

        Vector2 p = surfacePoints_[k];
        Scalar localT = interpolateAt(phi_, p.x(), p.y());
        Scalar iceT = 273.15 - 1.0;
        temps(i) = iceT - localT * 0.1;
        temps(i) = clamp(temps(i), 150.0, 300.0);
    }

    return temps;
}

LevelSetConfig createDefaultLevelSetConfig() {
    LevelSetConfig c;
    c.domainWidth = 0.6;
    c.domainHeight = 0.4;
    c.gridSizeX = 150;
    c.gridSizeY = 100;
    c.CFL = 0.5;
    c.wenoOrder = 5;
    c.reinitializationCoeff = 0.3;
    c.reinitializationSteps = 10;
    c.interfaceThreshold = 1.0e-10;
    c.minInterfaceWidth = 5.0e-3;
    c.useWENO = true;
    c.useSubcellFix = true;
    c.massConservation = true;
    c.massCorrectionFactor = 0.3;
    return c;
}

IceShapeConfig createDefaultIceShapeConfig() {
    IceShapeConfig c;
    c.type = AirfoilType::CIRCULAR_CYLINDER;
    c.chord = 0.5;
    c.thickness = 12.0;
    c.leadingEdgeRadius = 0.05;
    c.incidenceAngle = 3.0;
    c.initialIceAccretionFactor = 0.0;
    c.maxIceThickness = 0.02;
    c.roughnessHeight = 5.0e-5;
    c.impingementLimit = 3.5;
    c.collectionEfficiencyPeak = 0.9;
    c.stagnationPointOffset = 0.0;
    return c;
}

std::shared_ptr<LevelSetSolver> createLevelSetSolver(
    const LevelSetConfig& config,
    const IceShapeConfig& shapeConfig) {
    auto solver = std::make_shared<LevelSetSolver>(config, shapeConfig);
    solver->initialize();
    return solver;
}

}
}
