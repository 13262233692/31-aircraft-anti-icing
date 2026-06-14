#pragma once

#include "common/types.h"
#include "common/parallel_utils.h"
#include <memory>
#include <vector>
#include <array>

namespace anti_icing {
namespace levelset {

struct Grid2D {
    Index nx;
    Index ny;
    Scalar dx;
    Scalar dy;
    Scalar dxInv;
    Scalar dyInv;
    Vector2 origin;
    Scalar width;
    Scalar height;

    VectorX data;

    void initialize(Scalar w, Scalar h, Index nx_, Index ny_) {
        nx = nx_;
        ny = ny_;
        dx = w / static_cast<Scalar>(nx - 1);
        dy = h / static_cast<Scalar>(ny - 1);
        dxInv = 1.0 / dx;
        dyInv = 1.0 / dy;
        width = w;
        height = h;
        origin = Vector2(0.0, 0.0);
        data.resize(nx * ny);
        data.setZero();
    }

    Scalar& at(Index i, Index j) { return data(j * nx + i); }
    Scalar at(Index i, Index j) const { return data(j * nx + i); }

    Index index(Index i, Index j) const { return j * nx + i; }

    Vector2 coord(Index i, Index j) const {
        return Vector2(origin.x() + static_cast<Scalar>(i) * dx,
                       origin.y() + static_cast<Scalar>(j) * dy);
    }

    void setOrigin(const VectorX& d) { data = d; }
};

enum class AirfoilType {
    NACA0012,
    NACA2412,
    CIRCULAR_CYLINDER,
    ELLIPSE
};

struct IceShapeConfig {
    AirfoilType type;
    Scalar chord;
    Scalar thickness;
    Scalar leadingEdgeRadius;
    Scalar incidenceAngle;

    Scalar initialIceAccretionFactor;
    Scalar maxIceThickness;
    Scalar roughnessHeight;

    Scalar impingementLimit;
    Scalar collectionEfficiencyPeak;
    Scalar stagnationPointOffset;
};

struct LevelSetConfig {
    Scalar domainWidth;
    Scalar domainHeight;
    Index  gridSizeX;
    Index  gridSizeY;

    Scalar CFL;
    Index  wenoOrder;
    Scalar reinitializationCoeff;
    Index  reinitializationSteps;

    Scalar interfaceThreshold;
    Scalar minInterfaceWidth;

    bool   useWENO;
    bool   useSubcellFix;
    bool   massConservation;
    Scalar massCorrectionFactor;
};

struct IceAccretionData {
    VectorX iceGrowthRate;
    VectorX collectionEfficiency;
    VectorX surfaceHeatFlux;
    VectorX surfaceTemperature;
    VectorX surfaceCurvature;
    Scalar totalIceVolume;
    Scalar maxIceThickness;
    Scalar icedAreaFraction;
};

class LevelSetSolver {
public:
    LevelSetSolver(const LevelSetConfig& config,
                   const IceShapeConfig& shapeConfig);

    void initialize();
    void initializeAirfoil();
    void computeSignedDistance(const VectorX& surfaceCoords, const VectorX& surfaceNormals);
    void advance(Scalar dt, const VectorX& normalVelocity);

    void reinitialize(Index nSteps = 0);

    void computeSurfaceQuantities();

    void computeNormalVelocity(const IceAccretionData& accretion);

    void updateIceShape(Scalar dt, const IceAccretionData& accretion);

    bool needsRemeshing() const;

    bool hasExceededThreshold(Scalar threshold) const;

    const Grid2D& phi() const { return phi_; }
    const Grid2D& velocityField() const { return velocity_; }
    const Grid2D& curvature() const { return curvature_; }

    Vector2 getSurfacePoint(Index i) const;
    Vector2 getSurfaceNormal(Index i) const;
    Scalar getSurfaceCurvature(Index i) const;
    Index numSurfacePoints() const { return surfacePoints_.size(); }

    const std::vector<Vector2>& surfacePoints() const { return surfacePoints_; }
    const std::vector<Vector2>& surfaceNormals() const { return surfaceNormals_; }

    Scalar maxIceThickness() const { return maxIceThickness_; }
    Scalar totalIceVolume() const { return totalIceVolume_; }

    Scalar computeConvectiveCoeffModifier() const;

    VectorX getBoundaryConvectionFactors(Index nPoints) const;

    VectorX getBoundaryTemperature(Index nPoints) const;

    Index findStagnationPoint() const;

    Scalar leadingEdgeRadius() const;

private:
    Scalar computeWENODerivativeX(const Grid2D& f, Index i, Index j, bool upwind) const;
    Scalar computeWENODerivativeY(const Grid2D& f, Index i, Index j, bool upwind) const;

    Scalar godunovFlux(Scalar phiL, Scalar phiR,
                     Scalar u) const;

    void advectLevelSet(Scalar dt, const Grid2D& velX, const Grid2D& velY);

    void computeCurvature();

    void extractSurfacePoints();

    Scalar interpolateAt(const Grid2D& field, Scalar x, Scalar y) const;

    Scalar nacaAirfoil(Scalar x, Scalar y) const;

    LevelSetConfig config_;
    IceShapeConfig shapeConfig_;

    Grid2D phi_;
    Grid2D phiPrev_;
    Grid2D velocityX_;
    Grid2D velocityY_;
    Grid2D velocity_;
    Grid2D curvature_;

    std::vector<Vector2> surfacePoints_;
    std::vector<Vector2> surfaceNormals_;
    std::vector<Scalar> surfaceCurvatures_;

    Scalar maxIceThickness_;
    Scalar totalIceVolume_;
    Scalar initialVolume_;

    Index stagnationIndex_;
    Scalar chordRef_;

    Index reinitCounter_;
};

std::shared_ptr<LevelSetSolver> createLevelSetSolver(
    const LevelSetConfig& config,
    const IceShapeConfig& shapeConfig);

LevelSetConfig createDefaultLevelSetConfig();
IceShapeConfig createDefaultIceShapeConfig();

}
}
