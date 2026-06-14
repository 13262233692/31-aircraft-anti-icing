#pragma once

#include "common/types.h"
#include "common/parallel_utils.h"
#include <vector>
#include <array>
#include <memory>

namespace anti_icing {

struct Grid1D {
    Index nCells;
    Index nFaces;
    Index nNodes;

    VectorX x;
    VectorX xFace;
    VectorX dx;
    VectorX dxInv;
    VectorX faceArea;
    VectorX cellVolume;

    void initialize(Scalar L, Index n) {
        nCells = n;
        nFaces = n + 1;
        nNodes = n + 1;

        x.resize(nCells);
        xFace.resize(nFaces);
        dx.resize(nCells);
        dxInv.resize(nCells);
        faceArea.resize(nFaces);
        cellVolume.resize(nCells);

        Scalar h = L / static_cast<Scalar>(nCells);

        for (Index i = 0; i < nFaces; ++i) {
            xFace(i) = static_cast<Scalar>(i) * h;
        }

        for (Index i = 0; i < nCells; ++i) {
            x(i) = 0.5 * (xFace(i) + xFace(i + 1));
            dx(i) = xFace(i + 1) - xFace(i);
            dxInv(i) = 1.0 / dx(i);
        }

        faceArea.setConstant(1.0);
        cellVolume = dx;
    }
};

struct TetrahedronCell {
    std::array<Index, 4> nodes;
    Index materialId;
    Scalar volume;
    Matrix3 invJacobian;
};

struct BoundaryFace {
    std::array<Index, 3> nodes;
    Index boundaryId;
    Vector3 normal;
    Scalar area;
    Vector3 centroid;
};

struct Grid3D {
    Index nNodes;
    Index nCells;
    Index nBoundaryFaces;
    Index nBoundaries;

    Vector3Array nodes;
    std::vector<TetrahedronCell> cells;
    std::vector<BoundaryFace> boundaryFaces;
    std::vector<Index> boundaryFaceStart;

    Vector3Array cellCentroids;
    VectorX cellVolumes;

    void computeGeometricQuantities(bool useParallel = true) {
        cellCentroids.resize(nCells);
        cellVolumes.resize(nCells);

        parallel::parallelFor(nCells, [&](Index i) {
            const auto& cell = cells[i];
            Vector3 c = Vector3::Zero();
            for (int j = 0; j < 4; ++j) {
                c += nodes[cell.nodes[j]];
            }
            c /= 4.0;

            Vector3 v0 = nodes[cell.nodes[0]];
            Vector3 v1 = nodes[cell.nodes[1]] - v0;
            Vector3 v2 = nodes[cell.nodes[2]] - v0;
            Vector3 v3 = nodes[cell.nodes[3]] - v0;

            Scalar vol = v1.dot(v2.cross(v3)) / 6.0;

            Matrix3 J;
            J.col(0) = v1;
            J.col(1) = v2;
            J.col(2) = v3;

            cellCentroids[i] = c;
            cellVolumes(i) = std::abs(vol);
            cells[i].volume = std::abs(vol);
            cells[i].invJacobian = J.inverse();
        }, useParallel);
    }

    void computeBoundaryNormals(bool useParallel = true) {
        parallel::parallelFor(nBoundaryFaces, [&](Index i) {
            const auto& face = boundaryFaces[i];
            Vector3 v0 = nodes[face.nodes[0]];
            Vector3 v1 = nodes[face.nodes[1]] - v0;
            Vector3 v2 = nodes[face.nodes[2]] - v0;

            Vector3 n = v1.cross(v2);
            Scalar area = 0.5 * n.norm();
            n.normalize();

            Vector3 c = (nodes[face.nodes[0]] + nodes[face.nodes[1]] + nodes[face.nodes[2]]) / 3.0;

            boundaryFaces[i].normal = n;
            boundaryFaces[i].area = area;
            boundaryFaces[i].centroid = c;
        }, useParallel);
    }
};

std::shared_ptr<Grid3D> createStructuredTetGrid(Scalar Lx, Scalar Ly, Scalar Lz,
                                                 Index nx, Index ny, Index nz);

}
