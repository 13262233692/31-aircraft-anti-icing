#include "common/grid.h"

namespace anti_icing {

std::shared_ptr<Grid3D> createStructuredTetGrid(Scalar Lx, Scalar Ly, Scalar Lz,
                                                 Index nx, Index ny, Index nz) {
    auto grid = std::make_shared<Grid3D>();

    grid->nNodes = (nx + 1) * (ny + 1) * (nz + 1);
    grid->nCells = 6 * nx * ny * nz;

    grid->nodes.reserve(grid->nNodes);

    Scalar dx = Lx / static_cast<Scalar>(nx);
    Scalar dy = Ly / static_cast<Scalar>(ny);
    Scalar dz = Lz / static_cast<Scalar>(nz);

    auto idx = [&](Index i, Index j, Index k) {
        return i + (nx + 1) * (j + (ny + 1) * k);
    };

    for (Index k = 0; k <= nz; ++k) {
        for (Index j = 0; j <= ny; ++j) {
            for (Index i = 0; i <= nx; ++i) {
                grid->nodes.emplace_back(
                    static_cast<Scalar>(i) * dx,
                    static_cast<Scalar>(j) * dy,
                    static_cast<Scalar>(k) * dz
                );
            }
        }
    }

    grid->cells.reserve(grid->nCells);
    for (Index k = 0; k < nz; ++k) {
        for (Index j = 0; j < ny; ++j) {
            for (Index i = 0; i < nx; ++i) {
                Index n000 = idx(i, j, k);
                Index n100 = idx(i + 1, j, k);
                Index n010 = idx(i, j + 1, k);
                Index n110 = idx(i + 1, j + 1, k);
                Index n001 = idx(i, j, k + 1);
                Index n101 = idx(i + 1, j, k + 1);
                Index n011 = idx(i, j + 1, k + 1);
                Index n111 = idx(i + 1, j + 1, k + 1);

                grid->cells.push_back({{n000, n100, n110, n111}, 0, 0.0, Matrix3::Zero()});
                grid->cells.push_back({{n000, n110, n010, n011}, 0, 0.0, Matrix3::Zero()});
                grid->cells.push_back({{n000, n111, n101, n100}, 0, 0.0, Matrix3::Zero()});
                grid->cells.push_back({{n000, n111, n011, n001}, 0, 0.0, Matrix3::Zero()});
                grid->cells.push_back({{n000, n111, n011, n010}, 0, 0.0, Matrix3::Zero()});
                grid->cells.push_back({{n000, n111, n101, n100}, 0, 0.0, Matrix3::Zero()});
            }
        }
    }

    grid->nBoundaries = 6;
    grid->boundaryFaceStart.resize(grid->nBoundaries + 1, 0);

    std::vector<std::vector<BoundaryFace>> boundaryGroups(6);

    for (Index j = 0; j < ny; ++j) {
        for (Index k = 0; k < nz; ++k) {
            boundaryGroups[0].push_back({{idx(0, j, k), idx(0, j + 1, k), idx(0, j, k + 1)}, 0, Vector3::Zero(), 0.0, Vector3::Zero()});
            boundaryGroups[0].push_back({{idx(0, j + 1, k + 1), idx(0, j, k + 1), idx(0, j + 1, k)}, 0, Vector3::Zero(), 0.0, Vector3::Zero()});
        }
    }
    boundaryGroups[0].shrink_to_fit();
    for (auto& f : boundaryGroups[0]) f.boundaryId = 0;

    for (Index j = 0; j < ny; ++j) {
        for (Index k = 0; k < nz; ++k) {
            boundaryGroups[1].push_back({{idx(nx, j, k), idx(nx, j, k + 1), idx(nx, j + 1, k)}, 1, Vector3::Zero(), 0.0, Vector3::Zero()});
            boundaryGroups[1].push_back({{idx(nx, j + 1, k + 1), idx(nx, j + 1, k), idx(nx, j, k + 1)}, 1, Vector3::Zero(), 0.0, Vector3::Zero()});
        }
    }
    boundaryGroups[1].shrink_to_fit();
    for (auto& f : boundaryGroups[1]) f.boundaryId = 1;

    for (Index i = 0; i < nx; ++i) {
        for (Index k = 0; k < nz; ++k) {
            boundaryGroups[2].push_back({{idx(i, 0, k), idx(i, 0, k + 1), idx(i + 1, 0, k)}, 2, Vector3::Zero(), 0.0, Vector3::Zero()});
            boundaryGroups[2].push_back({{idx(i + 1, 0, k + 1), idx(i + 1, 0, k), idx(i, 0, k + 1)}, 2, Vector3::Zero(), 0.0, Vector3::Zero()});
        }
    }
    boundaryGroups[2].shrink_to_fit();
    for (auto& f : boundaryGroups[2]) f.boundaryId = 2;

    for (Index i = 0; i < nx; ++i) {
        for (Index k = 0; k < nz; ++k) {
            boundaryGroups[3].push_back({{idx(i, ny, k), idx(i + 1, ny, k), idx(i, ny, k + 1)}, 3, Vector3::Zero(), 0.0, Vector3::Zero()});
            boundaryGroups[3].push_back({{idx(i + 1, ny, k + 1), idx(i, ny, k + 1), idx(i + 1, ny, k)}, 3, Vector3::Zero(), 0.0, Vector3::Zero()});
        }
    }
    boundaryGroups[3].shrink_to_fit();
    for (auto& f : boundaryGroups[3]) f.boundaryId = 3;

    for (Index i = 0; i < nx; ++i) {
        for (Index j = 0; j < ny; ++j) {
            boundaryGroups[4].push_back({{idx(i, j, 0), idx(i + 1, j, 0), idx(i, j + 1, 0)}, 4, Vector3::Zero(), 0.0, Vector3::Zero()});
            boundaryGroups[4].push_back({{idx(i + 1, j + 1, 0), idx(i, j + 1, 0), idx(i + 1, j, 0)}, 4, Vector3::Zero(), 0.0, Vector3::Zero()});
        }
    }
    boundaryGroups[4].shrink_to_fit();
    for (auto& f : boundaryGroups[4]) f.boundaryId = 4;

    for (Index i = 0; i < nx; ++i) {
        for (Index j = 0; j < ny; ++j) {
            boundaryGroups[5].push_back({{idx(i, j, nz), idx(i, j + 1, nz), idx(i + 1, j, nz)}, 5, Vector3::Zero(), 0.0, Vector3::Zero()});
            boundaryGroups[5].push_back({{idx(i + 1, j + 1, nz), idx(i + 1, j, nz), idx(i, j + 1, nz)}, 5, Vector3::Zero(), 0.0, Vector3::Zero()});
        }
    }
    boundaryGroups[5].shrink_to_fit();
    for (auto& f : boundaryGroups[5]) f.boundaryId = 5;

    Index count = 0;
    for (Index b = 0; b < 6; ++b) {
        grid->boundaryFaceStart[b] = count;
        count += boundaryGroups[b].size();
    }
    grid->boundaryFaceStart[6] = count;
    grid->nBoundaryFaces = count;

    grid->boundaryFaces.reserve(grid->nBoundaryFaces);
    for (Index b = 0; b < 6; ++b) {
        for (const auto& f : boundaryGroups[b]) {
            grid->boundaryFaces.push_back(f);
        }
    }

    grid->computeGeometricQuantities();
    grid->computeBoundaryNormals();

    return grid;
}

}
