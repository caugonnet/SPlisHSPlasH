#ifndef __DFSPH_CUDA_Bender2019MapCUDA_cuh__
#define __DFSPH_CUDA_Bender2019MapCUDA_cuh__

#include "DFSPHDeviceState.cuh"
#include "DFSPHCudaBackend.h"

#include <cuda_runtime.h>
#include <cfloat>

namespace SPH
{
namespace cuda_dfsph
{

// Device-resident flattened Bender2019 volume/distance map for one rigid body.
// Ports Discregrid::CubicLagrangeDiscreteGrid interpolation to the GPU using the
// `shape_function_` (underscore) node ordering used by determineShapeFunctions.
struct DeviceBoundaryMap
{
	double domainMin[3];
	double domainMax[3];
	double cellSize[3];
	double invCellSize[3];
	unsigned int resolution[3];

	const double *nodes[2];
	const unsigned int *cells[2];    // 32 * nCells
	const unsigned int *cellMap[2];  // nCells
	unsigned int nCells;

	// Rigid transform (world = R * local + t), row-major R.
	double R[9];
	double t[3];
	double angVel[3];
	double linVel[3];
	double com[3];
	int isDynamic;
	int valid;
};

// shape_function_ (underscore ordering) with optional gradient. Matches
// Discregrid exactly (double precision).
__device__ inline void bmShapeFunction(const double xi[3], double N[32], double dN[32][3], bool wantGrad)
{
	const double x = xi[0], y = xi[1], z = xi[2];
	const double x2 = x * x, y2 = y * y, z2 = z * z;
	const double _1mx = 1.0 - x, _1my = 1.0 - y, _1mz = 1.0 - z;
	const double _1px = 1.0 + x, _1py = 1.0 + y, _1pz = 1.0 + z;
	const double _1m3x = 1.0 - 3.0 * x, _1m3y = 1.0 - 3.0 * y, _1m3z = 1.0 - 3.0 * z;
	const double _1p3x = 1.0 + 3.0 * x, _1p3y = 1.0 + 3.0 * y, _1p3z = 1.0 + 3.0 * z;
	const double _1mxt1my = _1mx * _1my, _1mxt1py = _1mx * _1py, _1pxt1my = _1px * _1my, _1pxt1py = _1px * _1py;
	const double _1mxt1mz = _1mx * _1mz, _1mxt1pz = _1mx * _1pz, _1pxt1mz = _1px * _1mz, _1pxt1pz = _1px * _1pz;
	const double _1myt1mz = _1my * _1mz, _1myt1pz = _1my * _1pz, _1pyt1mz = _1py * _1mz, _1pyt1pz = _1py * _1pz;
	const double _1mx2 = 1.0 - x2, _1my2 = 1.0 - y2, _1mz2 = 1.0 - z2;

	double fac = 1.0 / 64.0 * (9.0 * (x2 + y2 + z2) - 19.0);
	N[0] = fac * _1mxt1my * _1mz;
	N[1] = fac * _1pxt1my * _1mz;
	N[2] = fac * _1mxt1py * _1mz;
	N[3] = fac * _1pxt1py * _1mz;
	N[4] = fac * _1mxt1my * _1pz;
	N[5] = fac * _1pxt1my * _1pz;
	N[6] = fac * _1mxt1py * _1pz;
	N[7] = fac * _1pxt1py * _1pz;

	fac = 9.0 / 64.0 * _1mx2;
	double fact1m3x = fac * _1m3x, fact1p3x = fac * _1p3x;
	N[8] = fact1m3x * _1myt1mz;
	N[9] = fact1p3x * _1myt1mz;
	N[10] = fact1m3x * _1myt1pz;
	N[11] = fact1p3x * _1myt1pz;
	N[12] = fact1m3x * _1pyt1mz;
	N[13] = fact1p3x * _1pyt1mz;
	N[14] = fact1m3x * _1pyt1pz;
	N[15] = fact1p3x * _1pyt1pz;

	fac = 9.0 / 64.0 * _1my2;
	double fact1m3y = fac * _1m3y, fact1p3y = fac * _1p3y;
	N[16] = fact1m3y * _1mxt1mz;
	N[17] = fact1p3y * _1mxt1mz;
	N[18] = fact1m3y * _1pxt1mz;
	N[19] = fact1p3y * _1pxt1mz;
	N[20] = fact1m3y * _1mxt1pz;
	N[21] = fact1p3y * _1mxt1pz;
	N[22] = fact1m3y * _1pxt1pz;
	N[23] = fact1p3y * _1pxt1pz;

	fac = 9.0 / 64.0 * _1mz2;
	double fact1m3z = fac * _1m3z, fact1p3z = fac * _1p3z;
	N[24] = fact1m3z * _1mxt1my;
	N[25] = fact1p3z * _1mxt1my;
	N[26] = fact1m3z * _1mxt1py;
	N[27] = fact1p3z * _1mxt1py;
	N[28] = fact1m3z * _1pxt1my;
	N[29] = fact1p3z * _1pxt1my;
	N[30] = fact1m3z * _1pxt1py;
	N[31] = fact1p3z * _1pxt1py;

	if (!wantGrad)
		return;

	const double _9t3x2py2pz2m19 = 9.0 * (3.0 * x2 + y2 + z2) - 19.0;
	const double _9tx2p3y2pz2m19 = 9.0 * (x2 + 3.0 * y2 + z2) - 19.0;
	const double _9tx2py2p3z2m19 = 9.0 * (x2 + y2 + 3.0 * z2) - 19.0;
	const double _18x = 18.0 * x, _18y = 18.0 * y, _18z = 18.0 * z;
	const double _3m9x2 = 3.0 - 9.0 * x2, _3m9y2 = 3.0 - 9.0 * y2, _3m9z2 = 3.0 - 9.0 * z2;
	const double _2x = 2.0 * x, _2y = 2.0 * y, _2z = 2.0 * z;
	const double _18xm9t3x2py2pz2m19 = _18x - _9t3x2py2pz2m19;
	const double _18xp9t3x2py2pz2m19 = _18x + _9t3x2py2pz2m19;
	const double _18ym9tx2p3y2pz2m19 = _18y - _9tx2p3y2pz2m19;
	const double _18yp9tx2p3y2pz2m19 = _18y + _9tx2p3y2pz2m19;
	const double _18zm9tx2py2p3z2m19 = _18z - _9tx2py2p3z2m19;
	const double _18zp9tx2py2p3z2m19 = _18z + _9tx2py2p3z2m19;

	dN[0][0] = _18xm9t3x2py2pz2m19 * _1myt1mz;
	dN[0][1] = _1mxt1mz * _18ym9tx2p3y2pz2m19;
	dN[0][2] = _1mxt1my * _18zm9tx2py2p3z2m19;
	dN[1][0] = _18xp9t3x2py2pz2m19 * _1myt1mz;
	dN[1][1] = _1pxt1mz * _18ym9tx2p3y2pz2m19;
	dN[1][2] = _1pxt1my * _18zm9tx2py2p3z2m19;
	dN[2][0] = _18xm9t3x2py2pz2m19 * _1pyt1mz;
	dN[2][1] = _1mxt1mz * _18yp9tx2p3y2pz2m19;
	dN[2][2] = _1mxt1py * _18zm9tx2py2p3z2m19;
	dN[3][0] = _18xp9t3x2py2pz2m19 * _1pyt1mz;
	dN[3][1] = _1pxt1mz * _18yp9tx2p3y2pz2m19;
	dN[3][2] = _1pxt1py * _18zm9tx2py2p3z2m19;
	dN[4][0] = _18xm9t3x2py2pz2m19 * _1myt1pz;
	dN[4][1] = _1mxt1pz * _18ym9tx2p3y2pz2m19;
	dN[4][2] = _1mxt1my * _18zp9tx2py2p3z2m19;
	dN[5][0] = _18xp9t3x2py2pz2m19 * _1myt1pz;
	dN[5][1] = _1pxt1pz * _18ym9tx2p3y2pz2m19;
	dN[5][2] = _1pxt1my * _18zp9tx2py2p3z2m19;
	dN[6][0] = _18xm9t3x2py2pz2m19 * _1pyt1pz;
	dN[6][1] = _1mxt1pz * _18yp9tx2p3y2pz2m19;
	dN[6][2] = _1mxt1py * _18zp9tx2py2p3z2m19;
	dN[7][0] = _18xp9t3x2py2pz2m19 * _1pyt1pz;
	dN[7][1] = _1pxt1pz * _18yp9tx2p3y2pz2m19;
	dN[7][2] = _1pxt1py * _18zp9tx2py2p3z2m19;
	for (int r = 0; r < 8; ++r) { dN[r][0] /= 64.0; dN[r][1] /= 64.0; dN[r][2] /= 64.0; }

	const double _m3m9x2m2x = -_3m9x2 - _2x, _p3m9x2m2x = _3m9x2 - _2x;
	const double _1mx2t1m3x = _1mx2 * _1m3x, _1mx2t1p3x = _1mx2 * _1p3x;
	dN[8][0] = _m3m9x2m2x * _1myt1mz; dN[8][1] = -_1mx2t1m3x * _1mz; dN[8][2] = -_1mx2t1m3x * _1my;
	dN[9][0] = _p3m9x2m2x * _1myt1mz; dN[9][1] = -_1mx2t1p3x * _1mz; dN[9][2] = -_1mx2t1p3x * _1my;
	dN[10][0] = _m3m9x2m2x * _1myt1pz; dN[10][1] = -_1mx2t1m3x * _1pz; dN[10][2] = _1mx2t1m3x * _1my;
	dN[11][0] = _p3m9x2m2x * _1myt1pz; dN[11][1] = -_1mx2t1p3x * _1pz; dN[11][2] = _1mx2t1p3x * _1my;
	dN[12][0] = _m3m9x2m2x * _1pyt1mz; dN[12][1] = _1mx2t1m3x * _1mz; dN[12][2] = -_1mx2t1m3x * _1py;
	dN[13][0] = _p3m9x2m2x * _1pyt1mz; dN[13][1] = _1mx2t1p3x * _1mz; dN[13][2] = -_1mx2t1p3x * _1py;
	dN[14][0] = _m3m9x2m2x * _1pyt1pz; dN[14][1] = _1mx2t1m3x * _1pz; dN[14][2] = _1mx2t1m3x * _1py;
	dN[15][0] = _p3m9x2m2x * _1pyt1pz; dN[15][1] = _1mx2t1p3x * _1pz; dN[15][2] = _1mx2t1p3x * _1py;

	const double _m3m9y2m2y = -_3m9y2 - _2y, _p3m9y2m2y = _3m9y2 - _2y;
	const double _1my2t1m3y = _1my2 * _1m3y, _1my2t1p3y = _1my2 * _1p3y;
	dN[16][0] = -_1my2t1m3y * _1mz; dN[16][1] = _m3m9y2m2y * _1mxt1mz; dN[16][2] = -_1my2t1m3y * _1mx;
	dN[17][0] = -_1my2t1p3y * _1mz; dN[17][1] = _p3m9y2m2y * _1mxt1mz; dN[17][2] = -_1my2t1p3y * _1mx;
	dN[18][0] = _1my2t1m3y * _1mz; dN[18][1] = _m3m9y2m2y * _1pxt1mz; dN[18][2] = -_1my2t1m3y * _1px;
	dN[19][0] = _1my2t1p3y * _1mz; dN[19][1] = _p3m9y2m2y * _1pxt1mz; dN[19][2] = -_1my2t1p3y * _1px;
	dN[20][0] = -_1my2t1m3y * _1pz; dN[20][1] = _m3m9y2m2y * _1mxt1pz; dN[20][2] = _1my2t1m3y * _1mx;
	dN[21][0] = -_1my2t1p3y * _1pz; dN[21][1] = _p3m9y2m2y * _1mxt1pz; dN[21][2] = _1my2t1p3y * _1mx;
	dN[22][0] = _1my2t1m3y * _1pz; dN[22][1] = _m3m9y2m2y * _1pxt1pz; dN[22][2] = _1my2t1m3y * _1px;
	dN[23][0] = _1my2t1p3y * _1pz; dN[23][1] = _p3m9y2m2y * _1pxt1pz; dN[23][2] = _1my2t1p3y * _1px;

	const double _m3m9z2m2z = -_3m9z2 - _2z, _p3m9z2m2z = _3m9z2 - _2z;
	const double _1mz2t1m3z = _1mz2 * _1m3z, _1mz2t1p3z = _1mz2 * _1p3z;
	dN[24][0] = -_1mz2t1m3z * _1my; dN[24][1] = -_1mz2t1m3z * _1mx; dN[24][2] = _m3m9z2m2z * _1mxt1my;
	dN[25][0] = -_1mz2t1p3z * _1my; dN[25][1] = -_1mz2t1p3z * _1mx; dN[25][2] = _p3m9z2m2z * _1mxt1my;
	dN[26][0] = -_1mz2t1m3z * _1py; dN[26][1] = _1mz2t1m3z * _1mx; dN[26][2] = _m3m9z2m2z * _1mxt1py;
	dN[27][0] = -_1mz2t1p3z * _1py; dN[27][1] = _1mz2t1p3z * _1mx; dN[27][2] = _p3m9z2m2z * _1mxt1py;
	dN[28][0] = _1mz2t1m3z * _1my; dN[28][1] = -_1mz2t1m3z * _1px; dN[28][2] = _m3m9z2m2z * _1pxt1my;
	dN[29][0] = _1mz2t1p3z * _1my; dN[29][1] = -_1mz2t1p3z * _1px; dN[29][2] = _p3m9z2m2z * _1pxt1my;
	dN[30][0] = _1mz2t1m3z * _1py; dN[30][1] = _1mz2t1m3z * _1px; dN[30][2] = _m3m9z2m2z * _1pxt1py;
	dN[31][0] = _1mz2t1p3z * _1py; dN[31][1] = _1mz2t1p3z * _1px; dN[31][2] = _p3m9z2m2z * _1pxt1py;
	for (int r = 8; r < 32; ++r) { dN[r][0] *= 9.0 / 64.0; dN[r][1] *= 9.0 / 64.0; dN[r][2] *= 9.0 / 64.0; }
}

// Interpolate field at local-space point xLocal. Returns value; writes gradient
// (local space) when grad != nullptr. Returns DBL_MAX when outside/pruned.
__device__ inline double bmInterpolate(const DeviceBoundaryMap &m, int field,
										const double xLocal[3], double grad[3])
{
	// Domain containment.
	for (int d = 0; d < 3; ++d)
		if (xLocal[d] < m.domainMin[d] || xLocal[d] > m.domainMax[d])
			return DBL_MAX;

	unsigned int mi[3];
	for (int d = 0; d < 3; ++d)
	{
		int idx = static_cast<int>((xLocal[d] - m.domainMin[d]) * m.invCellSize[d]);
		if (idx < 0) idx = 0;
		if (static_cast<unsigned int>(idx) >= m.resolution[d]) idx = static_cast<int>(m.resolution[d]) - 1;
		mi[d] = static_cast<unsigned int>(idx);
	}
	unsigned int i = m.resolution[1] * m.resolution[0] * mi[2] + m.resolution[0] * mi[1] + mi[0];
	unsigned int i_ = m.cellMap[field][i];
	if (i_ == 0xffffffffu)
		return DBL_MAX;

	// subdomain(i) using the original single-index i (not compacted).
	double sdMin[3], sdMax[3], c0[3], c1[3], xi[3];
	for (int d = 0; d < 3; ++d)
	{
		sdMin[d] = m.domainMin[d] + static_cast<double>(mi[d]) * m.cellSize[d];
		sdMax[d] = sdMin[d] + m.cellSize[d];
		const double denom = sdMax[d] - sdMin[d];
		c0[d] = 2.0 / denom;
		c1[d] = (sdMax[d] + sdMin[d]) / denom;
		xi[d] = c0[d] * xLocal[d] - c1[d];
	}

	const bool wantGrad = (grad != nullptr);
	double N[32];
	double dN[32][3];
	bmShapeFunction(xi, N, dN, wantGrad);

	const unsigned int *cell = &m.cells[field][i_ * 32];
	const double *coeff = m.nodes[field];

	double phi = 0.0;
	if (wantGrad) { grad[0] = grad[1] = grad[2] = 0.0; }
	for (int j = 0; j < 32; ++j)
	{
		const double c = coeff[cell[j]];
		if (c == DBL_MAX)
		{
			if (wantGrad) { grad[0] = grad[1] = grad[2] = 0.0; }
			return DBL_MAX;
		}
		phi += c * N[j];
		if (wantGrad)
		{
			grad[0] += c * dN[j][0];
			grad[1] += c * dN[j][1];
			grad[2] += c * dN[j][2];
		}
	}
	if (wantGrad)
	{
		grad[0] *= c0[0];
		grad[1] *= c0[1];
		grad[2] *= c0[2];
	}
	return phi;
}

// Determine cell / shape functions for a local-space point, mirroring
// Discregrid::determineShapeFunctions. Returns false when outside/pruned.
// The resulting `cell` (node indices for `field`) can be reused to evaluate a
// different field, exactly as the CPU Bender2019 path reuses field-0 cells to
// read field-1 (volume) nodes.
struct BmShape
{
	unsigned int cell[32];
	double c0[3];
	double N[32];
	double dN[32][3];
	bool hasGrad;
};

__device__ inline bool bmDetermine(const DeviceBoundaryMap &m, int field,
								   const double xLocal[3], bool wantGrad, BmShape &sh)
{
	for (int d = 0; d < 3; ++d)
		if (xLocal[d] < m.domainMin[d] || xLocal[d] > m.domainMax[d])
			return false;

	unsigned int mi[3];
	for (int d = 0; d < 3; ++d)
	{
		int idx = static_cast<int>((xLocal[d] - m.domainMin[d]) * m.invCellSize[d]);
		if (idx < 0) idx = 0;
		if (static_cast<unsigned int>(idx) >= m.resolution[d]) idx = static_cast<int>(m.resolution[d]) - 1;
		mi[d] = static_cast<unsigned int>(idx);
	}
	unsigned int i = m.resolution[1] * m.resolution[0] * mi[2] + m.resolution[0] * mi[1] + mi[0];
	unsigned int i_ = m.cellMap[field][i];
	if (i_ == 0xffffffffu)
		return false;

	double sdMin, sdMax, c1[3], xi[3];
	for (int d = 0; d < 3; ++d)
	{
		sdMin = m.domainMin[d] + static_cast<double>(mi[d]) * m.cellSize[d];
		sdMax = sdMin + m.cellSize[d];
		const double denom = sdMax - sdMin;
		sh.c0[d] = 2.0 / denom;
		c1[d] = (sdMax + sdMin) / denom;
		xi[d] = sh.c0[d] * xLocal[d] - c1[d];
	}

	sh.hasGrad = wantGrad;
	bmShapeFunction(xi, sh.N, sh.dN, wantGrad);
	const unsigned int *cell = &m.cells[field][i_ * 32];
	for (int j = 0; j < 32; ++j) sh.cell[j] = cell[j];
	return true;
}

// Evaluate `field` at the previously determined shape functions. Writes the
// (local-space) gradient into grad when non-null (requires sh.hasGrad).
__device__ inline double bmEval(const DeviceBoundaryMap &m, int field,
								const BmShape &sh, double grad[3])
{
	const double *coeff = m.nodes[field];
	const bool wantGrad = (grad != nullptr) && sh.hasGrad;
	double phi = 0.0;
	if (wantGrad) { grad[0] = grad[1] = grad[2] = 0.0; }
	for (int j = 0; j < 32; ++j)
	{
		const double c = coeff[sh.cell[j]];
		if (c == DBL_MAX)
		{
			if (wantGrad) { grad[0] = grad[1] = grad[2] = 0.0; }
			return DBL_MAX;
		}
		phi += c * sh.N[j];
		if (wantGrad)
		{
			grad[0] += c * sh.dN[j][0];
			grad[1] += c * sh.dN[j][1];
			grad[2] += c * sh.dN[j][2];
		}
	}
	if (wantGrad)
	{
		grad[0] *= sh.c0[0];
		grad[1] *= sh.c0[1];
		grad[2] *= sh.c0[2];
	}
	return phi;
}

// Rigid-body point velocity at world point xj: v = linVel + angVel x (xj - com).
__device__ inline real3 bmPointVelocity(const DeviceBoundaryMap &m, real3 xj)
{
	real3 w = make_r3(static_cast<real>(m.angVel[0]), static_cast<real>(m.angVel[1]), static_cast<real>(m.angVel[2]));
	real3 com = make_r3(static_cast<real>(m.com[0]), static_cast<real>(m.com[1]), static_cast<real>(m.com[2]));
	real3 lv = make_r3(static_cast<real>(m.linVel[0]), static_cast<real>(m.linVel[1]), static_cast<real>(m.linVel[2]));
	return lv + cross(w, xj - com);
}

// Host-side helpers (implemented in Bender2019MapCUDA.cu).
DeviceBoundaryMap uploadBoundaryMap(const BoundaryMapDesc &desc);
void freeBoundaryMap(DeviceBoundaryMap &m);
void updateBoundaryMapTransform(DeviceBoundaryMap &m,
								const double R[9], const double t[3],
								const double angVel[3], const double linVel[3],
								const double com[3]);

} // namespace cuda_dfsph
} // namespace SPH

#endif
