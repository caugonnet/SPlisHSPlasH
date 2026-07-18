#include "DFSPHKernels.cuh"

#include <cub/device/device_radix_sort.cuh>

namespace SPH
{
namespace cuda_dfsph
{

static const int kBlock = 256;

static inline int gridBlocks(unsigned int n) { return static_cast<int>((n + kBlock - 1) / kBlock); }

// Iterate the 27-cell neighborhood of particle i. Emits neighbor index `jvar`.
// Skips out-of-range cells (no clamping, to avoid double-counting border cells).
#define FOR_NEIGHBORS(s, xi, jvar, body)                                                       \
	{                                                                                          \
		int3 _cc = cellCoord((s), (xi));                                                        \
		for (int _dz = -1; _dz <= 1; ++_dz)                                                     \
		for (int _dy = -1; _dy <= 1; ++_dy)                                                     \
		for (int _dx = -1; _dx <= 1; ++_dx)                                                     \
		{                                                                                      \
			int _nx = _cc.x + _dx, _ny = _cc.y + _dy, _nz = _cc.z + _dz;                        \
			if (_nx < 0 || _ny < 0 || _nz < 0 || _nx >= (s).gridDim[0] ||                       \
				_ny >= (s).gridDim[1] || _nz >= (s).gridDim[2])                                 \
				continue;                                                                      \
			unsigned int _cell = (unsigned int)(_nx + (s).gridDim[0] * (_ny + (s).gridDim[1] * _nz)); \
			unsigned int _s0 = (s).cellStart[_cell];                                            \
			if (_s0 == CELL_EMPTY) continue;                                                    \
			unsigned int _s1 = (s).cellEnd[_cell];                                              \
			for (unsigned int _k = _s0; _k < _s1; ++_k)                                         \
			{                                                                                  \
				unsigned int jvar = (s).sortedId[_k];                                           \
				body                                                                           \
			}                                                                                  \
		}                                                                                      \
	}

// ---------------------------------------------------------------------------
// Neighborhood
// ---------------------------------------------------------------------------
__global__ void kComputeCellKeys(DeviceState s)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= s.n) return;
	s.cellKey[i] = cellLinear(s, cellCoord(s, s.pos[i]));
	s.sortedId[i] = i;
}

void launchComputeCellKeys(const DeviceState &s, cudaStream_t stream)
{
	kComputeCellKeys<<<gridBlocks(s.n), kBlock, 0, stream>>>(s);
}

size_t querySortTempBytes(unsigned int capacity)
{
	size_t bytes = 0;
	cub::DeviceRadixSort::SortPairs<unsigned int, unsigned int>(
		nullptr, bytes, nullptr, nullptr, nullptr, nullptr, static_cast<int>(capacity));
	return bytes;
}

void launchSortByCell(DeviceState &s, unsigned int *keysAlt, unsigned int *valsAlt,
					  void *dTemp, size_t tempBytes, cudaStream_t stream)
{
	// Explicit in/out (no DoubleBuffer) so the sorted result is deterministically
	// in the alt buffers regardless of the number of radix passes. The caller
	// re-points s.cellKey / s.sortedId to the alt buffers for the rest of the step.
	size_t bytes = tempBytes;
	cub::DeviceRadixSort::SortPairs(dTemp, bytes, s.cellKey, keysAlt, s.sortedId, valsAlt,
								   static_cast<int>(s.n), 0, sizeof(unsigned int) * 8, stream);
	s.cellKey = keysAlt;
	s.sortedId = valsAlt;
}

__global__ void kResetCellRanges(DeviceState s)
{
	unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (c >= s.numCells) return;
	s.cellStart[c] = CELL_EMPTY;
	s.cellEnd[c] = CELL_EMPTY;
}

__global__ void kBuildCellRanges(DeviceState s)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= s.n) return;
	unsigned int key = s.cellKey[idx];
	unsigned int prev = (idx == 0) ? CELL_EMPTY : s.cellKey[idx - 1];
	if (idx == 0 || key != prev)
	{
		s.cellStart[key] = idx;
		if (idx > 0) s.cellEnd[prev] = idx;
	}
	if (idx == s.n - 1)
		s.cellEnd[key] = s.n;
}

void launchBuildCellRanges(const DeviceState &s, cudaStream_t stream)
{
	kResetCellRanges<<<gridBlocks(s.numCells), kBlock, 0, stream>>>(s);
	kBuildCellRanges<<<gridBlocks(s.n), kBlock, 0, stream>>>(s);
}

// ---------------------------------------------------------------------------
// Boundary (Bender2019 volume map)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void worldToLocal(const DeviceBoundaryMap &m, real3 x, double out[3])
{
	double d[3] = {static_cast<double>(x.x) - m.t[0], static_cast<double>(x.y) - m.t[1], static_cast<double>(x.z) - m.t[2]};
	// local = R^T * (x - t)
	for (int j = 0; j < 3; ++j)
		out[j] = m.R[0 * 3 + j] * d[0] + m.R[1 * 3 + j] * d[1] + m.R[2 * 3 + j] * d[2];
}

__device__ __forceinline__ void rotateToWorld(const DeviceBoundaryMap &m, const double v[3], double out[3])
{
	for (int k = 0; k < 3; ++k)
		out[k] = m.R[k * 3 + 0] * v[0] + m.R[k * 3 + 1] * v[1] + m.R[k * 3 + 2] * v[2];
}

__global__ void kComputeBoundary(DeviceState s, real supportRadius, real particleRadius)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= s.n) return;

	const bool active = (s.state[i] == 0);
	const real3 xi = s.pos[i];

	for (int b = 0; b < s.numMaps; ++b)
	{
		const unsigned int slot = static_cast<unsigned int>(b) * s.capacity + i;
		s.boundaryVolume[slot] = 0;
		s.boundaryXj[slot] = make_r3(0, 0, 0);
		const DeviceBoundaryMap &m = s.maps[b];
		if (!m.valid || !active) continue; // only Active particles

		double localXi[3];
		worldToLocal(m, xi, localXi);

		BmShape sh;
		bool chk = bmDetermine(m, 0, localXi, true, sh);
		if (!chk) continue;

		double normalLocal[3];
		double dist = bmEval(m, 0, sh, normalLocal);
		if (dist == DBL_MAX) continue;

		if (dist > 0.0 && static_cast<real>(dist) < supportRadius)
		{
			double volume = bmEval(m, 1, sh, nullptr);
			if (volume > 0.0 && volume != DBL_MAX)
			{
				double nWorld[3];
				rotateToWorld(m, normalLocal, nWorld);
				double nl = sqrt(nWorld[0] * nWorld[0] + nWorld[1] * nWorld[1] + nWorld[2] * nWorld[2]);
				if (nl > 1.0e-9)
				{
					nWorld[0] /= nl; nWorld[1] /= nl; nWorld[2] /= nl;
					real d = static_cast<real>(dist) + static_cast<real>(0.5) * particleRadius;
					real dmin = static_cast<real>(2.0) * particleRadius;
					if (d < dmin) d = dmin;
					s.boundaryVolume[slot] = static_cast<real>(volume);
					s.boundaryXj[slot] = make_r3(xi.x - d * static_cast<real>(nWorld[0]),
												 xi.y - d * static_cast<real>(nWorld[1]),
												 xi.z - d * static_cast<real>(nWorld[2]));
				}
			}
		}
	}
}

void launchComputeBoundary(const DeviceState &s, real, cudaStream_t stream)
{
	kComputeBoundary<<<gridBlocks(s.n), kBlock, 0, stream>>>(s, s.supportRadius, s.particleRadius);
}

// ---------------------------------------------------------------------------
// Density and DFSPH factor
// ---------------------------------------------------------------------------
__global__ void kComputeDensity(DeviceState s, CubicKernelC ker)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= s.n) return;

	const real3 xi = s.pos[i];
	const real V = s.volume;
	real density = V * ker.wZero;

	FOR_NEIGHBORS(s, xi, j,
		if (j != i)
		{
			real r = sqrt(sqnorm(xi - s.pos[j]));
			density += V * cubicW(ker, r);
		}
	);

	for (int b = 0; b < s.numMaps; ++b)
	{
		const unsigned int slot = static_cast<unsigned int>(b) * s.capacity + i;
		const real bv = s.boundaryVolume[slot];
		if (bv > 0)
		{
			real r = sqrt(sqnorm(xi - s.boundaryXj[slot]));
			density += bv * cubicW(ker, r);
		}
	}

	s.density[i] = density * s.density0;
}

void launchComputeDensity(const DeviceState &s, CubicKernelC kernel, cudaStream_t stream)
{
	kComputeDensity<<<gridBlocks(s.n), kBlock, 0, stream>>>(s, kernel);
}

__global__ void kComputeFactor(DeviceState s, CubicKernelC ker, real eps)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= s.n) return;

	const real3 xi = s.pos[i];
	const real V = s.volume;
	real sumGrad = 0;
	real3 gradPi = make_r3(0, 0, 0);

	FOR_NEIGHBORS(s, xi, j,
		if (j != i)
		{
			real3 gpj = (-V) * cubicGradW(ker, xi - s.pos[j]);
			sumGrad += sqnorm(gpj);
			gradPi -= gpj;
		}
	);

	for (int b = 0; b < s.numMaps; ++b)
	{
		const unsigned int slot = static_cast<unsigned int>(b) * s.capacity + i;
		const real bv = s.boundaryVolume[slot];
		if (bv > 0)
		{
			real3 gpj = (-bv) * cubicGradW(ker, xi - s.boundaryXj[slot]);
			gradPi -= gpj;
		}
	}

	sumGrad += sqnorm(gradPi);
	s.factor[i] = (sumGrad > eps) ? (static_cast<real>(1.0) / sumGrad) : static_cast<real>(0.0);
}

void launchComputeFactor(const DeviceState &s, CubicKernelC kernel, real eps, cudaStream_t stream)
{
	kComputeFactor<<<gridBlocks(s.n), kBlock, 0, stream>>>(s, kernel, eps);
}

// ---------------------------------------------------------------------------
// Non-pressure forces, CFL, integration
// ---------------------------------------------------------------------------
__global__ void kClearAccelGravity(DeviceState s, real3 g)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= s.n) return;
	if (s.mass[i] != 0 && s.state[i] == 0)
		s.accel[i] = g;
	else
		s.accel[i] = make_r3(0, 0, 0);
}

void launchClearAccelGravity(const DeviceState &s, real3 gravity, cudaStream_t stream)
{
	kClearAccelGravity<<<gridBlocks(s.n), kBlock, 0, stream>>>(s, gravity);
}

// Standard viscosity (matches Viscosity_Standard::step, fluid same-phase term):
//   a_i += d*visco*(m_j/rho_j) * (v_i-v_j).(x_i-x_j) / (|x_i-x_j|^2 + 0.01 h^2) * gradW
// d = 10 (3D) / 8 (2D), h = support radius. Boundary viscosity term is omitted
// because the target scenes set viscosityBoundary = 0.
__global__ void kViscosityStandard(DeviceState s, CubicKernelC ker, real viscosity, real dcoef, real h2)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= s.n) return;
	if (s.state[i] != 0) return;

	const real3 xi = s.pos[i];
	const real3 vi = s.vel[i];
	const real eps = static_cast<real>(0.01) * h2;
	real3 ai = make_r3(0, 0, 0);

	FOR_NEIGHBORS(s, xi, j,
		if (j != i)
		{
			const real3 xixj = xi - s.pos[j];
			const real coeff = dcoef * viscosity * (s.mass[j] / s.density[j]) *
							   dot(vi - s.vel[j], xixj) / (sqnorm(xixj) + eps);
			ai += coeff * cubicGradW(ker, xixj);
		}
	);

	s.accel[i] += ai;
}

void launchViscosityStandard(const DeviceState &s, CubicKernelC kernel, real viscosity, real dcoef, real h2, cudaStream_t stream)
{
	kViscosityStandard<<<gridBlocks(s.n), kBlock, 0, stream>>>(s, kernel, viscosity, dcoef, h2);
}

__global__ void kComputeMaxVelSq(DeviceState s, real dt, const real *dtPtr, real *scratch)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= s.n) return;
	if (dtPtr) dt = *dtPtr;
	if (s.state[i] != 0) { scratch[i] = 0; return; }
	real3 predicted = s.vel[i] + dt * s.accel[i];
	scratch[i] = sqnorm(predicted);
}

void launchComputeMaxVelSq(const DeviceState &s, real dt, real *scratch, cudaStream_t stream, const real *dtPtr)
{
	kComputeMaxVelSq<<<gridBlocks(s.n), kBlock, 0, stream>>>(s, dt, dtPtr, scratch);
}

__global__ void kApplyVelocityUpdate(DeviceState s, real dt, const real *dtPtr)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= s.n) return;
	if (dtPtr) dt = *dtPtr;
	if (s.state[i] != 0) return;
	s.vel[i] += dt * s.accel[i];
}

void launchApplyVelocityUpdate(const DeviceState &s, real dt, cudaStream_t stream, const real *dtPtr)
{
	kApplyVelocityUpdate<<<gridBlocks(s.n), kBlock, 0, stream>>>(s, dt, dtPtr);
}

__global__ void kApplyPosition(DeviceState s, real dt, const real *dtPtr)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= s.n) return;
	if (dtPtr) dt = *dtPtr;
	if (s.state[i] != 0) return;
	s.pos[i] += dt * s.vel[i];
}

void launchApplyPosition(const DeviceState &s, real dt, cudaStream_t stream, const real *dtPtr)
{
	kApplyPosition<<<gridBlocks(s.n), kBlock, 0, stream>>>(s, dt, dtPtr);
}

// ---------------------------------------------------------------------------
// Shared device helpers for the solver
// ---------------------------------------------------------------------------
__device__ __forceinline__ real deviceDensityChange(const DeviceState &s, CubicKernelC ker, unsigned int i)
{
	const real3 xi = s.pos[i];
	const real3 vi = s.vel[i];
	real delta = 0;
	FOR_NEIGHBORS(s, xi, j,
		if (j != i)
			delta += dot(vi - s.vel[j], cubicGradW(ker, xi - s.pos[j]));
	);
	delta *= s.volume;
	return delta;
}

__device__ __forceinline__ unsigned int deviceCountFluidNeighbors(const DeviceState &s, unsigned int i)
{
	const real3 xi = s.pos[i];
	unsigned int cnt = 0;
	FOR_NEIGHBORS(s, xi, j, if (j != i) cnt++;);
	return cnt;
}

// ---------------------------------------------------------------------------
// Solver initialisation
// ---------------------------------------------------------------------------
__global__ void kDivergenceInit(DeviceState s, CubicKernelC ker, real invH, int is2D, const real *dtPtr)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= s.n) return;
	if (dtPtr) invH = static_cast<real>(1.0) / *dtPtr;

	const real3 xi = s.pos[i];
	const real3 vi = s.vel[i];
	real densityAdv = deviceDensityChange(s, ker, i);
	for (int b = 0; b < s.numMaps; ++b)
	{
		const unsigned int slot = static_cast<unsigned int>(b) * s.capacity + i;
		const real bv = s.boundaryVolume[slot];
		if (bv > 0)
		{
			real3 vj = bmPointVelocity(s.maps[b], s.boundaryXj[slot]);
			densityAdv += bv * dot(vi - vj, cubicGradW(ker, xi - s.boundaryXj[slot]));
		}
	}
	if (densityAdv < 0) densityAdv = 0;

	unsigned int numNeighbors = deviceCountFluidNeighbors(s, i);
	if (!is2D) { if (numNeighbors < 20) densityAdv = 0; }
	else       { if (numNeighbors < 7)  densityAdv = 0; }

	s.densityAdv[i] = densityAdv;
	s.factor[i] *= invH;

	// warm start (USE_WARMSTART_V)
	real pv = s.pressureRho2V[i];
	if (densityAdv > 0)
		s.pressureRho2V[i] = static_cast<real>(0.5) * fmin(pv, static_cast<real>(0.5)) * invH;
	else
		s.pressureRho2V[i] = 0;
}

void launchDivergenceInit(const DeviceState &s, CubicKernelC kernel, real dt, cudaStream_t stream, const real *dtPtr)
{
	const real invH = static_cast<real>(1.0) / dt;
	kDivergenceInit<<<gridBlocks(s.n), kBlock, 0, stream>>>(s, kernel, invH, s.sim2D, dtPtr);
}

__global__ void kPressureInit(DeviceState s, CubicKernelC ker, real invH2, real h, const real *dtPtr)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= s.n) return;
	if (dtPtr) { h = *dtPtr; invH2 = static_cast<real>(1.0) / (h * h); }

	const real3 xi = s.pos[i];
	const real3 vi = s.vel[i];
	// computeDensityAdv: rho/rho0 + h*delta
	real delta = 0;
	FOR_NEIGHBORS(s, xi, j,
		if (j != i)
			delta += dot(vi - s.vel[j], cubicGradW(ker, xi - s.pos[j]));
	);
	delta *= s.volume;
	for (int b = 0; b < s.numMaps; ++b)
	{
		const unsigned int slot = static_cast<unsigned int>(b) * s.capacity + i;
		const real bv = s.boundaryVolume[slot];
		if (bv > 0)
		{
			real3 vj = bmPointVelocity(s.maps[b], s.boundaryXj[slot]);
			delta += bv * dot(vi - vj, cubicGradW(ker, xi - s.boundaryXj[slot]));
		}
	}
	real densityAdv = s.density[i] / s.density0 + h * delta;
	s.densityAdv[i] = densityAdv;
	s.factor[i] *= invH2;

	// warm start (USE_WARMSTART)
	real p = s.pressureRho2[i];
	if (densityAdv > 1.0)
		s.pressureRho2[i] = static_cast<real>(0.5) * fmin(p, static_cast<real>(0.00025)) * invH2;
	else
		s.pressureRho2[i] = 0;
}

void launchPressureInit(const DeviceState &s, CubicKernelC kernel, real dt, cudaStream_t stream, const real *dtPtr)
{
	const real invH2 = static_cast<real>(1.0) / (dt * dt);
	kPressureInit<<<gridBlocks(s.n), kBlock, 0, stream>>>(s, kernel, invH2, dt, dtPtr);
}

// ---------------------------------------------------------------------------
// Pressure accelerations and Jacobi iteration
// ---------------------------------------------------------------------------
__global__ void kComputePressureAccel(DeviceState s, CubicKernelC ker,
									   const real *pressure, real eps, int applyForce, BodyReactionAccum reaction)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= s.n) return;

	s.pressureAccel[i] = make_r3(0, 0, 0);
	if (s.state[i] != 0) return;

	const real p_rho2_i = pressure[i];
	const real3 xi = s.pos[i];
	real3 ai = make_r3(0, 0, 0);

	FOR_NEIGHBORS(s, xi, j,
		if (j != i)
		{
			const real p_rho2_j = pressure[j];
			const real pSum = p_rho2_i + p_rho2_j; // single fluid: density0_j/density0 == 1
			if (fabs(pSum) > eps)
			{
				real3 gpj = (-s.volume) * cubicGradW(ker, xi - s.pos[j]);
				ai += pSum * gpj;
			}
		}
	);

	if (fabs(p_rho2_i) > eps)
	{
		for (int b = 0; b < s.numMaps; ++b)
		{
			const unsigned int slot = static_cast<unsigned int>(b) * s.capacity + i;
			const real bv = s.boundaryVolume[slot];
			if (bv > 0)
			{
				real3 gpj = (-bv) * cubicGradW(ker, xi - s.boundaryXj[slot]);
				real3 a = p_rho2_i * gpj;
				ai += a;
				const DeviceBoundaryMap &m = s.maps[b];
				if (applyForce && m.isDynamic)
				{
					real3 force = (-s.mass[i]) * a;
					real3 xj = s.boundaryXj[slot];
					real3 com = make_r3(static_cast<real>(m.com[0]), static_cast<real>(m.com[1]), static_cast<real>(m.com[2]));
					real3 torque = cross(xj - com, force);
					atomicAdd(&reaction.force[3 * b + 0], static_cast<double>(force.x));
					atomicAdd(&reaction.force[3 * b + 1], static_cast<double>(force.y));
					atomicAdd(&reaction.force[3 * b + 2], static_cast<double>(force.z));
					atomicAdd(&reaction.torque[3 * b + 0], static_cast<double>(torque.x));
					atomicAdd(&reaction.torque[3 * b + 1], static_cast<double>(torque.y));
					atomicAdd(&reaction.torque[3 * b + 2], static_cast<double>(torque.z));
				}
			}
		}
	}

	s.pressureAccel[i] = ai;
}

void launchComputePressureAccel(const DeviceState &s, CubicKernelC kernel,
								const real *pressure, real eps, int applyBoundaryForces,
								BodyReactionAccum reaction, cudaStream_t stream)
{
	kComputePressureAccel<<<gridBlocks(s.n), kBlock, 0, stream>>>(s, kernel, pressure, eps, applyBoundaryForces, reaction);
}

__device__ __forceinline__ real deviceAijPj(const DeviceState &s, CubicKernelC ker, unsigned int i)
{
	const real3 xi = s.pos[i];
	const real3 ai = s.pressureAccel[i];
	real aij = 0;
	FOR_NEIGHBORS(s, xi, j,
		if (j != i)
			aij += dot(ai - s.pressureAccel[j], cubicGradW(ker, xi - s.pos[j]));
	);
	aij *= s.volume;
	for (int b = 0; b < s.numMaps; ++b)
	{
		const unsigned int slot = static_cast<unsigned int>(b) * s.capacity + i;
		const real bv = s.boundaryVolume[slot];
		if (bv > 0)
			aij += bv * dot(ai, cubicGradW(ker, xi - s.boundaryXj[slot]));
	}
	return aij;
}

__global__ void kSolveIterate(DeviceState s, CubicKernelC ker, real *pressure,
							  real hFactor, int isPressure, int is2D, real density0, real *errScratch,
							  const real *dtPtr)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= s.n) return;
	if (dtPtr) hFactor = isPressure ? (*dtPtr) * (*dtPtr) : *dtPtr;

	if (isPressure && s.state[i] != 0)
	{
		errScratch[i] = 0;
		return;
	}

	real aij_pj = deviceAijPj(s, ker, i) * hFactor;
	real s_i = isPressure ? (static_cast<real>(1.0) - s.densityAdv[i]) : (-s.densityAdv[i]);
	real residuum = s_i - aij_pj;
	if (residuum > 0) residuum = 0;

	if (!isPressure)
	{
		unsigned int numNeighbors = deviceCountFluidNeighbors(s, i);
		if (!is2D) { if (numNeighbors < 20) residuum = 0; }
		else       { if (numNeighbors < 7)  residuum = 0; }
	}

	real p = pressure[i];
	p = p - static_cast<real>(0.5) * (s_i - aij_pj) * s.factor[i];
	if (p < 0) p = 0;
	pressure[i] = p;

	errScratch[i] = -density0 * residuum;
}

void launchSolveIterate(const DeviceState &s, CubicKernelC kernel, real *pressure,
						real hFactor, int isPressure, real, real *errScratch, cudaStream_t stream,
						const real *dtPtr)
{
	kSolveIterate<<<gridBlocks(s.n), kBlock, 0, stream>>>(s, kernel, pressure, hFactor, isPressure, s.sim2D, s.density0, errScratch, dtPtr);
}

// ---------------------------------------------------------------------------
// Finalisers
// ---------------------------------------------------------------------------
__global__ void kDivergenceFinalizeApply(DeviceState s, real dt, const real *dtPtr)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= s.n) return;
	if (dtPtr) dt = *dtPtr;
	s.vel[i] += dt * s.pressureAccel[i];
	s.factor[i] *= dt;
	s.pressureRho2V[i] *= dt; // warm start persistence (USE_WARMSTART_V)
}

void launchDivergenceFinalizeApply(const DeviceState &s, real dt, cudaStream_t stream, const real *dtPtr)
{
	kDivergenceFinalizeApply<<<gridBlocks(s.n), kBlock, 0, stream>>>(s, dt, dtPtr);
}

__global__ void kPressureFinalizeApply(DeviceState s, real dt, const real *dtPtr)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= s.n) return;
	if (dtPtr) dt = *dtPtr;
	s.vel[i] += dt * s.pressureAccel[i];
	s.pressureRho2[i] *= dt * dt; // warm start persistence (USE_WARMSTART)
}

void launchPressureFinalizeApply(const DeviceState &s, real dt, cudaStream_t stream, const real *dtPtr)
{
	kPressureFinalizeApply<<<gridBlocks(s.n), kBlock, 0, stream>>>(s, dt, dtPtr);
}

__global__ void kZeroReaction(BodyReactionAccum r, int numBodies)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		for (int k = 0; k < 3 * numBodies; ++k) { r.force[k] = 0; r.torque[k] = 0; }
	}
}

void launchZeroReaction(BodyReactionAccum reaction, int numBodies, cudaStream_t stream)
{
	kZeroReaction<<<1, 1, 0, stream>>>(reaction, numBodies);
}

// Device-side replica of the host solver-loop predicate
// `while ((!chk || iters < minIter) && iters < maxIter)` with chk = (avg <= eta).
// Runs once per while-graph iteration, after the CUB error reduction.
__global__ void kSolverLoopCond(const real *errSum, unsigned int *iter, real *avgOut, int *cond,
								real invN, real eta, unsigned int minIter, unsigned int maxIter,
								const real *dtPtr, int etaOverDt)
{
	const unsigned int it = *iter + 1;
	*iter = it;
	if (dtPtr && etaOverDt) eta = eta / *dtPtr;
	const real avg = (*errSum) * invN;
	*avgOut = avg;
	const bool chk = (avg <= eta);
	*cond = ((!chk || it < minIter) && it < maxIter) ? 1 : 0;
}

void launchSolverLoopCond(const real *errSum, unsigned int *iter, real *avgOut, int *cond,
						  real invN, real eta, unsigned int minIter, unsigned int maxIter,
						  cudaStream_t stream, const real *dtPtr, int etaOverDt)
{
	kSolverLoopCond<<<1, 1, 0, stream>>>(errSum, iter, avgOut, cond, invN, eta, minIter, maxIter, dtPtr, etaOverDt);
}

// Device-side CFL control for the whole-step graph. dt2[0] = this step's base
// dt, dt2[1] = dt used for integration (CFL method 1, or a copy of dt2[0] when
// CFL is disabled). kCflRotateDt promotes the previous step's dtUsed to the new
// base dt at the start of each relaunch.
__global__ void kCflRotateDt(real *dt2)
{
	dt2[0] = dt2[1];
}

__global__ void kCflUpdateDt(const real *maxVelSq, real *dt2, real cflFactor, real diameter,
							 real cflMin, real cflMax, int enabled)
{
	if (!enabled)
	{
		dt2[1] = dt2[0];
		return;
	}
	real mv = *maxVelSq;
	if (mv < static_cast<real>(1.0e-9)) mv = static_cast<real>(1.0e-9);
	real h = cflFactor * static_cast<real>(0.4) * (diameter / sqrt(mv));
	if (h > cflMax) h = cflMax;
	if (h < cflMin) h = cflMin;
	dt2[1] = h;
}

// Pack render data straight into mapped OpenGL buffers: positions (3 floats,
// tight) and a color scalar (velocity magnitude, matching the GUI's default
// "velocity" color field).
__global__ void kPackRender(DeviceState s, real3 *outPos, real *outScalar)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= s.n) return;
	outPos[i] = s.pos[i];
	outScalar[i] = sqrt(sqnorm(s.vel[i]));
}

void launchPackRender(const DeviceState &s, real3 *outPos, real *outScalar, cudaStream_t stream)
{
	kPackRender<<<gridBlocks(s.n), kBlock, 0, stream>>>(s, outPos, outScalar);
}

void launchCflRotateDt(real *dt2, cudaStream_t stream)
{
	kCflRotateDt<<<1, 1, 0, stream>>>(dt2);
}

void launchCflUpdateDt(const real *maxVelSq, real *dt2, real cflFactor, real diameter,
					   real cflMin, real cflMax, int enabled, cudaStream_t stream)
{
	kCflUpdateDt<<<1, 1, 0, stream>>>(maxVelSq, dt2, cflFactor, diameter, cflMin, cflMax, enabled);
}

} // namespace cuda_dfsph
} // namespace SPH
