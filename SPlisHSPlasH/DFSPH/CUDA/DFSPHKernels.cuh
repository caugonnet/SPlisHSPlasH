#ifndef __DFSPH_CUDA_DFSPHKernels_cuh__
#define __DFSPH_CUDA_DFSPHKernels_cuh__

#include "DFSPHDeviceState.cuh"
#include "Bender2019MapCUDA.cuh"

#include <cuda_runtime.h>

namespace SPH
{
namespace cuda_dfsph
{

// Net reaction accumulators for one dynamic body (double precision for stable
// summation across many particles).
struct BodyReactionAccum
{
	double *force = nullptr;   // 3
	double *torque = nullptr;  // 3
};

// ---- Neighborhood ---------------------------------------------------------
void launchComputeCellKeys(const DeviceState &s, cudaStream_t stream);
// CUB radix sort of (cellKey -> sortedId). dTemp/tempBytes are persistent.
void launchSortByCell(DeviceState &s, unsigned int *keysAlt, unsigned int *valsAlt,
					  void *dTemp, size_t tempBytes, cudaStream_t stream);
size_t querySortTempBytes(unsigned int capacity);
void launchBuildCellRanges(const DeviceState &s, cudaStream_t stream);

// ---- Boundary (Bender2019) ------------------------------------------------
void launchComputeBoundary(const DeviceState &s, const DeviceBoundaryMap &bmap,
						   real dt, cudaStream_t stream);

// ---- DFSPH core -----------------------------------------------------------
void launchComputeDensity(const DeviceState &s, const DeviceBoundaryMap &bmap,
						  CubicKernelC kernel, cudaStream_t stream);
void launchComputeFactor(const DeviceState &s, const DeviceBoundaryMap &bmap,
						 CubicKernelC kernel, real eps, cudaStream_t stream);

void launchClearAccelGravity(const DeviceState &s, real3 gravity, cudaStream_t stream);
void launchViscosityStandard(const DeviceState &s, CubicKernelC kernel,
							 real viscosity, real dcoef, real h2, cudaStream_t stream);

// CFL: writes (vel + accel*dt)^2 per active particle into scratch (0 otherwise).
void launchComputeMaxVelSq(const DeviceState &s, real dt, real *scratch, cudaStream_t stream);
void launchApplyVelocityUpdate(const DeviceState &s, real dt, cudaStream_t stream);
void launchApplyPosition(const DeviceState &s, real dt, cudaStream_t stream);

// Solver initialisation (warm start + factor scaling). dt is the timestep the
// respective solver runs with.
void launchDivergenceInit(const DeviceState &s, const DeviceBoundaryMap &bmap,
						  CubicKernelC kernel, real dt, cudaStream_t stream);
void launchPressureInit(const DeviceState &s, const DeviceBoundaryMap &bmap,
						CubicKernelC kernel, real dt, cudaStream_t stream);

// Pressure accelerations from a given pressure array (p or p_v).
void launchComputePressureAccel(const DeviceState &s, const DeviceBoundaryMap &bmap,
								CubicKernelC kernel, const real *pressure, real eps,
								int applyBoundaryForces, BodyReactionAccum reaction,
								cudaStream_t stream);

// One Jacobi iteration. isPressure selects source term / hFactor semantics.
// Writes per-particle (-density0*residuum) into errScratch for reduction.
void launchSolveIterate(const DeviceState &s, const DeviceBoundaryMap &bmap,
						CubicKernelC kernel, real *pressure, real hFactor,
						int isPressure, real eps, real *errScratch, cudaStream_t stream);

// Finalisers.
void launchDivergenceFinalizeApply(const DeviceState &s, real dt, cudaStream_t stream);
void launchPressureFinalizeApply(const DeviceState &s, real dt, cudaStream_t stream);

// Host <-> device staging of interleaved real3 arrays (packing).
void launchZeroReaction(BodyReactionAccum reaction, cudaStream_t stream);

// Device-side solver-loop condition update (see kSolverLoopCond). Increments
// *iter, computes avg = *errSum * invN (also stored to *avgOut for stats) and
// writes the loop-continue flag into *cond using the exact host predicate.
void launchSolverLoopCond(const real *errSum, unsigned int *iter, real *avgOut, int *cond,
						  real invN, real eta, unsigned int minIter, unsigned int maxIter,
						  cudaStream_t stream);

} // namespace cuda_dfsph
} // namespace SPH

#endif
