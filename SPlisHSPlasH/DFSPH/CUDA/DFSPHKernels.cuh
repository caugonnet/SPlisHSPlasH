#ifndef __DFSPH_CUDA_DFSPHKernels_cuh__
#define __DFSPH_CUDA_DFSPHKernels_cuh__

#include "DFSPHDeviceState.cuh"
#include "Bender2019MapCUDA.cuh"

#include <cuda_runtime.h>

namespace SPH
{
namespace cuda_dfsph
{

// Net reaction accumulators per boundary body (double precision for stable
// summation across many particles). Layout: [3 * body + component].
struct BodyReactionAccum
{
	double *force = nullptr;   // 3 * numMaps
	double *torque = nullptr;  // 3 * numMaps
};

// ---- Neighborhood ---------------------------------------------------------
void launchComputeCellKeys(const DeviceState &s, cudaStream_t stream);
// CUB radix sort of (cellKey -> sortedId). dTemp/tempBytes are persistent.
void launchSortByCell(DeviceState &s, unsigned int *keysAlt, unsigned int *valsAlt,
					  void *dTemp, size_t tempBytes, cudaStream_t stream);
size_t querySortTempBytes(unsigned int capacity);
void launchBuildCellRanges(const DeviceState &s, cudaStream_t stream);

// Spatial particle reordering (see kGatherParticles).
struct GatherScratch
{
	real3 *pos;
	real3 *vel;
	real *mass;
	int *state;
	real *pressureRho2;
	real *pressureRho2V;
	unsigned int *origId;
};
void launchGatherParticles(const DeviceState &s, const unsigned int *perm,
						   const GatherScratch &out, cudaStream_t stream);

// ---- Boundary (Bender2019) ------------------------------------------------
void launchComputeBoundary(const DeviceState &s, real dt, cudaStream_t stream);

// ---- DFSPH core -----------------------------------------------------------
void launchComputeDensity(const DeviceState &s, CubicKernelC kernel, cudaStream_t stream);
void launchComputeFactor(const DeviceState &s, CubicKernelC kernel, real eps, cudaStream_t stream);

void launchClearAccelGravity(const DeviceState &s, real3 gravity, cudaStream_t stream);
void launchViscosityStandard(const DeviceState &s, CubicKernelC kernel,
							 real viscosity, real dcoef, real h2, cudaStream_t stream);

// CFL: writes (vel + accel*dt)^2 per active particle into scratch (0 otherwise).
void launchComputeMaxVelSq(const DeviceState &s, real dt, real *scratch, cudaStream_t stream,
						   const real *dtPtr = nullptr);
void launchApplyVelocityUpdate(const DeviceState &s, real dt, cudaStream_t stream,
							   const real *dtPtr = nullptr);
void launchApplyPosition(const DeviceState &s, real dt, cudaStream_t stream,
						 const real *dtPtr = nullptr);

// Solver initialisation (warm start + factor scaling). dt is the timestep the
// respective solver runs with.
void launchDivergenceInit(const DeviceState &s, CubicKernelC kernel, real dt, cudaStream_t stream,
						  const real *dtPtr = nullptr);
void launchPressureInit(const DeviceState &s, CubicKernelC kernel, real dt, cudaStream_t stream,
						const real *dtPtr = nullptr);

// Pressure accelerations from a given pressure array (p or p_v).
void launchComputePressureAccel(const DeviceState &s,
								CubicKernelC kernel, const real *pressure, real eps,
								int applyBoundaryForces, BodyReactionAccum reaction,
								cudaStream_t stream);

// One Jacobi iteration. isPressure selects source term / hFactor semantics.
// Writes per-particle (-density0*residuum) into errScratch for reduction.
void launchSolveIterate(const DeviceState &s,
						CubicKernelC kernel, real *pressure, real hFactor,
						int isPressure, real eps, real *errScratch, cudaStream_t stream,
						const real *dtPtr = nullptr);

// Finalisers.
void launchDivergenceFinalizeApply(const DeviceState &s, real dt, cudaStream_t stream,
								   const real *dtPtr = nullptr);
void launchPressureFinalizeApply(const DeviceState &s, real dt, cudaStream_t stream,
								 const real *dtPtr = nullptr);

// Host <-> device staging of interleaved real3 arrays (packing).
void launchZeroReaction(BodyReactionAccum reaction, int numBodies, cudaStream_t stream);

// Device-side solver-loop condition update (see kSolverLoopCond). Increments
// *iter, computes avg = *errSum * invN (also stored to *avgOut for stats) and
// writes the loop-continue flag into *cond using the exact host predicate.
void launchSolverLoopCond(const real *errSum, unsigned int *iter, real *avgOut, int *cond,
						  real invN, real eta, unsigned int minIter, unsigned int maxIter,
						  cudaStream_t stream, const real *dtPtr = nullptr, int etaOverDt = 0);

// Device-side dt control for the whole-step graph: dt2[0] = base dt of the
// current step, dt2[1] = dt used for integration. Rotate promotes the previous
// dtUsed at the start of a relaunch; update applies CFL method 1 (or copies
// dt2[0] when disabled).
void launchCflRotateDt(real *dt2, cudaStream_t stream);
// Pack positions + |velocity| into mapped GL buffers (CUDA/GL interop).
void launchPackRender(const DeviceState &s, real3 *outPos, real *outScalar, cudaStream_t stream);
void launchCflUpdateDt(const real *maxVelSq, real *dt2, real cflFactor, real diameter,
					   real cflMin, real cflMax, int enabled, cudaStream_t stream);

} // namespace cuda_dfsph
} // namespace SPH

#endif
