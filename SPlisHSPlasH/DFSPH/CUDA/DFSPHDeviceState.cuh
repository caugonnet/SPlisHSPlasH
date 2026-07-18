#ifndef __DFSPH_CUDA_DFSPHDeviceState_cuh__
#define __DFSPH_CUDA_DFSPHDeviceState_cuh__

#include "DFSPHCudaBackend.h"

#include <cuda_runtime.h>

namespace SPH
{
namespace cuda_dfsph
{

using real = cudsph_real;

#ifdef USE_DOUBLE
using real3 = double3;
__host__ __device__ __forceinline__ real3 make_r3(real x, real y, real z) { return make_double3(x, y, z); }
#else
using real3 = float3;
__host__ __device__ __forceinline__ real3 make_r3(real x, real y, real z) { return make_float3(x, y, z); }
#endif

__host__ __device__ __forceinline__ real3 operator+(real3 a, real3 b) { return make_r3(a.x + b.x, a.y + b.y, a.z + b.z); }
__host__ __device__ __forceinline__ real3 operator-(real3 a, real3 b) { return make_r3(a.x - b.x, a.y - b.y, a.z - b.z); }
__host__ __device__ __forceinline__ real3 operator*(real s, real3 a) { return make_r3(s * a.x, s * a.y, s * a.z); }
__host__ __device__ __forceinline__ real3 operator*(real3 a, real s) { return make_r3(s * a.x, s * a.y, s * a.z); }
__host__ __device__ __forceinline__ void operator+=(real3 &a, real3 b) { a.x += b.x; a.y += b.y; a.z += b.z; }
__host__ __device__ __forceinline__ void operator-=(real3 &a, real3 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; }
__host__ __device__ __forceinline__ real dot(real3 a, real3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ __forceinline__ real sqnorm(real3 a) { return dot(a, a); }
__host__ __device__ __forceinline__ real3 cross(real3 a, real3 b)
{
	return make_r3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

static const unsigned int CELL_EMPTY = 0xffffffffu;

// Cubic spline kernel constants (match SPH::CubicKernel). k = 8/(pi h^3),
// l = 48/(pi h^3).
struct CubicKernelC
{
	real radius;
	real k;
	real l;
	real wZero;
};

__host__ __device__ __forceinline__ CubicKernelC makeCubicKernel(real h)
{
	CubicKernelC c;
	const real pi = static_cast<real>(3.14159265358979323846);
	const real h3 = h * h * h;
	c.radius = h;
	c.k = static_cast<real>(8.0) / (pi * h3);
	c.l = static_cast<real>(48.0) / (pi * h3);
	c.wZero = c.k; // W(0): q=0 -> k*(1) = k
	return c;
}

__device__ __forceinline__ real cubicW(const CubicKernelC &c, real r)
{
	real res = 0;
	const real q = r / c.radius;
	if (q <= static_cast<real>(1.0))
	{
		if (q <= static_cast<real>(0.5))
		{
			const real q2 = q * q;
			const real q3 = q2 * q;
			res = c.k * (static_cast<real>(6.0) * q3 - static_cast<real>(6.0) * q2 + static_cast<real>(1.0));
		}
		else
		{
			const real f = static_cast<real>(1.0) - q;
			res = c.k * (static_cast<real>(2.0) * f * f * f);
		}
	}
	return res;
}

__device__ __forceinline__ real3 cubicGradW(const CubicKernelC &c, real3 r)
{
	real3 res = make_r3(0, 0, 0);
	const real rl = sqrt(sqnorm(r));
	const real q = rl / c.radius;
	if ((rl > static_cast<real>(1.0e-9)) && (q <= static_cast<real>(1.0)))
	{
		real3 gradq = (static_cast<real>(1.0) / rl) * r;
		gradq = (static_cast<real>(1.0) / c.radius) * gradq;
		if (q <= static_cast<real>(0.5))
			res = (c.l * q * (static_cast<real>(3.0) * q - static_cast<real>(2.0))) * gradq;
		else
		{
			const real f = static_cast<real>(1.0) - q;
			res = (c.l * (-f * f)) * gradq;
		}
	}
	return res;
}

// Fixed-capacity, device-resident DFSPH state. All pointers are device memory.
// Passed to kernels by value (POD).
struct DeviceState
{
	unsigned int n = 0;         // active particle count
	unsigned int capacity = 0;

	// SoA particle fields.
	real3 *pos = nullptr;
	real3 *vel = nullptr;
	real3 *accel = nullptr;
	real3 *pressureAccel = nullptr;
	real *mass = nullptr;
	real *density = nullptr;
	real *factor = nullptr;
	real *densityAdv = nullptr;
	real *pressureRho2 = nullptr;
	real *pressureRho2V = nullptr;
	int *state = nullptr;       // ParticleState (0 == Active)

	// Constants.
	real density0 = 1000;
	real volume = 0;
	real supportRadius = 0;
	real particleRadius = 0;
	int sim2D = 0;

	// Uniform grid (fixed capacity).
	real3 gridOrigin = make_r3(0, 0, 0);
	int gridDim[3] = {1, 1, 1};
	unsigned int numCells = 1;
	real cellSize = 1;

	unsigned int *cellKey = nullptr;    // capacity
	unsigned int *sortedId = nullptr;   // capacity
	unsigned int *cellStart = nullptr;  // numCells
	unsigned int *cellEnd = nullptr;    // numCells

	// Per-particle boundary contribution (single boundary supported for the
	// first slice; index 0). volume<=0 means no boundary neighbor.
	real *boundaryVolume = nullptr;     // capacity
	real3 *boundaryXj = nullptr;        // capacity
};

__device__ __forceinline__ int3 cellCoord(const DeviceState &s, real3 x)
{
	int3 c;
	c.x = static_cast<int>(floor((x.x - s.gridOrigin.x) / s.cellSize));
	c.y = static_cast<int>(floor((x.y - s.gridOrigin.y) / s.cellSize));
	c.z = static_cast<int>(floor((x.z - s.gridOrigin.z) / s.cellSize));
	return c;
}

__device__ __forceinline__ unsigned int cellLinear(const DeviceState &s, int3 c)
{
	if (c.x < 0) c.x = 0; if (c.x >= s.gridDim[0]) c.x = s.gridDim[0] - 1;
	if (c.y < 0) c.y = 0; if (c.y >= s.gridDim[1]) c.y = s.gridDim[1] - 1;
	if (c.z < 0) c.z = 0; if (c.z >= s.gridDim[2]) c.z = s.gridDim[2] - 1;
	return static_cast<unsigned int>(c.x + s.gridDim[0] * (c.y + s.gridDim[1] * c.z));
}

} // namespace cuda_dfsph
} // namespace SPH

#endif
