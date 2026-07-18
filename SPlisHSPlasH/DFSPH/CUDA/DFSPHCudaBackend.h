#ifndef __DFSPH_CUDA_DFSPHCudaBackend_h__
#define __DFSPH_CUDA_DFSPHCudaBackend_h__

// -----------------------------------------------------------------------------
// CUDA-free interface between the host-side facade (compiled at the project's
// C++11 standard) and the device-resident DFSPH backend (compiled as CUDA
// C++17 in a separate target). NOTHING in this header may pull in CUDA, CCCL or
// Eigen so it can be included from the main SPlisHSPlasH library without ABI or
// standard-version coupling. All data crosses the boundary as plain arrays of
// `cudsph_real` and POD descriptors.
// -----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>

namespace SPH
{
namespace cuda_dfsph
{
#ifdef USE_DOUBLE
	using cudsph_real = double;
#else
	using cudsph_real = float;
#endif

	// Which orchestration path the backend uses for one timestep. Mirrors the
	// plan's staged validation (direct streams -> CUDASTF stream -> reusable
	// graph -> device-side conditional loops).
	enum class Orchestrator : int
	{
		DirectStream = 0,   // hand-written cudaStream baseline
		StfStream,          // CUDASTF stream_ctx, host-controlled solver loops
		StfGraph,           // CUDASTF graph_ctx, host-controlled solver loops
		StfConditional      // CUDASTF stackable_ctx + while_graph_scope loops
	};

	// Read-only flattened Bender2019 volume/distance map (one rigid body). The
	// facade exports these from the Discregrid CubicLagrangeDiscreteGrid; the
	// backend copies them to device memory once at init.
	struct BoundaryMapDesc
	{
		// Grid geometry (local/rest space of the rigid body).
		double domainMin[3] = {0, 0, 0};
		double domainMax[3] = {0, 0, 0};
		double cellSize[3] = {0, 0, 0};
		double invCellSize[3] = {0, 0, 0};
		unsigned int resolution[3] = {0, 0, 0};

		// Two fields: 0 = signed distance, 1 = volume.
		// nodes[field]: coefficient array of length nodeCount[field].
		const double *nodes[2] = {nullptr, nullptr};
		std::size_t nodeCount[2] = {0, 0};

		// cells[field]: 32 node indices per cell, length 32 * nCells.
		const unsigned int *cells[2] = {nullptr, nullptr};
		// cellMap[field]: nCells entries (single-index -> compacted cell id or UINT_MAX).
		const unsigned int *cellMap[2] = {nullptr, nullptr};
		std::size_t nCells = 0;

		// Rigid transform (world = R * local + t) and its velocities.
		double rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}; // row-major
		double translation[3] = {0, 0, 0};
		double angularVelocity[3] = {0, 0, 0};
		double linearVelocity[3] = {0, 0, 0};
		double comPosition[3] = {0, 0, 0}; // center of mass (world)
		bool isDynamic = false;
	};

	// Everything the backend needs to allocate fixed-capacity device state and
	// upload the initial scene. Pointers are borrowed for the duration of the
	// initialize() call only.
	struct SceneDesc
	{
		unsigned int numParticles = 0;      // active == capacity for the fixed scenes
		unsigned int capacity = 0;

		const cudsph_real *positions = nullptr;   // 3 * numParticles
		const cudsph_real *velocities = nullptr;  // 3 * numParticles
		const cudsph_real *masses = nullptr;      // numParticles
		const int *particleState = nullptr;       // numParticles (0 = Active)

		cudsph_real density0 = 1000;
		cudsph_real volume = 0;                    // per-particle rest volume V
		cudsph_real supportRadius = 0;
		cudsph_real particleRadius = 0;
		bool sim2D = false;

		// Solver parameters (mirrors TimeStepDFSPH members).
		unsigned int minIterations = 2;
		unsigned int maxIterations = 100;
		cudsph_real maxError = static_cast<cudsph_real>(0.01);
		unsigned int maxIterationsV = 100;
		cudsph_real maxErrorV = static_cast<cudsph_real>(0.1);
		bool enableDivergenceSolver = true;

		// Standard XSPH-free viscosity coefficient (0 disables). Kept minimal for
		// the first vertical slice.
		cudsph_real viscosity = 0;

		// CFL time-step control (see Simulation::updateTimeStepSizeCFL).
		int cflMethod = 1;
		cudsph_real cflFactor = static_cast<cudsph_real>(1.0);
		cudsph_real cflMinTimeStepSize = static_cast<cudsph_real>(1e-5);
		cudsph_real cflMaxTimeStepSize = static_cast<cudsph_real>(0.005);

		int numBoundaries = 0;
		const BoundaryMapDesc *boundaries = nullptr; // numBoundaries entries
	};

	// Per-step inputs that can change between steps (gravity, dt, moving body
	// state). Positions/velocities remain device-authoritative.
	struct StepDesc
	{
		cudsph_real dt = 0;
		cudsph_real gravity[3] = {0, static_cast<cudsph_real>(-9.81), 0};
		Orchestrator orchestrator = Orchestrator::StfStream;

		// Updated dynamic-body kinematics (motor coupling stage 1). Length =
		// numBoundaries * (see BoundaryMapDesc fields). Null keeps init state.
		const double *bodyRotation = nullptr;    // 9 * numBoundaries
		const double *bodyTranslation = nullptr; // 3 * numBoundaries
		const double *bodyAngularVel = nullptr;  // 3 * numBoundaries
		const double *bodyLinearVel = nullptr;   // 3 * numBoundaries
		const double *bodyComPosition = nullptr; // 3 * numBoundaries
	};

	// Snapshot of the device state mirrored back to the host for the existing
	// renderer / exporters. All arrays are caller-owned and sized to capacity.
	struct HostMirror
	{
		cudsph_real *positions = nullptr;   // 3 * numParticles
		cudsph_real *velocities = nullptr;  // 3 * numParticles
		cudsph_real *density = nullptr;     // numParticles
		cudsph_real *factor = nullptr;      // numParticles
		cudsph_real *densityAdv = nullptr;  // numParticles
		cudsph_real *pressureRho2 = nullptr;   // numParticles
		cudsph_real *pressureRho2V = nullptr;  // numParticles
		cudsph_real *pressureAccel = nullptr;  // 3 * numParticles
	};

	// Diagnostics collected per step (iteration counts, timings, transfers).
	struct StepStats
	{
		unsigned int iterations = 0;
		unsigned int iterationsV = 0;
		double avgDensityErr = 0;
		double avgDensityErrV = 0;
		cudsph_real dtUsed = 0;       // dt actually used for integration this step
		float lastStepMs = 0;         // GPU time for the whole timestep
		unsigned long long graphBuilds = 0;
		unsigned long long graphLaunches = 0;
		unsigned long long hostSyncs = 0;
		unsigned int neighborHighWater = 0;
		bool neighborOverflow = false;
	};

	// Net reaction force/torque per dynamic body (motor coupling stage 2).
	struct BoundaryReaction
	{
		double force[3] = {0, 0, 0};
		double torque[3] = {0, 0, 0};
	};

	class DFSPHCudaBackend
	{
	public:
		virtual ~DFSPHCudaBackend() {}

		// Allocate fixed-capacity device state and upload the initial scene.
		virtual void initialize(const SceneDesc &scene) = 0;

		// Re-upload host positions/velocities/state (used on reset / state load).
		virtual void uploadState(const cudsph_real *positions,
								 const cudsph_real *velocities,
								 const int *particleState) = 0;

		// Advance one DFSPH timestep entirely on the device.
		virtual void step(const StepDesc &step, StepStats &stats) = 0;

		// Copy the requested device fields into the caller's host arrays.
		virtual void copyStateToHost(const HostMirror &mirror) = 0;

		// Pinned (page-locked) host memory for fast D2H mirroring. The facade is
		// CUDA-free, so allocation goes through the backend. Returns nullptr on
		// failure; buffers must be released with freeHostPinned before the
		// backend is destroyed.
		virtual cudsph_real *allocHostPinned(size_t count) = 0;
		virtual void freeHostPinned(cudsph_real *p) = 0;

		// CUDA/OpenGL interop: fill the given VBOs directly from device state
		// (positions: 3 floats/particle; scalar: velocity magnitude). Buffers are
		// registered lazily and re-registered when the ids change; must be called
		// from the thread owning the GL context, after step(). Returns false when
		// interop is unavailable (no GL context, registration failure) so the
		// caller can fall back to host-side rendering.
		virtual bool fillGlRenderBuffers(unsigned int posVbo, unsigned int scalarVbo) = 0;

		// Fetch net boundary reactions (valid after step()). Length = numBoundaries.
		virtual void getBoundaryReactions(BoundaryReaction *out) const = 0;

		// Current CFL-limited timestep proposal computed on device (0 if unset).
		virtual cudsph_real suggestedTimeStep() const = 0;
	};

	// Factory implemented in DFSPHCudaBackend.cu. Returns nullptr if no usable
	// CUDA device is present. Caller owns the returned pointer.
	DFSPHCudaBackend *createDFSPHCudaBackend();

} // namespace cuda_dfsph
} // namespace SPH

#endif
