// -----------------------------------------------------------------------------
// Device-resident DFSPH backend implementation.
//
// Owns the fixed-capacity device SoA state, the flattened Bender2019 volume map,
// the uniform-grid neighborhood, and persistent scratch. Advances one DFSPH
// timestep either through a hand-written cudaStream baseline (DirectStream) or a
// CUDASTF `context` that sequences the coarse phases with void-interface tokens
// (StfStream). The graph / conditional orchestrators are introduced in a later
// stage and currently reuse the StfStream host-controlled loops.
// -----------------------------------------------------------------------------

#include "DFSPHCudaBackend.h"
#include "DFSPHDeviceState.cuh"
#include "DFSPHKernels.cuh"
#include "Bender2019MapCUDA.cuh"

#include <cub/device/device_reduce.cuh>
#include <cuda/experimental/stf.cuh>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

namespace SPH
{
namespace cuda_dfsph
{

namespace stf = cuda::experimental::stf;

static void cudaCheck(cudaError_t e, const char *what)
{
	if (e != cudaSuccess)
	{
		std::fprintf(stderr, "[dfsph-cuda] CUDA error (%s): %s\n", what, cudaGetErrorString(e));
		std::abort();
	}
}

template <typename T>
static T *devAlloc(size_t count)
{
	T *p = nullptr;
	cudaCheck(cudaMalloc(&p, count * sizeof(T)), "cudaMalloc");
	return p;
}

class DFSPHCudaBackendImpl final : public DFSPHCudaBackend
{
public:
	DFSPHCudaBackendImpl() = default;
	~DFSPHCudaBackendImpl() override { destroy(); }

	void initialize(const SceneDesc &scene) override;
	void uploadState(const cudsph_real *positions, const cudsph_real *velocities, const int *particleState) override;
	void step(const StepDesc &sd, StepStats &stats) override;
	void copyStateToHost(const HostMirror &mirror) override;
	cudsph_real *allocHostPinned(size_t count) override
	{
		cudsph_real *p = nullptr;
		if (cudaMallocHost(&p, count * sizeof(cudsph_real)) != cudaSuccess)
			return nullptr;
		return p;
	}
	void freeHostPinned(cudsph_real *p) override
	{
		if (p) cudaFreeHost(p);
	}
	void getBoundaryReactions(BoundaryReaction *out) const override;
	cudsph_real suggestedTimeStep() const override { return m_suggestedDt; }

private:
	void destroy();
	void computeGridBounds(const SceneDesc &scene);
	void neighborhood(cudaStream_t stream);
	real reduceSum(real *scratch, cudaStream_t stream);
	real reduceMax(real *scratch, cudaStream_t stream);
	// Runs the full timestep on a raw stream. Used by both DirectStream and, via
	// the task bodies, by the CUDASTF path.
	void runTimestep(const StepDesc &sd, StepStats &stats, cudaStream_t stream, bool useStf);
	// One Jacobi solver loop executed entirely on-device via a CUDA Graph
	// conditional `while` node (CUDASTF stackable_ctx::while_graph_scope). No
	// host synchronization per iteration. Returns the iteration count and the
	// final average density error via avgErrOut.
	unsigned int runSolverLoopConditional(real *pressure, real hFactor, int isPressure,
										  real eta, unsigned int minIter, unsigned int maxIter,
										  real &avgErrOut, cudaStream_t stream);
	// Records one solver while-loop (counter reset task + conditional node) into
	// the current scope of sctx. `slot` selects the m_dIter/m_dErrOut entry
	// (0 = divergence, 1 = pressure) so both loops can coexist in one graph.
	// When dtPtr is non-null the kernels read dt from device memory (whole-step
	// graph with device-resident CFL): hFactor is derived per iteration and the
	// divergence eta (which scales with 1/dt) is divided on device.
	template <typename Tok>
	void recordSolverLoop(stf::stackable_ctx &sctx, Tok &tok,
						  real *pressure, real hFactor, int isPressure,
						  real eta, unsigned int minIter, unsigned int maxIter, unsigned int slot,
						  const real *dtPtr = nullptr);
	// The whole timestep as a single pushed CUDA graph (both solver loops as
	// nested conditional nodes). dt is device-resident (m_dDt): a rotate kernel
	// promotes the previous dtUsed at graph start and a device CFL kernel
	// computes the integration dt, so the same recorded graph is valid across
	// steps even under adaptive CFL.
	void runTimestepGraph(const StepDesc &sd, StepStats &stats, cudaStream_t stream);

	DeviceState m_s;
	CubicKernelC m_kernel;
	real m_eps = static_cast<real>(1.0e-5);
	real m_viscDCoef = static_cast<real>(10.0);
	real m_h2support = 0;

	// solver params
	unsigned int m_minIterations = 2, m_maxIterations = 100;
	unsigned int m_maxIterationsV = 100;
	real m_maxError = static_cast<real>(0.01), m_maxErrorV = static_cast<real>(0.1);
	bool m_enableDivergence = true;
	real m_viscosity = 0;
	int m_cflMethod = 1;
	real m_cflFactor = 1, m_cflMin = static_cast<real>(1e-5), m_cflMax = static_cast<real>(0.005);
	real m_suggestedDt = 0;

	// boundary maps (single supported for kernels; index 0)
	std::vector<DeviceBoundaryMap> m_maps;
	DeviceBoundaryMap m_map0{}; // primary map passed to kernels (valid==0 if none)

	// scratch
	unsigned int *m_baseKey = nullptr;  // fixed allocation for unsorted keys
	unsigned int *m_baseId = nullptr;   // fixed allocation for unsorted ids
	unsigned int *m_keysAlt = nullptr;
	unsigned int *m_valsAlt = nullptr;
	real *m_errScratch = nullptr;
	void *m_cubTemp = nullptr;
	size_t m_cubTempBytes = 0;
	real *m_dReduce = nullptr;   // device scalar result
	real *m_hReduce = nullptr;   // pinned host scalar
	unsigned int *m_dIter = nullptr; // device iteration counters for while-graph loops [2]: 0=divergence, 1=pressure
	real *m_dErrOut = nullptr;       // final avg density error per while-graph loop [2]
	real *m_dDt = nullptr;           // device-resident dt [2]: 0=base dt of the step, 1=dt used for integration
	real m_lastDtUsed = -1;          // host copy of m_dDt[1] after the last graph step (seed check)

	// Persistent CUDASTF context for the conditional while-graph solver loops.
	// Kept alive across timesteps so the executable-graph cache turns per-step
	// child-graph instantiation into a cheap cudaGraphExecUpdate.
	std::unique_ptr<stf::stackable_ctx> m_whileCtx;

	// Whole-timestep graph (StfGraph): recorded once, relaunched every step
	// (dt is device-resident, so adaptive CFL relaunches too). Lives in its own
	// context because push() is forbidden while a pop-epilogue is pending (the
	// launchable_graph holds the pop open).
	std::unique_ptr<stf::stackable_ctx> m_graphCtx;
	stf::stackable_ctx::launchable_graph m_stepGraph;
	real m_graphGrav[3] = {0, 0, 0};     // gravity baked into the step graph
	BodyReactionAccum m_reaction{};
	double *m_hReaction = nullptr; // pinned host [6]

	bool m_ready = false;
};

void DFSPHCudaBackendImpl::computeGridBounds(const SceneDesc &scene)
{
	real3 lo = make_r3(1e30f, 1e30f, 1e30f);
	real3 hi = make_r3(-1e30f, -1e30f, -1e30f);

	// Prefer the boundary map world-space domain (fluid stays inside the tank).
	bool haveBounds = false;
	if (scene.numBoundaries > 0 && scene.boundaries != nullptr)
	{
		for (int b = 0; b < scene.numBoundaries; ++b)
		{
			const BoundaryMapDesc &d = scene.boundaries[b];
			if (d.nodes[0] == nullptr) continue;
			for (int c = 0; c < 8; ++c)
			{
				double lx = (c & 1) ? d.domainMax[0] : d.domainMin[0];
				double ly = (c & 2) ? d.domainMax[1] : d.domainMin[1];
				double lz = (c & 4) ? d.domainMax[2] : d.domainMin[2];
				double wx = d.rotation[0] * lx + d.rotation[1] * ly + d.rotation[2] * lz + d.translation[0];
				double wy = d.rotation[3] * lx + d.rotation[4] * ly + d.rotation[5] * lz + d.translation[1];
				double wz = d.rotation[6] * lx + d.rotation[7] * ly + d.rotation[8] * lz + d.translation[2];
				lo.x = std::fmin(lo.x, (real)wx); lo.y = std::fmin(lo.y, (real)wy); lo.z = std::fmin(lo.z, (real)wz);
				hi.x = std::fmax(hi.x, (real)wx); hi.y = std::fmax(hi.y, (real)wy); hi.z = std::fmax(hi.z, (real)wz);
				haveBounds = true;
			}
		}
	}
	if (!haveBounds)
	{
		for (unsigned int i = 0; i < scene.numParticles; ++i)
		{
			real x = scene.positions[3 * i + 0];
			real y = scene.positions[3 * i + 1];
			real z = scene.positions[3 * i + 2];
			lo.x = std::fmin(lo.x, x); lo.y = std::fmin(lo.y, y); lo.z = std::fmin(lo.z, z);
			hi.x = std::fmax(hi.x, x); hi.y = std::fmax(hi.y, y); hi.z = std::fmax(hi.z, z);
		}
	}

	const real cell = scene.supportRadius;
	const real margin = static_cast<real>(2.0) * cell;
	lo = make_r3(lo.x - margin, lo.y - margin, lo.z - margin);
	hi = make_r3(hi.x + margin, hi.y + margin, hi.z + margin);

	m_s.gridOrigin = lo;
	m_s.cellSize = cell;
	for (int d = 0; d < 3; ++d)
	{
		real ext = ((d == 0) ? hi.x - lo.x : (d == 1) ? hi.y - lo.y : hi.z - lo.z);
		int n = static_cast<int>(std::ceil(ext / cell));
		if (n < 1) n = 1;
		m_s.gridDim[d] = n;
	}
	m_s.numCells = static_cast<unsigned int>(m_s.gridDim[0]) * m_s.gridDim[1] * m_s.gridDim[2];
}

void DFSPHCudaBackendImpl::initialize(const SceneDesc &scene)
{
	destroy();

	const unsigned int cap = scene.capacity > 0 ? scene.capacity : scene.numParticles;
	m_s.n = scene.numParticles;
	m_s.capacity = cap;
	m_s.density0 = scene.density0;
	m_s.volume = scene.volume;
	m_s.supportRadius = scene.supportRadius;
	m_s.particleRadius = scene.particleRadius;
	m_s.sim2D = scene.sim2D ? 1 : 0;

	m_minIterations = scene.minIterations;
	m_maxIterations = scene.maxIterations;
	m_maxIterationsV = scene.maxIterationsV;
	m_maxError = scene.maxError;
	m_maxErrorV = scene.maxErrorV;
	m_enableDivergence = scene.enableDivergenceSolver;
	m_viscosity = scene.viscosity;
	m_cflMethod = scene.cflMethod;
	m_cflFactor = scene.cflFactor;
	m_cflMin = scene.cflMinTimeStepSize;
	m_cflMax = scene.cflMaxTimeStepSize;
	m_viscDCoef = scene.sim2D ? static_cast<real>(8.0) : static_cast<real>(10.0);
	m_h2support = scene.supportRadius * scene.supportRadius;

	m_kernel = makeCubicKernel(scene.supportRadius);

	computeGridBounds(scene);

	// Particle SoA.
	m_s.pos = devAlloc<real3>(cap);
	m_s.vel = devAlloc<real3>(cap);
	m_s.accel = devAlloc<real3>(cap);
	m_s.pressureAccel = devAlloc<real3>(cap);
	m_s.mass = devAlloc<real>(cap);
	m_s.density = devAlloc<real>(cap);
	m_s.factor = devAlloc<real>(cap);
	m_s.densityAdv = devAlloc<real>(cap);
	m_s.pressureRho2 = devAlloc<real>(cap);
	m_s.pressureRho2V = devAlloc<real>(cap);
	m_s.state = devAlloc<int>(cap);
	m_s.boundaryVolume = devAlloc<real>(cap);
	m_s.boundaryXj = devAlloc<real3>(cap);

	// Grid.
	m_baseKey = devAlloc<unsigned int>(cap);
	m_baseId = devAlloc<unsigned int>(cap);
	m_s.cellKey = m_baseKey;
	m_s.sortedId = m_baseId;
	m_s.cellStart = devAlloc<unsigned int>(m_s.numCells);
	m_s.cellEnd = devAlloc<unsigned int>(m_s.numCells);
	m_keysAlt = devAlloc<unsigned int>(cap);
	m_valsAlt = devAlloc<unsigned int>(cap);
	m_errScratch = devAlloc<real>(cap);

	// Upload initial particle state.
	cudaCheck(cudaMemcpy(m_s.pos, scene.positions, 3 * scene.numParticles * sizeof(real), cudaMemcpyHostToDevice), "pos H2D");
	cudaCheck(cudaMemcpy(m_s.vel, scene.velocities, 3 * scene.numParticles * sizeof(real), cudaMemcpyHostToDevice), "vel H2D");
	cudaCheck(cudaMemcpy(m_s.mass, scene.masses, scene.numParticles * sizeof(real), cudaMemcpyHostToDevice), "mass H2D");
	cudaCheck(cudaMemcpy(m_s.state, scene.particleState, scene.numParticles * sizeof(int), cudaMemcpyHostToDevice), "state H2D");
	cudaCheck(cudaMemset(m_s.pressureRho2, 0, cap * sizeof(real)), "clear p");
	cudaCheck(cudaMemset(m_s.pressureRho2V, 0, cap * sizeof(real)), "clear pv");

	// Boundary maps.
	m_maps.clear();
	for (int b = 0; b < scene.numBoundaries; ++b)
		m_maps.push_back(uploadBoundaryMap(scene.boundaries[b]));
	if (!m_maps.empty())
		m_map0 = m_maps[0];
	else
		m_map0.valid = 0;

	// Scratch: size cub temp for the largest of sort / reduce.
	size_t sortBytes = querySortTempBytes(cap);
	size_t sumBytes = 0, maxBytes = 0;
	cub::DeviceReduce::Sum(nullptr, sumBytes, m_errScratch, m_errScratch, static_cast<int>(cap));
	cub::DeviceReduce::Max(nullptr, maxBytes, m_errScratch, m_errScratch, static_cast<int>(cap));
	m_cubTempBytes = sortBytes;
	if (sumBytes > m_cubTempBytes) m_cubTempBytes = sumBytes;
	if (maxBytes > m_cubTempBytes) m_cubTempBytes = maxBytes;
	cudaCheck(cudaMalloc(&m_cubTemp, m_cubTempBytes), "cub temp");

	m_dReduce = devAlloc<real>(1);
	cudaCheck(cudaMallocHost(&m_hReduce, sizeof(real)), "pinned reduce");
	m_dIter = devAlloc<unsigned int>(2);
	m_dErrOut = devAlloc<real>(2);
	cudaCheck(cudaMemset(m_dIter, 0, 2 * sizeof(unsigned int)), "clear iter");
	cudaCheck(cudaMemset(m_dErrOut, 0, 2 * sizeof(real)), "clear errout");
	m_dDt = devAlloc<real>(2);
	cudaCheck(cudaMemset(m_dDt, 0, 2 * sizeof(real)), "clear dt");
	m_lastDtUsed = -1;
	m_reaction.force = devAlloc<double>(3);
	m_reaction.torque = devAlloc<double>(3);
	cudaCheck(cudaMallocHost(&m_hReaction, 6 * sizeof(double)), "pinned reaction");

	m_ready = true;
}

void DFSPHCudaBackendImpl::uploadState(const cudsph_real *positions, const cudsph_real *velocities, const int *particleState)
{
	if (!m_ready) return;
	if (positions)
		cudaCheck(cudaMemcpy(m_s.pos, positions, 3 * m_s.n * sizeof(real), cudaMemcpyHostToDevice), "pos H2D");
	if (velocities)
		cudaCheck(cudaMemcpy(m_s.vel, velocities, 3 * m_s.n * sizeof(real), cudaMemcpyHostToDevice), "vel H2D");
	if (particleState)
		cudaCheck(cudaMemcpy(m_s.state, particleState, m_s.n * sizeof(int), cudaMemcpyHostToDevice), "state H2D");
	cudaCheck(cudaMemset(m_s.pressureRho2, 0, m_s.capacity * sizeof(real)), "clear p");
	cudaCheck(cudaMemset(m_s.pressureRho2V, 0, m_s.capacity * sizeof(real)), "clear pv");
}

void DFSPHCudaBackendImpl::neighborhood(cudaStream_t stream)
{
	// Always start from the fixed base buffers, then sort into the alt buffers.
	// launchSortByCell re-points m_s.cellKey / m_s.sortedId at the alt buffers,
	// which the boundary/density/factor/solver kernels then read.
	m_s.cellKey = m_baseKey;
	m_s.sortedId = m_baseId;
	launchComputeCellKeys(m_s, stream);
	launchSortByCell(m_s, m_keysAlt, m_valsAlt, m_cubTemp, m_cubTempBytes, stream);
	launchBuildCellRanges(m_s, stream);
}

real DFSPHCudaBackendImpl::reduceSum(real *scratch, cudaStream_t stream)
{
	size_t bytes = m_cubTempBytes;
	cub::DeviceReduce::Sum(m_cubTemp, bytes, scratch, m_dReduce, static_cast<int>(m_s.n), stream);
	cudaCheck(cudaMemcpyAsync(m_hReduce, m_dReduce, sizeof(real), cudaMemcpyDeviceToHost, stream), "reduce D2H");
	cudaCheck(cudaStreamSynchronize(stream), "reduce sync");
	return *m_hReduce;
}

real DFSPHCudaBackendImpl::reduceMax(real *scratch, cudaStream_t stream)
{
	size_t bytes = m_cubTempBytes;
	cub::DeviceReduce::Max(m_cubTemp, bytes, scratch, m_dReduce, static_cast<int>(m_s.n), stream);
	cudaCheck(cudaMemcpyAsync(m_hReduce, m_dReduce, sizeof(real), cudaMemcpyDeviceToHost, stream), "reduce D2H");
	cudaCheck(cudaStreamSynchronize(stream), "reduce sync");
	return *m_hReduce;
}

// Loop-continue predicate for while_graph_scope::update_cond. A namespace-scope
// functor (not an extended __device__ lambda) because nvcc forbids extended
// lambdas inside private member functions.
struct LoopCondNonZero
{
	__device__ bool operator()(stf::scalar_view<int> cond) const { return *cond != 0; }
};

template <typename Tok>
void DFSPHCudaBackendImpl::recordSolverLoop(stf::stackable_ctx &sctx, Tok &tok,
											real *pressure, real hFactor, int isPressure,
											real eta, unsigned int minIter, unsigned int maxIter, unsigned int slot,
											const real *dtPtr)
{
	const int etaOverDt = (dtPtr != nullptr && !isPressure) ? 1 : 0;
	const real invN = static_cast<real>(1.0) / static_cast<real>(m_s.n);
	unsigned int *dIter = m_dIter + slot;
	real *dAvg = m_dErrOut + slot;

	auto lCond = sctx.logical_data(stf::shape_of<stf::scalar_view<int>>()).set_symbol("loop_cond");
	// Counter reset on its own token: independent of the preceding solver
	// phases, so it can overlap them; the loop body joins both tokens.
	auto tokIter = sctx.token();
	sctx.task(tokIter.rw())->*[=](cudaStream_t s) {
		cudaMemsetAsync(dIter, 0, sizeof(unsigned int), s);
	};
	{
		auto wg = sctx.while_graph_scope();
		// Whole Jacobi iteration as one captured task: pressure accelerations,
		// Jacobi update + per-particle error, CUB error reduction, and the
		// device-side condition update. All solver state lives in raw device
		// buffers; the token serializes against the surrounding phases and the
		// loop-continue flag orders update_cond after the body.
		sctx.task(tok.rw(), tokIter.rw(), lCond.write())->*[=](cudaStream_t s, auto cond) {
			launchComputePressureAccel(m_s, m_map0, m_kernel, pressure, m_eps, 0, m_reaction, s);
			launchSolveIterate(m_s, m_map0, m_kernel, pressure, hFactor, isPressure, m_eps, m_errScratch, s, dtPtr);
			size_t bytes = m_cubTempBytes;
			cub::DeviceReduce::Sum(m_cubTemp, bytes, m_errScratch, m_dReduce, static_cast<int>(m_s.n), s);
			launchSolverLoopCond(m_dReduce, dIter, dAvg, cond.addr, invN, eta, minIter, maxIter, s, dtPtr, etaOverDt);
		};
		wg.update_cond(lCond.read())->*LoopCondNonZero{};
	}
}

unsigned int DFSPHCudaBackendImpl::runSolverLoopConditional(real *pressure, real hFactor, int isPressure,
															real eta, unsigned int minIter, unsigned int maxIter,
															real &avgErrOut, cudaStream_t stream)
{
	// The while-graph runs in its own CUDASTF context (and streams): order it
	// after the preceding raw-stream work.
	cudaCheck(cudaStreamSynchronize(stream), "pre-while sync");

	if (!m_whileCtx)
		m_whileCtx = std::make_unique<stf::stackable_ctx>();
	stf::stackable_ctx &sctx = *m_whileCtx;

	const unsigned int slot = isPressure ? 1 : 0;
	{
		auto tok = sctx.token();
		recordSolverLoop(sctx, tok, pressure, hFactor, isPressure, eta, minIter, maxIter, slot);
	}
	// Wait for the while-graph launched by the scope pop to complete.
	cudaCheck(cudaStreamSynchronize(sctx.fence()), "while fence");

	unsigned int iters = 0;
	real avg = 0;
	cudaCheck(cudaMemcpy(&iters, m_dIter + slot, sizeof(unsigned int), cudaMemcpyDeviceToHost), "iters D2H");
	cudaCheck(cudaMemcpy(&avg, m_dErrOut + slot, sizeof(real), cudaMemcpyDeviceToHost), "err D2H");
	avgErrOut = avg;
	return iters;
}

void DFSPHCudaBackendImpl::runTimestepGraph(const StepDesc &sd, StepStats &stats, cudaStream_t stream)
{
	const real dt0 = sd.dt;
	const real3 gravity = make_r3(sd.gravity[0], sd.gravity[1], sd.gravity[2]);

	// Refresh moving-body transform if provided (motor coupling stage 1).
	if (sd.bodyRotation && !m_maps.empty())
	{
		for (size_t b = 0; b < m_maps.size(); ++b)
		{
			updateBoundaryMapTransform(m_maps[b],
									   sd.bodyRotation + 9 * b, sd.bodyTranslation + 3 * b,
									   sd.bodyAngularVel + 3 * b, sd.bodyLinearVel + 3 * b,
									   sd.bodyComPosition + 3 * b);
		}
		m_map0 = m_maps[0];
	}

	cudaCheck(cudaStreamSynchronize(stream), "pre-graph sync");
	if (!m_graphCtx)
		m_graphCtx = std::make_unique<stf::stackable_ctx>();
	stf::stackable_ctx &sctx = *m_graphCtx;

	// Seed the device-resident dt when the caller's dt does not match the last
	// device-computed dtUsed (first step, or the user changed dt externally).
	// Otherwise the in-graph rotate kernel promotes dtUsed -> base dt.
	if (dt0 != m_lastDtUsed)
	{
		const real seed[2] = {dt0, dt0};
		cudaCheck(cudaMemcpy(m_dDt, seed, sizeof(seed), cudaMemcpyHostToDevice), "dt seed");
		m_lastDtUsed = dt0;
	}

	// The recorded graph is only a function of gravity and the boundary
	// transforms (dt lives in device memory; all pointers, counts and solver
	// parameters are fixed after initialize()). Rebuild when any of these
	// change; otherwise relaunch the stored executable graph. Moving bodies
	// bake a new m_map0 into the kernel arguments, so they force a rebuild
	// every step for now.
	const bool rebuild = !m_stepGraph.valid()
		|| m_graphGrav[0] != gravity.x || m_graphGrav[1] != gravity.y || m_graphGrav[2] != gravity.z
		|| sd.bodyRotation != nullptr;
	if (rebuild)
	{
		// Release the previous pop before pushing again (push() is forbidden
		// while a pop-epilogue is pending).
		m_stepGraph = {};
		sctx.push();
		auto tok = sctx.token();
		auto tokBoundary = sctx.token();

		// 0. Promote the previous step's dtUsed to this step's base dt.
		sctx.task(tok.rw())->*[=](cudaStream_t s) {
			launchCflRotateDt(m_dDt, s);
		};
		// 1. Neighborhood build. neighborhood() re-points m_s.cellKey/sortedId on
		// the host during recording, so later lambdas (which read m_s through
		// `this` when they are recorded) see the sorted buffers.
		sctx.task(tok.rw())->*[=](cudaStream_t s) {
			neighborhood(s);
		};
		// 2. Boundary contribution (volume-map evaluation). Depends only on the
		// particle positions, not on the sorted grid — a separate token lets STF
		// run it concurrently with the whole neighborhood build.
		sctx.task(tokBoundary.rw())->*[=](cudaStream_t s) {
			launchComputeBoundary(m_s, m_map0, dt0, s);
		};
		// 3-4. Density and DFSPH factor join both branches.
		sctx.task(tok.rw(), tokBoundary.rw())->*[=](cudaStream_t s) {
			launchComputeDensity(m_s, m_map0, m_kernel, s);
			launchComputeFactor(m_s, m_map0, m_kernel, m_eps, s);
		};

		// 5. Divergence solve (uses dt0).
		if (m_enableDivergence)
		{
			// etaV scales with 1/dt: pass the dt-independent base, the loop-cond
			// kernel divides by the device dt each iteration.
			const real etaVBase = m_maxErrorV * static_cast<real>(0.01) * m_s.density0;
			sctx.task(tok.rw())->*[=](cudaStream_t s) {
				launchDivergenceInit(m_s, m_map0, m_kernel, dt0, s, m_dDt + 0);
			};
			recordSolverLoop(sctx, tok, m_s.pressureRho2V, dt0, 0, etaVBase, 1, m_maxIterationsV, 0, m_dDt + 0);
			sctx.task(tok.rw())->*[=](cudaStream_t s) {
				launchZeroReaction(m_reaction, s);
				launchComputePressureAccel(m_s, m_map0, m_kernel, m_s.pressureRho2V, m_eps, 1, m_reaction, s);
				launchDivergenceFinalizeApply(m_s, dt0, s, m_dDt + 0);
			};
		}

		// 6-9. Forces, device-side CFL, velocity update, pressure init.
		sctx.task(tok.rw())->*[=](cudaStream_t s) {
			launchClearAccelGravity(m_s, gravity, s);
			if (m_viscosity > 0)
				launchViscosityStandard(m_s, m_kernel, m_viscosity, m_viscDCoef, m_h2support, s);
			// CFL (method 1) on device: predicted max velocity with the base dt,
			// then clamp into m_dDt[1]. cflMethod 0 copies the base dt instead.
			launchComputeMaxVelSq(m_s, dt0, m_errScratch, s, m_dDt + 0);
			size_t bytes = m_cubTempBytes;
			cub::DeviceReduce::Max(m_cubTemp, bytes, m_errScratch, m_dReduce, static_cast<int>(m_s.n), s);
			launchCflUpdateDt(m_dReduce, m_dDt, m_cflFactor, static_cast<real>(2.0) * m_s.particleRadius,
							  m_cflMin, m_cflMax, (m_cflMethod != 0) ? 1 : 0, s);
			launchApplyVelocityUpdate(m_s, dt0, s, m_dDt + 1);
			launchPressureInit(m_s, m_map0, m_kernel, dt0, s, m_dDt + 1);
		};

		// 10. Pressure solve (eta is dt-independent).
		const real eta = m_maxError * static_cast<real>(0.01) * m_s.density0;
		recordSolverLoop(sctx, tok, m_s.pressureRho2, dt0 * dt0, 1, eta, m_minIterations, m_maxIterations, 1, m_dDt + 1);

		// 11. Finalize, integrate positions, fetch reaction totals.
		sctx.task(tok.rw())->*[=](cudaStream_t s) {
			launchZeroReaction(m_reaction, s);
			launchComputePressureAccel(m_s, m_map0, m_kernel, m_s.pressureRho2, m_eps, 1, m_reaction, s);
			launchPressureFinalizeApply(m_s, dt0, s, m_dDt + 1);
			launchApplyPosition(m_s, dt0, s, m_dDt + 1);
			cudaMemcpyAsync(m_hReaction, m_reaction.force, 3 * sizeof(double), cudaMemcpyDeviceToHost, s);
			cudaMemcpyAsync(m_hReaction + 3, m_reaction.torque, 3 * sizeof(double), cudaMemcpyDeviceToHost, s);
		};

		m_stepGraph = sctx.pop_prologue_shared();
		m_graphGrav[0] = gravity.x; m_graphGrav[1] = gravity.y; m_graphGrav[2] = gravity.z;
		stats.graphBuilds++;

		// Debug: dump the recorded step graph as Graphviz when requested.
		if (const char *dot = std::getenv("DFSPH_CUDA_GRAPH_DOT"))
		{
			cudaCheck(cudaGraphDebugDotPrint(m_stepGraph.graph(), dot, cudaGraphDebugDotFlagsVerbose),
					  "graph dot dump");
			std::fprintf(stderr, "[dfsph-cuda] step graph dumped to %s\n", dot);
		}
	}

	m_stepGraph.launch();
	cudaCheck(cudaStreamSynchronize(m_stepGraph.stream()), "graph launch sync");

	unsigned int iters[2] = {0, 0};
	real avgs[2] = {0, 0};
	real dts[2] = {dt0, dt0};
	cudaCheck(cudaMemcpy(iters, m_dIter, 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost), "iters D2H");
	cudaCheck(cudaMemcpy(avgs, m_dErrOut, 2 * sizeof(real), cudaMemcpyDeviceToHost), "errs D2H");
	cudaCheck(cudaMemcpy(dts, m_dDt, 2 * sizeof(real), cudaMemcpyDeviceToHost), "dt D2H");
	stats.iterationsV = m_enableDivergence ? iters[0] : 0;
	stats.iterations = iters[1];
	stats.avgDensityErrV = m_enableDivergence ? avgs[0] : 0;
	stats.avgDensityErr = avgs[1];
	m_suggestedDt = dts[1];
	stats.dtUsed = dts[1];
	m_lastDtUsed = dts[1];
}

void DFSPHCudaBackendImpl::runTimestep(const StepDesc &sd, StepStats &stats, cudaStream_t stream, bool /*useStf*/)
{
	const real dt0 = sd.dt;
	const real3 gravity = make_r3(sd.gravity[0], sd.gravity[1], sd.gravity[2]);

	// Refresh moving-body transform if provided (motor coupling stage 1).
	if (sd.bodyRotation && !m_maps.empty())
	{
		for (size_t b = 0; b < m_maps.size(); ++b)
		{
			updateBoundaryMapTransform(m_maps[b],
									   sd.bodyRotation + 9 * b, sd.bodyTranslation + 3 * b,
									   sd.bodyAngularVel + 3 * b, sd.bodyLinearVel + 3 * b,
									   sd.bodyComPosition + 3 * b);
		}
		m_map0 = m_maps[0];
	}

	// 1. Neighborhood.
	neighborhood(stream);

	// 2. Boundary contribution (Bender2019 volume map).
	launchComputeBoundary(m_s, m_map0, dt0, stream);

	// 3. Density + 4. DFSPH factor.
	launchComputeDensity(m_s, m_map0, m_kernel, stream);
	launchComputeFactor(m_s, m_map0, m_kernel, m_eps, stream);

	const bool deviceLoops = (sd.orchestrator == Orchestrator::StfConditional);

	// 5. Divergence solve (uses dt0).
	unsigned int itersV = 0;
	if (m_enableDivergence)
	{
		launchDivergenceInit(m_s, m_map0, m_kernel, dt0, stream);
		const real eta = (static_cast<real>(1.0) / dt0) * m_maxErrorV * static_cast<real>(0.01) * m_s.density0;
		if (deviceLoops)
		{
			real avg = 0;
			itersV = runSolverLoopConditional(m_s.pressureRho2V, dt0, 0, eta, 1, m_maxIterationsV, avg, stream);
			stats.avgDensityErrV = avg;
		}
		else
		{
			bool chk = false;
			while ((!chk || itersV < 1) && itersV < m_maxIterationsV)
			{
				launchComputePressureAccel(m_s, m_map0, m_kernel, m_s.pressureRho2V, m_eps, 0, m_reaction, stream);
				launchSolveIterate(m_s, m_map0, m_kernel, m_s.pressureRho2V, dt0, 0, m_eps, m_errScratch, stream);
				real err = reduceSum(m_errScratch, stream);
				real avg = err / static_cast<real>(m_s.n);
				stats.avgDensityErrV = avg;
				chk = (avg <= eta);
				++itersV;
			}
		}
		launchZeroReaction(m_reaction, stream);
		launchComputePressureAccel(m_s, m_map0, m_kernel, m_s.pressureRho2V, m_eps, 1, m_reaction, stream);
		launchDivergenceFinalizeApply(m_s, dt0, stream);
	}
	stats.iterationsV = itersV;

	// 6. Clear accelerations + gravity, 7. viscosity.
	launchClearAccelGravity(m_s, gravity, stream);
	if (m_viscosity > 0)
		launchViscosityStandard(m_s, m_kernel, m_viscosity, m_viscDCoef, m_h2support, stream);

	// 8. CFL time step (method 1). Uses (vel + accel*dt0).
	// cflMethod 0 disables adaptive stepping: keep the caller-provided dt.
	real dtUsed = dt0;
	if (m_cflMethod != 0)
	{
		launchComputeMaxVelSq(m_s, dt0, m_errScratch, stream);
		real maxVel2 = reduceMax(m_errScratch, stream);
		if (maxVel2 < static_cast<real>(1.0e-9)) maxVel2 = static_cast<real>(1.0e-9);
		const real diameter = static_cast<real>(2.0) * m_s.particleRadius;
		real h = m_cflFactor * static_cast<real>(0.4) * (diameter / std::sqrt(maxVel2));
		if (h > m_cflMax) h = m_cflMax;
		if (h < m_cflMin) h = m_cflMin;
		dtUsed = h;
	}
	m_suggestedDt = dtUsed;

	// 9. Apply non-pressure velocity update with the new dt.
	launchApplyVelocityUpdate(m_s, dtUsed, stream);

	// 10. Pressure solve (uses dtUsed).
	unsigned int iters = 0;
	{
		launchPressureInit(m_s, m_map0, m_kernel, dtUsed, stream);
		const real eta = m_maxError * static_cast<real>(0.01) * m_s.density0;
		const real hFactor = dtUsed * dtUsed;
		if (deviceLoops)
		{
			real avg = 0;
			iters = runSolverLoopConditional(m_s.pressureRho2, hFactor, 1, eta, m_minIterations, m_maxIterations, avg, stream);
			stats.avgDensityErr = avg;
		}
		else
		{
			bool chk = false;
			while ((!chk || iters < m_minIterations) && iters < m_maxIterations)
			{
				launchComputePressureAccel(m_s, m_map0, m_kernel, m_s.pressureRho2, m_eps, 0, m_reaction, stream);
				launchSolveIterate(m_s, m_map0, m_kernel, m_s.pressureRho2, hFactor, 1, m_eps, m_errScratch, stream);
				real err = reduceSum(m_errScratch, stream);
				real avg = err / static_cast<real>(m_s.n);
				stats.avgDensityErr = avg;
				chk = (avg <= eta);
				++iters;
			}
		}
		launchZeroReaction(m_reaction, stream);
		launchComputePressureAccel(m_s, m_map0, m_kernel, m_s.pressureRho2, m_eps, 1, m_reaction, stream);
		launchPressureFinalizeApply(m_s, dtUsed, stream);
	}
	stats.iterations = iters;

	// 11. Final position integration.
	launchApplyPosition(m_s, dtUsed, stream);

	// Fetch reaction totals (valid after the pressure finalize).
	cudaCheck(cudaMemcpyAsync(m_hReaction, m_reaction.force, 3 * sizeof(double), cudaMemcpyDeviceToHost, stream), "reaction force D2H");
	cudaCheck(cudaMemcpyAsync(m_hReaction + 3, m_reaction.torque, 3 * sizeof(double), cudaMemcpyDeviceToHost, stream), "reaction torque D2H");

	cudaCheck(cudaStreamSynchronize(stream), "step sync");
	stats.dtUsed = dtUsed;
}

void DFSPHCudaBackendImpl::step(const StepDesc &sd, StepStats &stats)
{
	if (!m_ready) return;

	cudaEvent_t t0, t1;
	cudaCheck(cudaEventCreate(&t0), "event");
	cudaCheck(cudaEventCreate(&t1), "event");

	// StfGraph runs the whole step as one CUDA graph with nested conditional
	// while nodes; dt (including adaptive CFL) is device-resident.
	const bool wholeStepGraph = (sd.orchestrator == Orchestrator::StfGraph);

	if (sd.orchestrator == Orchestrator::DirectStream || sd.orchestrator == Orchestrator::StfConditional || wholeStepGraph)
	{
		// StfConditional also drives the coarse phases on a raw stream; the
		// CUDASTF while-graph contexts live inside the two solver loops.
		cudaStream_t stream;
		cudaCheck(cudaStreamCreate(&stream), "stream");
		cudaEventRecord(t0, stream);
		if (wholeStepGraph)
			runTimestepGraph(sd, stats, stream);
		else
			runTimestep(sd, stats, stream, false);
		cudaEventRecord(t1, stream);
		cudaEventSynchronize(t1);
		cudaStreamDestroy(stream);
		if (wholeStepGraph)
			stats.graphLaunches = 1;
		else
			stats.graphLaunches = (sd.orchestrator == Orchestrator::StfConditional) ? 3 : 1;
	}
	else
	{
		// CUDASTF stream_ctx orchestration. Coarse phases are wrapped as tasks on
		// a void-interface token so CUDASTF sequences them; the host-controlled
		// solver loops run inside runTimestep using the task's stream.
		stf::context ctx;
		auto tok = ctx.token();
		cudaEventRecord(t0, static_cast<cudaStream_t>(nullptr));
		ctx.task(tok.rw())->*[&](cudaStream_t s) {
			runTimestep(sd, stats, s, true);
		};
		ctx.finalize();
		cudaEventRecord(t1, static_cast<cudaStream_t>(nullptr));
		cudaEventSynchronize(t1);
		stats.graphLaunches = 1;
	}

	float ms = 0;
	cudaEventElapsedTime(&ms, t0, t1);
	stats.lastStepMs = ms;
	// Host-controlled loops sync once per iteration; the conditional path only
	// syncs around each while-graph (pre-sync + finalize) plus CFL and step end;
	// the whole-step graph syncs once around the single graph launch.
	if (wholeStepGraph)
		stats.hostSyncs = 2;
	else if (sd.orchestrator == Orchestrator::StfConditional)
		stats.hostSyncs = 6;
	else
		stats.hostSyncs = stats.iterations + stats.iterationsV + 2;
	cudaEventDestroy(t0);
	cudaEventDestroy(t1);
}

void DFSPHCudaBackendImpl::copyStateToHost(const HostMirror &mirror)
{
	if (!m_ready) return;
	const unsigned int n = m_s.n;
	// Async copies + one sync: each blocking cudaMemcpy would pay a full device
	// synchronization. Fastest when the destination arrays are pinned (the
	// facade mirrors positions/velocities/density into pinned staging).
	auto d2h = [&](void *dst, const void *src, size_t bytes) {
		if (dst) cudaCheck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, nullptr), "state D2H");
	};
	d2h(mirror.positions, m_s.pos, 3 * n * sizeof(real));
	d2h(mirror.velocities, m_s.vel, 3 * n * sizeof(real));
	d2h(mirror.density, m_s.density, n * sizeof(real));
	d2h(mirror.factor, m_s.factor, n * sizeof(real));
	d2h(mirror.densityAdv, m_s.densityAdv, n * sizeof(real));
	d2h(mirror.pressureRho2, m_s.pressureRho2, n * sizeof(real));
	d2h(mirror.pressureRho2V, m_s.pressureRho2V, n * sizeof(real));
	d2h(mirror.pressureAccel, m_s.pressureAccel, 3 * n * sizeof(real));
	cudaCheck(cudaStreamSynchronize(nullptr), "state D2H sync");
}

void DFSPHCudaBackendImpl::getBoundaryReactions(BoundaryReaction *out) const
{
	if (!out) return;
	// Only one dynamic body is accumulated for now (index 0).
	for (int d = 0; d < 3; ++d)
	{
		out[0].force[d] = m_hReaction ? m_hReaction[d] : 0.0;
		out[0].torque[d] = m_hReaction ? m_hReaction[3 + d] : 0.0;
	}
}

void DFSPHCudaBackendImpl::destroy()
{
	if (!m_ready) return;
	if (m_stepGraph.valid())
		m_stepGraph = {}; // runs the pending pop_epilogue on m_graphCtx
	if (m_graphCtx)
	{
		m_graphCtx->finalize();
		m_graphCtx.reset();
	}
	if (m_whileCtx)
	{
		m_whileCtx->finalize();
		m_whileCtx.reset();
	}
	cudaFree(m_s.pos); cudaFree(m_s.vel); cudaFree(m_s.accel); cudaFree(m_s.pressureAccel);
	cudaFree(m_s.mass); cudaFree(m_s.density); cudaFree(m_s.factor); cudaFree(m_s.densityAdv);
	cudaFree(m_s.pressureRho2); cudaFree(m_s.pressureRho2V); cudaFree(m_s.state);
	cudaFree(m_s.boundaryVolume); cudaFree(m_s.boundaryXj);
	cudaFree(m_baseKey); cudaFree(m_baseId); cudaFree(m_s.cellStart); cudaFree(m_s.cellEnd);
	cudaFree(m_keysAlt); cudaFree(m_valsAlt); cudaFree(m_errScratch);
	cudaFree(m_cubTemp); cudaFree(m_dReduce); cudaFree(m_dIter); cudaFree(m_dErrOut); cudaFree(m_dDt);
	if (m_hReduce) cudaFreeHost(m_hReduce);
	cudaFree(m_reaction.force); cudaFree(m_reaction.torque);
	if (m_hReaction) cudaFreeHost(m_hReaction);
	for (auto &m : m_maps) freeBoundaryMap(m);
	m_maps.clear();
	m_s = DeviceState{};
	m_baseKey = m_baseId = m_keysAlt = m_valsAlt = nullptr;
	m_errScratch = m_dReduce = nullptr;
	m_dIter = nullptr;
	m_dErrOut = nullptr;
	m_dDt = nullptr;
	m_hReduce = nullptr;
	m_cubTemp = nullptr;
	m_hReaction = nullptr;
	m_reaction = BodyReactionAccum{};
	m_ready = false;
}

DFSPHCudaBackend *createDFSPHCudaBackend()
{
	int count = 0;
	if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
		return nullptr;
	return new DFSPHCudaBackendImpl();
}

} // namespace cuda_dfsph
} // namespace SPH
