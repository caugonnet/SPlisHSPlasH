#include "TimeStepDFSPHCUDA.h"
#include "FeatureProbes.h"

#include "SPlisHSPlasH/TimeManager.h"
#include "SPlisHSPlasH/Simulation.h"
#include "SPlisHSPlasH/BoundaryModel_Bender2019.h"
#include "SPlisHSPlasH/NonPressureForceBase.h"
#include "Utilities/Timing.h"
#include "Utilities/Logger.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>

using namespace SPH;
using namespace GenParam;

std::string TimeStepDFSPHCUDA::METHOD_NAME = "DFSPH_CUDA";
int TimeStepDFSPHCUDA::ORCHESTRATOR = -1;
int TimeStepDFSPHCUDA::ENUM_ORCH_DIRECT = -1;
int TimeStepDFSPHCUDA::ENUM_ORCH_STF_STREAM = -1;
int TimeStepDFSPHCUDA::ENUM_ORCH_STF_GRAPH = -1;
int TimeStepDFSPHCUDA::ENUM_ORCH_STF_COND = -1;
int TimeStepDFSPHCUDA::RUN_FEATURE_PROBES = -1;

// Backing storage for one flattened Bender2019 map. Kept alive on the facade so
// the BoundaryMapDesc pointers stay valid across the initialize() call.
struct TimeStepDFSPHCUDA::MapStorage
{
	std::vector<double> nodes[2];
	std::vector<unsigned int> cells[2];
	std::vector<unsigned int> cellMap[2];
};

TimeStepDFSPHCUDA::TimeStepDFSPHCUDA()
	: TimeStepDFSPH()
	, m_backend(nullptr)
	, m_deviceReady(false)
	, m_initialized(false)
	, m_orchestrator(static_cast<int>(cuda_dfsph::Orchestrator::StfStream))
	, m_runFeatureProbes(true)
{
	m_backend = cuda_dfsph::createDFSPHCudaBackend();
	m_deviceReady = (m_backend != nullptr);
	if (!m_deviceReady)
		LOG_WARN << "DFSPH_CUDA: no usable CUDA device found, falling back to CPU DFSPH.";
}

TimeStepDFSPHCUDA::~TimeStepDFSPHCUDA(void)
{
	delete m_backend;
	m_backend = nullptr;
}

void TimeStepDFSPHCUDA::initParameters()
{
	TimeStepDFSPH::initParameters();

	ORCHESTRATOR = createEnumParameter("cudaOrchestrator", "CUDA orchestrator", &m_orchestrator);
	setGroup(ORCHESTRATOR, "Simulation|DFSPH");
	setDescription(ORCHESTRATOR, "How the GPU timestep is scheduled (baseline stream, CUDASTF stream/graph, or device-side conditional loops).");
	EnumParameter *ep = static_cast<EnumParameter*>(getParameter(ORCHESTRATOR));
	ep->addEnumValue("Direct stream", ENUM_ORCH_DIRECT);
	ep->addEnumValue("CUDASTF stream", ENUM_ORCH_STF_STREAM);
	ep->addEnumValue("CUDASTF graph", ENUM_ORCH_STF_GRAPH);
	ep->addEnumValue("CUDASTF conditional", ENUM_ORCH_STF_COND);

	RUN_FEATURE_PROBES = createBoolParameter("cudaFeatureProbes", "Run CUDA feature probes", &m_runFeatureProbes);
	setGroup(RUN_FEATURE_PROBES, "Simulation|DFSPH");
	setDescription(RUN_FEATURE_PROBES, "Run the CCCL/CUDASTF capability probes once before the first GPU step.");
}

void TimeStepDFSPHCUDA::reset()
{
	TimeStepDFSPH::reset();
	m_initialized = false;
}

void TimeStepDFSPHCUDA::resize()
{
	TimeStepDFSPH::resize();
	m_initialized = false;
}

bool TimeStepDFSPHCUDA::exportBoundaryMap(unsigned int boundaryIndex, MapStorage &storage,
										 cuda_dfsph::BoundaryMapDesc &desc)
{
	Simulation *sim = Simulation::getCurrent();
	BoundaryModel_Bender2019 *bm =
		static_cast<BoundaryModel_Bender2019*>(sim->getBoundaryModel(boundaryIndex));
	Discregrid::DiscreteGrid *map = bm->getMap();
	if (map == nullptr)
		return false;

	// Serialize the CubicLagrangeDiscreteGrid through its public save() API and
	// parse the flat binary (standard-layout raw records) back into device-ready
	// arrays. This avoids modifying the external Discregrid checkout.
	char tmpl[] = "/tmp/dfsph_bmapXXXXXX";
	int fd = mkstemp(tmpl);
	if (fd < 0)
		return false;
	close(fd);
	map->save(tmpl);

	std::ifstream in(tmpl, std::ios::binary);
	if (!in.good())
	{
		std::remove(tmpl);
		return false;
	}
	std::streambuf *buf = in.rdbuf();
	auto rd = [&](void *dst, std::size_t bytes) -> bool {
		return buf->sgetn(reinterpret_cast<char*>(dst), bytes) == static_cast<std::streamsize>(bytes);
	};

	// Layout matches CubicLagrangeDiscreteGrid::save().
	double domainMin[3], domainMax[3];
	rd(domainMin, sizeof(domainMin));
	rd(domainMax, sizeof(domainMax));
	unsigned int resolution[3];
	rd(resolution, sizeof(resolution));
	double cellSize[3], invCellSize[3];
	rd(cellSize, sizeof(cellSize));
	rd(invCellSize, sizeof(invCellSize));
	std::size_t nCells = 0, nFields = 0;
	rd(&nCells, sizeof(nCells));
	rd(&nFields, sizeof(nFields));

	// nodes
	std::size_t nNodeArrays = 0;
	rd(&nNodeArrays, sizeof(nNodeArrays));
	std::vector<std::vector<double>> nodes(nNodeArrays);
	for (std::size_t f = 0; f < nNodeArrays; ++f)
	{
		std::size_t cnt = 0;
		rd(&cnt, sizeof(cnt));
		nodes[f].resize(cnt);
		if (cnt)
			rd(nodes[f].data(), cnt * sizeof(double));
	}
	// cells
	std::size_t nCellArrays = 0;
	rd(&nCellArrays, sizeof(nCellArrays));
	std::vector<std::vector<unsigned int>> cells(nCellArrays);
	for (std::size_t f = 0; f < nCellArrays; ++f)
	{
		std::size_t cnt = 0;
		rd(&cnt, sizeof(cnt));
		cells[f].resize(cnt * 32);
		for (std::size_t c = 0; c < cnt; ++c)
			rd(&cells[f][c * 32], 32 * sizeof(unsigned int));
	}
	// cell map
	std::size_t nMapArrays = 0;
	rd(&nMapArrays, sizeof(nMapArrays));
	std::vector<std::vector<unsigned int>> cellMap(nMapArrays);
	for (std::size_t f = 0; f < nMapArrays; ++f)
	{
		std::size_t cnt = 0;
		rd(&cnt, sizeof(cnt));
		cellMap[f].resize(cnt);
		if (cnt)
			rd(cellMap[f].data(), cnt * sizeof(unsigned int));
	}
	in.close();
	std::remove(tmpl);

	if (nNodeArrays < 2 || nCellArrays < 2 || nMapArrays < 2)
	{
		LOG_WARN << "DFSPH_CUDA: boundary map has fewer than 2 fields; expected distance+volume.";
		return false;
	}

	for (int f = 0; f < 2; ++f)
	{
		storage.nodes[f] = std::move(nodes[f]);
		storage.cells[f] = std::move(cells[f]);
		storage.cellMap[f] = std::move(cellMap[f]);
		desc.nodes[f] = storage.nodes[f].data();
		desc.nodeCount[f] = storage.nodes[f].size();
		desc.cells[f] = storage.cells[f].data();
		desc.cellMap[f] = storage.cellMap[f].data();
	}
	desc.nCells = nCells;
	for (int d = 0; d < 3; ++d)
	{
		desc.domainMin[d] = domainMin[d];
		desc.domainMax[d] = domainMax[d];
		desc.cellSize[d] = cellSize[d];
		desc.invCellSize[d] = invCellSize[d];
		desc.resolution[d] = resolution[d];
	}

	// Rigid transform + kinematics.
	RigidBodyObject *rbo = bm->getRigidBodyObject();
	const Vector3r t = rbo->getPosition();
	const Matrix3r R = rbo->getRotation().toRotationMatrix();
	for (int r = 0; r < 3; ++r)
	{
		for (int c = 0; c < 3; ++c)
			desc.rotation[r * 3 + c] = static_cast<double>(R(r, c));
		desc.translation[r] = static_cast<double>(t[r]);
	}
	const Vector3r av = rbo->getAngularVelocity();
	const Vector3r lv = rbo->getVelocity();
	const Vector3r com = rbo->getPosition();
	for (int d = 0; d < 3; ++d)
	{
		desc.angularVelocity[d] = static_cast<double>(av[d]);
		desc.linearVelocity[d] = static_cast<double>(lv[d]);
		desc.comPosition[d] = static_cast<double>(com[d]);
	}
	desc.isDynamic = rbo->isDynamic();
	return true;
}

void TimeStepDFSPHCUDA::gatherHostState()
{
	Simulation *sim = Simulation::getCurrent();
	FluidModel *model = sim->getFluidModel(0);
	const unsigned int n = model->numActiveParticles();

	m_hPos.resize(3 * n);
	m_hVel.resize(3 * n);
	m_hMass.resize(n);
	m_hState.resize(n);
	for (unsigned int i = 0; i < n; ++i)
	{
		const Vector3r &x = model->getPosition(i);
		const Vector3r &v = model->getVelocity(i);
		m_hPos[3 * i + 0] = static_cast<cuda_dfsph::cudsph_real>(x[0]);
		m_hPos[3 * i + 1] = static_cast<cuda_dfsph::cudsph_real>(x[1]);
		m_hPos[3 * i + 2] = static_cast<cuda_dfsph::cudsph_real>(x[2]);
		m_hVel[3 * i + 0] = static_cast<cuda_dfsph::cudsph_real>(v[0]);
		m_hVel[3 * i + 1] = static_cast<cuda_dfsph::cudsph_real>(v[1]);
		m_hVel[3 * i + 2] = static_cast<cuda_dfsph::cudsph_real>(v[2]);
		m_hMass[i] = static_cast<cuda_dfsph::cudsph_real>(model->getMass(i));
		m_hState[i] = static_cast<int>(model->getParticleState(i));
	}
}

void TimeStepDFSPHCUDA::scatterHostState()
{
	Simulation *sim = Simulation::getCurrent();
	FluidModel *model = sim->getFluidModel(0);
	const unsigned int n = model->numActiveParticles();

	m_hDensity.resize(n);
	m_hFactor.resize(n);
	m_hDensityAdv.resize(n);
	m_hPressureRho2.resize(n);
	m_hPressureRho2V.resize(n);
	m_hPressureAccel.resize(3 * n);

	cuda_dfsph::HostMirror mirror;
	mirror.positions = m_hPos.data();
	mirror.velocities = m_hVel.data();
	mirror.density = m_hDensity.data();
	mirror.factor = m_hFactor.data();
	mirror.densityAdv = m_hDensityAdv.data();
	mirror.pressureRho2 = m_hPressureRho2.data();
	mirror.pressureRho2V = m_hPressureRho2V.data();
	mirror.pressureAccel = m_hPressureAccel.data();
	m_backend->copyStateToHost(mirror);

	for (unsigned int i = 0; i < n; ++i)
	{
		model->setPosition(i, Vector3r(static_cast<Real>(m_hPos[3 * i + 0]),
									   static_cast<Real>(m_hPos[3 * i + 1]),
									   static_cast<Real>(m_hPos[3 * i + 2])));
		model->setVelocity(i, Vector3r(static_cast<Real>(m_hVel[3 * i + 0]),
									   static_cast<Real>(m_hVel[3 * i + 1]),
									   static_cast<Real>(m_hVel[3 * i + 2])));
		model->setDensity(i, static_cast<Real>(m_hDensity[i]));
		m_simulationData.setFactor(0, i, static_cast<Real>(m_hFactor[i]));
		m_simulationData.setDensityAdv(0, i, static_cast<Real>(m_hDensityAdv[i]));
		m_simulationData.setPressureRho2(0, i, static_cast<Real>(m_hPressureRho2[i]));
		m_simulationData.setPressureRho2_V(0, i, static_cast<Real>(m_hPressureRho2V[i]));
		m_simulationData.setPressureAccel(0, i, Vector3r(static_cast<Real>(m_hPressureAccel[3 * i + 0]),
														 static_cast<Real>(m_hPressureAccel[3 * i + 1]),
														 static_cast<Real>(m_hPressureAccel[3 * i + 2])));
	}
}

void TimeStepDFSPHCUDA::buildSceneDesc(cuda_dfsph::SceneDesc &scene)
{
	Simulation *sim = Simulation::getCurrent();
	FluidModel *model = sim->getFluidModel(0);
	const unsigned int n = model->numActiveParticles();

	gatherHostState();

	scene.numParticles = n;
	scene.capacity = n;
	scene.positions = m_hPos.data();
	scene.velocities = m_hVel.data();
	scene.masses = m_hMass.data();
	scene.particleState = m_hState.data();
	scene.density0 = static_cast<cuda_dfsph::cudsph_real>(model->getDensity0());
	scene.volume = static_cast<cuda_dfsph::cudsph_real>(model->getVolume(0));
	scene.supportRadius = static_cast<cuda_dfsph::cudsph_real>(sim->getSupportRadius());
	scene.particleRadius = static_cast<cuda_dfsph::cudsph_real>(sim->getParticleRadius());
	scene.sim2D = sim->is2DSimulation();

	scene.minIterations = m_minIterations;
	scene.maxIterations = m_maxIterations;
	scene.maxError = static_cast<cuda_dfsph::cudsph_real>(m_maxError);
	scene.maxIterationsV = m_maxIterationsV;
	scene.maxErrorV = static_cast<cuda_dfsph::cudsph_real>(m_maxErrorV);
	scene.enableDivergenceSolver = m_enableDivergenceSolver;

	// Standard viscosity coefficient (only Standard viscosity is ported for the
	// first slice). Found by parameter name to avoid coupling to the module type.
	scene.viscosity = 0;
	NonPressureForceBase *visc = model->getViscosityBase();
	if (visc != nullptr)
	{
		for (unsigned int p = 0; p < visc->numParameters(); ++p)
		{
			ParameterBase *pb = visc->getParameter(p);
			if (pb != nullptr && pb->getName() == "viscosity")
			{
				scene.viscosity = static_cast<cuda_dfsph::cudsph_real>(
					static_cast<RealParameter*>(pb)->getValue());
				break;
			}
		}
	}

	scene.cflMethod = sim->getValue<int>(Simulation::CFL_METHOD);
	scene.cflFactor = static_cast<cuda_dfsph::cudsph_real>(sim->getValue<Real>(Simulation::CFL_FACTOR));
	scene.cflMinTimeStepSize = static_cast<cuda_dfsph::cudsph_real>(sim->getValue<Real>(Simulation::CFL_MIN_TIMESTEPSIZE));
	scene.cflMaxTimeStepSize = static_cast<cuda_dfsph::cudsph_real>(sim->getValue<Real>(Simulation::CFL_MAX_TIMESTEPSIZE));

	const unsigned int nBoundaries = sim->numberOfBoundaryModels();
	m_mapStorage.clear();
	m_mapDescs.clear();
	m_mapStorage.resize(nBoundaries);
	m_mapDescs.resize(nBoundaries);
	int validMaps = 0;
	for (unsigned int b = 0; b < nBoundaries; ++b)
	{
		if (exportBoundaryMap(b, m_mapStorage[b], m_mapDescs[b]))
			++validMaps;
	}
	scene.numBoundaries = static_cast<int>(nBoundaries);
	scene.boundaries = m_mapDescs.empty() ? nullptr : m_mapDescs.data();
	if (validMaps != static_cast<int>(nBoundaries))
		LOG_WARN << "DFSPH_CUDA: only " << validMaps << " of " << nBoundaries << " boundary maps exported.";
}

void TimeStepDFSPHCUDA::ensureInitialized()
{
	if (m_initialized)
		return;

	if (m_runFeatureProbes)
	{
		const bool ok = cuda_dfsph::dfsph_cuda_run_feature_probes(true);
		if (!ok)
			LOG_WARN << "DFSPH_CUDA: one or more feature probes failed; results may be unreliable.";
	}

	cuda_dfsph::SceneDesc scene;
	buildSceneDesc(scene);
	m_backend->initialize(scene);
	m_initialized = true;
}

void TimeStepDFSPHCUDA::step()
{
	// CPU fallback when there is no device.
	if (!m_deviceReady)
	{
		TimeStepDFSPH::step();
		return;
	}

	Simulation *sim = Simulation::getCurrent();
	TimeManager *tm = TimeManager::getCurrent();
	if (sim->numberOfFluidModels() != 1)
	{
		LOG_WARN << "DFSPH_CUDA currently supports a single fluid model; using CPU DFSPH.";
		TimeStepDFSPH::step();
		return;
	}

	ensureInitialized();

	cuda_dfsph::StepDesc sd;
	sd.dt = static_cast<cuda_dfsph::cudsph_real>(tm->getTimeStepSize());
	const Real *grav = sim->getVecValue<Real>(Simulation::GRAVITATION);
	sd.gravity[0] = static_cast<cuda_dfsph::cudsph_real>(grav[0]);
	sd.gravity[1] = static_cast<cuda_dfsph::cudsph_real>(grav[1]);
	sd.gravity[2] = static_cast<cuda_dfsph::cudsph_real>(grav[2]);
	sd.orchestrator = static_cast<cuda_dfsph::Orchestrator>(m_orchestrator);

	cuda_dfsph::StepStats stats;
	{
		START_TIMING("DFSPH_CUDA_step");
		m_backend->step(sd, stats);
		STOP_TIMING_AVG;
	}

	m_iterations = stats.iterations;
	m_iterationsV = stats.iterationsV;

	// Mirror device state to the host so existing visualization/export works.
	scatterHostState();

	// Adopt the device-computed CFL timestep and advance simulation time by the
	// dt actually used for integration this step.
	const Real dtUsed = (stats.dtUsed > 0) ? static_cast<Real>(stats.dtUsed) : static_cast<Real>(sd.dt);
	tm->setTimeStepSize(dtUsed);
	tm->setTime(tm->getTime() + dtUsed);

	sim->emitParticles();
	sim->animateParticles();
}
