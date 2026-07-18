#ifndef __TimeStepDFSPHCUDA_h__
#define __TimeStepDFSPHCUDA_h__

// Host-facing facade for the device-resident DFSPH backend. It reuses the
// parameters and particle fields of the CPU TimeStepDFSPH but overrides the
// timestep to run entirely on the GPU (orchestrated with CUDASTF), mirroring
// the device state back to the host after each step so the existing renderer,
// exporters and debug tools keep working unchanged.
//
// This header is deliberately CUDA-free and C++11-safe: it is included by the
// main SPlisHSPlasH library (Simulation.cpp) which is compiled at the project's
// default standard. The CUDA/CCCL implementation lives behind the opaque
// cuda_dfsph::DFSPHCudaBackend interface.

#include "SPlisHSPlasH/Common.h"
#include "SPlisHSPlasH/DFSPH/TimeStepDFSPH.h"
#include "DFSPHCudaBackend.h"

#include <vector>

namespace SPH
{
	/** \brief Divergence-Free SPH timestep executed on the GPU via CCCL kernels
	 *  and CUDASTF orchestration. Falls back to the CPU implementation when no
	 *  usable CUDA device is available. */
	class TimeStepDFSPHCUDA : public TimeStepDFSPH
	{
	protected:
		cuda_dfsph::DFSPHCudaBackend *m_backend;
		bool m_deviceReady;
		bool m_initialized;
		int m_orchestrator;             // cuda_dfsph::Orchestrator as int (parameter)
		bool m_runFeatureProbes;
		bool m_fullMirror;              // mirror solver-internal fields every step (debug)

		// Host staging buffers reused across steps (avoid per-step allocation).
		// The per-step mirror targets (positions/velocities/density) are pinned
		// host memory allocated through the backend for fast D2H transfers.
		cuda_dfsph::cudsph_real *m_hPos;
		cuda_dfsph::cudsph_real *m_hVel;
		cuda_dfsph::cudsph_real *m_hDensity;
		unsigned int m_pinnedCapacity;
		std::vector<cuda_dfsph::cudsph_real> m_hMass;
		std::vector<int> m_hState;
		std::vector<cuda_dfsph::cudsph_real> m_hFactor;
		std::vector<cuda_dfsph::cudsph_real> m_hDensityAdv;
		std::vector<cuda_dfsph::cudsph_real> m_hPressureRho2;
		std::vector<cuda_dfsph::cudsph_real> m_hPressureRho2V;
		std::vector<cuda_dfsph::cudsph_real> m_hPressureAccel;

		// Flattened Bender2019 boundary maps kept alive for initialize().
		struct MapStorage;
		std::vector<MapStorage> m_mapStorage;
		std::vector<cuda_dfsph::BoundaryMapDesc> m_mapDescs;

		void ensureInitialized();
		void ensurePinned(unsigned int n);
		void buildSceneDesc(cuda_dfsph::SceneDesc &scene);
		void gatherHostState();
		void scatterHostState();
		bool exportBoundaryMap(unsigned int boundaryIndex, MapStorage &storage,
							   cuda_dfsph::BoundaryMapDesc &desc);

	public:
		static int ORCHESTRATOR;
		static int ENUM_ORCH_DIRECT;
		static int ENUM_ORCH_STF_STREAM;
		static int ENUM_ORCH_STF_GRAPH;
		static int ENUM_ORCH_STF_COND;
		static int RUN_FEATURE_PROBES;
		static int FULL_MIRROR;

		static std::string METHOD_NAME;

		TimeStepDFSPHCUDA();
		virtual ~TimeStepDFSPHCUDA(void);

		virtual void step();
		virtual void reset();
		virtual void resize();
		// Scene-file namespace: DFSPH_CUDA is a drop-in DFSPH solver, so it reads
		// the scene's "DFSPH" block (maxError, minIterations, ...). Returning
		// METHOD_NAME here would make the loader look for a "DFSPH_CUDA" block
		// and silently fall back to default solver parameters.
		virtual std::string getMethodName() { return TimeStepDFSPH::METHOD_NAME; }

		virtual void initParameters();

		bool deviceReady() const { return m_deviceReady; }
	};
}

#endif
