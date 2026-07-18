#ifndef __DFSPH_CUDA_FeatureProbes_h__
#define __DFSPH_CUDA_FeatureProbes_h__

// Plain C++ (CUDA-free) declaration so hosts compiled at C++11 (e.g. the
// facade and the CPU-side simulator) can trigger the toolchain feature probes
// without including any CCCL / CUDA headers.

namespace SPH
{
namespace cuda_dfsph
{
	/** Run all CUDA/CCCL/CUDASTF feature probes required by the DFSPH CUDA
	 *  backend. Returns true only if every probe passes. When @p verbose is
	 *  true a one-line PASS/FAIL summary is printed for each probe. */
	bool dfsph_cuda_run_feature_probes(bool verbose = true);
}
}

#endif
