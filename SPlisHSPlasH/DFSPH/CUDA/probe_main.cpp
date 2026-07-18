// Standalone runner for the DFSPH CUDA feature probes.
// Built as the `DFSPHCudaProbe` target when USE_CUDA_DFSPH=ON.
#include "FeatureProbes.h"
#include <cstdio>

int main()
{
    const bool ok = SPH::cuda_dfsph::dfsph_cuda_run_feature_probes(true);
    std::printf("[dfsph-cuda probe] overall: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
