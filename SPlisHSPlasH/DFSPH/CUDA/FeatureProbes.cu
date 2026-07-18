// Feature probes for the CUDA DFSPH backend.
//
// This translation unit verifies, at build/run time, that the four toolchain
// capabilities the plan depends on are actually available and behave as we
// assume before any solver kernels are written:
//
//   1. CUB device reductions captured inside a CUDA graph and replayed.
//   2. CUB deferred problem sizes (`cuda::args::deferred`) for reductions whose
//      `num_items` lives in device memory (graph-stable convergence checks).
//   3. CUDASTF relaunchable whole-graph execution (stream_ctx + graph_ctx).
//   4. CUDASTF device-side convergence loops via `while_graph_scope`.
//
// Each probe returns true on success. `dfsph_cuda_run_feature_probes()` runs
// them all and is exposed through a plain C++ (CUDA-free) declaration so the
// facade / tests can call it without pulling in CCCL headers.

#include "FeatureProbes.h"

#include <cub/cub.cuh>
#include <cuda/experimental/stf.cuh>

#include <cstdio>
#include <vector>

namespace SPH
{
namespace cuda_dfsph
{
namespace
{

#define PROBE_CUDA_CHECK(expr)                                                          \
    do                                                                                  \
    {                                                                                   \
        cudaError_t _err = (expr);                                                      \
        if (_err != cudaSuccess)                                                        \
        {                                                                               \
            std::fprintf(stderr, "[dfsph-cuda probe] CUDA error %s at %s:%d: %s\n",     \
                         #expr, __FILE__, __LINE__, cudaGetErrorString(_err));          \
            return false;                                                               \
        }                                                                               \
    } while (0)

// --------------------------------------------------------------------------
// Probe 1: CUB DeviceReduce captured in a CUDA graph and replayed.
// --------------------------------------------------------------------------
bool probe_cub_graph_reduce()
{
    const int n = 4096;
    std::vector<float> h_in(n, 1.0f);

    float *d_in = nullptr;
    float *d_out = nullptr;
    PROBE_CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    PROBE_CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    PROBE_CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // Pre-query and persist temp storage (no allocation during replay).
    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    PROBE_CUDA_CHECK(cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n));
    PROBE_CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    cudaStream_t stream;
    PROBE_CUDA_CHECK(cudaStreamCreate(&stream));

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    PROBE_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n, stream);
    PROBE_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    PROBE_CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    bool ok = true;
    for (int rep = 0; rep < 3 && ok; ++rep)
    {
        PROBE_CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
        PROBE_CUDA_CHECK(cudaStreamSynchronize(stream));
        float result = 0.0f;
        PROBE_CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        ok = (result == static_cast<float>(n));
    }

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_temp);
    cudaFree(d_out);
    cudaFree(d_in);
    return ok;
}

// --------------------------------------------------------------------------
// Probe 2: CUB deferred problem size (device-resident num_items).
// A graph captured once must produce correct results when the device-side
// count changes between replays.
// --------------------------------------------------------------------------
bool probe_cub_deferred_reduce()
{
    const int capacity = 8192;
    std::vector<float> h_in(capacity, 1.0f);

    float *d_in = nullptr;
    float *d_out = nullptr;
    int *d_count = nullptr;
    PROBE_CUDA_CHECK(cudaMalloc(&d_in, capacity * sizeof(float)));
    PROBE_CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    PROBE_CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    PROBE_CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), capacity * sizeof(float), cudaMemcpyHostToDevice));

    // Device-resident problem size: the reduction reads num_items from d_count
    // in stream order, so the same captured graph is correct for any count.
    const auto deferred_items = cuda::args::deferred{d_count};

    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    // Query using the deferred num_items (upper bound = capacity).
    cudaError_t query = cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, deferred_items);
    if (query != cudaSuccess)
    {
        std::fprintf(stderr, "[dfsph-cuda probe] deferred reduce query failed: %s\n",
                     cudaGetErrorString(query));
        cudaFree(d_count);
        cudaFree(d_out);
        cudaFree(d_in);
        return false;
    }
    PROBE_CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    cudaStream_t stream;
    PROBE_CUDA_CHECK(cudaStreamCreate(&stream));

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    PROBE_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, deferred_items, stream);
    PROBE_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    PROBE_CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    bool ok = true;
    const int counts[3] = {capacity, 1000, 4321}; // sums must track the device count
    for (int t = 0; t < 3 && ok; ++t)
    {
        PROBE_CUDA_CHECK(cudaMemcpy(d_count, &counts[t], sizeof(int), cudaMemcpyHostToDevice));
        PROBE_CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
        PROBE_CUDA_CHECK(cudaStreamSynchronize(stream));
        float result = 0.0f;
        PROBE_CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        ok = (result == static_cast<float>(counts[t]));
        if (!ok)
            std::fprintf(stderr, "[dfsph-cuda probe] deferred reduce: got %f expected %d\n",
                         result, counts[t]);
    }

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_temp);
    cudaFree(d_count);
    cudaFree(d_out);
    cudaFree(d_in);
    return ok;
}

// --------------------------------------------------------------------------
// Probe 3: CUDASTF relaunchable whole-graph execution.
// Build a small AXPY graph and execute it several times.
// --------------------------------------------------------------------------
__global__ void probe_axpy(float a, cuda::experimental::stf::slice<const float> x,
                           cuda::experimental::stf::slice<float> y)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nthreads = gridDim.x * blockDim.x;
    for (size_t i = tid; i < x.size(); i += nthreads)
        y(i) += a * x(i);
}

bool probe_stf_graph_relaunch()
{
    using namespace cuda::experimental::stf;

    const size_t N = 1024;
    std::vector<float> X(N, 1.0f), Y(N, 0.0f);

    graph_ctx ctx;
    auto lX = ctx.logical_data(make_slice(X.data(), N));
    auto lY = ctx.logical_data(make_slice(Y.data(), N));

    const int reps = 5;
    const float alpha = 2.0f;
    for (int r = 0; r < reps; ++r)
    {
        ctx.task(lX.read(), lY.rw())->*[&](cudaStream_t s, auto dX, auto dY) {
            probe_axpy<<<8, 128, 0, s>>>(alpha, dX, dY);
        };
    }
    ctx.finalize();

    bool ok = true;
    for (size_t i = 0; i < N && ok; ++i)
        ok = (Y[i] == alpha * reps);
    if (!ok)
        std::fprintf(stderr, "[dfsph-cuda probe] stf graph relaunch: Y[0]=%f expected %f\n",
                     Y[0], alpha * reps);
    return ok;
}

// --------------------------------------------------------------------------
// Probe 4: CUDASTF device-side convergence loop via while_graph_scope.
// Newton square-root iteration until a reduced max-change drops below tol.
// --------------------------------------------------------------------------
bool probe_stf_while_graph()
{
#if _CCCL_CTK_BELOW(12, 4)
    std::fprintf(stderr, "[dfsph-cuda probe] while_graph_scope requires CUDA 12.4+.\n");
    return false;
#else
    using namespace cuda::experimental::stf;

    constexpr size_t N = 512;
    constexpr double tol = 1e-12;
    std::vector<double> host_S(N), host_X(N);
    for (size_t i = 0; i < N; ++i)
    {
        host_S[i] = 1.0 + static_cast<double>(i);
        host_X[i] = host_S[i];
    }

    stackable_ctx ctx;
    auto lS = ctx.logical_data(make_slice(host_S.data(), N)).set_symbol("S");
    lS.set_read_only();
    auto lX = ctx.logical_data(make_slice(host_X.data(), N)).set_symbol("X");
    auto lmax_err = ctx.logical_data(shape_of<scalar_view<double>>()).set_symbol("max_err");

    {
        auto while_guard = ctx.while_graph_scope();
        ctx.parallel_for(box(N), lX.rw(), lS.read(), lmax_err.reduce(reducer::maxval<double>{}))
            ->*[] __device__(size_t i, auto x, auto s, auto &max_err) {
                  double x_old = x(i);
                  double x_new = 0.5 * (x_old + s(i) / x_old);
                  x(i) = x_new;
                  max_err = fabs(x_new - x_old);
              };
        while_guard.update_cond(lmax_err.read())->*[tol] __device__(auto max_err) {
            return (*max_err > tol);
        };
    }
    ctx.finalize();

    bool ok = true;
    for (size_t i = 0; i < N && ok; ++i)
    {
        double expected = sqrt(1.0 + static_cast<double>(i));
        ok = (fabs(host_X[i] - expected) < 1e-8);
    }
    if (!ok)
        std::fprintf(stderr, "[dfsph-cuda probe] while_graph_scope produced wrong result.\n");
    return ok;
#endif
}

} // namespace

bool dfsph_cuda_run_feature_probes(bool verbose)
{
    struct Probe
    {
        const char *name;
        bool (*fn)();
    };
    const Probe probes[] = {
        {"cub_graph_reduce", probe_cub_graph_reduce},
        {"cub_deferred_reduce", probe_cub_deferred_reduce},
        {"stf_graph_relaunch", probe_stf_graph_relaunch},
        {"stf_while_graph", probe_stf_while_graph},
    };

    bool all_ok = true;
    for (const auto &p : probes)
    {
        const bool ok = p.fn();
        all_ok = all_ok && ok;
        if (verbose)
            std::printf("[dfsph-cuda probe] %-22s : %s\n", p.name, ok ? "PASS" : "FAIL");
    }
    return all_ok;
}

} // namespace cuda_dfsph
} // namespace SPH
