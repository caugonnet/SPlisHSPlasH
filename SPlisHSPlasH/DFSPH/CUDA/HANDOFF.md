# CUDA DFSPH backend — agent handoff

Status snapshot for the "CUDA DFSPH Acceleration" plan
(`~/.cursor/plans/cuda_dfsph_acceleration_20f4ddd3.plan.md`). The goal is an
opt-in, device-resident DFSPH solver that dogfoods **CCCL** (CUB primitives,
`cuda::args::deferred`) and **CUDASTF** orchestration, first validated on
`data/Scenes/DoubleDamBreak.json`, then extended to `MotorScene.json`.

Do **not** edit the plan file. Todo IDs below match the plan.

---

## Current status

| Todo | State | Notes |
|------|-------|-------|
| `build-probes` | **done** | CUDA target, facade, solver registration, CCCL/CUDASTF probes all build & the probes ran on the GPU earlier. |
| `device-foundation` | **done (validated 2026-07-18)** | Device SoA, uniform grid + CUB radix sort, Bender2019 map flatten/interpolate, persistent scratch all implemented and validated via the DoubleDamBreak parity runs below. |
| `double-dam-correctness` | **done (validated 2026-07-18)** | GPU DoubleDamBreak runs end-to-end (no crash/NaN, all 4 probes PASS). Fixed-dt parity vs. CPU DFSPH: step 1 matches to fp32 roundoff (max position diff 1.7e-6 m, density 1e-4 relative). See "Parity results" below. |
| `conditional-graphs` | **done (validated 2026-07-18)** | Both solver loops run as CUDA Graph conditional `while` nodes via `stackable_ctx::while_graph_scope` when `cudaOrchestrator=StfConditional`. Bit-identical to the host-controlled loops over 100 steps. Zero host syncs per iteration. `StfGraph` still falls through to StfStream (milestone 7/9 territory). |
| `whole-step-graph` (milestone 9) | **done for fixed dt (2026-07-18)** | `StfGraph` + `cflMethod=0`: the entire timestep (neighborhood, both conditional while loops, forces, integration) is recorded ONCE into a `launchable_graph` (`pop_prologue_shared`) and relaunched every step — 1 graph launch + 2 host syncs per step. Bit-identical to Direct stream. Fastest orchestrator at small/medium N. Adaptive CFL falls back to the StfStream path (the CFL readback splits the step). |
| `async-visualization` | **mostly done (2026-07-18)** | Host mirror reduced from ~4 ms to <1 ms at 85k: lean mirror (pos/vel/density only; `cudaFullMirror` param restores the debug fields), pinned staging via new backend `allocHostPinned`, async D2H + one sync, and the scatter loop capped at 8 OpenMP threads (36-thread default cost ~2.8 ms in wake-up/contention with driver threads). Triple-buffered GUI snapshots still open. |
| `motor-coupling` | pending | Reaction force/torque accumulation kernels + `updateBoundaryMapTransform` exist; PBD handoff not wired. |
| `validate-benchmark` | pending | No parity/benchmark suite yet. |

### What is verified vs. not
- ✅ `SPlisHSPlasH_CUDA` static lib compiles cleanly (all 4 `.cu` + device link), CUDA 13.0, `sm_86`.
- ✅ `libSPlisHSPlasH.a` links with the facade (`TimeStepDFSPHCUDA`) and registers the `DFSPH_CUDA` method.
- ✅ `SPHSimulator` links and runs. Fixed 2026-07-18: the exe link failed with
  undefined `__cudaRegisterLinkedBinary_*` — resolved by adding
  `CUDA_RESOLVE_DEVICE_SYMBOLS ON` to `SPlisHSPlasH_CUDA` (device link now done
  once inside the CUDA static lib instead of in downstream C++ targets).
- ✅ GPU timestep runs DoubleDamBreak end-to-end headless; all 4 CCCL/CUDASTF
  probes PASS in-process; ~1.2–1.8 ms per `DFSPH_CUDA_step` (4732 particles,
  RTX 3080 Ti, includes full host mirror each step).
- ✅ Bug fixed 2026-07-18: the backend applied its internal CFL (method 1)
  unconditionally, ignoring `cflMethod=0`. `runTimestep` now keeps the incoming
  dt when `m_cflMethod == 0`. Before the fix, "parity" runs silently stepped at
  a different dt than the CPU.

### Parity results (2026-07-18, DoubleDamBreak, fp32, zSort off)

Method: bake settings into scene copies (`data/Scenes/DDB_parity_{cpu,gpu}.json`;
note `SPHSimulator` only honors the **last** `--param` flag, so multiple
`--param`s do NOT work), `cflMethod=0`, `timeStepSize=0.001`, `stopAt=0.1`,
VTK export at 200 fps, compare by particle id with
`SPlisHSPlasH/DFSPH/CUDA/tools/compare_vtk.py`.

- Step 1: max position diff **1.7e-6 m** (fp32 roundoff), density 1e-4 relative.
- 100 steps: max position diff grows to ~5e-3 m (rms 8.5e-4; particle radius
  0.025 m). Control experiment (CPU 1-thread vs N-thread) stays at ~6e-7,
  so the growth is driven by the GPU's different neighbor-summation order
  injecting ~1e-6-level noise each step plus occasional ±1 solver-iteration
  differences at the tolerance threshold — not by a formula error.
- All three CUDASTF orchestrators (`cudaOrchestrator` 1/2/3) are **bit-identical**
  to Direct stream over the full run (expected: they currently share
  `runTimestep`; graph/conditional fall through to the host-controlled loop).
- With adaptive CFL enabled, CPU and GPU runs disagree on step count (float
  clamp accumulation differs in the last bit → CPU squeezes in an extra step);
  use `cflMethod=0` for frame-aligned comparisons.

---

## How to build

Build config used: `build-cpu/` (Release, `USE_AVX=ON`, `USE_DOUBLE_PRECISION=OFF` ⇒ device `real` is `float`).

```bash
cd /home/caugonnet/git/SPlisHSPlasH/build-cpu
cmake . -DUSE_CUDA_DFSPH=ON -DCUDA_DFSPH_ARCHITECTURES=86
cmake --build . --target SPlisHSPlasH_CUDA -j 4      # device lib
cmake --build . --target SPlisHSPlasH   -j 4          # main lib + facade
cmake --build . --target SPHSimulator   -j 4          # <-- finish this
```

`CCCL_DIR` defaults to `~/git/caugonnet_cccl`. GPUs present: RTX 3080 Ti (sm_86,
use this) and Quadro P620 (sm_61).

Fast iteration on device code without a full CMake build (what was used to shake
out compile errors):

```bash
cd SPlisHSPlasH/DFSPH/CUDA
CCCL=~/git/caugonnet_cccl
INC="-I$CCCL/libcudacxx/include -I$CCCL/cub -I$CCCL/thrust -I$CCCL/cudax/include"
nvcc -std=c++17 -arch=sm_86 --expt-relaxed-constexpr --extended-lambda $INC -dc DFSPHCudaBackend.cu -o /tmp/x.o
```
(CCCL/CUDASTF headers are heavy: ~50 s per TU.)

## How to run (next step)

```bash
cd /home/caugonnet/git/SPlisHSPlasH
./build-cpu/bin/SPHSimulator data/Scenes/DoubleDamBreak.json
```
`DoubleDamBreak.json` currently selects `simulationMethod: 4` (CPU DFSPH). To use
the GPU backend, either set the simulation method to `DFSPH_CUDA` in the GUI
(Simulation panel) or change the scene. The method is registered as
`TimeStepDFSPHCUDA::METHOD_NAME = "DFSPH_CUDA"`, enum `DFSPH_CUDA` in
`Simulation.h`. The facade falls back to CPU DFSPH if there is no GPU or if the
scene has >1 fluid model.

The facade exposes two parameters (Simulation|DFSPH group):
- `cudaOrchestrator`: Direct stream / CUDASTF stream / CUDASTF graph / CUDASTF conditional.
- `cudaFeatureProbes`: run the CCCL/CUDASTF probes once before the first step.

---

## Architecture (façade + isolated CUDA lib)

The main project is C++11/AVX and must not see CUDA/CCCL/Eigen-in-CUDA. So:

```
SPlisHSPlasH (C++, AVX)
  └─ TimeStepDFSPHCUDA (facade, .cpp)      # subclass of TimeStepDFSPH
        │  POD structs only (DFSPHCudaBackend.h)
        ▼
  SPlisHSPlasH_CUDA (static lib, CUDA C++17)
        DFSPHCudaBackend.cu   # orchestrator, device state, CUDASTF
        DFSPHKernels.cu/.cuh  # all physics kernels + CUB wrappers
        Bender2019MapCUDA.*   # flattened volume-map interpolation
        DFSPHDeviceState.cuh  # SoA + cubic spline kernel + grid helpers
        FeatureProbes.*       # CCCL/CUDASTF capability probes
```

`DFSPHCudaBackend.h` is the **only** shared header. Everything crosses the
boundary as arrays of `cudsph_real` (= `float` unless `USE_DOUBLE`) and POD
descriptors (`SceneDesc`, `StepDesc`, `HostMirror`, `StepStats`, `BoundaryReaction`).

### Data flow per step (facade `TimeStepDFSPHCUDA::step`)
1. `ensureInitialized()` → `buildSceneDesc()` gathers host AoS → `backend->initialize()` (once). Runs feature probes once.
2. Build `StepDesc` (dt, gravity, orchestrator).
3. `backend->step()` runs the whole timestep on device.
4. `scatterHostState()` mirrors device → host `FluidModel` so the existing
   renderer/exporters keep working (correctness phase; snapshotting is a later todo).
5. Adopt device-computed CFL dt (`stats.dtUsed`), advance time, emit/animate.

---

## Physics: exact CPU formulas ported (single fluid, Bender2019)

All ported from the **non-AVX** CPU paths for clarity (`TimeStep.cpp`,
`TimeStepDFSPH.cpp`, `Viscosity_Standard.cpp`). Single-fluid assumption:
`density0_j/density0 == 1`, per-particle volume `V = model->getVolume(0)`.

- **Density**: `rho_i = (V*W0 + Σ_j V*W(r) + V_b*W(|x_i - x_bj|)) * rho0`, fluid neighbours exclude self.
- **DFSPH factor**: `1/sumGrad` if `>1e-5` else 0; boundary adds `-V_b·gradW` into `grad_p_i`.
- **densityAdv / densityChange**: fluid term uses bare `gradW` then `*= V`; boundary uses `V_b·(v_i-v_bj)·gradW`, `v_bj` from rigid point velocity.
- **pressureAccel**: fluid `if(|pSum|>eps) a += pSum·(-V·gradW)`; boundary `if(|p_rho2_i|>eps) a += p_rho2_i·(-V_b·gradW)`; boundary force/torque accumulated only when `applyForce && map.isDynamic`.
- **aij_pj**: fluid `(a_i-a_j)·gradW` then `*= V`; boundary `V_b·a_i·gradW`.
- **Jacobi update**: `p = max(p - 0.5·(s_i - aij_pj)·factor, 0)`; `err = -rho0·min(s_i-aij_pj,0)`.
  - Divergence: `s_i = -densityAdv`, `hFactor = dt`, deficiency (`<20` 3D / `<7` 2D fluid neighbours) zeroes residuum. Source `densityChange`, warm start `USE_WARMSTART_V`.
  - Pressure: `s_i = 1 - densityAdv`, `hFactor = dt²`, skip non-Active. Warm start `USE_WARMSTART`.
- **Standard viscosity** (only fluid term; scenes use `viscosityBoundary=0`):
  `a_i += d·visc·(m_j/rho_j)·(v_i-v_j)·(x_i-x_j)/(|x_i-x_j|²+0.01h²)·gradW`, `d=10` (3D)/`8` (2D), `h=support radius`.
- **clear+gravity**: `a = grav` if `mass!=0 && Active` else 0.
- **CFL (method 1)**: `maxVel = max (v+a·dt0)²`, `dt = clamp(cflFactor·0.4·(2r)/√maxVel, cflMin, cflMax)`. Method 2 currently behaves like method 1 (documented limitation; DoubleDamBreak uses method 1).

### Step ordering (matches CPU `TimeStepDFSPH::step`, CFL moved into backend)
neighbourhood → boundary map → density → factor → **divergence solve (dt0)** →
clear+gravity → viscosity → **CFL → dtUsed** → `v += dtUsed·a` →
**pressure solve (dtUsed)** → `x += dtUsed·v`. Reported `stats.dtUsed = dtUsed`.

---

## Neighborhood & Bender2019 map — important subtleties

- **Uniform grid**: `cellSize = supportRadius`. Bounds taken from the boundary
  map's world-space AABB (fluid can't leave the tank) + 2·cell margin; falls back
  to particle AABB. Grid is fixed-capacity; `numCells` fixed at init.
- **Sort**: CUB `DeviceRadixSort::SortPairs` with **explicit in/out** buffers
  (`m_baseKey/m_baseId` → `m_keysAlt/m_valsAlt`), *not* `DoubleBuffer`. `neighborhood()`
  resets `s.cellKey/s.sortedId` to the base buffers each step then re-points them
  at the alt buffers after the sort. (An earlier DoubleBuffer version aliased the
  same buffer across steps — do not reintroduce it.)
- **Neighbor macro** `FOR_NEIGHBORS` iterates the 27-cell stencil and **skips
  out-of-range cells without clamping** (clamping would merge border cells and
  double-count). Cell *keys* are clamped via `cellLinear`; neighbor *search* is not.
  All fluid loops must `if (j != i)` to exclude self (CompactNSearch excludes self).
- **Bender2019 map**: exported by serialising Discregrid's
  `CubicLagrangeDiscreteGrid` via its `save()` API and re-parsing the binary in
  `TimeStepDFSPHCUDA::exportBoundaryMap` (avoids patching Discregrid). Two fields:
  0 = signed distance, 1 = volume. Interpolation is a faithful device port of
  `shape_function_` + `interpolate` (kept in **double** precision).
  - **Critical**: the CPU reuses the field-0 `cell` (node indices) to read field-1
    nodes. The device mirrors this via `bmDetermine(field 0)` then
    `bmEval(field 1, same shape)`. Do not recompute the cell per field.
  - `local = Rᵀ·(x - t)`, normal back to world via `R·n_local`; `boundaryXj = x - d·n`,
    `d = max(dist + 0.5r, 2r)`.

---

## Files (all under `SPlisHSPlasH/DFSPH/CUDA/`)

| File | Purpose |
|------|---------|
| `DFSPHCudaBackend.h` | POD interface (shared with C++ side). |
| `TimeStepDFSPHCUDA.{h,cpp}` | Host facade, map export, host mirror, scene build. |
| `DFSPHCudaBackend.cu` | `DFSPHCudaBackendImpl`: device alloc, grid, `runTimestep`, CUDASTF path. |
| `DFSPHKernels.{cuh,cu}` | All physics kernels + CUB sort/reduce launchers. |
| `DFSPHDeviceState.cuh` | `DeviceState` SoA, `real3` ops, cubic spline kernel, grid helpers. |
| `Bender2019MapCUDA.{cuh,cu}` | `DeviceBoundaryMap`, shape fn, interpolate, upload/free/transform. |
| `FeatureProbes.{h,cu}`, `probe_main.cpp` | CCCL/CUDASTF capability probes + standalone runner. |
| `CMakeLists.txt` | `SPlisHSPlasH_CUDA` static lib + `DFSPHCudaProbe` exe. |

Also touched: root `CMakeLists.txt` (`USE_CUDA_DFSPH`, CCCL, arch), `SPlisHSPlasH/CMakeLists.txt` (subdir + link), `Simulation.{h,cpp}` (enum + solver creation), `TimeStepDFSPHCUDA.cpp` viscosity coefficient read.

---

## CUDASTF usage today

`Orchestrator::DirectStream` = plain `cudaStream_t` sequence (the reliable
baseline). `StfStream`/`StfGraph` build a `cuda::experimental::stf::context`,
create one `ctx.token()`, and run the whole `runTimestep` inside a single
`ctx.task(tok.rw())` body. Host-controlled solver loops read the CUB reduction
to host each iteration (`reduceSum`/`reduceMax` → pinned host +
`cudaStreamSynchronize`), which is the allowed correctness-phase host sync.

### Conditional while-graphs (`StfConditional`, added 2026-07-18)

`runSolverLoopConditional()` (DFSPHCudaBackend.cu) runs one Jacobi loop as a
CUDA Graph conditional `while` node:

- A **persistent** `stf::stackable_ctx` (`m_whileCtx`, member) is created lazily
  and kept alive across timesteps. This matters: on every `while_graph_scope`
  pop, STF queries its executable-graph cache (keyed by node/edge count) and
  patches the cached exec graph via `cudaGraphExecUpdate` instead of a fresh
  `cudaGraphInstantiate`. A per-step context re-instantiated both loop graphs
  every step and was ~1 ms/step slower.
- Loop body = one `sctx.task(lCond.write())` whose captured stream records:
  `launchComputePressureAccel` → `launchSolveIterate` → `cub::DeviceReduce::Sum`
  → `launchSolverLoopCond` (new kernel in DFSPHKernels.cu). The cond kernel
  replicates the host predicate `(!chk || it < minIter) && it < maxIter`
  exactly, increments a device iteration counter (`m_dIter`), and writes the
  continue-flag into a `scalar_view<int>` logical data read by
  `wg.update_cond(...)->*LoopCondNonZero{}`.
- All solver state stays in raw device buffers; only the continue-flag flows
  through STF logical data (it orders update_cond after the body).
- nvcc quirk: extended `__device__` lambdas are forbidden inside private member
  functions — hence the namespace-scope `LoopCondNonZero` functor.
- Ordering with the raw-stream phases: `cudaStreamSynchronize(stream)` before
  the scope, `cudaStreamSynchronize(sctx.fence())` after (fence() is only legal
  at root level). Iteration count/final error read back once per loop.
- `step()` routes `StfConditional` through the raw-stream path (the STF value
  lives inside the loops), `while_graph_scope()` defaults to launch value 1 →
  do-while semantics, matching the CPU loops.

Validation: bit-identical positions vs. the host-controlled loops over the full
100-step DoubleDamBreak parity run (all frames, max|dx| = 0).

### Whole-timestep relaunchable graph (`StfGraph` + fixed dt, added 2026-07-18)

`runTimestepGraph()` records the ENTIRE timestep once into
`m_stepGraph = sctx.pop_prologue_shared()` (a storable
`stackable_ctx::launchable_graph`) and calls `m_stepGraph.launch()` on
subsequent steps: one graph launch + one fence sync + two tiny stat readbacks
per step. Structure: one `sctx.push()`, plain `sctx.task(tok.rw())` phases
(neighborhood+density+factor / divergence init / finalize / forces+velocity /
pressure finalize+integrate+reaction-D2H) with the two solver loops recorded by
the shared `recordSolverLoop()` helper as nested conditional while nodes.

Key implementation facts:
- `recordSolverLoop(sctx, tok, pressure, hFactor, isPressure, eta, minIter,
  maxIter, slot)` — shared by StfConditional (own scope per loop, per-step) and
  StfGraph (recorded once). Per-loop stat slots `m_dIter[2]` / `m_dErrOut[2]`
  (0=divergence, 1=pressure) because both loops now live in one graph and would
  otherwise clobber each other before readback; the cond kernel also stores the
  avg error per iteration (`avgOut`).
- The step graph lives in its OWN context (`m_graphCtx`), separate from the
  conditional-path context (`m_whileCtx`): `push()` is forbidden while a
  pop-epilogue is pending (the launchable_graph holds the pop open), so mixing
  both patterns in one context would abort.
- Rebuild triggers: dt or gravity change, or moving-body transforms
  (`sd.bodyRotation != nullptr` — `m_map0` is baked into kernel args by value).
  Release the old handle (`m_stepGraph = {}`) BEFORE `sctx.push()`.
- The graph is byte-identical across steps because: dt fixed, all device
  pointers fixed, and the neighborhood base→alt buffer swap is deterministic
  (host-side `m_s` pointer mutation happens at record time only).
- Adaptive CFL (`cflMethod != 0`) needs a host readback mid-step → `step()`
  routes StfGraph to the old StfStream behavior in that case (verified to run).
- Teardown order in destroy(): `m_stepGraph = {}` (runs the pending
  pop_epilogue) → `m_graphCtx->finalize()`.

Validation: bit-identical to Direct stream over the 100-step parity run.

Performance (avg `DFSPH_CUDA_step` ms, RTX 3080 Ti, fixed dt):

| Config | Direct(0) | StfStream(1) | WholeGraph(2) | Conditional(3) |
|---|---|---|---|---|
| 4.7k, scene tol | 1.35 | 1.19 | **1.18** | 1.35 |
| 4.7k, tight tol (maxErr 0.001) | 2.12 | — | **1.96** | 2.12 |
| 85k, scene tol | 2.16 | 2.21 | 2.16 | 2.45 |

Whole-graph is the fastest at small/medium N (~13% over direct baseline; 8%
under tight tolerances) and matches at 85k where the run is compute-bound.
Direct stream also pays a per-step cudaStreamCreate/Destroy, which is part of
why StfStream/WholeGraph beat it at 4.7k. Earlier lessons: a per-step
`stackable_ctx` (no cache) cost ~1 ms/step extra; per-step re-recording into a
persistent ctx (exec-update instead of instantiate) still cost ~0.35 ms —
record-once/relaunch-many is the right pattern.

---

## Recommended next steps (in order)

1. ~~Finish `SPHSimulator` link and run DoubleDamBreak~~ **done 2026-07-18**.
2. ~~Numerical parity harness~~ **done 2026-07-18** (`tools/compare_vtk.py` +
   `data/Scenes/DDB_parity_*.json`; see "Parity results"). Possible hardening:
   also compare intermediate quantities (factor, densityAdv) and iteration
   counts per step — the facade currently doesn't expose GPU iteration counters
   through the counting/averaging log (`Total number: DFSPH - iterations` is
   missing from GPU runs).
3. ~~Move the two solver loops into `while_graph_scope`~~ **done 2026-07-18**
   (see "Conditional while-graphs" above; bit-identical, break-even to +6% perf).
   Bug found & fixed along the way: `TimeStepDFSPHCUDA::getMethodName()`
   returned "DFSPH_CUDA", so `SceneLoader::readTimeStepParameterObject` never
   found the scene's "DFSPH" block and the GPU ran with default solver
   tolerances. It now returns `TimeStepDFSPH::METHOD_NAME` ("DFSPH").
   Next perf step here: milestone 9 (whole-timestep graph), which subsumes the
   remaining per-step launch/sync overhead — this is where the conditional
   nodes should start paying off visibly.
3b. ~~Whole-timestep relaunchable graph~~ **done for fixed dt 2026-07-18**
   (see "Whole-timestep relaunchable graph" above). Remaining extension:
   device-resident dt (kernels reading dt from device memory) would let the
   adaptive-CFL path live inside the reusable graph too — needs kernels to take
   `const real* dt` instead of a baked host scalar.
4. ~~Reduce the per-step host mirror~~ **done 2026-07-18**: SimStep at 85k is
   now ~3.3 ms steady-state (2.35 GPU + 0.95 mirror) vs 40.2 ms CPU ⇒ ~12×
   end-to-end. Remaining ideas: true triple-buffered snapshots for the GUI
   render path, and skipping the mirror entirely on steps with no export.
   Profiling gotchas hit here (worth remembering): the ~150 ms one-time init
   (probes + map export + graph record) is amortized into "Average time"
   counters — check `DFSPH_CUDA_init`; and the CLI `--stopAt` did not override
   the scene's `stopAt` in these runs, so "longer" runs may silently still be
   100 steps.
5. MotorScene one-way then two-way coupling (`motor-coupling`); the reaction
   accumulators (`BodyReactionAccum`, atomic double force/torque) and
   `updateBoundaryMapTransform` are already in place.

## Known limitations / gotchas
- Single fluid model only; multiple fluids fall back to CPU.
- Only boundary index 0 is passed to kernels (`m_map0`); multi-boundary needs a device array of maps.
- CFL method 2 approximated as method 1.
- Boundary viscosity term not ported (scenes set it to 0).
- No parity tests, no `Tests/CUDA/` targets yet.
- `USE_DOUBLE_PRECISION` must match between main project and the CUDA lib (handled via `USE_DOUBLE` define in the CUDA CMake); current build is single precision.
