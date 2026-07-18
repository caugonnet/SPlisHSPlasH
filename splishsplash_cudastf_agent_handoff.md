# SPlisHSPlasH as a CUDASTF Demonstrator

## Purpose

This document hands off an investigation into using
[SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH)
as a substantial, visually impressive CUDASTF application.

The main question is whether a modern CUDA port of a representative
SPlisHSPlasH solver, especially DFSPH, would be a good vehicle for demonstrating:

- task-based orchestration of a real graphics/simulation workload;
- composition of custom CUDA kernels with CCCL algorithms;
- repeatable CUDA Graph fragments;
- CUDA Graph conditional `while` nodes;
- device-side convergence without host synchronization;
- reusable graphs across simulation timesteps;
- irregular data structures such as spatial sorting and neighbour lists;
- a more visually compelling example than MiniWeather.

This is not intended to be a full port of all SPlisHSPlasH solvers initially.
The first target should be a carefully chosen, self-contained simulation path.

---

## Why SPlisHSPlasH is Interesting

SPlisHSPlasH is an MIT-licensed C++ SPH framework for fluid simulation.

It supports visually compelling scenarios such as:

- dam breaks;
- free-surface fluids;
- viscosity;
- surface tension;
- vorticity;
- multiphase fluids;
- boundary interactions;
- rigid-fluid coupling;
- foam and rendering/export pipelines.

It is a stronger stress test for CUDASTF than MiniWeather because it combines:

- regular per-particle computations;
- irregular neighbourhood traversal;
- sorting and particle permutation;
- reductions;
- iterative pressure solvers;
- dynamic convergence;
- potentially changing particle counts;
- possible interaction with a GPU neighbour-search library;
- repeated timesteps with mostly stable memory topology.

MiniWeather remains a cleaner pedagogical example, but SPlisHSPlasH could be a
more convincing flagship application.

---

## Implementation Language

The numerical solvers in SPlisHSPlasH are written in C++.

Python is used for bindings and scripting, not as the main implementation
language for the numerical kernels.

The current upstream implementation is primarily:

- C++;
- OpenMP on CPU;
- optional vectorization in some paths;
- optional GPU neighbourhood search through `cuNSearch`.

A CUDASTF port would therefore most naturally be a C++/CUDA port of existing
C++ loop nests and solver stages.

---

## Existing CUDA Work

There has already been CUDA work related to SPlisHSPlasH, but there is no
maintained, complete CUDA backend in current upstream SPlisHSPlasH.

### Historical upstream pull request

An old upstream pull request attempted CUDA implementations of WCSPH and DFSPH:

- <https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/pull/53>

The work reportedly included:

- GPU-resident particle data;
- CUDA WCSPH;
- CUDA DFSPH;
- CUDA density computation;
- reduced host-device transfers;
- kernel fusion;
- assumptions that point sets are not dynamically created or removed.

The pull request remained unmerged and appears abandoned.

It may still be useful as:

- a source of early CUDA kernel designs;
- a reference for data layout;
- a record of performance pitfalls;
- a comparison point for a modern CUDASTF implementation.

### Academic CUDA DFSPH implementation

A separate research implementation ported DFSPH to CUDA from an older
SPlisHSPlasH version.

Paper:

- *Optimizations for Predictive-Corrective Particle-Based Fluid Simulation on GPU*
- <https://perso.liris.cnrs.fr/npronost/Papers/Optimizations%20for%20predictive-corrective%20particle-based%20fluid%20simulation%20on%20GPU%20%5BTVC2022%5D.pdf>

Repository:

- <https://git.liris.cnrs.fr/npronost/sph_dynamic_window>

This implementation reportedly:

- keeps simulation data on the GPU;
- uses a cell-linked spatial structure;
- constructs neighbourhood information with counting-sort-like techniques;
- reduces global-memory traffic;
- reorganizes or fuses kernels;
- uses CUDA/OpenGL interoperability;
- disables or modifies some dynamic behaviour for reproducibility/performance.

This code is probably the best existing technical reference for a modern port,
but it is a research snapshot based on an old SPlisHSPlasH revision rather than
a maintained backend.

---

## Candidate Initial Solver

The best initial target is probably DFSPH.

A representative timestep has the following high-level structure:

```text
update_neighbourhood
compute_density_and_solver_factors

divergence_solve:
    initialize_divergence_state

    while divergence_error > threshold:
        compute_divergence_correction
        update_velocity
        recompute_divergence_error
        reduce_error
        update_loop_condition

compute_non_pressure_forces
integrate_velocity

density_solve:
    initialize_density_state

    while density_error > threshold:
        compute_density_correction
        update_velocity
        predict_density_error
        reduce_error
        update_loop_condition

integrate_position
update_boundaries_and_emitters
```

The two iterative pressure-related loops are particularly attractive for CUDA
Graph conditional nodes.

---

## Potential CUDASTF Logical Data

A first design might expose the following logical data objects:

```cpp
logical_data<particle_positions> positions;
logical_data<particle_velocities> velocities;
logical_data<particle_accelerations> accelerations;
logical_data<particle_densities> densities;
logical_data<particle_factors> solver_factors;

logical_data<cell_keys> spatial_keys;
logical_data<particle_indices> particle_indices;
logical_data<cell_offsets> cell_offsets;

logical_data<neighbor_offsets> neighbor_offsets;
logical_data<neighbor_indices> neighbor_indices;

logical_data<solver_errors> solver_errors;
logical_data<solver_state> divergence_state;
logical_data<solver_state> density_state;

logical_data<boundary_data> boundaries;
logical_data<temporary_storage> cub_scratch;
```

The exact representation should be based on the solver path chosen from current
SPlisHSPlasH.

---

## Where CCCL Fits

A modern CUDA port should likely use CCCL heavily for global data movement and
collective operations, while keeping SPH physics interactions in custom kernels.

The expected split is:

### Good candidates for CUB or Thrust

- cell-key sorting;
- particle-index sorting;
- run-length encoding of occupied cells;
- prefix sums for cell offsets;
- prefix sums for CSR-style neighbour lists;
- reductions for convergence;
- reductions for CFL timestep selection;
- active-particle compaction;
- escaped-particle filtering;
- particle reordering;
- scans and selections for variable-size worklists.

### Good candidates for custom CUDA kernels

- density computation;
- pressure forces;
- divergence correction;
- density correction;
- viscosity;
- surface tension;
- vorticity;
- boundary interactions;
- position and velocity integration;
- traversal of neighbouring cells;
- construction of explicit neighbour lists;
- fused updates over several per-particle fields.

### Likely CCCL primitives

| Simulation operation | Candidate primitive |
|---|---|
| Sort particles by spatial cell | `cub::DeviceRadixSort::SortPairs` |
| Detect occupied-cell runs | `cub::DeviceRunLengthEncode` |
| Build cell offsets | `cub::DeviceScan::ExclusiveSum` |
| Build CSR neighbour offsets | `cub::DeviceScan::ExclusiveSum` |
| Compute convergence error | `cub::DeviceReduce::Sum` or `Max` |
| Compute CFL timestep | `cub::DeviceReduce::Max` |
| Compact active particles | `cub::DeviceSelect::Flagged` |
| Remove invalid particles | `cub::DeviceSelect::If` |
| Prototype transforms/gathers | Thrust algorithms |
| Per-block reductions | `cub::BlockReduce` |
| Warp-level neighbour work | CUB warp primitives |

A likely rule is:

- use Thrust for a first readable implementation;
- migrate stable device-wide collectives to CUB;
- use CUB block/warp primitives inside custom kernels;
- fuse reordering of multiple SoA fields into one custom kernel rather than
  issuing one gather per field.

---

## Neighbourhood-Search Options

Neighbourhood construction is likely the most important architectural choice.

### Option A: Implicit neighbour traversal

Sort particles by cell and store a cell range table.

Each particle directly visits particles in the adjacent 27 cells:

```text
compute_cell_key
sort_by_cell
build_cell_ranges

for each particle:
    for each adjacent cell:
        for each particle in cell range:
            if within_support_radius:
                process_interaction
```

Advantages:

- no explicit neighbour-list allocation;
- lower memory footprint;
- simple dependency structure.

Disadvantages:

- repeated neighbour discovery in every solver iteration;
- potentially expensive because DFSPH revisits neighbours many times.

### Option B: Explicit CSR-style neighbour lists

Build:

```text
neighbor_offsets[num_particles + 1]
neighbor_indices[num_neighbor_pairs]
```

Pipeline:

```text
compute_cell_key
sort_by_cell
build_cell_ranges
count_neighbours
exclusive_scan_neighbor_counts
fill_neighbor_indices
```

Advantages:

- pressure iterations reuse the same neighbour list;
- repeated solver passes become cheaper and simpler.

Disadvantages:

- variable total neighbour count;
- requires capacity management;
- explicit neighbour-list construction costs memory and time;
- graph reuse must account for maximum neighbour capacity.

The agent should determine which representation is used by current
SPlisHSPlasH solvers and by the existing CUDA research implementations.

---

## Repeatable CUDA Graph Opportunities

SPlisHSPlasH is attractive because the same computation pattern repeats at
several levels.

### Repeated timesteps

The complete timestep graph is largely stable:

```text
neighbourhood_build
solver_precomputation
divergence_solve
non_pressure_forces
velocity_integration
density_solve
position_integration
```

Memory addresses and capacities can remain stable even though particle values
change every timestep.

This may allow one instantiated graph to be reused over many timesteps.

### Repeatable pressure-solver fragments

Each pressure iteration consists of a repeated kernel/reduction pipeline:

```text
pressure_update
velocity_update
error_computation
error_reduction
condition_update
```

This maps naturally to a repeatable CUDASTF graph fragment.

### Conditional `while` nodes

The divergence and density loops are likely strong use cases for CUDA Graph
conditional nodes.

A conceptual body is:

```text
while conditional_handle != 0:
    custom_pressure_kernel
    custom_error_kernel
    cub_reduce_error
    update_conditional_handle_kernel
```

This could avoid:

- host synchronization after every solver iteration;
- graph reconstruction per iteration;
- host-side convergence checks;
- repeated CPU submission overhead.

This is likely one of the strongest arguments for using SPlisHSPlasH as a
CUDASTF demonstrator.

---

## Interaction Between CUB and CUDA Graphs

A CUDASTF task should be able to invoke CUB on the task stream.

Conceptually:

```cpp
ctx.task(cell_counts.read(),
         cell_offsets.write(),
         cub_scratch.rw())
    ->*[=](cudaStream_t stream,
           auto counts,
           auto offsets,
           auto scratch) {
        cub::DeviceScan::ExclusiveSum(
            scratch.data(),
            scratch.size_bytes(),
            counts.data(),
            offsets.data(),
            particle_capacity,
            stream);
    };
```

Important design constraints:

1. Query CUB temporary-storage sizes before graph creation.
2. Allocate all scratch storage persistently.
3. Avoid allocation during stream capture or conditional-body execution.
4. Use stable device pointers.
5. Prefer fixed capacities.
6. Treat active counts separately from allocated capacities.
7. Verify capture support for every CCCL primitive actually used.
8. Check whether conditional graph bodies can safely contain the captured CCCL
   sequences needed by the solver.

A key difficulty is that CUB APIs commonly receive `num_items` as a host-side
argument when the graph is built or captured.

A changing device-side active particle count cannot automatically change the
launch topology of an already instantiated CUB algorithm.

Possible approaches include:

- fixed-capacity operations with inactive-element predicates;
- graph variants for several capacity buckets;
- rebuilding or updating graphs only when capacity changes;
- custom kernels that accept a device-side active count;
- keeping particle count fixed in the first demonstrator.

---

## Main Risks

### Excessive kernel granularity

A direct translation of every CPU loop into an individual kernel or Thrust call
may perform poorly.

The old CUDA work reportedly introduced fusion to reduce launch overhead and
global-memory traffic.

The port should therefore identify coarse logical tasks while still allowing
internal fusion.

CUDASTF tasks do not need to correspond one-to-one with kernels.

A task may encapsulate:

- one custom fused kernel;
- several kernels;
- a CCCL algorithm;
- a captured graph fragment;
- a whole solver phase.

### Data layout and particle permutation

Spatial sorting changes particle order.

All per-particle fields must remain consistent:

- position;
- velocity;
- acceleration;
- density;
- pressure-related state;
- phase/material identifiers;
- boundary interaction state;
- application-specific attributes.

Reordering many arrays independently may be too expensive.

A fused SoA permutation kernel may be preferable.

### Dynamic particle counts

Emitters, sinks, deletion and changing point sets complicate graph reuse.

The initial scope should probably exclude:

- dynamic creation or removal of fluid phases;
- arbitrary point-set creation;
- buffer reallocation;
- unconstrained emitter growth.

A fixed maximum capacity plus an active count is likely preferable.

### Neighbour-list capacity

Explicit neighbour lists require a bounded allocation strategy.

Possible approaches:

- fixed maximum neighbours per particle;
- global overallocated CSR storage;
- capacity checks followed by graph rebuild;
- two-pass count/scan/fill with persistent maximum capacity.

### Integration with `cuNSearch`

Current SPlisHSPlasH can use `cuNSearch` for GPU neighbour search.

Questions include:

- Does it leave results resident on the GPU?
- What data format does it produce?
- Can its operations be captured into CUDA Graphs?
- Does it allocate internally?
- Does it synchronize?
- Can it accept an external stream?
- Is the output usable directly by custom CUDA kernels?
- Does it reorder particles?
- Is a custom CCCL-based implementation preferable for CUDASTF?

### Performance comparison

The correct baseline is not merely upstream CPU SPlisHSPlasH.

Potential comparison targets:

- current OpenMP SPlisHSPlasH;
- current AVX-enabled CPU implementation;
- historical upstream CUDA pull request;
- academic CUDA DFSPH implementation;
- a hand-written stream-based CUDA implementation;
- CUDASTF stream backend;
- CUDASTF graph backend;
- repeatable graph backend;
- conditional-while graph backend.

---

## Recommended Initial Scope

A realistic first port should deliberately constrain the problem.

Suggested configuration:

- one fluid phase;
- DFSPH;
- static boundaries;
- no dynamic point-set creation;
- no particle emission initially;
- fixed particle capacity;
- fixed precision choice;
- one representative dam-break scene;
- offline export rather than immediate GUI integration;
- explicit profiling of every phase;
- one GPU initially.

Suggested first scene:

- a standard double-dam-break or comparable current SPlisHSPlasH example.

Suggested rendering path:

- reuse SPlisHSPlasH export;
- output VTK or Partio;
- render through the existing visualization pipeline;
- defer CUDA/OpenGL interoperability until the solver works.

---

## Suggested Milestones

### Milestone 1: Code archaeology

Identify in current SPlisHSPlasH:

- the DFSPH timestep entry point;
- every stage of one timestep;
- the dominant CPU loops;
- the convergence loops;
- all particle fields touched by DFSPH;
- neighbourhood-search interfaces;
- boundary-model interactions;
- dynamic allocation sites;
- emitter and deletion paths;
- current threading/vectorization.

Deliverable:

- a call graph and dependency graph for one DFSPH timestep.

### Milestone 2: Existing CUDA comparison

Study:

- upstream CUDA pull request;
- academic CUDA implementation;
- any forks with newer CUDA work;
- current `cuNSearch` integration.

Deliverable:

- a table of reusable kernels, obsolete assumptions, data layouts and known
  performance problems.

### Milestone 3: Minimal device-resident state

Create a minimal CUDA data model for:

- positions;
- velocities;
- densities;
- solver factors;
- boundary data;
- spatial keys;
- cell ranges;
- solver-error state.

Deliverable:

- one device-resident timestep skeleton with no host round-trips between stages.

### Milestone 4: CCCL neighbourhood pipeline

Implement:

```text
key_generation
radix_sort
cell_run_detection
cell_offset_construction
```

Then choose implicit or explicit neighbour traversal.

Deliverable:

- validated GPU neighbourhood data matching CPU results.

### Milestone 5: One pressure loop

Port one DFSPH iterative loop using:

- custom physics kernels;
- CUB reduction;
- a host-controlled loop initially.

Deliverable:

- numerically validated GPU iteration and performance profile.

### Milestone 6: CUDASTF task decomposition

Express the phase using logical data and tasks.

Compare:

- one task per kernel;
- one task per solver phase;
- fused custom-kernel tasks;
- tasks wrapping CCCL operations.

Deliverable:

- a recommended granularity model.

### Milestone 7: Repeatable graphs

Represent the pressure-iteration body as a repeatable graph fragment.

Deliverable:

- comparison against normal stream submission.

### Milestone 8: Conditional `while`

Move convergence control onto the device using a CUDA Graph conditional node.

Deliverable:

- no host synchronization per pressure iteration;
- numerical validation;
- launch and runtime comparison.

### Milestone 9: Full timestep graph

Combine:

- neighbourhood construction;
- divergence solve;
- force computation;
- density solve;
- integration.

Deliverable:

- reusable timestep graph over many frames.

### Milestone 10: Visual demonstration

Run and render a compelling scene.

Deliverable:

- video or image sequence;
- performance results;
- task graph visualization;
- comparison with MiniWeather.

---

## Questions the Agent Should Answer First

1. What is the current, exact DFSPH timestep call graph?
2. Which stages dominate runtime in current CPU SPlisHSPlasH?
3. Which stages already have reusable CUDA implementations?
4. What is the current state of the old CUDA pull request?
5. Are there newer CUDA forks not yet identified?
6. What GPU data does `cuNSearch` expose?
7. Can `cuNSearch` run asynchronously on a caller-provided stream?
8. Is `cuNSearch` CUDA Graph capture-compatible?
9. Does current SPlisHSPlasH already reorder particles spatially?
10. How many particle arrays must be permuted together?
11. Are explicit neighbour lists reused through both pressure loops?
12. What is the typical pressure-iteration count?
13. How stable is the active particle count in standard scenes?
14. Can a first scene avoid emitters and particle deletion?
15. Which CCCL operations are graph-capture-safe in the required configuration?
16. Can all CUB temporary storage be allocated ahead of graph construction?
17. How should `num_items` be handled for graph reuse?
18. Does the conditional body require fixed launch dimensions?
19. Which kernels should be fused to avoid reproducing old performance issues?
20. What is the smallest visually impressive end-to-end milestone?

---

## Concrete Agent Assignment

Investigate and propose a modern CUDASTF-based DFSPH port of SPlisHSPlasH.

The response should include:

1. The exact current SPlisHSPlasH source files and functions involved in one
   DFSPH timestep.
2. A detailed timestep DAG.
3. A list of all major per-particle arrays and their read/write dependencies.
4. A comparison of implicit versus explicit neighbour representations.
5. A mapping from every major stage to:
   - custom CUDA kernel;
   - CUB primitive;
   - Thrust primitive;
   - external library call;
   - host-side operation.
6. An assessment of the old CUDA pull request.
7. An assessment of the academic CUDA implementation.
8. An assessment of `cuNSearch`.
9. A proposed CUDASTF logical-data model.
10. A proposed CUDASTF task granularity.
11. A plan for persistent scratch storage.
12. A plan for repeatable graph fragments.
13. A plan for conditional `while` nodes.
14. A strategy for fixed capacity and active particle counts.
15. A staged implementation plan with validation tests.
16. A benchmark plan comparing streams, ordinary graphs, repeatable graphs and
    conditional graphs.
17. A recommendation on the best initial scene.
18. A list of technical blockers and unknowns.

Do not begin by porting the entire framework.

Start by producing a source-level architecture report for one constrained DFSPH
configuration, then implement the smallest device-resident vertical slice.

---

## Expected Validation

Correctness should be evaluated at several levels.

### Primitive-level validation

- sorted keys and indices;
- cell offsets;
- neighbour counts;
- neighbour identities;
- reductions;
- scans;
- particle permutation.

### Solver-level validation

- densities;
- solver factors;
- velocity corrections;
- convergence errors;
- number of pressure iterations;
- integrated positions and velocities.

### End-to-end validation

- deterministic fixed-timestep scene where possible;
- comparison against CPU SPlisHSPlasH;
- tolerance-based state comparisons;
- visual comparison;
- conservation or stability metrics when meaningful.

### Performance validation

Measure:

- kernel time;
- CCCL algorithm time;
- graph launch overhead;
- number of kernels per timestep;
- global-memory traffic where possible;
- host synchronization count;
- pressure-loop iteration overhead;
- neighbourhood-build cost;
- total frame time;
- scaling with particle count.

---

## Working Hypothesis

The strongest expected result is not that every SPlisHSPlasH loop maps directly
to a CUDASTF task.

The likely successful design is hierarchical:

```text
CUDASTF timestep DAG
    |
    +-- neighbourhood task
    |       +-- custom key kernel
    |       +-- CUB radix sort
    |       +-- CUB run-length encode
    |       +-- CUB scan
    |       +-- custom neighbour construction
    |
    +-- divergence-solver repeatable subgraph
    |       +-- fused pressure kernels
    |       +-- block reductions
    |       +-- CUB final reduction
    |       +-- device-side conditional update
    |
    +-- force task
    |       +-- fused custom kernels
    |
    +-- density-solver repeatable subgraph
    |       +-- fused pressure kernels
    |       +-- block reductions
    |       +-- CUB final reduction
    |       +-- device-side conditional update
    |
    +-- integration task
```

CUDASTF would own the coarse dependency structure and graph reuse.

CCCL would provide robust global collectives.

Custom fused CUDA kernels would implement the local SPH physics.

Conditional CUDA Graph nodes would own convergence control.

This combination is the main reason SPlisHSPlasH may be a particularly strong
CUDASTF demonstrator.
