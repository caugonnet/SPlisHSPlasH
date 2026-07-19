#include "Bender2019MapCUDA.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace SPH
{
namespace cuda_dfsph
{

static void mapCudaCheck(cudaError_t e, const char *what)
{
	if (e != cudaSuccess)
	{
		std::fprintf(stderr, "[dfsph-cuda] boundary map CUDA error (%s): %s\n", what, cudaGetErrorString(e));
		std::abort();
	}
}

// Upload a flattened Bender2019 map to the device. Allocates node/cell/cellMap
// arrays once; the returned struct is POD and can be passed to kernels.
DeviceBoundaryMap uploadBoundaryMap(const BoundaryMapDesc &desc)
{
	DeviceBoundaryMap m;
	std::memset(&m, 0, sizeof(m));
	m.valid = 0;

	if (desc.nodes[0] == nullptr || desc.nodes[1] == nullptr)
		return m;

	for (int d = 0; d < 3; ++d)
	{
		m.domainMin[d] = desc.domainMin[d];
		m.domainMax[d] = desc.domainMax[d];
		m.cellSize[d] = desc.cellSize[d];
		m.invCellSize[d] = desc.invCellSize[d];
		m.resolution[d] = desc.resolution[d];
	}
	m.nCells = static_cast<unsigned int>(desc.nCells);
	for (int i = 0; i < 9; ++i)
		m.R[i] = desc.rotation[i];
	for (int d = 0; d < 3; ++d)
	{
		m.t[d] = desc.translation[d];
		m.angVel[d] = desc.angularVelocity[d];
		m.linVel[d] = desc.linearVelocity[d];
		m.com[d] = desc.comPosition[d];
	}
	m.isDynamic = desc.isDynamic ? 1 : 0;

	for (int f = 0; f < 2; ++f)
	{
		double *dNodes = nullptr;
		unsigned int *dCells = nullptr;
		unsigned int *dCellMap = nullptr;
		const size_t nodeBytes = desc.nodeCount[f] * sizeof(double);
		const size_t cellBytes = desc.nCells * 32 * sizeof(unsigned int);
		const size_t mapBytes = desc.nCells * sizeof(unsigned int);
		mapCudaCheck(cudaMalloc(&dNodes, nodeBytes), "malloc nodes");
		mapCudaCheck(cudaMalloc(&dCells, cellBytes), "malloc cells");
		mapCudaCheck(cudaMalloc(&dCellMap, mapBytes), "malloc cellMap");
		mapCudaCheck(cudaMemcpy(dNodes, desc.nodes[f], nodeBytes, cudaMemcpyHostToDevice), "copy nodes");
		mapCudaCheck(cudaMemcpy(dCells, desc.cells[f], cellBytes, cudaMemcpyHostToDevice), "copy cells");
		mapCudaCheck(cudaMemcpy(dCellMap, desc.cellMap[f], mapBytes, cudaMemcpyHostToDevice), "copy cellMap");
		m.nodes[f] = dNodes;
		m.cells[f] = dCells;
		m.cellMap[f] = dCellMap;
	}
	m.valid = 1;
	return m;
}

void freeBoundaryMap(DeviceBoundaryMap &m)
{
	for (int f = 0; f < 2; ++f)
	{
		if (m.nodes[f]) cudaFree(const_cast<double*>(m.nodes[f]));
		if (m.cells[f]) cudaFree(const_cast<unsigned int*>(m.cells[f]));
		if (m.cellMap[f]) cudaFree(const_cast<unsigned int*>(m.cellMap[f]));
		m.nodes[f] = nullptr;
		m.cells[f] = nullptr;
		m.cellMap[f] = nullptr;
	}
	m.valid = 0;
}

// Refresh only the rigid transform / kinematics (motor coupling, cheap).
void updateBoundaryMapTransform(DeviceBoundaryMap &m,
								const double R[9], const double t[3],
								const double angVel[3], const double linVel[3],
								const double com[3])
{
	for (int i = 0; i < 9; ++i)
		m.R[i] = R[i];
	for (int d = 0; d < 3; ++d)
	{
		m.t[d] = t[d];
		m.angVel[d] = angVel[d];
		m.linVel[d] = linVel[d];
		m.com[d] = com[d];
	}
}

} // namespace cuda_dfsph
} // namespace SPH
