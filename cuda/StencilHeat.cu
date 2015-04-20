#include "StencilHeat.h"

template <typename T>
void stencil_heat_2d(
	const CudaExecConfig& cnf, DeviceBuffer<T>& src, 
	DeviceBuffer<T>& dst, const T value)
{
	dim3 grid = cnf.get_grid();
	dim3 block = cnf.get_block();
	stencil_heat_2d_kernel<<<grid, block>>>(
		src.get_ptr(), dst.get_ptr(), dst.get_extent(), value);
}

template <typename T>
void stencil_heat_2d(
	const CudaExecConfig& cnf, 
	std::shared_ptr<DeviceBuffer<T>> src,
	std::shared_ptr<DeviceBuffer<T>> dst, const T value)
{
	stencil_heat_2d(cnf, *src.get(), *dst.get(), value);
}

template <typename T>
void stencil_heat_2d(
	const CudaExecConfig& cnf,
	thrust::device_vector<T>& srcVec,
	thrust::device_vector<T>& destVec,
	const T value
	)
{
	const T* src = thrust::raw_pointer_cast(&srcVec[0]);
	T* dst = thrust::raw_pointer_cast(&destVec[0]);
	stencil_heat_2d_kernel << <cnf.get_grid(), cnf.get_block() >> >(src, dst, cnf.get_extent(), value);
}

// Instances
template
void stencil_heat_2d(
	const CudaExecConfig& cnf,
	DeviceBuffer<float, Extent2>& srcBuf,
	DeviceBuffer<float, Extent2>& destBuf,
	const float value);
	
template
void stencil_heat_2d(
	const CudaExecConfig& cnf,
	std::shared_ptr<DeviceBuffer<float>> src,
	std::shared_ptr<DeviceBuffer<float>> dst, const float value);
