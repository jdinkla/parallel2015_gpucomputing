#include "StencilHeatWithOffset.h"

template <typename T>
void stencil_heat_2d(
	const CudaExecConfig& cnf, DeviceBuffer<T>& src, DeviceBuffer<T>& dst, 
	const Extent2& without, const Pos2& offset, const T value)
{
	dim3 grid = cnf.get_grid();
	dim3 block = cnf.get_block();
	stencil_heat_2d_kernel << <grid, block >> >(
		src.get_ptr(), dst.get_ptr(), dst.get_extent(), without, offset, value);
}

template <typename T>
void stencil_heat_2d(
	const CudaExecConfig& cnf,
	std::shared_ptr<DeviceBuffer<T>> src,
	std::shared_ptr<DeviceBuffer<T>> dst, 
	const Extent2& without, const Pos2& offset, const T value)
{
	stencil_heat_2d(cnf, *src.get(), *dst.get(), without, offset, value);
}

// Instances
template
void stencil_heat_2d(
	const CudaExecConfig& cnf,
	DeviceBuffer<float, Extent2>& srcBuf,
	DeviceBuffer<float, Extent2>& destBuf,
	const Extent2& without, const Pos2& offset,
	const float value);

template
void stencil_heat_2d(
	const CudaExecConfig& cnf,
	std::shared_ptr<DeviceBuffer<float>> src,
	std::shared_ptr<DeviceBuffer<float>> dst, 
	const Extent2& without, const Pos2& offset, 
	const float value);

