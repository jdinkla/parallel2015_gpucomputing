/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#define __CL_ENABLE_EXCEPTIONS

#include "OpenCLUtilities.h"
#include "cl_1_2.hpp"
#include <iostream>
#include <vector>
#include "Defs.h"
#include "StringUtilities.h"

using namespace std;

void info_platforms()
{
	vector<cl::Platform> platforms;
	try
	{
		cout << "---------- platforms " << endl;
		cl::Platform::get(&platforms);
		for (auto& p : platforms)
		{
			cout << "name:       '" << p.getInfo<CL_PLATFORM_NAME>() << "'" << endl;
			cout << "vendor:     '" << p.getInfo<CL_PLATFORM_VENDOR>() << "'" << endl;
			cout << "version:    '" << p.getInfo<CL_PLATFORM_VERSION>() << "'" << endl;
			cout << "profile:    '" << p.getInfo<CL_PLATFORM_PROFILE>() << "'" << endl;
			cout << "extensions: '" << p.getInfo<CL_PLATFORM_EXTENSIONS>() << "'" << endl;
			cout << endl;
		}

	}
	catch (cl::Error e)
	{
		cerr << "ERROR " << e.what() << ", code=" << e.err() << endl;
	}
}

void info_devices()
{
	vector<cl::Platform> platforms;
	try
	{
		cout << "---------- devices " << endl;
		cl::Platform::get(&platforms);
		for (auto& p : platforms)
		{
			vector<cl::Device> devices;
			p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
			cout << "platform:   '" << p.getInfo<CL_PLATFORM_NAME>() << "' has " << devices.size() << " devices" << endl;
			for (auto& d : devices)
			{
				cout << "  name:       '" << d.getInfo<CL_DEVICE_NAME>() << "'" << endl;
				cout << "  vendor:     '" << d.getInfo<CL_DEVICE_VENDOR>() << "'" << endl;
				cout << "  available:  '" << d.getInfo<CL_DEVICE_AVAILABLE>() << "'" << endl;
				cout << "  comp avail: '" << d.getInfo<CL_DEVICE_COMPILER_AVAILABLE>() << "'" << endl;
				cout << "  extensions: '" << d.getInfo<CL_DEVICE_EXTENSIONS>() << "'" << endl;
				cout << endl;
			}

			cout << endl;
		}

	}
	catch (cl::Error e)
	{
		cerr << "ERROR " << e.what() << ", code=" << e.err() << endl;
	}
}

void info_devices_detailed()
{
	vector<cl::Platform> platforms;
	try
	{
		cout << "---------- devices " << endl;
		cl::Platform::get(&platforms);
		for (auto& p : platforms)
		{
			vector<cl::Device> devices;
			p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
			cout << "platform:   '" << p.getInfo<CL_PLATFORM_NAME>() << "' has " << devices.size() << " devices" << endl;
			for (auto& d : devices)
			{
				cout << "  CL_DEVICE_TYPE: " << d.getInfo<CL_DEVICE_TYPE>() << endl;
				cout << "  CL_DEVICE_VENDOR_ID: " << d.getInfo<CL_DEVICE_VENDOR_ID>() << endl;
				cout << "  CL_DEVICE_MAX_COMPUTE_UNITS: " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
				cout << "  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << d.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << endl;
				cout << "  CL_DEVICE_MAX_WORK_GROUP_SIZE: " << d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl;
				//cout << "  CL_DEVICE_MAX_WORK_ITEM_SIZES: " << d.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>() << endl;
				cout << "  CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR: " << d.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR>() << endl;
				cout << "  CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT: " << d.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT>() << endl;
				cout << "  CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT: " << d.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT>() << endl;
				cout << "  CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG: " << d.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG>() << endl;
				cout << "  CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: " << d.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>() << endl;
				cout << "  CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE: " << d.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>() << endl;
				cout << "  CL_DEVICE_MAX_CLOCK_FREQUENCY: " << d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << endl;
				cout << "  CL_DEVICE_ADDRESS_BITS: " << d.getInfo<CL_DEVICE_ADDRESS_BITS>() << endl;
				cout << "  CL_DEVICE_MAX_READ_IMAGE_ARGS: " << d.getInfo<CL_DEVICE_MAX_READ_IMAGE_ARGS>() << endl;
				cout << "  CL_DEVICE_MAX_WRITE_IMAGE_ARGS: " << d.getInfo<CL_DEVICE_MAX_WRITE_IMAGE_ARGS>() << endl;
				cout << "  CL_DEVICE_MAX_MEM_ALLOC_SIZE: " << d.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << endl;
				cout << "  CL_DEVICE_IMAGE2D_MAX_WIDTH: " << d.getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>() << endl;
				cout << "  CL_DEVICE_IMAGE2D_MAX_HEIGHT: " << d.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>() << endl;
				cout << "  CL_DEVICE_IMAGE3D_MAX_WIDTH: " << d.getInfo<CL_DEVICE_IMAGE3D_MAX_WIDTH>() << endl;
				cout << "  CL_DEVICE_IMAGE3D_MAX_HEIGHT: " << d.getInfo<CL_DEVICE_IMAGE3D_MAX_HEIGHT>() << endl;
				cout << "  CL_DEVICE_IMAGE3D_MAX_DEPTH: " << d.getInfo<CL_DEVICE_IMAGE3D_MAX_DEPTH>() << endl;
				cout << "  CL_DEVICE_IMAGE_SUPPORT: " << d.getInfo<CL_DEVICE_IMAGE_SUPPORT>() << endl;
				cout << "  CL_DEVICE_MAX_PARAMETER_SIZE: " << d.getInfo<CL_DEVICE_MAX_PARAMETER_SIZE>() << endl;
				cout << "  CL_DEVICE_MAX_SAMPLERS: " << d.getInfo<CL_DEVICE_MAX_SAMPLERS>() << endl;
				cout << "  CL_DEVICE_MEM_BASE_ADDR_ALIGN: " << d.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>() << endl;
				cout << "  CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE: " << d.getInfo<CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE>() << endl;
				cout << "  CL_DEVICE_SINGLE_FP_CONFIG: " << d.getInfo<CL_DEVICE_SINGLE_FP_CONFIG>() << endl;
				cout << "  CL_DEVICE_GLOBAL_MEM_CACHE_TYPE: " << d.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE>() << endl;
				cout << "  CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE: " << d.getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>() << endl;
				cout << "  CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: " << d.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>() << endl;
				cout << "  CL_DEVICE_GLOBAL_MEM_SIZE: " << d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << endl;
				cout << "  CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: " << d.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>() << endl;
				cout << "  CL_DEVICE_MAX_CONSTANT_ARGS: " << d.getInfo<CL_DEVICE_MAX_CONSTANT_ARGS>() << endl;
				cout << "  CL_DEVICE_LOCAL_MEM_TYPE: " << d.getInfo<CL_DEVICE_LOCAL_MEM_TYPE>() << endl;
				cout << "  CL_DEVICE_LOCAL_MEM_SIZE: " << d.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << endl;
				cout << "  CL_DEVICE_ERROR_CORRECTION_SUPPORT: " << d.getInfo<CL_DEVICE_ERROR_CORRECTION_SUPPORT>() << endl;
				cout << "  CL_DEVICE_PROFILING_TIMER_RESOLUTION: " << d.getInfo<CL_DEVICE_PROFILING_TIMER_RESOLUTION>() << endl;
				cout << "  CL_DEVICE_ENDIAN_LITTLE: " << d.getInfo<CL_DEVICE_ENDIAN_LITTLE>() << endl;
				cout << "  CL_DEVICE_AVAILABLE: " << d.getInfo<CL_DEVICE_AVAILABLE>() << endl;
				cout << "  CL_DEVICE_COMPILER_AVAILABLE: " << d.getInfo<CL_DEVICE_COMPILER_AVAILABLE>() << endl;
				cout << "  CL_DEVICE_EXECUTION_CAPABILITIES: " << d.getInfo<CL_DEVICE_EXECUTION_CAPABILITIES>() << endl;
				cout << "  CL_DEVICE_QUEUE_PROPERTIES: " << d.getInfo<CL_DEVICE_QUEUE_PROPERTIES>() << endl;
#if !defined(MAC) && !defined(OPENCL_AMD)
				cout << "  CL_DEVICE_QUEUE_ON_HOST_PROPERTIES: " << d.getInfo<CL_DEVICE_QUEUE_ON_HOST_PROPERTIES>() << endl;
#endif
				cout << "  CL_DEVICE_NAME: " << d.getInfo<CL_DEVICE_NAME>() << endl;
				cout << "  CL_DEVICE_VENDOR: " << d.getInfo<CL_DEVICE_VENDOR>() << endl;
				cout << "  CL_DRIVER_VERSION: " << d.getInfo<CL_DRIVER_VERSION>() << endl;
				cout << "  CL_DEVICE_PROFILE: " << d.getInfo<CL_DEVICE_PROFILE>() << endl;
				cout << "  CL_DEVICE_VERSION: " << d.getInfo<CL_DEVICE_VERSION>() << endl;
				cout << "  CL_DEVICE_EXTENSIONS: " << d.getInfo<CL_DEVICE_EXTENSIONS>() << endl;
				cout << "  CL_DEVICE_PLATFORM: " << d.getInfo<CL_DEVICE_PLATFORM>() << endl;
				cout << "  CL_DEVICE_DOUBLE_FP_CONFIG: " << d.getInfo<CL_DEVICE_DOUBLE_FP_CONFIG>() << endl;
				cout << "  CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF: " << d.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF>() << endl;
				cout << "  CL_DEVICE_HOST_UNIFIED_MEMORY: " << d.getInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>() << endl;
				cout << "  CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR: " << d.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR>() << endl;
				cout << "  CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT: " << d.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT>() << endl;
				cout << "  CL_DEVICE_NATIVE_VECTOR_WIDTH_INT: " << d.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_INT>() << endl;
				cout << "  CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG: " << d.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG>() << endl;
				cout << "  CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT: " << d.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT>() << endl;
				cout << "  CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE: " << d.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE>() << endl;
				cout << "  CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF: " << d.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF>() << endl;
				cout << "  CL_DEVICE_OPENCL_C_VERSION: " << d.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << endl;
				//cout << "  CL_DEVICE_LINKER_AVAILABLE: " << d.getInfo<CL_DEVICE_LINKER_AVAILABLE>() << endl;
				//cout << "  CL_DEVICE_BUILT_IN_KERNELS: " << d.getInfo<CL_DEVICE_BUILT_IN_KERNELS>() << endl;
				////cout << "  CL_DEVICE_IMAGE_MAX_BUFFER_SIZE: " << d.getInfo<CL_DEVICE_IMAGE_MAX_BUFFER_SIZE>() << endl;
				////cout << "  CL_DEVICE_IMAGE_MAX_ARRAY_SIZE: " << d.getInfo<CL_DEVICE_IMAGE_MAX_ARRAY_SIZE>() << endl;
				//cout << "  CL_DEVICE_PARENT_DEVICE: " << d.getInfo<CL_DEVICE_PARENT_DEVICE>() << endl;
				////cout << "  CL_DEVICE_PARTITION_MAX_SUB_DEVICES: " << d.getInfo<CL_DEVICE_PARTITION_MAX_SUB_DEVICES>() << endl;
				////cout << "  CL_DEVICE_PARTITION_PROPERTIES: " << d.getInfo<CL_DEVICE_PARTITION_PROPERTIES>() << endl;
				//cout << "  CL_DEVICE_PARTITION_AFFINITY_DOMAIN: " << d.getInfo<CL_DEVICE_PARTITION_AFFINITY_DOMAIN>() << endl;
				////cout << "  CL_DEVICE_PARTITION_TYPE: " << d.getInfo<CL_DEVICE_PARTITION_TYPE>() << endl;
				//cout << "  CL_DEVICE_REFERENCE_COUNT: " << d.getInfo<CL_DEVICE_REFERENCE_COUNT>() << endl;
				//cout << "  CL_DEVICE_PREFERRED_INTEROP_USER_SYNC: " << d.getInfo<CL_DEVICE_PREFERRED_INTEROP_USER_SYNC>() << endl;
				////cout << "  CL_DEVICE_PRINTF_BUFFER_SIZE: " << d.getInfo<CL_DEVICE_PRINTF_BUFFER_SIZE>() << endl;
				////cout << "  CL_DEVICE_IMAGE_PITCH_ALIGNMENT: " << d.getInfo<CL_DEVICE_IMAGE_PITCH_ALIGNMENT>() << endl;
				//cout << "  CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT: " << d.getInfo<CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT>() << endl;
				//cout << "  CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS: " << d.getInfo<CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS>() << endl;
				//cout << "  CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE: " << d.getInfo<CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE>() << endl;
				//cout << "  CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES: " << d.getInfo<CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES>() << endl;
				//cout << "  CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE: " << d.getInfo<CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE>() << endl;
				//cout << "  CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE: " << d.getInfo<CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE>() << endl;
				//cout << "  CL_DEVICE_MAX_ON_DEVICE_QUEUES: " << d.getInfo<CL_DEVICE_MAX_ON_DEVICE_QUEUES>() << endl;
				//cout << "  CL_DEVICE_MAX_ON_DEVICE_EVENTS: " << d.getInfo<CL_DEVICE_MAX_ON_DEVICE_EVENTS>() << endl;
				//cout << "  CL_DEVICE_SVM_CAPABILITIES: " << d.getInfo<CL_DEVICE_SVM_CAPABILITIES>() << endl;
				//cout << "  CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE: " << d.getInfo<CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE>() << endl;
				//cout << "  CL_DEVICE_MAX_PIPE_ARGS: " << d.getInfo<CL_DEVICE_MAX_PIPE_ARGS>() << endl;
				//cout << "  CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS: " << d.getInfo<CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS>() << endl;
				//cout << "  CL_DEVICE_PIPE_MAX_PACKET_SIZE: " << d.getInfo<CL_DEVICE_PIPE_MAX_PACKET_SIZE>() << endl;
				//cout << "  CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT: " << d.getInfo<CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT>() << endl;
				//cout << "  CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT: " << d.getInfo<CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT>() << endl;
				//cout << "  CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT: " << d.getInfo<CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT>() << endl;
				cout << endl;
			}
			cout << endl;
		}
	}
	catch (cl::Error e)
	{
		cerr << "ERROR " << e.what() << ", code=" << e.err() << endl;
	}

}

void info_all()
{
	info_platforms();
	info_devices();
}

void info_tree()
{
	string indent = "    ";
	vector<cl::Platform> platforms;
	try
	{
		cl::Platform::get(&platforms);
		for (auto& p : platforms)
		{
			cout << "+--- '" 
				<< trim(p.getInfo<CL_PLATFORM_NAME>()) 
				<< "', '" 
				//<< p.getInfo<CL_PLATFORM_VENDOR>() 
				//<< "', "
				<< trim(p.getInfo<CL_PLATFORM_VERSION>())
				<< "'" 
				<< endl;
			vector<cl::Device> devices;
			p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
			for (auto& d : devices)
			{
				cout << "|  +-- "
					<< "'" 
					<< trim(d.getInfo<CL_DEVICE_NAME>())
					<< "'"
					<< endl;
			}
			cout << "|" << endl;
		}

	}
	catch (cl::Error e)
	{
		cerr << "ERROR " << e.what() << ", code=" << e.err() << endl;
	}
}


cl::Platform get_platform_by_name(std::string name)
{
	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) 
	{
		throw new std::runtime_error("OpenCL not found");
	}
	int i = 0;
	for (auto& platform : platforms) {
		const string name2 = platform.getInfo<CL_PLATFORM_NAME>();
		const std::size_t found = name2.find(name);
		if (found != std::string::npos) 
		{
			break;
		}
		i++;
	}
	return platforms[i];
}

cl::Device get_device_by_name(cl::Platform& platform, std::string name)
{
	vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	if (devices.size() == 0) 
	{
		throw new std::runtime_error("No devices found");
	}
	int i = 0;
	for (auto& device : devices) {
		const string name2 = device.getInfo<CL_DEVICE_NAME>();
		const std::size_t found = name2.find(name);
		if (found != std::string::npos) 
		{
			break;
		}
		i++;
	}
	return devices[i];
}

vector<cl::Platform> get_platforms()
{
	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	return platforms;
}

vector<cl::Device> get_devices(cl::Platform& platform)
{
	vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	return devices;
}

cl::Context get_context(cl::Platform& platform, cl::Device& device)
{
	cl_context_properties properties[]
		= { CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0 };
	cl::Context context(device, properties);
	return context;
}

cl::Program get_program(cl::Device& device, cl::Context& context, string code)
{
	cl::Program::Sources sources(1, { code.c_str(), code.length() + 1 });
	cl::Program program = cl::Program(context, sources); program.build({ device });
	return program;
}

double duration_in_ms(cl::Event& event)
{
	const cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	const cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	return (end - start) / 1000.0;
}

