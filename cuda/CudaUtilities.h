/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include "Defs.h"
#include <ostream>

#if defined(MAC) || defined(LINUX)
#define __declspec(x)
#endif

namespace CUDA
{

//// Überprüft das übergebene Error-Flag
//void check_rc(cudaError_t rc, const char* msg);

// Überprüft den Zustand des Error-Flags von CUDA
__declspec(deprecated)
void check_cuda(const char* msg = 0);

// Überprüft den Zustand des Error-Flags von CUDA
void check(const char* msg);

// Gibt die Größe des freien Speichers in Bytes zurück.
size_t get_free_device_mem();

// Gibt die Größe des Speichers in Bytes zurück.
size_t get_total_device_mem();

// Gibt die Anzahl der GPUs zurück
int get_number_of_gpus();

// resets all the devices
void reset_all();

// Determines the amount of maximal allocatable memory
size_t determine_max_mem(const int device_id, const size_t total);

// returns the cuda device id of the GPU that is currently rendering
int get_opengl_device();

// set the current CUDA device to the device that is used by OpenGL
void set_device_according_to_opengl();

// synchronize all devices
void sync_all();

}

inline std::ostream &operator<<(std::ostream& ostr, const dim3& d)
{
	return ostr << d.x << "," << d.y << "," << d.z;
}

inline std::ostream &operator<<(std::ostream& ostr, const uchar4& d)
{
	return ostr << (int)d.x << "," << (int)d.y << "," << (int)d.z << "," << (int)d.w;
}


