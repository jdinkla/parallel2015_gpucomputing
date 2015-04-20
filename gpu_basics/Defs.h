/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

/* 
	Hier werden die Plattformen unterschieden
	Windows, Linux, Mac
	jeweils für C11 oder ohne C11

	Unten wird NVCC definiert
*/

#undef WINDOWS
#undef LINUX
#undef MAC
#undef NVCC

// --------------- Windows ---------------
#if defined(_MSC_VER) || defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS 1
#define OS_NAME "Windows"
#endif

// --------------- Mac ---------------
#if defined(__APPLE__) || defined(__MACOSX)
#define MAC 1
#define OS_NAME "Mac"
#endif

// --------------- Linux ---------------
#ifdef __linux__
#define LINUX 1
#define OS_NAME "Linux"
#endif

// --------------- C++ 11 ---------------
#ifdef WINDOWS
// __cplusplus is broke 
#if !(_MSC_VER >= 1800)
#error Visual Studio 2013 is needed
#endif
#else
#if !(__cplusplus >= 201103L) && !defined(_MSC_VER)
#error C++ 11 is needed	
#endif
#endif

// --------------- NVIDIA ---------------
// if this is compiled with NVCC, include the runtime_api
// otherwise we define the definitions from cuda_runtime_api ourself 
#if defined(__NVCC__) || defined(__CUDACC__) || defined(COMPILE_FOR_CUDA)

#define NVCC 1
#include <cuda_runtime_api.h>
	
#else

#include "DefsByMe.h"

#endif

