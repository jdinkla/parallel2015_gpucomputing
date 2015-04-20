/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

// #define PROFILE_NVIDIA

#ifdef PROFILE_NVIDIA

#include <nvToolsExtCuda.h>

#define START_RANGE(msg) auto x7836824_range = nvtxRangeStartA(msg); 
#define END_RANGE nvtxRangeEnd(x7836824_range);

#else

#define START_RANGE(msg)
#define END_RANGE

#endif
