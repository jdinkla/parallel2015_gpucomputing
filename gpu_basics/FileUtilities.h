/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <string>
#include "Defs.h"
#include "Extent2.h"
#include "HostBuffer.h"

std::string read_file(std::string name);

void save_bmp(std::string filename, uchar4* ptr, const Extent2& ext);

void save_bmp(std::string filename, uchar4* ptr, const int width, const int height);

void save_bmp(std::string filename, HostBuffer<float>& host, const float max_heat);


