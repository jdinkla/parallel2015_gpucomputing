/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <string>
#include "cl_1_2.hpp"

void info_platforms();

void info_devices();

void info_devices_detailed();

void info_all();

void info_tree();

cl::Platform get_platform_by_name(std::string name);

cl::Device get_device_by_name(cl::Platform& platform, std::string name);

std::vector<cl::Platform> get_platforms();

std::vector<cl::Device> get_devices(cl::Platform& platform);

cl::Context get_context(cl::Platform& platform, cl::Device& device);

cl::Program get_program(cl::Device& device, cl::Context& context, std::string code);

double duration_in_ms(cl::Event& event);
