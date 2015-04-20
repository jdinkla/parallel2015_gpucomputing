/*
* Copyright (c) 2014 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <vector>
#include "Defs.h"

typedef std::pair<int, int> pair2d;
typedef std::vector<dim3> blocks_t;

blocks_t get_naive_blocks();
blocks_t get_blocks();
blocks_t get_special_blocks();

size_t get_max_size_2d(const bool silent = false);
