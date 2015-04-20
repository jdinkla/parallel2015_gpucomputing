/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once

#include <amp.h>
#include <vector>

void amp_info();

// returns all not emulated accelerators
std::vector<concurrency::accelerator> get_accs();
