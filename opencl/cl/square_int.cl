kernel void square(global int* input_output)
{
	size_t id = get_global_id(0);
	int value = input_output[id];
	input_output[id] = value * value;
}

