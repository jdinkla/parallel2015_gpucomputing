kernel void square(global float* input_output)
{
    size_t id = get_global_id(0);
    float value = input_output[id];
    input_output[id] = value * value;
}

