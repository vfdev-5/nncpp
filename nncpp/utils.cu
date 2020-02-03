// STD
#include <cassert>
#include <iostream>


// NNCPP
#include "cuda_utils.cuh"


namespace nncpp
{


float _compute_op(const Tensor & input, void kernel_reduce_func(CUDATensorWrapper, CUDATensorWrapper), float initValue=0.0f)
{
    assert(input.device == Device::CUDA);
    CUDATensorWrapper itw(input);
    Tensor output = Tensor::zeros(1, 1, 1, 1, Device::CUDA);
    output.at(0) = initValue;
    CUDATensorWrapper otw(output);
    uint gridSize = setup_grid_size(input.numel(), BLOCK_SIZE);
    kernel_reduce_func<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(itw, otw);
    CHECK(cudaGetLastError());
    return output.at(0);          
}


Tensor _compute_op_on_dim(const Tensor & input, size_t dim, void kernel_reduce_func(CUDATensorWrapper, size_t, CUDATensorWrapper), float initValue=0.0f)
{
    assert(dim < 4);
    assert(input.device == Device::CUDA);
    CUDATensorWrapper itw(input);
    
    size_t outShape[4]{input.shape[0], input.shape[1], input.shape[2], input.shape[3]};
    outShape[dim] = 1;
    Tensor output = Tensor::zeros(outShape[0], outShape[1], outShape[2], outShape[3], Device::CUDA);
    for (size_t i = 0; i < output.numel(); i++)
    {
        output.at(i) = initValue;
    }

    CUDATensorWrapper otw(output);
    uint gridSize = setup_grid_size(input.numel(), BLOCK_SIZE);
    kernel_reduce_func<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(itw, dim, otw);
    CHECK(cudaGetLastError());
    return std::move(output);
}


float max(const Tensor & input)
{
    return _compute_op(input, kernel_reduce_max, -1.0f * 1e10);
}


float min(const Tensor & input)
{
    return _compute_op(input, kernel_reduce_min, 1.0f * 1e10);
}


float sum(const Tensor & input)
{
    return _compute_op(input, kernel_reduce_sum);
}


Tensor sum(const Tensor & input, size_t dim)
{
    return _compute_op_on_dim(input, dim, kernel_reduce_sum_on_dim);
}


Tensor max(const Tensor & input, size_t dim)
{
    return _compute_op_on_dim(input, dim, kernel_reduce_max_on_dim, -1.0f * 1e10);
}


Tensor min(const Tensor & input, size_t dim)
{
    return _compute_op_on_dim(input, dim, kernel_reduce_min_on_dim, 1.0f * 1e10);
}


}