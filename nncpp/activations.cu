// STD
#include <cassert>
#include <iostream>

// CUDA
#include <cooperative_groups.h>


// NNCpp
#include "tensor.hpp"
#include "activations.hpp"
#include "cuda_tensor_wrapper.cuh"
#include "cuda_utils.cuh"


namespace cg = cooperative_groups;


namespace nncpp
{


__global__ void kerner_relu(CUDATensorWrapper input, CUDATensorWrapper output)
{    
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < input.numel())
    {
        if (input.at(i) < 0.0f)
        {            
            output.at(i) = 0.0f;
        }
        else if (input.const_data() != output.data())
        { // copy data if output is not input            
            output.at(i) = input.at(i);
        }
        
    }
}


__global__ void kerner_relu_backward(CUDATensorWrapper grad, CUDATensorWrapper input, CUDATensorWrapper output)
{    
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < input.numel())
    {
        float value = grad.at(i);
        if (input.at(i) < 0.0f)
        {            
            value = 0.0f;
        }
        output.at(i) = value;
    }
}


__global__ void kerner_sigmoid(CUDATensorWrapper input, CUDATensorWrapper output)
{    
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < input.numel())
    {
        float exp_i = expf(input.at(i));     
        output.at(i) = exp_i / (1.0f + exp_i);        
    }
}


__global__ void kerner_sigmoid_backward(CUDATensorWrapper grad, CUDATensorWrapper input, CUDATensorWrapper output)
{    
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < input.numel())
    {        
        float value = input.at(i);
        output.at(i) = -1.0f * expf(value) * expm1f(value) * grad.at(i);
    }
}


__global__ void kerner_softmax(CUDATensorWrapper input, size_t dim, CUDATensorWrapper output, CUDATensorWrapper buffer)
{   
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    // softmax(input, dim) = exp(input - max(input)) / sum(exp(input - max(input)), dim)
    // a) compute max(input)
    _kernel_reduce_op(input, buffer, op_max, nullptr, _atomicMax);
    cg::sync(cta); 

    // b.1) compute output = exp(input - max(input))    
    float max_input = buffer.at(0);
    uint gridSize = blockDim.x * gridDim.x;
    {
        size_t outputNumel = output.numel();
        uint i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < outputNumel)
        {
            output.at(i) = expf(input.at(i) - max_input);
            i += gridSize;
        }    
    }
    cg::sync(cta); 

    // b.2) compute buffer = sum(output, dim)
    // initialize buffer to zero:
    buffer.at(0) = 0.0f;
    _kernel_reduce_op_on_dim(output, dim, buffer, op_sum);
    cg::sync(cta); 
    
    // c) compute output / buffer
    // output.shape=(N,C,H,W) and buffer.shape=(X,Y,Z)
    {
        size_t bufferNumel = buffer.numel();
        uint i = blockIdx.x * blockDim.x + threadIdx.x;
        size_t size = output.shape[dim];
        float denom;
        size_t bufIndices[4], outIndex;

        while (i < bufferNumel)
        {
            denom = buffer.at(i);
            buffer.convert_from_linear(i, bufIndices);
            for (size_t j = 0; j < size; j++)
            {   
                bufIndices[dim] = j;
                outIndex = output.convert_to_linear(bufIndices);            
                output.at(outIndex) = output.at(outIndex) / denom;
            }
            i += gridSize;
        }    
    }
    cg::sync(cta);
}


void _elementwise_activation_inplace(Tensor & input, void kernel_func(CUDATensorWrapper, CUDATensorWrapper))
{
    assert(input.device == Device::CUDA);    
    int grid_size = setup_grid_size(input.numel(), BLOCK_SIZE);
    CUDATensorWrapper tw(input);
    kernel_func<<<grid_size, BLOCK_SIZE>>>(tw, tw);
    CHECK(cudaGetLastError());
}


Tensor _elementwise_activation(const Tensor & input, void kernel_func(CUDATensorWrapper, CUDATensorWrapper))
{
    assert(input.device == Device::CUDA);
    Tensor output = Tensor::zeros_like(input);
    int grid_size = setup_grid_size(input.numel(), BLOCK_SIZE);
    CUDATensorWrapper itw(input);
    CUDATensorWrapper otw(output);
    kernel_func<<<grid_size, BLOCK_SIZE>>>(itw, otw);
    CHECK(cudaGetLastError());
    return std::move(output);
}


Tensor _elementwise_activation_backward(
    const Tensor & grad,
    const Tensor & input,
    void kernel_func(CUDATensorWrapper, CUDATensorWrapper, CUDATensorWrapper))
{
    assert(grad.device == Device::CUDA);    
    assert(input.device == Device::CUDA);

    Tensor output = Tensor::zeros_like(grad);
    int grid_size = setup_grid_size(input.numel(), BLOCK_SIZE);
    CUDATensorWrapper itw(input);
    CUDATensorWrapper otw(output);
    CUDATensorWrapper gtw(grad);
    kernel_func<<<grid_size, BLOCK_SIZE>>>(gtw, itw, otw);
    CHECK(cudaGetLastError());
    return std::move(output);
}


void relu_(Tensor & input)
{
    _elementwise_activation_inplace(input, kerner_relu);
}


Tensor relu(const Tensor & input)
{
    return _elementwise_activation(input, kerner_relu);
}


void sigmoid_(Tensor & input)
{
    _elementwise_activation_inplace(input, kerner_sigmoid);
}


Tensor sigmoid(const Tensor & input)
{
    return _elementwise_activation(input, kerner_sigmoid);
}


void softmax_(Tensor & input, size_t dim)
{
    assert(dim < 4);
    assert(input.device == Device::CUDA);    
    int grid_size = setup_grid_size(input.numel(), BLOCK_SIZE);
    CUDATensorWrapper tw(input);

    size_t shape[]{input.shape[0], input.shape[1], input.shape[2], input.shape[3]};
    shape[dim] = 1;
    auto buffer = Tensor::zeros(shape[0], shape[1], shape[2], shape[3], Device::CUDA);
    CUDATensorWrapper buffertw(buffer);
    kerner_softmax<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(tw, dim, tw, buffertw);
    CHECK(cudaGetLastError());
}


Tensor softmax(const Tensor & input, size_t dim)
{
    assert(dim < 4);
    assert(input.device == Device::CUDA);
    Tensor output = Tensor::zeros_like(input);
    int grid_size = setup_grid_size(input.numel(), BLOCK_SIZE);
    CUDATensorWrapper itw(input);
    CUDATensorWrapper otw(output);

    size_t shape[]{input.shape[0], input.shape[1], input.shape[2], input.shape[3]};
    shape[dim] = 1;
    auto buffer = Tensor::zeros(shape[0], shape[1], shape[2], shape[3], Device::CUDA);
    CUDATensorWrapper buffertw(buffer);
    kerner_softmax<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(itw, dim, otw, buffertw);
    CHECK(cudaGetLastError());

    return std::move(output);
}


Tensor ReLU::forward(const Tensor & t)
{    
    _context.clear();
    _context.push_back(t);
    return relu(t);
}


Tensor ReLU::backward(const Tensor & grad)
{   
    assert(!_context.empty());
    auto input = _context[0];
    return _elementwise_activation_backward(grad, input, kerner_relu_backward);
}


Tensor Sigmoid::forward(const Tensor & t)
{
    _context.clear();
    _context.push_back(t);
    return sigmoid(t);
}


Tensor Sigmoid::backward(const Tensor & grad)
{   
    assert(!_context.empty());
    auto input = _context[0];
    return _elementwise_activation_backward(grad, input, kerner_sigmoid_backward);
}

}