// STD
#include <cassert>
#include <iostream>

// NNCpp
#include "tensor.hpp"
#include "activations.hpp"
#include "cuda_tensor_wrapper.cuh"


namespace nncpp
{

const int BLOCK_SIZE = 512;


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


int setup_grid_size(size_t numel)
{
    int grid_size = (numel % BLOCK_SIZE == 0) ? numel / BLOCK_SIZE + 1 : numel / BLOCK_SIZE;
    grid_size = (grid_size == 0) ? 1 : grid_size;
    return grid_size;
}


void _elementwise_activation_inplace(Tensor & input, void kernel_func(CUDATensorWrapper, CUDATensorWrapper))
{
    assert(input.device == Device::CUDA);    
    int grid_size = setup_grid_size(input.numel());
    auto tw = CUDATensorWrapper(input);
    kernel_func<<<grid_size, BLOCK_SIZE>>>(tw, tw);
    CHECK(cudaGetLastError());
}


Tensor _elementwise_activation(const Tensor & input, void kernel_func(CUDATensorWrapper, CUDATensorWrapper))
{
    assert(input.device == Device::CUDA);
    Tensor output = Tensor::zeros_like(input);
    int grid_size = setup_grid_size(input.numel());
    auto itw = CUDATensorWrapper(input);
    auto otw = CUDATensorWrapper(output);
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
    int grid_size = setup_grid_size(input.numel());
    auto itw = CUDATensorWrapper(input);
    auto otw = CUDATensorWrapper(output);
    auto gtw = CUDATensorWrapper(grad);
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