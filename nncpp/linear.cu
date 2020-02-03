// STD
#include <math.h>
#include <cassert>

// NNCPP
#include "linear.hpp"
#include "cuda_tensor_wrapper.cuh"
#include "cuda_utils.cuh"


namespace nncpp
{


__global__ void kernel_linear_backward(CUDATensorWrapper grad, CUDATensorWrapper input, CUDATensorWrapper output, 
                                       CUDATensorWrapper grad_weight, CUDATensorWrapper grad_bias);


Linear::Linear(size_t inChannels, size_t outChannels) :
    weight(Tensor::randn(inChannels, outChannels, 1, 1, Device::CUDA, 0.0, sqrt(2.0 / inChannels))),
    bias(Tensor::zeros(outChannels, 1, 1, 1, Device::CUDA)),
    grad_weight(Tensor::zeros(inChannels, outChannels, 1, 1, Device::CUDA)),
    grad_bias(Tensor::zeros(outChannels, 1, 1, 1, Device::CUDA))
{
}


Tensor Linear::forward(const Tensor & input)
{   
    assert(input.device == Device::CUDA);
    assert(input.shape[1] == weight.shape[0]);
    // specifics of NNCPP tensor implementation: shape always NCHW
    assert(input.shape[2] == 1);
    assert(input.shape[3] == 1);

    _context.push_back(input);
    Tensor output = Tensor::zeros(input.shape[0], weight.shape[1], input.shape[2], input.shape[3], input.device);
    
    CUDATensorWrapper itw(input);
    CUDATensorWrapper wtw(weight);
    CUDATensorWrapper btw(bias);
    CUDATensorWrapper otw(output);

    dim3 bs(MATMUL_BLOCK_SIZE, MATMUL_BLOCK_SIZE);
    dim3 gs(setup_grid_size(output.shape[0], MATMUL_BLOCK_SIZE), setup_grid_size(output.shape[1], MATMUL_BLOCK_SIZE));

    kernel_mm<<< gs, bs >>>(itw, wtw, btw, otw);
    CHECK(cudaGetLastError());

    return std::move(output);
}


Tensor Linear::backward(const Tensor & grad)
{ // Gradients explained here: https://sgugger.github.io/a-simple-neural-net-in-numpy.html#a-simple-neural-net-in-numpy

    assert(!_context.empty());
    auto input = _context[0];

    assert(grad.device == Device::CUDA);    
    assert(input.device == Device::CUDA);

    CUDATensorWrapper itw(input);
    CUDATensorWrapper gtw(grad);
    CUDATensorWrapper gwtw(grad_weight);
    CUDATensorWrapper gbtw(grad_bias);    

    Tensor output = Tensor::zeros_like(grad);
    CUDATensorWrapper otw(output);

    int grid_size = setup_grid_size(input.numel(), BLOCK_SIZE);
        
    kernel_linear_backward<<<grid_size, BLOCK_SIZE>>>(gtw, itw, otw, gwtw, gbtw);
    CHECK(cudaGetLastError());

    return std::move(output);
}


void Linear::zero_grad()
{
    auto gs = setup_grid_size(grad_weight.numel(), BLOCK_SIZE);
    CUDATensorWrapper gwtw(grad_weight);
    CUDATensorWrapper gbtw(grad_bias);   
    kernel_set_to_zero<<< gs, BLOCK_SIZE >>>(gwtw, gbtw);
    CHECK(cudaGetLastError());
}


__global__ void kernel_linear_backward(CUDATensorWrapper grad, CUDATensorWrapper input, CUDATensorWrapper output, 
                                       CUDATensorWrapper grad_weight, CUDATensorWrapper grad_bias)
{

    // grad_weight = sum(input * grad, dim=0)
    // grad_bias = sum(grad, dim=0)
    // output = mm(grad, weight.t)

}


}