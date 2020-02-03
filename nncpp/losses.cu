
// STD
#include <cassert>

// NNCPP
#include "losses.hpp"
#include "cuda_utils.cuh"


namespace nncpp
{


__global__ void kernel_mse_loss(CUDATensorWrapper yPred, CUDATensorWrapper yTrue, CUDATensorWrapper loss);
 
  
float MSELoss::forward(const Tensor & yPred, const Tensor & yTrue)
{
    assert(yPred.device == Device::CUDA);
    assert(yTrue.device == Device::CUDA);
    assert(yPred.shape[0] == yTrue.shape[0]);
    assert(yPred.shape[1] == yTrue.shape[1]);
    assert(yPred.shape[2] == yTrue.shape[2]);
    assert(yPred.shape[3] == yTrue.shape[3]);
    
    _context.clear();
    _context.push_back(yPred);
    _context.push_back(yTrue);

    Tensor loss = Tensor::zeros_like(yPred);
  
    CUDATensorWrapper yptw(yPred);
    CUDATensorWrapper yttw(yTrue);
    CUDATensorWrapper ltw(loss);

    uint gridSize = setup_grid_size(yptw.numel(), BLOCK_SIZE);

    kernel_mse_loss<<< gridSize, BLOCK_SIZE >>>(yptw, yttw, ltw);
    CHECK(cudaGetLastError());

    return loss.at(0, 0, 0, 0);
}


Tensor MSELoss::backward()
{
    assert(_context.size() == 2);
    auto yPred = _context[0];
    auto yTrue = _context[1];

    size_t b = yPred.shape[0];
    size_t c = yPred.shape[1];

    Tensor grad = Tensor::zeros_like(yPred);

    CUDATensorWrapper yptw(yPred);
    CUDATensorWrapper yttw(yTrue);
    CUDATensorWrapper gtw(grad);

    int grid_size = setup_grid_size(yPred.numel(), BLOCK_SIZE);
        
    kernel_gen_binary_op<<<grid_size, BLOCK_SIZE>>>((-1.0f / b / c), yttw, (1.0f / b / c), yptw, grad, op_sum);
    CHECK(cudaGetLastError());

    return std::move(grad);
}


__global__ void kernel_mse_loss(CUDATensorWrapper yPred, CUDATensorWrapper yTrue, CUDATensorWrapper loss)
{

    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint gridSize = gridDim.x * blockDim.x;

    auto lossNumel = loss.numel();
    while (i < lossNumel)
    {
        loss.at(i) = 0.5f * powf(yPred.at(i) - yTrue.at(i), 2.0);
        i += gridSize;
    }

    _kernel_reduce_op(loss, loss, op_sum, nullptr, atomicAdd);
}


}

