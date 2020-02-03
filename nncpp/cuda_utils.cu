// CUDA
#include <cooperative_groups.h>


// NNCPP
#include "cuda_utils.cuh"

namespace cg = cooperative_groups;


namespace nncpp
{


uint setup_grid_size(size_t numel, size_t blockSize)
{        
    uint grid_size = (numel + blockSize - 1) / blockSize;
    return grid_size;
}


__global__ void kernel_reduce_max(CUDATensorWrapper input, CUDATensorWrapper output)
{
    _kernel_reduce_op(input, output, op_max, nullptr, _atomicMax);
}


__global__ void kernel_reduce_max_on_dim(CUDATensorWrapper input, size_t dim, CUDATensorWrapper output)
{  
    _kernel_reduce_op_on_dim(input, dim, output, op_max);
}


__global__ void kernel_reduce_min(CUDATensorWrapper input, CUDATensorWrapper output)
{
    _kernel_reduce_op(input, output, op_min, nullptr, _atomicMin);
}


__global__ void kernel_reduce_min_on_dim(CUDATensorWrapper input, size_t dim, CUDATensorWrapper output)
{  
    _kernel_reduce_op_on_dim(input, dim, output, op_min);
}


__global__ void kernel_reduce_sum(CUDATensorWrapper input, CUDATensorWrapper output)
{
    _kernel_reduce_op(input, output, op_sum, nullptr, atomicAdd);
}


__global__ void kernel_reduce_sum_on_dim(CUDATensorWrapper input, size_t dim, CUDATensorWrapper output)
{  
    _kernel_reduce_op_on_dim(input, dim, output, op_sum);
}


__global__ void kernel_gen_binary_op(float a, CUDATensorWrapper t1, float b, CUDATensorWrapper t2, CUDATensorWrapper output, reduction_op op, void * opArgs)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint gridSize = gridDim.x * blockDim.x;

    auto outNumel = output.numel();
    while (i < outNumel)
    {
        output.at(i) = op(a * t1.at(i), b * t2.at(i), opArgs);
        i += gridSize;
    }

}


__device__ void _block_op_reduce(float *smem, const cg::thread_block & cta, reduction_op op, void * opArgs)
{ // Internal method to compute max reduction. Inspired by "reduce6" from "reduction_kernel.cu" from 6_Advanced official NVIDIA samples

    uint blockSize = blockDim.x;
    uint tid = threadIdx.x;
    // Op Reduce:
    for (uint s = blockSize / 2; s > 32; s /= 2)
    {
        if (tid < s)
        {
          smem[tid] = op(smem[tid], smem[tid + s], opArgs);
        }
        cg::sync(cta);
    }
    // Warp Op Reduce:
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    if (tid < 32)
    {   
        if (blockSize >= 64)
        {
            smem[tid] = op(smem[tid], smem[tid + 32], opArgs);
        }
        for (uint offset = tile32.size() / 2; offset > 0; offset /= 2)
        {
            smem[tid] = op(smem[tid], tile32.shfl_down(smem[tid], offset), opArgs);
        }    
    }
}


__device__ void _kernel_reduce_op(CUDATensorWrapper input, CUDATensorWrapper output, reduction_op op, void * opArgs, atomic_reduction_op blocks_reduce_op)
{    
    extern __shared__ float smem[];
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    uint blockSize = blockDim.x;
    uint tid = threadIdx.x;
    uint i = blockIdx.x * (blockSize * 2) + threadIdx.x;
    uint gridSize = (blockSize * 2) * gridDim.x;
      
    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    size_t numel = input.numel();
    float reduction = output.at(0);

    while (i < numel)
    {
        reduction = op(reduction, input.at(i), opArgs);

        // ensure we don't read out of bounds
        if (i + blockSize < numel)
        {
            reduction = op(reduction, input.at(i + blockSize), opArgs);
        }
        i += gridSize;        
    }

    // each thread puts its local result into shared memory
    smem[tid] = reduction;
    cg::sync(cta);

    _block_op_reduce(smem, cta, op, opArgs);

    // reduce the results per block to the output using atomic operation
    if (tid == 0)
    { 
        blocks_reduce_op(&output.at(0), smem[tid]);
    } 
}


__device__ void _kernel_reduce_op_on_dim(CUDATensorWrapper input, size_t dim, CUDATensorWrapper output, reduction_op op, void * opArgs)
{    
    extern __shared__ float smem[];
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    uint blockSize = blockDim.x;
    uint i = blockIdx.x * blockSize + threadIdx.x;
    uint gridSize = blockSize * gridDim.x;
      
    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    size_t outputNumel = output.numel();
    size_t size = input.shape[dim];
    float reduction;
    size_t outIndices[4], inIndex;

    while (i < outputNumel)
    {
        reduction = output.at(i);
        output.convert_from_linear(i, outIndices);
        for (size_t j = 0; j < size; j++)
        {   
            outIndices[dim] = j;
            inIndex = input.convert_to_linear(outIndices);            
            reduction = op(reduction, input.at(inIndex), opArgs);
        }
        output.at(i) = reduction;
        i += gridSize;
    }
}


__device__ float _atomicMax(float* address, float val)
{ // Implementation adapted from https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions

    unsigned int* address_as_int = (unsigned int*) address;
    unsigned int old = *address_as_int, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int( fmaxf( val, __int_as_float(assumed) ) ) );
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __int_as_float(old);
}


__device__ float _atomicMin(float* address, float val)
{ // Implementation adapted from https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions

    unsigned int* address_as_int = (unsigned int*) address;
    unsigned int old = *address_as_int, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int( fminf( val, __int_as_float(assumed) ) ) );
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __int_as_float(old);
}


__global__ void kernel_mm(CUDATensorWrapper input, CUDATensorWrapper weight, CUDATensorWrapper bias, CUDATensorWrapper output)
{ // Inspired from matrixMul.cu file from CUDA samples 0_Simple/matrixMul

    uint bx = blockIdx.x * MATMUL_BLOCK_SIZE;
    uint by = blockIdx.y * MATMUL_BLOCK_SIZE;
    uint tx = threadIdx.x;
    uint ty = threadIdx.y;

    float result = 0.0f;
    size_t inStride = input.strides[0];
    size_t wStride = weight.strides[0];

    size_t inBegin = by * inStride;
    size_t inEnd = inBegin + inStride;
    size_t inStep = MATMUL_BLOCK_SIZE;
    size_t inNumel = input.numel();    

    size_t wBegin = bx;
    size_t wStep = MATMUL_BLOCK_SIZE * wStride;
    size_t wNumel = weight.numel();

    for (size_t inIndex = inBegin, wIndex = wBegin; 
         inIndex < inEnd; 
         inIndex += inStep, wIndex += wStep)
    {   
        __shared__ float smemInput[MATMUL_BLOCK_SIZE][MATMUL_BLOCK_SIZE];
        __shared__ float smemWeight[MATMUL_BLOCK_SIZE][MATMUL_BLOCK_SIZE];
    
        bool c1 = (inIndex + tx + inStride * ty) < inNumel;
        bool c2 = (wIndex + tx + wStride * ty) < wNumel;

        // special care for matrices smaller than MATMUL_BLOCK_SIZE * MATMUL_BLOCK_SIZE
        if ((inNumel - inIndex < MATMUL_BLOCK_SIZE * MATMUL_BLOCK_SIZE) && (tx >= inStride))
            c1 = false;
        if ((wNumel - wIndex < MATMUL_BLOCK_SIZE * MATMUL_BLOCK_SIZE) && (tx >= wStride))
            c2 = false;
    
        if (c1)
            smemInput[ty][tx] = input.at(inIndex + tx + inStride * ty);
        if (c2)
            smemWeight[ty][tx] = weight.at(wIndex + tx + wStride * ty);

        __syncthreads();

#if 0
        if (tx == 0 && ty == 0 && true)
        {
            printf("smemInput:\n");
            for (int i=0; i < MATMUL_BLOCK_SIZE; i++)
            {
                for (int j=0; j < MATMUL_BLOCK_SIZE; j++)
                {
                    printf("%.1f ", smemInput[i][j]);
                }
                printf("\n");
            }
            printf("\n");

            printf("smemWeight:\n");
            for (int i=0; i < MATMUL_BLOCK_SIZE; i++)
            {
                for (int j=0; j < MATMUL_BLOCK_SIZE; j++)
                {
                    printf("%.1f ", smemWeight[i][j]);
                }
                printf("\n");
            }
            printf("\n");
        }
#endif

#pragma unroll        
        for (uint i = 0; i < MATMUL_BLOCK_SIZE; i++)
        {
            result += smemInput[ty][i] * smemWeight[i][tx];
        }    

        __syncthreads();
    }

    // add bias
    size_t i = bx + tx;
    if (i < bias.numel())
        result += bias.at(i);
    
    size_t outStride = output.strides[0];
    size_t outIndex = (by + ty) * outStride + (bx + tx);
    size_t outNumel = output.numel();
    if (outIndex < outNumel)
    {
        // special care for matrices smaller than MATMUL_BLOCK_SIZE * MATMUL_BLOCK_SIZE
        if ((outNumel < MATMUL_BLOCK_SIZE * MATMUL_BLOCK_SIZE) && (tx >= outStride))
            return;

        // printf("%i | %i | %i | %i : outIndex=%llu, result=%f\n", bx, by, tx, ty, outIndex, result);
        output.at(outIndex) = result;    
    }
}


__global__ void kernel_set_to_zero(CUDATensorWrapper weight, CUDATensorWrapper bias)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint gridSize = gridDim.x * blockDim.x;

    auto wNumel = weight.numel();
    while (i < wNumel)
    {
        weight.at(i) = 0.0f;
        i += gridSize;
    }

    auto bNumel = bias.numel();
    while (i < bNumel)
    {
        bias.at(i) = 0.0f;
        i += gridSize;
    }
}


}