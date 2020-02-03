#pragma once


// NNCPP
#include "tensor.hpp"
#include "cuda_tensor_wrapper.cuh"


namespace nncpp
{

uint setup_grid_size(size_t numel, size_t blockSize);

typedef float reduction_op(float, float, void * opArgs);
typedef float atomic_reduction_op(float*, float);

__device__ void _kernel_reduce_op(CUDATensorWrapper input, CUDATensorWrapper output, reduction_op op, void * opArgs, atomic_reduction_op blocks_reduce_op);
__device__ void _kernel_reduce_op_on_dim(CUDATensorWrapper input, size_t dim, CUDATensorWrapper output, reduction_op op, void * opArgs=nullptr);
__device__ float _atomicMax(float* address, float val);
__device__ float _atomicMin(float* address, float val);

inline __device__ float op_sum(float x, float y, void * opArgs = nullptr) { return x + y; }
inline __device__ float op_max(float x, float y, void * opArgs = nullptr) { return fmaxf(x, y); }
inline __device__ float op_min(float x, float y, void * opArgs = nullptr) { return fminf(x, y); }

__global__ void kernel_reduce_max(CUDATensorWrapper input, CUDATensorWrapper output);
__global__ void kernel_reduce_min(CUDATensorWrapper input, CUDATensorWrapper output);
__global__ void kernel_reduce_sum(CUDATensorWrapper input, CUDATensorWrapper output);

__global__ void kernel_reduce_max_on_dim(CUDATensorWrapper input, size_t dim, CUDATensorWrapper output);
__global__ void kernel_reduce_min_on_dim(CUDATensorWrapper input, size_t dim, CUDATensorWrapper output);
__global__ void kernel_reduce_sum_on_dim(CUDATensorWrapper input, size_t dim, CUDATensorWrapper output);

__global__ void kernel_mm(CUDATensorWrapper input, CUDATensorWrapper weight, CUDATensorWrapper bias, CUDATensorWrapper output);
__global__ void kernel_set_to_zero(CUDATensorWrapper weight, CUDATensorWrapper bias);

__global__ void kernel_gen_binary_op(float a, CUDATensorWrapper t1, float b, CUDATensorWrapper t2, CUDATensorWrapper output, reduction_op op, void * opArgs=nullptr);

}