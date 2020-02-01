#pragma once

// STD
#include <memory>

// NNCPP
#include "tensor.hpp"


namespace nncpp
{


const size_t BLOCK_SIZE = 512;
const size_t MATMUL_BLOCK_SIZE = 32;


#define CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}       


class CUDATensorWrapper
{

public:
    CUDATensorWrapper(Tensor & t) : 
        _t_data(t.data()),
        _t_numel(t.numel()),
        strides{t.strides[0], t.strides[1], t.strides[2], t.strides[3]},
        shape{t.shape[0], t.shape[1], t.shape[2], t.shape[3]}
    {}

    CUDATensorWrapper(const Tensor & t) :        
        _t_data( (float *) t.const_data() ),
        _t_numel(t.numel()),
        strides{t.strides[0], t.strides[1], t.strides[2], t.strides[3]},
        shape{t.shape[0], t.shape[1], t.shape[2], t.shape[3]}
    {}

    CUDATensorWrapper(float * data, size_t numel, size_t inputStrides[4], size_t inputShape[4]) :
        _t_data( data ),
        _t_numel(numel),
        strides{inputStrides[0], inputStrides[1], inputStrides[2], inputStrides[3]},
        shape{inputShape[0], inputShape[1], inputShape[2], inputShape[3]}
    {}

    const size_t strides[4];
    const size_t shape[4];

    __host__ __device__ float & at(size_t linear)
    { return _t_data[linear]; }

    __host__ __device__ const float & at(size_t linear) const
    { return _t_data[linear]; }

    __host__ __device__ size_t numel() const
    { return _t_numel; }

    __host__ __device__ float * data() 
    { return _t_data; }

    __host__ __device__ const float * const_data() const
    { return _t_data; }
    
    inline __host__ __device__ size_t convert_to_linear(size_t indices[4]);
    inline __host__ __device__ void convert_from_linear(size_t linear, size_t indices[4]);    

protected:
    
    float * _t_data;
    size_t _t_numel;    

};


inline __host__ __device__ size_t CUDATensorWrapper::convert_to_linear(size_t indices[4])
{
    return indices[0] * strides[0] + indices[1] * strides[1] + indices[2] * strides[2] + indices[3] * strides[3];
}


inline __host__ __device__ void CUDATensorWrapper::convert_from_linear(size_t linear, size_t indices[4])
{     
    int t = linear;
    indices[0] = linear / strides[0];
    t %= strides[0];
    indices[1] = t / strides[1];
    t %= strides[1];
    indices[2] = t / strides[2];
    t %= strides[2];
    indices[3] = t / strides[3];
}


}

