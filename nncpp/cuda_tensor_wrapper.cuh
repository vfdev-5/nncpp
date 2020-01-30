#pragma once

// STD
#include <memory>

// NNCPP
#include "tensor.hpp"


namespace nncpp
{


const int BLOCK_SIZE = 512;


#define CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}       


class CUDATensorWrapper
{

public:
    CUDATensorWrapper(Tensor & t) : 
        _t_data(t.data()),
        _t_numel(t.numel())
    {}

    CUDATensorWrapper(const Tensor & t) :        
        _t_data( (float *) t.const_data() ),
        _t_numel(t.numel())
    {}

    // inline static CUDATensorWrapper zeros(size_t numel)
    // { return CUDATensorWrapper}

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

protected:
    float * _t_data;
    size_t _t_numel;

};

}

