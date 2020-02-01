// STD
#include <iostream>
#include <random>

// NNCpp
#include "tensor.hpp"


namespace nncpp
{


void _unifiedmem_free(float * data)
{
    cudaFree(data);
}


void _unifiedmem_init(float ** data, size_t n, size_t c, size_t h, size_t w, float init_value)
{
    size_t size = n * c * h * w;
    cudaMallocManaged(data, size * sizeof(float));
    for (size_t i=0; i < size; i++)
    {
        (*data)[i] = init_value;
    }
}

void _unifiedmem_zeros(float ** data, size_t n, size_t c, size_t h, size_t w, void * args)
{   
    _unifiedmem_init(data, n, c, h, w, 0.0f);
}


void _unifiedmem_ones(float ** data, size_t n, size_t c, size_t h, size_t w, void * args)
{   
    _unifiedmem_init(data, n, c, h, w, 1.0f);
}


void _unifiedmem_randn(float ** data, size_t n, size_t c, size_t h, size_t w, void * args)
{   
    size_t size = n * c * h * w;
    cudaMallocManaged(data, size * sizeof(float));

    float * ms = (float *) args;

    std::normal_distribution<float> d{ms[0], ms[1]};
    
    for (size_t i=0; i < size; i++)
    {
        (*data)[i] = d(random_generator);
    }
}


void _cuda_sync()
{
    cudaDeviceSynchronize();
}


}


std::ostream& operator<<(std::ostream& os, nncpp::Tensor & t)
{
    std::string device = "CPU";
    if (t.device == nncpp::Device::CUDA)
    {
        cudaDeviceSynchronize();
        device = "CUDA";
    }    
    uint max_length = 5;
    os << "Tensor: on " << device << ", (" << t.shape[0] << ", " << t.shape[1] << ", " << t.shape[2] << ", " << t.shape[3] << ")"<< std::endl;
    os << "[ " << std::endl;
    for (size_t i = 0; i < std::min(t.shape[0], (size_t) max_length); i++)
    {
        os << "  [ ";
        for (size_t j = 0; j < std::min(t.shape[1], (size_t) max_length); j++)
        {
            os << "[";
            for (size_t k = 0; k < std::min(t.shape[2], (size_t) max_length); k++)
            {
                os << "[";
                for (size_t l = 0; l < std::min(t.shape[3], (size_t) max_length); l++)
                {   
                    os << (float) t.at(i, j, k, l) << " ";
                }
                os << "]";
            }
            os << "] ";
        }
        os << "]" << std::endl;
    }
    os << "]" << std::endl;    
    return os;
}
