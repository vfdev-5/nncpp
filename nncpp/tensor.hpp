#pragma once

// STD
#include <iostream>
#include <tuple>
#include <memory>
#include <random>


namespace nncpp
{
 

typedef enum {
  CPU = 0,
  CUDA = 1
} Device;


class Tensor
{

public:

    static Tensor zeros(size_t n, size_t c, size_t h, size_t w, Device device=Device::CPU);
    static Tensor ones(size_t n, size_t c, size_t h, size_t w, Device device=Device::CPU);    
    static Tensor randn(size_t n, size_t c, size_t h, size_t w, Device device=Device::CPU, float mean=0.0, float stddev=1.0);

    static Tensor zeros_like(const Tensor & t);
    static Tensor ones_like(const Tensor & t);

    const size_t shape[4];
    const Device device;

    float at(size_t i, size_t j, size_t k, size_t l) const
    { return at(i, j, k, l); }
    
    float & at(size_t i, size_t j, size_t k, size_t l);
    
    float at(size_t linear) const
    { return at(linear); }

    float & at(size_t linear);
    
    size_t numel() const
    { return shape[0] * shape[1] * shape[2] * shape[3]; }

    float * data() { return _data.get(); }
    const float * const_data() const { return _data.get(); }
    const size_t strides[4];

protected:
    Tensor(const size_t inputShape[4], Device inputDevice) :
        shape{inputShape[0], inputShape[1], inputShape[2], inputShape[3]},
        strides{inputShape[1] * inputShape[2] * inputShape[3], 
                inputShape[2] * inputShape[3], 
                inputShape[3], 
                1},
        device(inputDevice)
    {}

    std::shared_ptr<float> _data;

    static void create_tensor(Tensor & t, void func(float ** data, size_t n, size_t c, size_t h, size_t w, void * args), void * args=nullptr);
};


static std::mt19937 random_generator;

}

std::ostream& operator<<(std::ostream& os, nncpp::Tensor & t);
