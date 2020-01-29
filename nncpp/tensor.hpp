#pragma once

// STD
#include <tuple>
#include <memory>


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

    static Tensor zeros_like(const Tensor & t);
    static Tensor ones_like(const Tensor & t);

    size_t shape[4];
    Device device;

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

protected:
    Tensor() :
        shape{0, 0, 0, 0}        
    {}

    std::shared_ptr<float> _data;

    static void create_tensor(Tensor & t, size_t n, size_t c, size_t h, size_t w, Device d, 
                              void func(float ** data, size_t n, size_t c, size_t h, size_t w));
};

}

std::ostream& operator<<(std::ostream& os, nncpp::Tensor & t);
