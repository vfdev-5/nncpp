// STD
#include <cassert>
#include <iostream>

// NNCpp
#include "tensor.hpp"


namespace nncpp
{

void _unifiedmem_free(float * data);
void _unifiedmem_zeros(float ** data, size_t n, size_t c, size_t h, size_t w);
void _unifiedmem_ones(float ** data, size_t n, size_t c, size_t h, size_t w);
void _cuda_sync();


void check_device(Device device);


Tensor Tensor::zeros(size_t n, size_t c, size_t h, size_t w, Device device) 
{
    check_device(device);
    Tensor output;
    create_tensor(output, n, c, h, w, device, _unifiedmem_zeros);
    return output;
}


Tensor Tensor::ones(size_t n, size_t c, size_t h, size_t w, Device device) 
{
    check_device(device);
    Tensor output;
    create_tensor(output, n, c, h, w, device, _unifiedmem_ones);    
    return output;
}


Tensor Tensor::zeros_like(const Tensor & t) 
{
    check_device(t.device);
    Tensor output;
    create_tensor(output, t.shape[0], t.shape[1], t.shape[2], t.shape[3], t.device, _unifiedmem_zeros);
    return output;
}


Tensor Tensor::ones_like(const Tensor & t)
{
    check_device(t.device);
    Tensor output;
    create_tensor(output, t.shape[0], t.shape[1], t.shape[2], t.shape[3], t.device, _unifiedmem_ones);    
    return output;
}



void assert_indices(const Tensor & t, size_t i, size_t j, size_t k, size_t l);
size_t linear_index(const Tensor & t, size_t i, size_t j, size_t k, size_t l);


float & Tensor::at(size_t i, size_t j, size_t k, size_t l)
{
    if (device == Device::CUDA) 
        _cuda_sync();

    assert_indices(*this, i, j, k, l);
    auto index = linear_index(*this, i, j, k, l);
    return this->_data.get()[index];
}


float & Tensor::at(size_t linear)
{
    if (device == Device::CUDA) 
        _cuda_sync();

    assert(linear >= 0 && linear < numel());
    return this->_data.get()[linear];
}


void assert_indices(const Tensor & t, size_t i, size_t j, size_t k, size_t l)
{
    assert(i >= 0 && i < t.shape[0]);
    assert(j >= 0 && j < t.shape[1]);
    assert(k >= 0 && k < t.shape[2]);
    assert(l >= 0 && l < t.shape[3]);
}


size_t linear_index(const Tensor & t, size_t i, size_t j, size_t k, size_t l)
{
    return i * t.shape[1] * t.shape[2] * t.shape[3] + 
           j * t.shape[2] * t.shape[3] + 
           k * t.shape[3] + l;
}


void check_device(Device device)
{
   if (device != Device::CPU && device != Device::CUDA) 
    {
        std::cerr << "Unknown device: " << device << std::endl;
        assert(false);
    }
}


void Tensor::create_tensor(Tensor & t, size_t n, size_t c, size_t h, size_t w, Device device, void func(float ** data, size_t n, size_t c, size_t h, size_t w))
{
    float * data;
    func(&data, n, c, h, w);
    t._data = std::shared_ptr<float>(data, [&](float *p){
        _unifiedmem_free(p);
    });

    t.device = device;
    t.shape[0] = n;
    t.shape[1] = c;
    t.shape[2] = h;
    t.shape[3] = w;    
}


}
