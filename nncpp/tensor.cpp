// STD
#include <cassert>
#include <iostream>

// NNCpp
#include "tensor.hpp"


namespace nncpp
{

void _unifiedmem_free(float * data);
void _unifiedmem_zeros(float ** data, size_t n, size_t c, size_t h, size_t w, void * args = nullptr);
void _unifiedmem_ones(float ** data, size_t n, size_t c, size_t h, size_t w, void * args = nullptr);
void _unifiedmem_randn(float ** data, size_t n, size_t c, size_t h, size_t w, void * args = nullptr);
void _cuda_sync();


void check_device(Device device);


Tensor Tensor::zeros(size_t n, size_t c, size_t h, size_t w, Device device) 
{
    check_device(device);
    size_t shape[]{n, c, h, w};
    Tensor output(shape, device);
    create_tensor(output, _unifiedmem_zeros);
    return std::move(output);
}


Tensor Tensor::ones(size_t n, size_t c, size_t h, size_t w, Device device) 
{
    check_device(device);
    size_t shape[]{n, c, h, w};
    Tensor output(shape, device);
    create_tensor(output, _unifiedmem_ones);    
    return std::move(output);
}


Tensor Tensor::randn(size_t n, size_t c, size_t h, size_t w, Device device, float mean, float stddev)
{
    check_device(device);
    size_t shape[]{n, c, h, w};
    Tensor output(shape, device);
    float ms[]{mean, stddev};
    create_tensor(output, _unifiedmem_randn, ms);
    return std::move(output);
}


Tensor Tensor::zeros_like(const Tensor & t) 
{
    check_device(t.device);
    Tensor output(t.shape, t.device);
    create_tensor(output, _unifiedmem_zeros);
    return std::move(output);
}


Tensor Tensor::ones_like(const Tensor & t)
{
    check_device(t.device);
    Tensor output(t.shape, t.device);
    create_tensor(output, _unifiedmem_ones);
    return std::move(output);
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
    assert(i < t.shape[0]);
    assert(j < t.shape[1]);
    assert(k < t.shape[2]);
    assert(l < t.shape[3]);
}


size_t linear_index(const Tensor & t, size_t i, size_t j, size_t k, size_t l)
{
    return i * t.strides[0] + j * t.strides[1] + k * t.strides[2] + l * t.strides[3];
}


void check_device(Device device)
{
   if (device != Device::CPU && device != Device::CUDA) 
    {
        std::cerr << "Unknown device: " << device << std::endl;
        assert(false);
    }
}


void Tensor::create_tensor(Tensor & t, void func(float ** data, size_t n, size_t c, size_t h, size_t w, void * args), void * args)
{
    float * data;
    func(&data, t.shape[0], t.shape[1], t.shape[2], t.shape[3], args);
    t._data = std::shared_ptr<float>(data, [&](float *p){
        _unifiedmem_free(p);
    });
}


}
