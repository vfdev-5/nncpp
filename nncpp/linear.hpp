#pragma once

// NNCPP
#include "tensor.hpp"
#include "module.hpp"


namespace nncpp
{

class Linear : public Module
{

public:

    Linear(size_t inChannels, size_t outChannels);

    Tensor forward(const Tensor & t) override;
    Tensor backward(const Tensor & grad) override;
    void zero_grad() override;

    Tensor weight;
    Tensor bias;

    Tensor grad_weight;
    Tensor grad_bias;

};

}