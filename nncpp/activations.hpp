#pragma once

// NNCPP
#include "tensor.hpp"
#include "module.hpp"

namespace nncpp
{


void relu_(Tensor & t);

Tensor relu(const Tensor & t);

void sigmoid_(Tensor & t);

Tensor sigmoid(const Tensor & t);

void softmax_(Tensor & t, size_t dim);

Tensor softmax(const Tensor & t, size_t dim);


class ReLU: public Module
{

public:
    Tensor forward(const Tensor & t) override;
    Tensor backward(const Tensor & grad) override;

};


class Sigmoid: public Module
{

public:
    Tensor forward(const Tensor & t) override;
    Tensor backward(const Tensor & grad) override;

};


}

