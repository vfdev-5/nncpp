#pragma once

// STD
#include <vector>

// NNCPP
#include "tensor.hpp"


namespace nncpp
{


void relu_(Tensor & t);

Tensor relu(const Tensor & t);

void sigmoid_(Tensor & t);

Tensor sigmoid(const Tensor & t);

void softmax_(Tensor & t, size_t dim);

Tensor softmax(const Tensor & t, size_t dim);


class Identity
{

public:
    virtual Tensor forward(const Tensor & t) 
    { return t; }
    
    virtual Tensor backward(const Tensor & grad) 
    { return grad; }

    virtual Tensor operator()(const Tensor & t) 
    { return forward(t); }

protected:

    std::vector<Tensor> _context;
    

};


class ReLU: public Identity
{

public:
    Tensor forward(const Tensor & t) override;
    Tensor backward(const Tensor & grad) override;

};


class Sigmoid: public Identity
{

public:
    Tensor forward(const Tensor & t) override;
    Tensor backward(const Tensor & grad) override;

};


}

