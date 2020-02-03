#pragma once

// STD
#include <vector>

// NNCPP
#include "tensor.hpp"


namespace nncpp
{


class Module
{

public:

    virtual Tensor forward(const Tensor & t) = 0;
    
    virtual Tensor backward(const Tensor & grad) = 0;

    virtual Tensor operator()(const Tensor & t) 
    { return forward(t); }

    virtual void zero_grad() {}

protected:

    std::vector<Tensor> _context;   
};


class Identity : public Module
{

public:
    virtual Tensor forward(const Tensor & t) 
    { return t; }
    
    virtual Tensor backward(const Tensor & grad) 
    { return grad; }
    
};

}