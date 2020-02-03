#pragma once


// NNCPP
#include "module.hpp"


namespace nncpp
{


class MSELoss : public Module
{

public:
    float forward(const Tensor & yPred, const Tensor & yTrue);
    Tensor backward();

    float operator()(const Tensor & yPred, const Tensor & yTrue) 
    { return forward(yPred, yTrue); }

};


}