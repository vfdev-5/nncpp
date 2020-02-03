#pragma once


// NNCPP
#include "tensor.hpp"


namespace nncpp
{

float max(const Tensor & input);
float min(const Tensor & input);
float sum(const Tensor & input);

Tensor sum(const Tensor & input, size_t dim);
Tensor min(const Tensor & input, size_t dim);
Tensor max(const Tensor & input, size_t dim);

}
