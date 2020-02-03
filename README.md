# [WIP] NNCpp - Neural Networks in C++ with CUDA ops

**Warning:** _This is my playground project to implement tensor ops, neural networks, gradient computations etc._
_Code structure and assumptions made may be restrictive. CUDA Kernel implementations are not optimized for performances_
_and may not work in all cases._


- NNCpp library
  - Tensor (CPU/CUDA) on [Unified Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd)
  - Layers: activations, convolution on CUDA only
  - Model: list of layers
  - Criterion: CrossEntropy, MSE
- Example applications
- Tests

## Requirements

- g++
- cuda 10.1, nvcc
- gtests:
```
apt-get install -y libgtest-dev
cd /usr/src/gtest && cmake . && make && mv libg* /usr/lib/
```

## Installation

```
mkdir build && cd $_
cmake .. && make
```

## NNCpp API

- Tensor : "tensor.hpp"

  - All tensors are 4D. 
  - Data allocated in ["Unified Memory"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd) for all devices: CPU/CUDA.


```c++
#include <iostream>
#include "tensor.hpp"

auto t_cpu = nncpp::Tensor::zeros(n, c, h, w, Device::CPU);
auto t_cuda = nncpp::Tensor::ones(n, c, h, w, Device::CUDA);

std::cout << t_cpu << std::endl;
std::cout << t_cuda << std::endl;

t_cpu.at(0, 0, 0, 0) = 1.0f;
t_cuda.at(0, 0, 0, 0) = 2.0f;
```

- NN operations: "activations.hpp", 

## Run applications



### Extend applications 


## Tests
```
cd build
cmake .. & make && ./test-nncpp
```

### Profiling 

```
cd build
nvprof ./test-nncpp
```