# NNCpp - Neural Networks in C++ with CUDA ops

- NNCpp library
  - Tensor (CPU/CUDA)
  - Layers: activations, convolution
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

```c++
#include "tensor.hpp"

auto t_cpu = nncpp::Tensor::zeros(n, c, h, w, Device::CPU);
auto t_cuda = nncpp::Tensor::ones(n, c, h, w, Device::CUDA);

t_cpu.get(0, 0, 0, 0);
t_cuda.set(1, 1, 1, 1, 2.3);
```

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