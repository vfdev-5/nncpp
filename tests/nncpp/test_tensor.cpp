// STD
#include <iostream>

// GTest
#include <gtest/gtest.h>

// Project
#include "tensor.hpp"

using namespace nncpp;

TEST(TestTensor, testCopy)
{  
    {
        Tensor t = Tensor::zeros(1, 2, 3, 4);
        {            
            Tensor t2 = t;
        }
    }
}


TEST(TestTensor, testShape)
{    
    int n(4), c(3), h(2), w(2);

    Tensor t = Tensor::zeros(n, c, h, w);

    ASSERT_EQ(t.shape[0], n);
    ASSERT_EQ(t.shape[1], c);
    ASSERT_EQ(t.shape[2], h);
    ASSERT_EQ(t.shape[3], w);
    
}

TEST(TestTensor, testNumel)
{    
    int n(4), c(3), h(2), w(2);

    Tensor t = Tensor::zeros(n, c, h, w);

    ASSERT_EQ(t.numel(), n * c * h * w);
}


TEST(TestTensor, testDeviceCPU)
{    
    int n(4), c(3), h(2), w(2);
    Tensor t = Tensor::zeros(n, c, h, w, Device::CPU);
    ASSERT_EQ(t.device, Device::CPU);
}


TEST(TestTensor, testDeviceCUDA)
{    
    int n(4), c(3), h(2), w(2);
    Tensor t = Tensor::zeros(n, c, h, w, Device::CUDA);
    ASSERT_EQ(t.device, Device::CUDA);
}


TEST(TestTensor, testValues)
{    
    int n(4), c(3), h(2), w(2);
    {
        Tensor t = Tensor::zeros(n, c, h, w);

        ASSERT_FLOAT_EQ(t.at(0, 0, 0, 0), 0.0);

        t.at(0, 0, 0, 1) = 1.2;
        ASSERT_FLOAT_EQ(t.at(0, 0, 0, 1), 1.2);

        t.at(3, 2, 1, 0) = 2.2;
        ASSERT_FLOAT_EQ(t.at(3, 2, 1, 0), 2.2);
    }
    {
        Tensor t = Tensor::ones(n, c, h, w);
        for (size_t i=0; i < n; i++)
        {
            for (size_t j=0; j < c; j++)
            {
                for (size_t k=0; k < h; k++)
                {
                    for (size_t l=0; l < w; l++)
                    {
                        ASSERT_FLOAT_EQ(t.at(i, j, k, l), 1.0);
                    }
                }
            }
        }
    }
}


TEST(TestTensor, testValuesCUDA)
{    
    int n(4), c(3), h(2), w(2);
    {
        Tensor t = Tensor::zeros(n, c, h, w, Device::CUDA);

        ASSERT_FLOAT_EQ(t.at(0, 0, 0, 0), 0.0);

        t.at(0, 0, 0, 1) = 1.2;
        ASSERT_FLOAT_EQ(t.at(0, 0, 0, 1), 1.2);

        t.at(3, 2, 1, 0) = 2.2;
        ASSERT_FLOAT_EQ(t.at(3, 2, 1, 0), 2.2);
    }
    {
        Tensor t = Tensor::ones(n, c, h, w, Device::CUDA);
        for (size_t i=0; i < n; i++)
        {
            for (size_t j=0; j < c; j++)
            {
                for (size_t k=0; k < h; k++)
                {
                    for (size_t l=0; l < w; l++)
                    {
                        ASSERT_FLOAT_EQ(t.at(i, j, k, l), 1.0);
                    }
                }
            }
        }
    }
}
