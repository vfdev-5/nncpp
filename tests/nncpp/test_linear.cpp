// STD
#include <iostream>
#include <cmath>

// GTest
#include <gtest/gtest.h>

// Project
#include "tensor.hpp"
#include "linear.hpp"

using namespace nncpp;


Tensor setup_test_tensor(bool small=false)
{
    Tensor t = (small) ? Tensor::ones(4, 10, 1, 1, Device::CUDA) : Tensor::randn(32, 128, 1, 1, Device::CUDA);
    return t;
}


TEST(TestLinear, testLinearForward)
{

    for (bool small: {true, false})
    {
        Tensor t = setup_test_tensor(small);
        
        random_generator.seed(12);
        auto linear = (small) ? Linear(10, 2) : Linear(128, 10);
        if (small)
        {
            for (size_t i=0; i < linear.weight.numel(); i++)
            {
                linear.weight.at(i) = 1.0f;
            }
            for (size_t i=0; i < linear.bias.numel(); i++)
            {
                linear.bias.at(i) = 1.0f;
            }
        }
        
        Tensor true_output = Tensor::zeros(t.shape[0], linear.weight.shape[1], 1, 1);

        for (size_t i = 0; i < true_output.shape[0]; i++)
        {
            for (size_t j = 0; j < true_output.shape[1]; j++)
            {
                for (size_t k = 0; k < t.shape[1]; k++)
                {
                    true_output.at(i, j, 0, 0) += t.at(i, k, 0, 0) * linear.weight.at(k, j, 0, 0);
                }
                true_output.at(i, j, 0, 0) += linear.bias.at(j, 0, 0, 0);
            }
        }
        
        Tensor output = linear(t);

        ASSERT_EQ(output.numel(), true_output.numel());
        ASSERT_EQ(output.shape[0], true_output.shape[0]);
        ASSERT_EQ(output.shape[1], true_output.shape[1]);
        for (size_t i=0; i < output.numel(); i++)
        {
            ASSERT_NEAR(output.at(i), true_output.at(i), 1e-5);
        }    
    }
}

