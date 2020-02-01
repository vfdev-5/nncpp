// STD
#include <iostream>
#include <cmath>

// GTest
#include <gtest/gtest.h>

// Project
#include "tensor.hpp"
#include "activations.hpp"

using namespace nncpp;


Tensor setup_test_tensor1()
{
    int n(4), c(3), h(1), w(1);
    Tensor t = Tensor::ones(n, c, h, w, Device::CUDA);

    for (size_t i=0; i<t.shape[0] / 2; i++)
    {
        for (size_t j=0; j<t.shape[1]; j++)
        {
            t.at(i, j, 0, 0) = -t.at(i, j, 0, 0);
        }            
    }
    return t;
}


TEST(TestActivations, testReluInplace)
{        
    Tensor t = setup_test_tensor1();

    relu_(t);

    for (size_t i=0; i < t.shape[0] / 2; i++)
    {
        for (size_t j=0; j < t.shape[1]; j++)
        {
            ASSERT_FLOAT_EQ(t.at(i, j, 0, 0), 0.0);
        }
    }    
    for (size_t i=t.shape[0] / 2; i < t.shape[0]; i++)
    {
        for (size_t j=0; j < t.shape[1]; j++)
        {
            ASSERT_FLOAT_EQ(t.at(i, j, 0, 0), 1.0);
        }
    }
}


TEST(TestActivations, testRelu)
{    
    Tensor t = setup_test_tensor1();

    auto out = relu(t);

    for (size_t i=0; i < t.shape[0] / 2; i++)
    {
        for (size_t j=0; j < t.shape[1]; j++)
        {
            ASSERT_FLOAT_EQ(out.at(i, j, 0, 0), 0.0);
            ASSERT_FLOAT_EQ(t.at(i, j, 0, 0), -1.0);
        }
    }    
    for (size_t i=t.shape[0] / 2; i < t.shape[0]; i++)
    {
        for (size_t j=0; j < t.shape[1]; j++)
        {
            ASSERT_FLOAT_EQ(out.at(i, j, 0, 0), 1.0);
            ASSERT_FLOAT_EQ(t.at(i, j, 0, 0), 1.0);
        }
    }
}


TEST(TestActivations, testReLUForward)
{    
    auto layer = ReLU(); 
    Tensor t = setup_test_tensor1();

    auto out = layer(t);

    for (size_t i=0; i < t.shape[0] / 2; i++)
    {
        for (size_t j=0; j < t.shape[1]; j++)
        {
            ASSERT_FLOAT_EQ(out.at(i, j, 0, 0), 0.0);
            ASSERT_FLOAT_EQ(t.at(i, j, 0, 0), -1.0);
        }
    }    
    for (size_t i=t.shape[0] / 2; i < t.shape[0]; i++)
    {
        for (size_t j=0; j < t.shape[1]; j++)
        {
            ASSERT_FLOAT_EQ(out.at(i, j, 0, 0), 1.0);
            ASSERT_FLOAT_EQ(t.at(i, j, 0, 0), 1.0);
        }
    }
}


TEST(TestActivations, testReLUBackward)
{    
    Tensor input = setup_test_tensor1();
    Tensor grads = setup_test_tensor1();
    auto layer = ReLU();

    layer(input);
    auto output = layer.backward(grads);

    for (size_t i=0; i < output.shape[0] / 2; i++)
    {
        for (size_t j=0; j < output.shape[1]; j++)
        {
            ASSERT_FLOAT_EQ(output.at(i, j, 0, 0), 0.0);
            ASSERT_FLOAT_EQ(grads.at(i, j, 0, 0), -1.0);
        }
    }    
    for (size_t i=output.shape[0] / 2; i < output.shape[0]; i++)
    {
        for (size_t j=0; j < output.shape[1]; j++)
        {
            ASSERT_FLOAT_EQ(output.at(i, j, 0, 0), 1.0);
            ASSERT_FLOAT_EQ(grads.at(i, j, 0, 0), 1.0);
        }
    }
}


Tensor setup_test_tensor2()
{
    int n(4), c(3), h(10), w(10);
    Tensor t = Tensor::zeros(n, c, h, w, Device::CUDA);

    float v = - n * c * h * w * 0.5;
    for (size_t i=0; i < t.shape[0]; i++)
    {
        for (size_t j=0; j < t.shape[1]; j++)
        {
            for (size_t k=0; k < t.shape[2]; k++)
            {
                for (size_t l=0; l < t.shape[3]; l++)
                {            
                    t.at(i, j, k, l) = v * 0.1;
                    v += 1.0f;
                }
            }
        }
    }
    return t;
}


TEST(TestActivations, testSigmoidInplace)
{
    Tensor input = setup_test_tensor2();
    Tensor output = setup_test_tensor2();
    sigmoid_(output);

    for (size_t i=0; i < output.shape[0]; i++)
    {
        for (size_t j=0; j < output.shape[1]; j++)
        {
            for (size_t k=0; k < output.shape[2]; k++)
            {
                for (size_t l=0; l < output.shape[3]; l++)
                {            
                    float v = input.at(i, j, k, l);
                    v = expf(v) / (1.0f + expf(v));
                    ASSERT_FLOAT_EQ(output.at(i, j, k, l), v);
                }
            }
        }
    }            
}


TEST(TestActivations, testSigmoid)
{    
    Tensor input = setup_test_tensor2();    
    Tensor output = sigmoid(input);

    for (size_t i=0; i < output.shape[0]; i++)
    {
        for (size_t j=0; j < output.shape[1]; j++)
        {
            for (size_t k=0; k < output.shape[2]; k++)
            {
                for (size_t l=0; l < output.shape[3]; l++)
                {            
                    float v = input.at(i, j, k, l);
                    v = expf(v) / (1.0f + expf(v));
                    ASSERT_FLOAT_EQ(output.at(i, j, k, l), v);
                }
            }
        }
    }
}


TEST(TestActivations, testSigmoidForward)
{    
    Tensor input = setup_test_tensor2();
    auto layer = Sigmoid();
    Tensor output = layer(input);

    for (size_t i=0; i < output.shape[0]; i++)
    {
        for (size_t j=0; j < output.shape[1]; j++)
        {
            for (size_t k=0; k < output.shape[2]; k++)
            {
                for (size_t l=0; l < output.shape[3]; l++)
                {            
                    float v = input.at(i, j, k, l);
                    v = expf(v) / (1.0f + expf(v));
                    ASSERT_FLOAT_EQ(output.at(i, j, k, l), v);
                }
            }
        }
    }
}


TEST(TestActivations, testSigmoidBackward)
{    
    Tensor input = setup_test_tensor2();
    auto layer = Sigmoid();
    layer(input);
    Tensor grads = Tensor::ones_like(input);
    auto output = layer.backward(grads);

    for (size_t i=0; i < output.shape[0]; i++)
    {
        for (size_t j=0; j < output.shape[1]; j++)
        {
            for (size_t k=0; k < output.shape[2]; k++)
            {
                for (size_t l=0; l < output.shape[3]; l++)
                {            
                    double v = input.at(i, j, k, l);
                    v = -1.0 * exp(v) * expm1(v) * grads.at(i, j, k, l);
                    ASSERT_NEAR(output.at(i, j, k, l), v, 1e-4);
                }
            }
        }
    }
}


TEST(TestActivations, testSoftmaxInplace)
{    
    Tensor input = setup_test_tensor2();    
    Tensor output = setup_test_tensor2();    
    softmax_(output, 1);
    
    float denom;
    for (size_t i=0; i < output.shape[0]; i++)
    {
        for (size_t k=0; k < output.shape[2]; k++)
        {
            for (size_t l=0; l < output.shape[3]; l++)
            {    
                denom = 0.0f;        
                for (size_t j=0; j < output.shape[1]; j++)
                {
                    float v = input.at(i, j, k, l);
                    denom += expf(v);
                }
                for (size_t j=0; j < output.shape[1]; j++)
                {
                    float v = input.at(i, j, k, l);
                    ASSERT_NEAR(output.at(i, j, k, l), expf(v) / denom, 1e-4);
                }
            }
        }
    }
}


TEST(TestActivations, testSoftmax)
{    
    Tensor input = setup_test_tensor2();    
    Tensor output = softmax(input, 1);
    
    float denom;
    for (size_t i=0; i < output.shape[0]; i++)
    {
        for (size_t k=0; k < output.shape[2]; k++)
        {
            for (size_t l=0; l < output.shape[3]; l++)
            {    
                denom = 0.0f;        
                for (size_t j=0; j < output.shape[1]; j++)
                {
                    float v = input.at(i, j, k, l);
                    denom += expf(v);
                }
                for (size_t j=0; j < output.shape[1]; j++)
                {
                    float v = input.at(i, j, k, l);
                    ASSERT_NEAR(output.at(i, j, k, l), expf(v) / denom, 1e-4);
                }
            }
        }
    }
}
