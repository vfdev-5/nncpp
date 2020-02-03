// STD
#include <iostream>
#include <cmath>

// GTest
#include <gtest/gtest.h>

// Project
#include "tensor.hpp"
#include "utils.hpp"

using namespace nncpp;


Tensor setup_tensor()
{
    Tensor input = Tensor::zeros(4, 3, 2, 2, Device::CUDA);
    int n = input.numel();
    for (size_t i=0; i < input.numel(); i++)
    {
        input.at(i) = i - n * 0.5;
    }
    return input;
}


TEST(TestUtils, testSum)
{
    Tensor input = setup_tensor();
    float true_sum = 0.0f;
    
    for (size_t i=0; i < input.numel(); i++)
    {
        true_sum += input.at(i);
    }

    float output = sum(input);
    ASSERT_FLOAT_EQ(output, true_sum);
}


TEST(TestUtils, testMax)
{
    Tensor input = setup_tensor();
    float true_max = -1.0f * 1e10;
    
    for (size_t i=0; i < input.numel(); i++)
    {
        true_max = fmaxf(true_max, input.at(i));
    }

    float output = max(input);
    ASSERT_FLOAT_EQ(output, true_max);
}


TEST(TestUtils, testMin)
{
    Tensor input = setup_tensor();
    float true_min = 1.0f * 1e10;
    
    for (size_t i=0; i < input.numel(); i++)
    {
        true_min = fminf(true_min, input.at(i));
    }

    float output = min(input);
    ASSERT_FLOAT_EQ(output, true_min);
}


TEST(TestUtils, testSumOnDim)
{
    Tensor input = setup_tensor();

    for (size_t dim=0; dim < 4; dim++)
    {   
        size_t outShape[4]{input.shape[0], input.shape[1], input.shape[2], input.shape[3]};
        outShape[dim] = 1;

        Tensor true_output = Tensor::zeros(outShape[0], outShape[1], outShape[2], outShape[3]);
                
        for (size_t i = 0; i < input.shape[0]; i++)
        {
            for (size_t j = 0; j < input.shape[1]; j++)
            {
                for (size_t k = 0; k < input.shape[2]; k++)                
                {
                    for (size_t l = 0; l < input.shape[3]; l++)
                    {
                        size_t outIndices[]{i, j, k, l};
                        outIndices[dim] = 0;
                        true_output.at(outIndices[0], 
                                       outIndices[1], 
                                       outIndices[2], 
                                       outIndices[3]) += input.at(i, j, k, l);

                    }
                }
            }            
        }

        Tensor output = sum(input, dim);
        
        ASSERT_EQ(output.shape[0], true_output.shape[0]);
        ASSERT_EQ(output.shape[1], true_output.shape[1]);
        ASSERT_EQ(output.shape[2], true_output.shape[2]);
        ASSERT_EQ(output.shape[3], true_output.shape[3]);

        for (size_t i=0; i < true_output.numel(); i++)
        {
            ASSERT_FLOAT_EQ(output.at(i), true_output.at(i));
        }
    } 
}


TEST(TestUtils, testMaxOnDim)
{
    Tensor input = setup_tensor();

    for (size_t dim=0; dim < 4; dim++)
    {   
        size_t outShape[4]{input.shape[0], input.shape[1], input.shape[2], input.shape[3]};
        outShape[dim] = 1;

        Tensor true_output = Tensor::zeros(outShape[0], outShape[1], outShape[2], outShape[3]);

        // for (size_t i = 0; i < input.shape[0]; i++)
        // {
        //     for (size_t j = 0; j < input.shape[1]; j++)
        //     {
        //         for (size_t k = 0; k < input.shape[2]; k++)                
        //         {
        //             for (size_t l = 0; l < input.shape[3]; l++)
        //             {
        //                 size_t outIndices[]{i, j, k, l};
        //                 outIndices[dim] = 0;
        //                 true_output.at(i) = -1.0f * 1e10;
        //                 true_output.at(outIndices[0], 
        //                                outIndices[1], 
        //                                outIndices[2], 
        //                                outIndices[3]) += input.at(i, j, k, l);

        //             }
        //         }
        //     }            
        // }

        // for (size_t i = 0; i < true_output.numel(); i++)
        // {
        //     true_output.at(i) = -1.0f * 1e10;
        //     for (size_t j = 0; j < input.shape[dim]; j++)
        //     {
        //         true_output.at(i) = fmaxf(true_output.at(i), input.at(i + j * input.strides[dim]));
        //     }            
        // }

        // Tensor output = max(input, dim);
        
        // ASSERT_EQ(output.shape[0], true_output.shape[0]);
        // ASSERT_EQ(output.shape[1], true_output.shape[1]);
        // ASSERT_EQ(output.shape[2], true_output.shape[2]);
        // ASSERT_EQ(output.shape[3], true_output.shape[3]);

        // for (size_t i=0; i < true_output.numel(); i++)
        // {
        //     ASSERT_FLOAT_EQ(output.at(i), true_output.at(i));
        // }
    } 
}


TEST(TestUtils, testMinOnDim)
{
    Tensor input = setup_tensor();

    for (size_t dim=0; dim < 4; dim++)
    {   
        size_t outShape[4]{input.shape[0], input.shape[1], input.shape[2], input.shape[3]};
        outShape[dim] = 1;

        Tensor true_output = Tensor::zeros(outShape[0], outShape[1], outShape[2], outShape[3]);
        
        // for (size_t i = 0; i < true_output.numel(); i++)
        // {
        //     true_output.at(i) = 1.0f * 1e10;
        //     for (size_t j = 0; j < input.shape[dim]; j++)
        //     {
        //         true_output.at(i) = fminf(true_output.at(i), input.at(i + j * input.strides[dim]));
        //     }            
        // }

        // Tensor output = min(input, dim);
        
        // ASSERT_EQ(output.shape[0], true_output.shape[0]);
        // ASSERT_EQ(output.shape[1], true_output.shape[1]);
        // ASSERT_EQ(output.shape[2], true_output.shape[2]);
        // ASSERT_EQ(output.shape[3], true_output.shape[3]);

        // for (size_t i=0; i < true_output.numel(); i++)
        // {
        //     ASSERT_FLOAT_EQ(output.at(i), true_output.at(i));
        // }
    } 
}