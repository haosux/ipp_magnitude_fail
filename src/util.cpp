//==============================================================
// Copyright(C) 2023-2024 Intel Corporation
// Licensed under the Intel Proprietary License
// =============================================================

#include "util.h"
#include <iostream>

void displayArray(int arr[], int size)
{
    printf("arr is: ");
    for (int i = 0; i < size; i++)
    {
        printf("%4d ", arr[i]);
    }
    printf("\n");
}

void displayArray(cv::Mat& image, bool DEBUG)
{
    if (!DEBUG)
        return;
    std::cout << "img (grad_x) = \n"
              << cv::format(image, cv::Formatter::FMT_C) << ";" << std::endl
              << std::endl
              << std::endl;

    // printf("img = rows: %d, cols: %d\n",image.rows, image.cols);
    // for (int i = 0; i < image.rows; ++i) {
    //    for (int j = 0; j < image.cols; ++j) {
    //        printf("%4u, ", image.data[i*image.cols + j]);
    //    }
    //    printf("\n");
    //}
    // printf("\n");
    // printf("\n");
}


void displayArrayInt16(cv::Mat& image, bool DEBUG)
{
    if (!DEBUG)
        return;

    for (int x = 0; x < image.rows; x++)
    {
        for (int y = 0; y < image.cols; y++)
        {
            printf("%7d ", image.at<int16_t>(x, y));
        }
        printf("\n");
    }
}

void displayArrayFloat(cv::Mat& image, bool DEBUG)
{
    if (!DEBUG)
        return;

    for (int x = 0; x < image.rows; x++)
    {
        for (int y = 0; y < image.cols; y++)
        {
            printf("%7.2f ", image.at<float>(x, y));
        }
        printf("\n");
    }
}

void comparecv_32f(cv::Mat& image, cv::Mat& output)
{
    double same = 0.0f;
    for (int x = 0; x < image.rows; x++)
    {
        for (int y = 0; y < image.cols; y++)
        {
            if (std::abs(image.at<float>(x, y) - output.at<float>(x, y)) < 0.0001)
            {
                same++;
            }
            else
            {
                printf("<%d, %d> left is %f, right is %f\n", x, y, image.at<float>(x, y),
                       output.at<float>(x, y));
            }
        }
    }

    printf("correctness is %.3f %%\n", same / image.total() * 100);
}

void comparecv_phase_32f(cv::Mat& cvPha, cv::Mat& ippPha)
{
    double same = 0.0f;
    for (int x = 0; x < cvPha.rows; x++)
    {
        for (int y = 0; y < cvPha.cols; y++)
        {
            if (std::abs(cvPha.at<float>(x, y) - ippPha.at<float>(x, y)) < 0.0001)
            {
                same++;
            }
            else if (std::abs(cvPha.at<float>(x, y) + ippPha.at<float>(x, y) - 3.14) < 0.01)
            {
                same++;
            }
            else
            {
                printf("<%2d, %2d> left is %f, right is %f, sum is %f\n", x, y,
                       cvPha.at<float>(x, y), ippPha.at<float>(x, y),
                       cvPha.at<float>(x, y) + ippPha.at<float>(x, y));
            }
        }
    }

    printf("correctness is %.3f %%\n", same / cvPha.total() * 100);
}
