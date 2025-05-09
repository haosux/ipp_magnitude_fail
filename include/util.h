
//==============================================================
// Copyright(C) 2023-2024 Intel Corporation
// Licensed under the Intel Proprietary License
// =============================================================

#ifndef CVOI_USECASE_LITHIUM_UTIL_H_
#define CVOI_USECASE_LITHIUM_UTIL_H_

#include <opencv2/opencv.hpp>

void displayArray(int arr[], int size);
void displayArray(cv::Mat& image, bool DEBUG);
void displayArrayFloat(cv::Mat& image, bool DEBUG);
void displayArrayInt16(cv::Mat& image, bool DEBUG);

void comparecv_32f(cv::Mat& image, cv::Mat& output);
void comparecv_phase_32f(cv::Mat& image, cv::Mat& output);

#endif // CVOI_USECASE_LITHIUM_UTIL_H_