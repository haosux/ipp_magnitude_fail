//==============================================================
// Copyright(C) 2023-2024 Intel Corporation
// Licensed under the Intel Proprietary License
// =============================================================

#ifndef CVOI_USECASE_LITHIUM_IPP_SOBEL_H_
#define CVOI_USECASE_LITHIUM_IPP_SOBEL_H_

#include <opencv2/opencv.hpp>

namespace cvoi
{
namespace ipp
{
namespace mt_tl
{
/**
 * @brief Computes the x direction image derivatives boosted by ipp multi-threaded api.
 *
 * @param[in] img  input image.
 * @param[out] output output image of the same size and the same number of channels as src .
 *
 */
void sobel_dx(const cv::Mat& image, cv::Mat& output);

/**
 * @brief Computes the y direction image derivatives boosted by ipp multi-threaded api.
 *
 * @param[in] img  input image.
 * @param[out] output output image of the same size and the same number of channels as src .
 *
 */
void sobel_dy(const cv::Mat& image, cv::Mat& output);

} // namespace mt_tl
} // namespace ipp
} // namespace cvoi

#endif // CVOI_USECASE_LITHIUM_IPP_SOBEL_H_
