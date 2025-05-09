//==============================================================
// Copyright(C) 2023-2024 Intel Corporation
// Licensed under the Intel Proprietary License
// =============================================================

#ifndef CVOI_USECASE_LITHIUM_IPP_MAGNITUDE_H_
#define CVOI_USECASE_LITHIUM_IPP_MAGNITUDE_H_

#include <opencv2/opencv.hpp>
#include <ipp.h>

namespace cvoi
{
namespace ipp
{
/**
 * @brief Calculates the rotation angle of 2D vectors.
 *
 * @param[in] image_dx  floating-point array of x-coordinates of the vectors.
 * @param[in] image_dy  floating-point array of y-coordinates of the vectors; it must have the
 * same size as x.
 * @param[out]  output  output array of the same size and type as x.
 *
 */
void phase(const cv::Mat& image_dx, const cv::Mat& image_dy, cv::Mat& output);

/**
 * @brief Calculates the magnitude of 2D vectors.
 *
 * @param[in] image_dx  floating-point array of x-coordinates of the vectors.
 * @param[in] image_dy  floating-point array of y-coordinates of the vectors; it must have the
 * same size as x.
 * @param[out]  output  output array of the same size and type as x.
 *
 */
void magnitude(const cv::Mat& image_dx, const cv::Mat& image_dy, cv::Mat& output);

/**
 * @brief Calculates the magnitude of 2D vectors in complex mode.
 *
 * @param[in] image_dx  floating-point array of x-coordinates of the vectors.
 * @param[in] image_dy  floating-point array of y-coordinates of the vectors; it must have the
 * same size as x.
 * @param[in] complex pointer to a list of complex indices, the size of the list is
 * image_dx.total().
 * @param[out]  output  output array of the same size and type as x.
 *
 */
void magnitude_complex(const cv::Mat& image_dx, const cv::Mat& image_dy, Ipp32fc* complex,
                       cv::Mat& output);

} // namespace ipp

} // namespace cvoi

#endif // CVOI_USECASE_LITHIUM_IPP_MAGNITUDE_H_