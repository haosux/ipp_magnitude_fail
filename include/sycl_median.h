//==============================================================
// Copyright(C) 2023-2024 Intel Corporation
// Licensed under the Intel Proprietary License
// =============================================================

#ifndef CVOI_USECASE_LITHIUM_SYCL_MEDIAN_H_
#define CVOI_USECASE_LITHIUM_SYCL_MEDIAN_H_

#include <CL/sycl.hpp>
#include <opencv2/opencv.hpp>

namespace cvoi
{
namespace ipp
{
/**
 * @brief Do median filter to the given image. This function uses Binary Tree Traversal algorithm
 * for median calculation, and cv::copyMakeBorder() for border padding.
 * 对图像进行 median 滤波，使用二分法逼近的方法求中值， 预先使用
 * opencv::copyMakeBorder求原图像的扩边图像.
 *
 * @param[in] myQueue  SYCL queue.
 * @param[in] src_image the input image to be calculated. Image type: CV_8UC1.
 * @param[out] copy_border_image new image for the input image with border padding.
 * 默认输入图片为单通道 uint8_t 类型
 * @param[out] outimage output image, image type: CV_8UC1
 *
 * @return calculated image with type of CV_8UC1. 默认输出图片为单通道 uint8_t 类型
 */
void sycl_median_with_border(cl::sycl::queue myQueue, cv::Mat& src_image,
                             cv::Mat& copy_border_image, cv::Mat& outimage);

} // namespace ipp

} // namespace cvoi

#endif // CVOI_USECASE_LITHIUM_SYCL_MEDIAN_H_
