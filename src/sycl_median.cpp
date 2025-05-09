//==============================================================
// Copyright(C) 2023-2024 Intel Corporation
// Licensed under the Intel Proprietary License
// =============================================================

#include "sycl_median.h"

namespace cvoi
{
namespace ipp
{
void sycl_median_with_border(cl::sycl::queue myQueue, cv::Mat& src_image,
                             cv::Mat& copy_border_image, cv::Mat& outimage)
{
    const int KERNEL_SIZE = 9;
    const int RADIUS = 4; // The radius of given kernel size. For kernel 9x9, the radius is 4.

    cv::copyMakeBorder(src_image, copy_border_image, RADIUS, RADIUS, RADIUS, RADIUS,
                       cv::BORDER_REPLICATE);

    outimage.create(src_image.size(), CV_8UC1);

    // Allocate memory on SYCL device for input images. 创建设备内存，存储输入图片，结果内存
    uint8_t* device_src = cl::sycl::malloc_device<uint8_t>(copy_border_image.total(), myQueue);
    uint8_t* device_res = cl::sycl::malloc_device<uint8_t>(src_image.total(), myQueue);

    // 将输入图片拷贝到设备内存中 device_src
    // 是已经扩边后的图片内存，宽度较原来的图片大 2 x RADIUS
    /// Copy input image (with border padding, the width is extended to the original_width+Radius*2)
    /// into SYCL device memory.
    myQueue.memcpy(device_src, copy_border_image.data, copy_border_image.total() * sizeof(uint8_t));

    myQueue.submit(
        [&](cl::sycl::handler& cgh)
        {
            size_t rows = src_image.rows;
            size_t cols = src_image.cols;

            sycl::range<2> global{rows, cols};
            sycl::range<2> local{16, 16};

            cgh.parallel_for(
                cl::sycl::range<2>{rows, cols},
                [=](cl::sycl::id<2> item)
                {
                    int x = item[0]; // row ID
                    int y = item[1]; // column ID

                    int idx = 0;
                    int kernel_idx = 0;
                    int row_idx = 0;
                    int col_idx = 0;
                    int res_idx = x * cols + y;
                    // float convolve_res = 0.0f;
                    int count = 0;
                    // pixels 数组用来保存 9x9 大小的矩形框内的所有像素值，用来求中位数
                    // 对于边界外的像素值，采用 replicate 方法
                    // array pixels[] stores ROI part(size:9x9) of input picture.
                    // Border padding is used replicate method.
                    uint8_t pixels[KERNEL_SIZE * KERNEL_SIZE] = {0};
                    uint8_t histogram[256] = {0};

                    for (int i = -RADIUS; i <= RADIUS; i++)
                    {
                        for (int j = -RADIUS; j <= RADIUS; j++)
                        {
                            row_idx = x + RADIUS + i; ///< row index in ROI
                            col_idx = y + RADIUS + j; ///< column index in ROI

                            pixels[count] = device_src[row_idx * (cols + 2 * RADIUS) + col_idx];
                            count++;
                        }
                    } // end for (int i = -RADIUS; i <= RADIUS; i++)

                    float fMedianEstimate = 127.0f; ///< Medial checking threshold.
                    float fMinBound = 0.0f; ///< Minimum value of input data 中位数范围最小值
                    float fMaxBound = 255.0f; ///< Maxmum value of input data 中位数范围最大值
                    for (int isearch = 0; isearch < 8; isearch++)
                    {
                        uint8_t highCount = 0;
                        for (int i = 0; i < count; i++)
                        {
                            highCount += (fMedianEstimate < pixels[i]);
                        }

                        if (highCount > 40)
                        { // 如果有 41 个数大于 128 ，更新范围  128  191.5  255
                          ///< If the number of ROI itemswhose value >128 is at least 9x9/2, then
                          ///< update range of [128, 255]
                            fMinBound = fMedianEstimate;
                        }
                        else
                        { // 如果有 41 个数小于 128 ，更新范围  0    64     128
                          ///< If the number of ROI items whose value >128 is less than 9x9/2, then
                          ///< update range of [0, 128)
                            fMaxBound = fMedianEstimate;
                        }
                        fMedianEstimate = 0.5f * (fMinBound + fMaxBound);
                    }
                    device_res[res_idx] = (unsigned int)(fMedianEstimate + 0.5f);
                }); // End cgh.parallel_for
        });         // End myQueue.submit
    myQueue.wait();

    myQueue.memcpy(outimage.data, device_res, src_image.total() * sizeof(uint8_t));
    myQueue.wait();

    sycl::free(device_src, myQueue);
    sycl::free(device_res, myQueue);
}

} // namespace ipp

} // namespace cvoi