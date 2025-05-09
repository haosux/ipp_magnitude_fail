
//==============================================================
// Copyright(C) 2023-2024 Intel Corporation
// Licensed under the Intel Proprietary License
// =============================================================

#include "ippi_tl.h"
#include "util.h"
#include <ipp.h>
#include <opencv2/opencv.hpp>

namespace cvoi
{
namespace ipp
{
namespace mt_tl
{
void sobel_dy(const cv::Mat& image, cv::Mat& output)
{
    output.create(image.size(), CV_16SC1);

    Ipp8u* pSrc = image.data; // source data

    IppiSize roiSize = {image.cols, image.rows};
    IppiSizeL roiSizeL = {image.cols, image.rows};
    IppStatus statusT;
    int bufferSizeV = 0; // horizon convolution buffer size
    Ipp8u* pBuffer;
    Ipp16s* hDst = (Ipp16s*)output.data; // horizon  result
    IppiBorderType borderType_T = ippBorderRepl;

    int step8 = image.cols * sizeof(Ipp8u);
    int step16 = image.cols * sizeof(Ipp16s);
    ippiFilterSobelHorizBorderGetBufferSize_T(roiSize, ippMskSize3x3, ipp8u, ipp16s, 1,
                                              /* numChannels */ &bufferSizeV);

    pBuffer = ippsMalloc_8u(bufferSizeV);

    statusT = ippiFilterSobelHorizBorder_8u16s_C1R_T(pSrc, step8, hDst, step16, roiSize,
                                                     ippMskSize3x3, borderType_T, 0, pBuffer);

    ippsFree(pBuffer);

    return;
}

void sobel_dx(const cv::Mat& image, cv::Mat& output)
{
    output.create(image.size(), CV_16SC1);

    Ipp8u* pSrc = image.data; // source data

    IppiSize roiSize = {image.cols, image.rows};
    IppiSizeL roiSizeL = {image.cols, image.rows};
    IppStatus statusT;
    int bufferSizeV = 0; // horizon convolution buffer size
    Ipp8u* pBuffer;
    Ipp16s* hDst = (Ipp16s*)output.data; // horizon  result
    IppiBorderType borderType_T = ippBorderRepl;

    int step8 = image.cols * sizeof(Ipp8u);
    int step16 = image.cols * sizeof(Ipp16s);
    ippiFilterSobelVertBorderGetBufferSize_T(roiSize, ippMskSize3x3, ipp8u, ipp16s, 1,
                                             /* numChannels */ &bufferSizeV);

    pBuffer = ippsMalloc_8u(bufferSizeV);

    statusT = ippiFilterSobelVertBorder_8u16s_C1R_T(pSrc, step8, hDst, step16, roiSize,
                                                    ippMskSize3x3, borderType_T, 0, pBuffer);

    ippsFree(pBuffer);

    return;
}

} // namespace mt_tl
} // namespace ipp
} // namespace cvoi
