
//==============================================================
// Copyright(C) 2023-2024 Intel Corporation
// Licensed under the Intel Proprietary License
// =============================================================

#include "ippi_tl.h"
#include "util.h"
#include <ipp.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <cassert>

namespace cvoi
{
namespace ipp
{
void phase(const cv::Mat& image_dx, const cv::Mat& image_dy, cv::Mat& output)
{
    assert(image_dx.type() == CV_16S);
    assert(image_dx.size() == image_dy.size());

    output.create(image_dx.size(), CV_32FC1);

#ifndef MULTI_THREADING_IN_OP
    ippsPhase_16s32f((Ipp16s*)image_dx.data, (Ipp16s*)image_dy.data, (Ipp32f*)output.data,
                     output.total());
#else
    int len = output.total();

    int mid = len / 2;
    int secondlen = len - mid;

    int len1 = len / 4;
    int len2 = len1 * 2;
    int len3 = len1 * 3;
    int len4 = len - len3;
    ippsPhase_32f((Ipp32f*)image_dx.data, (Ipp32f*)image_dy.data, (Ipp32f*)output.data, mid);
    ippsPhase_32f((Ipp32f*)image_dx.data + mid, (Ipp32f*)image_dy.data + mid,
                  (Ipp32f*)output.data + mid, secondlen);

    std::thread t1(ippsPhase_32f, (Ipp32f*)image_dx.data, (Ipp32f*)image_dy.data,
                   (Ipp32f*)output.data, mid);
    std::thread t2(ippsPhase_32f, (Ipp32f*)image_dx.data + mid, (Ipp32f*)image_dy.data + mid,
                   (Ipp32f*)output.data + mid, secondlen);
    t1.join();
    t2.join();

    std::thread t1(ippsPhase_32f, (Ipp32f*)image_dx.data, (Ipp32f*)image_dy.data,
                   (Ipp32f*)output.data, len1);
    std::thread t2(ippsPhase_32f, (Ipp32f*)image_dx.data + len1, (Ipp32f*)image_dy.data + len1,
                   (Ipp32f*)output.data + len1, len1);
    std::thread t3(ippsPhase_32f, (Ipp32f*)image_dx.data + len2, (Ipp32f*)image_dy.data + len2,
                   (Ipp32f*)output.data + len2, len1);
    std::thread t4(ippsPhase_32f, (Ipp32f*)image_dx.data + len3, (Ipp32f*)image_dy.data + len3,
                   (Ipp32f*)output.data + len3, len4);
    t1.join();
    t2.join();
    t3.join();
    t4.join();
#endif
}

void magnitude(const cv::Mat& image_dx, const cv::Mat& image_dy, cv::Mat& output)
{
    assert(image_dx.type() == CV_16S);
    assert(image_dx.size() == image_dy.size());

    output.create(image_dx.size(), CV_32FC1);

    ippsMagnitude_16s32f((Ipp16s*)image_dx.data, (Ipp16s*)image_dy.data, (Ipp32f*)output.data,
                         output.total());
}

} // namespace ipp

} // namespace cvoi

static void ipp_magnitude_complex(cv::Mat& image_dx, cv::Mat& image_dy, Ipp32fc* complex,
                                  cv::Mat& output)
{
    output.create(image_dx.size(), CV_32FC1);

    // 接收数据
    Ipp32f* ptrDx = (Ipp32f*)image_dx.data;
    Ipp32f* ptrDy = (Ipp32f*)image_dy.data;

    Ipp32fc a;

    // 申请空间
    // Ipp32fc* complex = new Ipp32fc[image_dx.total()];

    IppiSize roiSize = {image_dx.cols, image_dx.rows};

    ippsRealToCplx_32f(ptrDx, ptrDy, complex, image_dx.total());

    auto start = std::chrono::steady_clock::now();
    ippiMagnitude_32fc32f_C1R(complex, image_dx.cols * sizeof(Ipp32fc), (Ipp32f*)output.data,
                              output.step[0], roiSize);
    auto stop = std::chrono::steady_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
    printf("ipp magnitude api                took %.3f milliseconds\n", time);

    // delete[] complex;
}
