//==============================================================
// Copyright(C) 2023-2024 Intel Corporation
// Licensed under the Intel Proprietary License
// =============================================================
// This test file is used to showcase how the operator APIs is called.

#include "util.h"
#include <ipp.h>
#include <opencv2/opencv.hpp>
#include "common/slog.hpp"
#include "common/stimer.hpp"

static void test_cv_sobel(cv::Mat& src_image, int LOOP_NUM)
{
    cv::Mat cv_dx;
    int ksize = 3;
    int scale = 1;
    int delta = 0;

    CvoiTimer aTimer;
    for (int i = 0; i < LOOP_NUM; i++)
    {
        cv::Sobel(src_image, cv_dx, CV_16S, 1, 0, ksize, scale, delta, cv::BORDER_REPLICATE);
    }
    aTimer.printElapsed("opencv sobel", LOOP_NUM);
}

void test_ipp_sobel(cv::Mat& src_image, int LOOP_NUM)
{
    cv::Mat ipp_dx;

    CvoiTimer aTimer;
    for (int i = 0; i < LOOP_NUM; i++)
    {
        cvoi::ipp::mt_tl::sobel_dx(src_image, ipp_dx);
    }
    aTimer.printElapsed("IPP sobel", LOOP_NUM);
}

#ifdef DEVICE_INFO_STANDALONE
int main()
{
    const bool DEBUG = false;
    const int LOOP_NUM = 20;

    cv::Mat cv_dx;
    cv::Mat cv_dy;
    cv::Mat ipp_dx;
    cv::Mat ipp_dy;

    cv::Mat src_image = cv::imread("data/out.png", cv::IMREAD_GRAYSCALE);

    printf("\n\n");
    printf("\n\n source image \n\n");
    displayArray(src_image, DEBUG);

    int ksize = 3;
    int scale = 1;
    int delta = 0;

    cv::Sobel(src_image, cv_dx, CV_16S, 1, 0, ksize, scale, delta, cv::BORDER_REPLICATE);
    printf("\n\n");
    printf("\n\n cv sobel dx image \n\n");
    displayArray(cv_dx, DEBUG);

    cvoi::ipp::mt_tl::sobel_dx(src_image, ipp_dx);
    printf("\n\n");
    printf("\n\n ipp sobel dx image \n\n");
    displayArray(ipp_dx, DEBUG);

    cv::Sobel(src_image, cv_dy, CV_16S, 0, 1, ksize, scale, delta, cv::BORDER_REPLICATE);
    printf("\n\n");
    printf("\n\n cv sobel dy image \n\n");
    displayArray(cv_dy, DEBUG);

    cvoi::ipp::mt_tl::sobel_dy(src_image, ipp_dy);
    printf("\n\n");
    printf("\n\n ipp sobel dy image \n\n");
    displayArray(ipp_dy, DEBUG);

    test_cv_sobel(src_image, LOOP_NUM);

    test_ipp_sobel(src_image, LOOP_NUM);
}
#endif
