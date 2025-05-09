//==============================================================
// Copyright(C) 2023-2024 Intel Corporation
// Licensed under the Intel Proprietary License
// =============================================================
// This test file is used to showcase how the operator APIs is called.
#include "ippi_tl.h"
#include "util.h"
#include <ipp.h>
#include <opencv2/opencv.hpp>
#include "common/slog.hpp"
#include "common/stimer.hpp"

static void test_ipp_phase(const cv::Mat& ipp_dx, cv::Mat& ipp_dy, int LOOP_NUM)
{
    cv::Mat ipp_pha;

    CvoiTimer aTimer;
    for (int i = 0; i < LOOP_NUM; i++)
    {
        cvoi::ipp::phase(ipp_dx, ipp_dy, ipp_pha);
    }
    aTimer.printElapsed("IPP phase", LOOP_NUM);
}

static void test_cv_phase(cv::Mat& cv_dx, cv::Mat& cv_dy, int LOOP_NUM)
{
    cv::Mat cv_phase;

    CvoiTimer aTimer;
    for (int i = 0; i < LOOP_NUM; i++)
    {
        cv::phase(cv_dx, cv_dy, cv_phase);
    }
    aTimer.printElapsed("opencv phase", LOOP_NUM);
}

static void test_ipp_magnitude(cv::Mat& ipp_dx, cv::Mat& ipp_dy, int LOOP_NUM)
{
    cv::Mat ipp_mag;

    CvoiTimer aTimer;
    for (int i = 0; i < LOOP_NUM; i++)
    {
        cvoi::ipp::magnitude(ipp_dx, ipp_dy, ipp_mag);
    }
    aTimer.printElapsed("IPP magnitude", LOOP_NUM);
}

static void test_cv_magnitude_phase(cv::Mat& cv_dx, cv::Mat& cv_dy, int LOOP_NUM)
{
    cv::Mat cv_mag;
    cv::Mat cv_pha;

    CvoiTimer aTimer;
    for (int i = 0; i < LOOP_NUM; i++)
    {
        cv::magnitude(cv_dx, cv_dy, cv_mag);
        cv::phase(cv_dx, cv_dy, cv_pha);
    }
    aTimer.printElapsed("opencv magnitude+phase", LOOP_NUM);
}

static void test_cv_calc(cv::Mat& src_image, int LOOP_NUM)
{
    cv::Mat cv_dx;
    cv::Mat cv_dy;
    cv::Mat cv_mag;
    cv::Mat cv_pha;

    CvoiTimer aTimer;
    for (int i = 0; i < LOOP_NUM; i++)
    {
        cv::Sobel(src_image, cv_dx, CV_32FC1, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
        cv::Sobel(src_image, cv_dy, CV_32FC1, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
        cv::magnitude(cv_dx, cv_dy, cv_mag);
        cv::phase(cv_dx, cv_dy, cv_pha);
    }
    aTimer.printElapsed("opencv sobel+magnitude+phase", LOOP_NUM);
}

#ifdef DEVICE_INFO_STANDALONE
int main()
{
    const bool DEBUG = false;
    const int LOOP_NUM = 15;

    cv::Mat cv_dx;
    cv::Mat cv_dy;
    cv::Mat ipp_dx;
    cv::Mat ipp_dy;

    cv::Mat src_image = cv::imread("data/out.png", cv::IMREAD_GRAYSCALE);

    int ksize = 3;
    int scale = 1;
    int delta = 0;

    cv_dx.convertTo(cv_dx, CV_32FC1);
    cv_dy.convertTo(cv_dy, CV_32FC1);

    test_cv_phase(cv_dx, cv_dy, LOOP_NUM);
    test_ipp_phase(ipp_dx, ipp_dy, LOOP_NUM);
    test_cv_magnitude_phase(cv_dx, cv_dy, LOOP_NUM);
    test_cv_calc(cv_dx, cv_dy, LOOP_NUM);
    test_ipp_magnitude(ipp_dx, ipp_dy, LOOP_NUM);
}
#endif
