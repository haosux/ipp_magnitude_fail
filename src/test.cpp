
//==============================================================
// Copyright(C) 2023-2024 Intel Corporation
// Licensed under the Intel Proprietary License
// =============================================================

#include <iostream>
#include <opencv2/opencv.hpp>
#include "ipp_magnitude.h"
#include "ipp_sobel.h"
#include "util.h"

using std::cout;
using std::endl;
// using sycl::queue;

/**
 * @brief An encapsulation function for IPP optimized operators.
 *
 * @param[in] src_image the source image to be filtered.
 * @param[out] ipp_dx Sobel DX output
 * @param[out] ipp_dy Sobel DY output
 * @param[out] ipp_mag magnitude filter output
 * @param[out] ipp_pha phase filter output
 *
 */
void ipp_calc(cv::Mat& src_image, cv::Mat& ipp_dx, cv::Mat& ipp_dy, cv::Mat& ipp_mag,
              cv::Mat& ipp_pha)
{
    // ipp_mag.create(src_image.size(), CV_32FC1);
    // ipp_pha.create(src_image.size(), CV_32FC1);

    cvoi::ipp::mt_tl::sobel_dx(src_image, ipp_dx);
    cvoi::ipp::mt_tl::sobel_dy(src_image, ipp_dy);

    // cvoi::ipp::phase(ipp_dx, ipp_dy, ipp_pha);
    cvoi::ipp::magnitude(ipp_dx, ipp_dy, ipp_mag);


    printf("\n\nipp_calc  display  ipp  dx  Mat \n\n");
    displayArrayInt16(ipp_dx, true);

    printf("\n\nipp_calc  display  ipp  dy  Mat \n\n");
    displayArrayInt16(ipp_dy, true);

    printf("\n\nipp_calc  display  ipp  mag  Mat \n\n");
    displayArrayFloat(ipp_mag, true);
}

/**
 * @brief test function for SYCL Median operator
 *
 * @param[inout] myQueue sycl queue
 * @param[in] cvimage    input image to be calculated
 * @param[in] LOOP_NUM   Loop number
 */
void test_detect_ipp_sycl(/*cl::sycl::queue myQueue,*/ cv::Mat& cvImage, cv::Mat& outMag,
                          cv::Mat& outPha, int LOOP_NUM)
{
    cv::Mat outImage;
    cv::Mat borderImage;
    // cv::Mat outMag;
    // cv::Mat outPha;
    cv::Mat ippdx;
    cv::Mat ippdy;

    for (int i = 0; i < LOOP_NUM; i++)
    {

        ipp_calc(cvImage, ippdx, ippdy, outMag, outPha);

    }
}

/**
 * @brief test function for opencv operators.
 *
 * @param[in] cvimage    input image to be calculated
 * @param[in] LOOP_NUM   Loop number
 */
void test_detect_opencv(cv::Mat& cvImage, cv::Mat& cvMag, cv::Mat& cvPha, int LOOP_NUM)
{
    cv::Mat outImage;
    cv::Mat dx;
    cv::Mat dy;
    // cv::Mat cvMag;
    // cv::Mat cvPha;

    for (int i = 0; i < LOOP_NUM; i++)
    {

        // cv::medianBlur(cvImage, outImage, 9);

        cv::Sobel(cvImage, dx, CV_32FC1, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
        cv::Sobel(cvImage, dy, CV_32FC1, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
        cv::magnitude(dx, dy, cvMag);
        // cv::phase(dx, dy, cvPha);

    }
}




int main(int argc, char** argv)
{
    const bool DEBUG = true;



    // std::string filename = "/data/so_small_starry_25.png";
    // std::string filename = "/data/b2.png";
    std::string filename = "/data/r1.png";
    std::string srcPath = CMAKE_CURRENT_SOURCE_DIR + filename;

    const int LOOP_NUM = 1;
    cout << endl
         << endl
        //  << "This program run dilation on picture " << argv[1] << " for " << LOOP_NUM << " times."
         << "This program run dilation on picture " << srcPath << " for " << LOOP_NUM << " times."
         << endl
         << endl
         << endl;



    // cv::Mat src_image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat src_image = cv::imread(srcPath, cv::IMREAD_GRAYSCALE);

    cv::Mat cvMag;
    cv::Mat cvPha;
    cv::Mat ippMag;
    cv::Mat ippPha;

    test_detect_opencv(src_image, cvMag, cvPha, LOOP_NUM);
    test_detect_ipp_sycl(/*mQueue,*/ src_image, ippMag, ippPha, LOOP_NUM);

    if (DEBUG)
        printf("display opencv mag Mat \n\n");
    displayArrayFloat(cvMag, DEBUG);


    // printf("\ncompare correctness opencv magnitude vs ipp magnitude \n");
    // comparecv_32f(cvMag, ippMag);


}
