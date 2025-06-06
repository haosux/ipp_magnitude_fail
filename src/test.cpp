
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
void test_detect_ipp_sycl(cv::Mat& cvImage, cv::Mat& outMag,
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
         << "This program run dilation on picture " << srcPath << " for " << LOOP_NUM << " times."
         << endl
         << endl
         << endl;


    cv::Mat src_image = cv::imread(srcPath, cv::IMREAD_GRAYSCALE);

    cv::Mat cvMag;
    cv::Mat cvPha;
    cv::Mat ippMag;
    cv::Mat ippPha;

    printf("\n\ninput image  display \n\n");
    displayArray(src_image, true);

    test_detect_ipp_sycl(src_image, ippMag, ippPha, LOOP_NUM);




}
