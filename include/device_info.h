//==============================================================
// Copyright(C) 2023-2024 Intel Corporation
// Licensed under the Intel Proprietary License
// =============================================================
#ifndef CVOI_USECASE_LITHIUM_DEVICE_INFO_H_
#define CVOI_USECASE_LITHIUM_DEVICE_INFO_H_

#include <CL/sycl.hpp>

// c:\Program Files(x86)\Intel\oneAPI\compiler\latest\windows\bin > .\sycl-ls.exe
//
//                                      Platform Name                                       Device
//                                      Name
//[opencl:gpu:0]                        Intel(R) OpenCL HD Graphics,                        Intel(R)
// Iris(R) Xe Graphics 3.0[31.0.101.3413] [opencl:cpu : 1]                      Intel(R) OpenCL,
// 11th Gen Intel(R) Core(TM) i7 - 1165G7 @ 2.80GHz 3.0[2022.13.3.0.16_160000] [opencl:acc:2]
// Intel(R) FPGA Emulation Platform for OpenCL(TM),    Intel(R) FPGA Emulation
// Device 1.2[2022.13.3.0.16_160000] [ext_oneapi_level_zero:gpu:0]         Intel(R) Level - Zero,
// Intel(R) Iris(R) Xe Graphics 1.3[1.3.23904] [host:host:0] SYCL host platform, SYCL host
// device 1.2[1.2]

void display_device_info(cl::sycl::queue mQueue);

class level_zero_selector : public sycl::device_selector
{
public:
    int operator()(const sycl::device& dev) const override
    {
        if (dev.get_info<sycl::info::device::name>().find("Intel") != std::string::npos &&
            dev.get_platform().get_info<sycl::info::platform::name>().find("Level") !=
                std::string::npos)
        {
            return 1;
        }
        return -1;
    }
};

class gpu_opencl_selector : public sycl::device_selector
{
public:
    int operator()(const sycl::device& dev) const override
    {
        if (dev.get_info<sycl::info::device::name>().find("Intel") != std::string::npos &&
            dev.get_platform().get_info<sycl::info::platform::name>().find("OpenCL Gra") !=
                std::string::npos)
        {
            return 1;
        }
        return -1;
    }
};

class cpu_opencl_selector : public sycl::device_selector
{
public:
    int operator()(const sycl::device& dev) const override
    {
        if (dev.get_info<sycl::info::device::name>().find("Intel(R) Core") != std::string::npos &&
            dev.get_platform().get_info<sycl::info::platform::name>().find("OpenCL") !=
                std::string::npos)
        {
            return 1;
        }
        return -1;
    }
};

#endif // CVOI_USECASE_LITHIUM_DEVICE_INFO_H_