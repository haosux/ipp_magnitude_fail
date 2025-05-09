//==============================================================
// Copyright(C) 2023-2024 Intel Corporation
// Licensed under the Intel Proprietary License
// =============================================================

#include "device_info.h"
#include <iomanip>
#include <iostream>

void display_device_info(cl::sycl::queue mQueue)
{
    int c_width = 24;
    auto device = mQueue.get_device();
    auto p_name = device.get_platform().get_info<sycl::info::platform::name>();
    std::cout << std::setw(c_width) << "Platform Name: " << p_name << "\n";

    auto p_version = device.get_platform().get_info<sycl::info::platform::version>();
    std::cout << std::setw(c_width) << "Platform Version: " << p_version << "\n";

    auto d_name = device.get_info<sycl::info::device::name>();
    std::cout << std::setw(c_width) << "Device Name: " << d_name << "\n";

    auto max_work_group = device.get_info<sycl::info::device::max_work_group_size>();
    std::cout << std::setw(c_width) << "Max Work Group Size: " << max_work_group << "\n";

    auto max_compute_units = device.get_info<sycl::info::device::max_compute_units>();
    std::cout << std::setw(c_width) << "Max Compute Units: " << max_compute_units << "\n";

    auto global_mem_size = device.get_info<sycl::info::device::global_mem_size>();
    std::cout << std::setw(c_width) << "Global Mem Size: " << global_mem_size
              << "\n"; // 13604175872  12GB

    auto local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
    std::cout << std::setw(c_width) << "Local  Mem Size: " << local_mem_size << "\n"; // 65536  64KB

    std::cout << std::endl << std::endl;
}

#ifdef DEVICE_INFO_STANDALONE
int main()
{
    cl::sycl::queue mQueue{gpu_opencl_selector{}};

    display_device_info(mQueue);
}
#endif