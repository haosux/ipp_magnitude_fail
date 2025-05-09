# Lithium Battery Defect Detection Pre-Process

This project emulates the CV pre-processes for Lithium Battery defect detection pipelines, in which below operators are involved:
1. [cv::medianBlur] Median Blur, setting convolution kernel size as 9.
2. [cv::Sobel-dx] Horizontal Sobel Filter, setting convolution kernel size as 3.
3. [cv::Sobel-dy] Vertical Sobel Filter, setting convolution kernel size as 3.
4. [cv::magnitude] Calculate Magnitude of 2D vectors formed from the both(horizontal & vertical) outputs of Sobel Filters.
5. [cv::phase] Calculate the rotation of the 2D vector formed from the both(horizontal & vertical) outputs of Sobel Filters.

In this project, the 5 corresponding operators are optimized by Intel OneAPI toolkit:
- `cv::medianBlue` is optimized by oneAPI DPC++ SYCL parallelization.
- Other operators (`cv::Sobel`, `cv::magnitude`, `cv::phase`) is optimized by rewriting with IPP libraries.

## Prerequisites
### Hardware Configuration

- Intel Gen-11 or later platform
- [Optional] Intel Arc Graphics Card

### Software Dependency

- Intel OneAPI Base toolkit

## Build & Run

### On a Linux System

Build and run program

```bash
source /opt/intel/oneapi/setvars.sh
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=icpx ..
cmake --build .
./test_lithium_pipeline ../data/globe.jpg
```

### On a Windows System

modify OpenCV_DIR in CMakelists.txt
```bash
set(OpenCV_DIR "C:/Users/RS/Downloads/opencv_4.5.5/build")
```

Build and run program

```bash
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

mkdir build
cd build
cmake -G Ninja -DCMAKE_CXX_COMPILER=icx .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

.\test_lithium_pipeline ..\data\globe.jpg
```

## Output Sample
Below shows an output example launching the program on an Intel Tiger Lake platform.

```bash

This program run dilation on picture ..\data\globe.jpg for 20 times.


         Platform Name: Intel(R) OpenCL HD Graphics
      Platform Version: OpenCL 3.0
           Device Name: Intel(R) Iris(R) Xe Graphics
   Max Work Group Size: 512
     Max Compute Units: 96
       Global Mem Size: 13604134912
       Local  Mem Size: 65536


opencv took 86.761 milliseconds
opencv took 79.420 milliseconds
opencv took 81.603 milliseconds
opencv took 82.764 milliseconds
opencv took 87.251 milliseconds
opencv took 85.734 milliseconds
opencv took 85.289 milliseconds
opencv took 79.471 milliseconds
opencv took 81.838 milliseconds
opencv took 81.543 milliseconds
opencv took 82.453 milliseconds
opencv took 79.599 milliseconds
opencv took 81.443 milliseconds
opencv took 81.353 milliseconds
opencv took 80.464 milliseconds
opencv took 79.129 milliseconds
opencv took 80.700 milliseconds
opencv took 79.529 milliseconds
opencv took 80.216 milliseconds
opencv took 79.614 milliseconds
sycl median + ippx4 took 966.102 milliseconds
sycl median + ippx4 took 27.161 milliseconds
sycl median + ippx4 took 27.220 milliseconds
sycl median + ippx4 took 27.763 milliseconds
sycl median + ippx4 took 27.975 milliseconds
sycl median + ippx4 took 27.791 milliseconds
sycl median + ippx4 took 28.292 milliseconds
sycl median + ippx4 took 34.509 milliseconds
sycl median + ippx4 took 27.350 milliseconds
sycl median + ippx4 took 27.154 milliseconds
sycl median + ippx4 took 27.890 milliseconds
sycl median + ippx4 took 27.039 milliseconds
sycl median + ippx4 took 27.035 milliseconds
sycl median + ippx4 took 28.002 milliseconds
sycl median + ippx4 took 27.746 milliseconds
sycl median + ippx4 took 27.545 milliseconds
sycl median + ippx4 took 28.105 milliseconds
sycl median + ippx4 took 27.832 milliseconds
sycl median + ippx4 took 27.866 milliseconds
sycl median + ippx4 took 27.165 milliseconds

C:\Users\RS\Videos\github\cv_phase2\build>PAUSE
Press any key to continue . . .

```
## Deep Dive
### Optimization on CPU

Besides performance optimizations on Intel GPU, this project also supports optimization with Intel CPU.

In order to enable CPU optimization, you need to replace the line in function main(), which is located under file `test.cpp`:
```cpp
    queue mQueue{sycl::gpu_selector_v};
```
with this line:
```cpp
    queue mQueue{sycl::cpu_selector_v};
```

The performance of optimization with CPU is good enough according to our tests on Intel Core i7-1165G7 platform.
| Optimization Mode | Performance (ms) |
| --- | --- |
| Legacy opencv | 96.6 |
| Optimization with igpu | 30.7 |
| Optimization with cpu | 38.3 |


### SYCL AOT Compilation Prioritization

The Intel® oneAPI DPC++ Compiler converts a SYCL program into an intermediate language called SPIR-V and stores that in the binary produced by the compilation process. The advantage of producing this intermediate file instead of the binary is that this code can be run on any hardware platform by translating the SPIR-V code into the assembly code of the platform at runtime. This process of translating the intermediate code present in the binary is called JIT compilation (Just-In-Time compilation). JIT compilation can happen on demand at runtime.

Although the advantages of code's compatibility, JIT compilation has negative contribution to runtime performance, especially when the SYCL intermediate file is translated at the first run. You can also find the phenomena from the printed lob shown in above section.

In order to improve its runtime performance for the first run, AOT compilation can be enabled.

The overhead of JIT compilation at runtime can be avoided by Ahead-Of-Time (AOT) compilation. With AOT compilation, the binary will contain the actual assembly code of the platform that was selected at compile time instead of the SPIR-V intermediate code. The advantage is that we do not need to JIT compile the code from SPIR-V to assembly during execution, which makes the code run faster.

Please follow the below steps to enable JIT compilation mode:

### 1. Find target values or graphics device code for the GPU device you want to use.

You may find the target values for Intel GPU from [the user guide of oneAPI compiler](https://intel.github.io/llvm-docs/UsersManual.html). Below is a quick reference list:

```The following triples are supported by default:
* spir64 - this is the default generic SPIR-V target;
* spir64_x86_64 - generate code ahead of time for x86_64 CPUs;
* spir64_fpga - generate code ahead of time for Intel FPGA;
* spir64_gen - generate code ahead of time for Intel Processor Graphics;
Full target triples can also be used:
* spir64-unknown-unknown, spir64_x86_64-unknown-unknown,
  spir64_fpga-unknown-unknown, spir64_gen-unknown-unknown
Available in special build configuration:
* nvptx64-nvidia-cuda - generate code ahead of time for CUDA target;
* native_cpu - allows to run SYCL applications with no need of an
additional backend (note that this feature is WIP and experimental, and
currently overrides all the other specified SYCL targets when enabled.)

Special target values specific to Intel, NVIDIA and AMD Processor Graphics
support are accepted, providing a streamlined interface for AOT. Only one of
these values at a time is supported.
* intel_gpu_pvc - Ponte Vecchio Intel graphics architecture
* intel_gpu_pvc_vg - Ponte Vecchio VG Intel graphics architecture
* intel_gpu_acm_g12, intel_gpu_dg2_g12 - Alchemist G12 Intel graphics architecture
* intel_gpu_acm_g11, intel_gpu_dg2_g11 - Alchemist G11 Intel graphics architecture
* intel_gpu_acm_g10, intel_gpu_dg2_g10 - Alchemist G10 Intel graphics architecture
* intel_gpu_dg1, intel_gpu_12_10_0 - DG1 Intel graphics architecture
* intel_gpu_adl_n - Alder Lake N Intel graphics architecture
* intel_gpu_adl_p - Alder Lake P Intel graphics architecture
* intel_gpu_rpl_s - Raptor Lake Intel graphics architecture (equal to intel_gpu_adl_s)
* intel_gpu_adl_s - Alder Lake S Intel graphics architecture
* intel_gpu_rkl - Rocket Lake Intel graphics architecture
* intel_gpu_tgllp, intel_gpu_12_0_0 - Tiger Lake Intel graphics architecture
* intel_gpu_jsl - Jasper Lake Intel graphics architecture (equal to intel_gpu_ehl)
* intel_gpu_ehl - Elkhart Lake Intel graphics architecture
* intel_gpu_icllp, intel_gpu_11_0_0 - Ice Lake Intel graphics architecture
* intel_gpu_cml, intel_gpu_9_7_0 - Comet Lake Intel graphics architecture
* intel_gpu_aml, intel_gpu_9_6_0 - Amber Lake Intel graphics architecture
* intel_gpu_whl, intel_gpu_9_5_0 - Whiskey Lake Intel graphics architecture
* intel_gpu_glk, intel_gpu_9_4_0 - Gemini Lake Intel graphics architecture
* intel_gpu_bxt - Broxton Intel graphics architecture (equal to intel_gpu_apl)
* intel_gpu_apl, intel_gpu_9_3_0 - Apollo Lake Intel graphics architecture
* intel_gpu_cfl, intel_gpu_9_2_9 - Coffee Lake Intel graphics architecture
* intel_gpu_kbl, intel_gpu_9_1_9 - Kaby Lake Intel graphics architecture
* intel_gpu_skl, intel_gpu_9_0_9 - Skylake Intel graphics architecture
* intel_gpu_bdw, intel_gpu_8_0_0 - Broadwell Intel graphics architecture
```

If you are unsure which graphics processor you have, you can map your device’s PCI ID to an entry on this page. To determine your PCI device ID, you can use the lspci command on your Linux*-based distribution:
```bash
lspci -nn |grep  -Ei 'VGA|DISPLAY'
```

Example output from that command:
```bash
00:02.0 VGA compatible controller [0300]: Intel Corporation Alder Lake-P Integrated Graphics Controller [8086:46a6] (rev 0c)
```

In the previous example, the PCI device ID is **8086:46a6**。

### 2. Add extra compiler options to indicate the specific GPU target.

In `CMakeLists.txt` of this project, update `CMAKE_CXX_FLAGS` as bellow:
```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=intel_gpu_adl_p ")
```
or
```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=spir64_gen -Xs '-device 0x46a6' ")
```

### (Optional) 3. Add GPU device selector in source code.
If there are more than 1 GPU device on your platform, you need to set ENV variables to filter out the one you prefer. Please follow the [guideline](https://intel.github.io/llvm-docs/EnvironmentVariables.html#oneapi-device-selector) to select the specific GPU device.

If it fails to filter out the GPU device by ENV variable settings, you'd add your own device selector class in your source code. Below is an example to select the integrated GPU from an A770 GPU mounted.
1. Find sycl device info by calling sycl-ls command:
```bash
$ sycl-ls
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2023.16.12.0.12_195853.xmain-hotfix]
[opencl:cpu:1] Intel(R) OpenCL, 12th Gen Intel(R) Core(TM) i7-12700H OpenCL 3.0 (Build 0) [2023.16.12.0.12_195853.xmain-hotfix]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770M Graphics OpenCL 3.0 NEO  [23.17.26241.33]
[opencl:gpu:3] Intel(R) OpenCL Graphics, Intel(R) Graphics [0x46a6] OpenCL 3.0 NEO  [23.17.26241.33]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770M Graphics 1.3 [1.3.26241]
[ext_oneapi_level_zero:gpu:1] Intel(R) Level-Zero, Intel(R) Graphics [0x46a6] 1.3 [1.3.26241]
```
2. Write your own selector class.
```c++
class my_selector : public sycl::device_selector
{
public:
    int operator()(const sycl::device& dev) const override
    {
        if (dev.get_info<sycl::info::device::name>().find("Intel") != std::string::npos &&
            dev.get_platform().get_info<sycl::info::platform::name>().find("OpenCL Gra") !=
                std::string::npos &&
            dev.get_info<sycl::info::device::name>().find("A770") ==
                std::string::npos )
        {
            return 1;
        }
        return -1;
    }
};
```
3. Call your own selector class when generating SYCL Queue.
```c++
int main(int argc, char** argv)
{
  ...
  sycl::queue mQueue{my_selector{}};
  ...
```
