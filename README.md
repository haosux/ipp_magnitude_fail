# ipp magnitude 问题说明

本工程说明 ipp magnitude 算子在处理输入都为 0 值时，计算错误的问题


# Build & Run

## On a Linux System

```bash
source /opt/intel/oneapi/setvars.sh
mkdir build
cd build
cmake ..
make 

./test_lithium_pipeline

```