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

我们看到两个输入第一行的三个像素都是零

![image](https://github.com/user-attachments/assets/a2053a54-0e12-4474-ab8b-28c755b8b421)


但是经由 ipp 计算后，第一行的三个像素都是未定义值

![image](https://github.com/user-attachments/assets/e2a07a9d-c1a8-43d0-a9de-0c98b8666466)
