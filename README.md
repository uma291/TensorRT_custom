# TensorRT custom plugin

Just add some new custom tensorRT plugin

## New plugin

- [BatchedNMSLandmark_TRT](./plugin/batchedNMSLandmarkPlugin/), BatchedNMSLandmarkDynamic_TRT: Batched NMS with face landmark
- [BatchedNMSLandmarkConf_TRT](./plugin/batchedNMSLandmarkConfPlugin/), BatchedNMSLandmarkConfDynamic_TRT: Batched NMS with face lanmdark & confidence
- [EfficientNMSLandmark_TRT](./plugin/efficientNMSLandmarkPlugin/): Efficient NMS with face landmark
- [EfficientNMSCustom_TRT](./plugin/efficientNMSCustomPlugin/): Same Efficient NMS, but return boxes indices
- [RoIAlignDynamic](./plugin/roIAlignPlugin/): Same ONNX RoIAlign, copy from [MMCVRoIAlign](https://github.com/open-mmlab/mmdeploy)
- [RoIAlign2Dynamic](./plugin/roIAlign2Plugin/): Same as pyramidROIAlignPlugin, but only one feature_map.

## Prerequisites

- Deepstream 6.0.1

## Install

Follow guide from

Please refer to the guide under [github.com/NVIDIA-AI-IOT/deepstream_tao_apps](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/blob/master/TRT-OSS/x86/README.md)

### 1. Installl Cmake (>= 3.13)

TensorRT OSS requires cmake >= v3.13, so install cmake 3.13 if your cmake version is lower than 3.13

```
wget https://github.com/Kitware/CMake/releases/download/v3.19.4/cmake-3.19.4.tar.gz
tar xvf cmake-3.19.4.tar.gz
cd cmake-3.19.4/
mkdir $HOME/install
./configure --prefix=$HOME/install
make -j$(nproc)
sudo make install
```

### 2. Build TensorRT OSS Plugin

| DeepStream Release | TRT Version | TRT_OSS_CHECKOUT_TAG | Support |
| ------------------ | ----------- | -------------------- | ------- |
| 5.0                | TRT 7.0.0   | release/7.0          | No      |
| 5.0.1              | TRT 7.0.0   | release/7.0          | No      |
| 5.1                | TRT 7.2.X   | 21.03                | No      |
| 6.0 EA             | TRT 7.2.2   | 21.03                | No      |
| 6.0 GA             | TRT 8.0.1   | release/8.0          | No      |
| 6.0.1              | TRT 8.2.1   | release/8.2          | Yes     |
| 6.1                | TRT 8.2.5.1 | release/8.2          | Yes     |

```
git clone -b release/8.2 https://github.com/nvidia/TensorRT
cd TensorRT/
git submodule update --init --recursive
export TRT_SOURCE=`pwd`
cd $TRT_SOURCE
mkdir -p build && cd build
## NOTE: as mentioned above, please make sure your GPU_ARCHS in TRT OSS CMakeLists.txt
## if GPU_ARCHS is not in TRT OSS CMakeLists.txt, add -DGPU_ARCHS=xy as below, for xy, refer to below "How to Get GPU_ARCHS" section
$HOME/install/bin/cmake .. -DGPU_ARCHS=xy  -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu/ -DCMAKE_C_COMPILER=/usr/bin/gcc -DTRT_BIN_DIR=`pwd`/out
make nvinfer_plugin -j$(nproc)
```

After building ends successfully, libnvinfer_plugin.so\* will be generated under `pwd`/out/ or ./build.

### 3. Replace "libnvinfer_plugin.so\*"

```
// backup original libnvinfer_plugin.so.x.y, e.g. libnvinfer_plugin.so.8.0.0
sudo mv /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8.p.q ${HOME}/libnvinfer_plugin.so.8.p.q.bak
// only replace the real file, don't touch the link files, e.g. libnvinfer_plugin.so, libnvinfer_plugin.so.8
sudo cp $TRT_SOURCE/`pwd`/out/libnvinfer_plugin.so.8.m.n  /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8.p.q
sudo ldconfig
```

## How to Get GPU_ARCHS

Can use either method to get GPU_ARCHs

1. GPU_ARCHS value can be got by "deviceQuery" CUDA sample

```
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```

2. If there is not "/usr/local/cuda/samples" in your system, you could use the deviceQuery.cpp in this folder,

```
nvcc deviceQuery.cpp -o deviceQuery
./deviceQuery
```

There will be output like below, which indicates the "GPU_ARCHS" is **75**.

```
./deviceQuery

Detected 2 CUDA Capable device(s)

Device 0: "Tesla T4"
  CUDA Driver Version / Runtime Version          10.2 / 10.2
  CUDA Capability Major/Minor version number:    7.5
```

# Acknowledgments

- [NNDam/TensorRT-CPP](https://github.com/NNDam/TensorRT-CPP)
- [MMCVRoIAlign](https://github.com/open-mmlab/mmdeploy)
