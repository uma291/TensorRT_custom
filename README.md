# TensorRT custom plugin

Just add some new custom tensorRT plugin

## New plugin

- BatchedNMSLandmark_TRT, BatchedNMSLandmarkDynamic_TRT: Batched NMS with face landmark
- BatchedNMSLandmarkConf_TRT, BatchedNMSLandmarkConfDynamic_TRT: Batched NMS with face lanmdark & confidence
- RoIAlignDynamic: Same ONNX RoIAlign, copy from [MMCVRoIAlign](https://github.com/open-mmlab/mmdeploy)

## Prerequisites

- Deepstream 6.0.1

## Install

Follow guide from [github.com/NVIDIA-AI-IOT/deepstream_tao_apps](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/blob/master/TRT-OSS/x86/README.md)

# Acknowledgments

- [NNDam/TensorRT-CPP](https://github.com/NNDam/TensorRT-CPP)
- [MMCVRoIAlign](https://github.com/open-mmlab/mmdeploy)
