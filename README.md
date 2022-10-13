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

- Deepstream 6.0.1 or Deepstream 6.1

## Install

Follow guide from [github.com/NVIDIA-AI-IOT/deepstream_tao_apps](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/blob/master/TRT-OSS/x86/README.md)

# Acknowledgments

- [NNDam/TensorRT-CPP](https://github.com/NNDam/TensorRT-CPP)
- [MMCVRoIAlign](https://github.com/open-mmlab/mmdeploy)
