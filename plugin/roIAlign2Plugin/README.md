# RoIAlign2Plugin

**Table Of Contents**
- [Changelog](#changelog)
- [Description](#description)
- [Structure](#structure)
- [Parameters](#parameters)
- [Compatibility Modes](#compatibility-modes)
- [Additional Resources](#additional-resources)
- [License](#license)
- [Known issues](#known-issues)

## Changelog

February 2022
Major refactoring of the plugin to add new features and compatibility modes.

June 2019
This is the first release of this `README.md` file.

## Description

The `RoIAlignP2lugin` plugin performs the ROIAlign operations on the output feature maps of an FPN (Feature Pyramid Network). This is used in many implementations of FasterRCNN and MaskRCNN. This operation is also known as ROIPooling.

## Structure

#### Inputs

This plugin works in NCHW format. It takes five input tensors:

- `feature_map` with shape `[N, C, 256, 256]`, usually corresponds to `P2`.

`rois` is the proposal ROI coordinates, these usually come from a Proposals layer or an NMS operation, such as the [EfficientNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) plugin. Its shape is `[N, R, 4]` where `N` is the batch_size, `R` is the number of ROI candidates and `4` is the number of coordinates.


#### Outputs

This plugin generates one output tensor of shape `[N, R, C, pooled_size, pooled_size]` where `C` is the same number of channels as the feature maps, and `pooled_size` is the configured height (and width) of the feature area after ROIAlign.

## Parameters

This plugin has the plugin creator class `RoIAlignPluginPluginCreator` and the plugin class `RoIAlignPl2ugin`.

The following parameters are used to create a `RoIAlignP2lugin` instance:

| Type    | Parameter              | Default    | Description
|---------|------------------------|------------|--------------------------------------------------------
| `int`   | `pooled_size`          | 7          | The spatial size of a feature area after ROIAlgin will be `[pooled_size, pooled_size]`
| `int[]` | `image_size`           | 1024,1024  | An 2-element array with the input image size of the entire network, in `[image_height, image_width]` layout
| `int`   | `sampling_ratio`       | 0          | If set to 1 or larger, the number of samples to take for each output element. If set to 0, this will be calculated adaptively by the size of the ROI.
| `int`   | `roi_coords_absolute`  | 1          | If set to 0, the ROIs are normalized in [0-1] range. If set to 1, the ROIs are in full image space.
| `int`   | `roi_coords_swap`      | 0          | If set to 0, the ROIs are in `[x1,y1,x2,y2]` format (PyTorch standard). If set to 1, they are in `[y1,x1,y2,x2]` format (TensorFlow standard).
| `int`   | `roi_coords_transform` | 2          | The coordinate transformation method to use for the ROI Align operation. If set to 2, `half_pixel` sampling will be performed. If set to 1, `output_half_pixel` will be performed. If set to 0, no pixel offset will be applied. More details on compatibility modes below.

## Compatibility Modes

There exist many implementations of FasterRCNN and MaskRCNN, and unfortunately, there is no consensus on a canonical way to execute the ROI Pooling of an FPN. This plugin attempts to support multiple common implementations, configurable via the various parameters that have been exposed.

#### Detectron 2

To replicate the standard ROI Pooling behavior of [Detectron 2](https://github.com/facebookresearch/detectron2), set the parameters as follows:

- `roi_coords_transform`: 2. This implementation uses half_pixel coordinate offsets.
- `roi_coords_swap`: 0. This implementation follows the PyTorch standard for coordinate layout.
- `roi_coords_absolute`: 1. This implementation works will full-size ROI coordinates.
- `sampling_ratio`: 0. This implementation uses an adaptive sampling ratio determined from each ROI area.

#### MaskRCNN Benchmark

To replicate the standard ROI Pooling behavior of [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), set the parameters as follows:

- `roi_coords_transform`: 1. This implementation uses output_half_pixel coordinate offsets.
- `roi_coords_swap`: 0. This implementation follows the PyTorch standard for coordinate layout.
- `roi_coords_absolute`: 1. This implementation works will full-size ROI coordinates.
- `sampling_ratio`: 2. This implementation performs two samples per output element.

#### Other Implementations

Other FPN ROI Pooling implementations may be adapted by having a better understanding of how the various parameters work internally.

**Coordinate Transformation**: This flag primarily defines various offsets applied to coordinates when performing the bilinear interpolation sampling for ROI Align. The three supported values work as follows:
- `roi_coords_transform` = -1: This is a back-compatibility that calculates the scale by subtracting one to both the input and output dimensions. This is similar to the `align_corners` resize method.
- `roi_coords_transform` = 0: This is a naive implementation where no pixel offset is applied anywhere. It is similar to the `asymmetric` resize method.
- `roi_coords_transform` = 1: This performs half pixel offset by applying a 0.5 offset only in the output element sampling. This is similar to the `output_half_pixel` ROI Align method.
- `roi_coords_transform` = 2: This performs half pixel offset by applying a 0.5 offset in the output element sampling, but also to the input map coordinate. This is similar to the `half_pixel` ROI Align method, and is the favored method of performing ROI Align.

## Additional Resources

The following resources provide a deeper understanding of the `RoIAlignP2lugin` plugin:

- [MaskRCNN](https://github.com/matterport/Mask_RCNN)
- [FPN](https://arxiv.org/abs/1612.03144)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


## Known issues

There are no known issues in this plugin.
