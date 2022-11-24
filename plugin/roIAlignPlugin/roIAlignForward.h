#ifndef TRT_ROI_ALIGN_HELPER_H
#define TRT_ROI_ALIGN_HELPER_H

#include <float.h>

#include "plugin.h"
using namespace nvinfer1;
using namespace nvinfer1::plugin;

template <typename scalar_t>
void TRTRoIAlignForwardCUDAKernelLauncher(scalar_t const* input, scalar_t const* rois, scalar_t* output,
    scalar_t* argmax_y, scalar_t* argmax_x, int output_size, int channels, int height, int width, int aligned_height,
    int aligned_width, scalar_t spatial_scale, int sampling_ratio, int pool_mode, bool aligned, cudaStream_t stream);

void TRTRoIAlignForwardCUDAKernelLauncher_float(float const* input, float const* rois, float* output, float* argmax_y,
    float* argmax_x, int output_size, int channels, int height, int width, int aligned_height, int aligned_width,
    float spatial_scale, int sampling_ratio, int pool_mode, bool aligned, cudaStream_t stream);

#endif