#include "roIAlignForward.h"

#define THREADS_PER_BLOCK 512

#define CUDA_1D_KERNEL_LOOP(i, n)                                                                                      \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

inline int GET_BLOCKS(int const N)
{
    int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int max_block_num = 4096;
    return min(optimal_block_num, max_block_num);
}

template <typename T>
__device__ T bilinear_interpolate(
    T const* input, int const height, int const width, T y, T x, int const index /* index for debug only*/)
{
    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width)
        return 0;

    if (y <= 0)
        y = 0;
    if (x <= 0)
        x = 0;

    int y_low = (int) y;
    int x_low = (int) x;
    int y_high;
    int x_high;

    if (y_low >= height - 1)
    {
        y_high = y_low = height - 1;
        y = (T) y_low;
    }
    else
    {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1)
    {
        x_high = x_low = width - 1;
        x = (T) x_low;
    }
    else
    {
        x_high = x_low + 1;
    }

    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1. - ly, hx = 1. - lx;
    // do bilinear interpolation
    T v1 = input[y_low * width + x_low];
    T v2 = input[y_low * width + x_high];
    T v3 = input[y_high * width + x_low];
    T v4 = input[y_high * width + x_high];
    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    return val;
}

template <typename T>
__global__ void roi_align_forward_cuda_kernel(int const nthreads, T const* input, T const* rois, T* output, T* argmax_y,
    T* argmax_x, int const pooled_height, int const pooled_width, T const spatial_scale, int const sampling_ratio,
    int const pool_mode, // 0 - max pool, 1 - avg pool
    bool const aligned, int const channels, int const height, int const width)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;

        T const* offset_rois = rois + n * 5;
        int roi_batch_ind = offset_rois[0];

        // Do not using rounding; this implementation detail is critical
        T offset = aligned ? (T) 0.5 : (T) 0.0;
        T roi_start_w = offset_rois[1] * spatial_scale - offset;
        T roi_start_h = offset_rois[2] * spatial_scale - offset;
        T roi_end_w = offset_rois[3] * spatial_scale - offset;
        T roi_end_h = offset_rois[4] * spatial_scale - offset;

        T roi_width = roi_end_w - roi_start_w;
        T roi_height = roi_end_h - roi_start_h;
        if (!aligned)
        { // for backward-compatibility only
            roi_width = max(roi_width, (T) 1.);
            roi_height = max(roi_height, (T) 1.);
        }

        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

        T const* offset_input = input + (roi_batch_ind * channels + c) * height * width;

        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h
            = (sampling_ratio > 0) ? sampling_ratio : static_cast<int>(ceilf(roi_height / pooled_height));
        int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : static_cast<int>(ceilf(roi_width / pooled_width));

        if (pool_mode == 0)
        {
            // We do max pooling inside a bin
            T maxval = -FLT_MAX;
            T maxidx_y = -1.f, maxidx_x = -1.f;
            for (int iy = 0; iy < roi_bin_grid_h; iy++)
            {
                T const y = roi_start_h + ph * bin_size_h
                    + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h);
                for (int ix = 0; ix < roi_bin_grid_w; ix++)
                {
                    T const x = roi_start_w + pw * bin_size_w
                        + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);
                    T val = bilinear_interpolate(offset_input, height, width, y, x, index);
                    if (val > maxval)
                    {
                        maxval = val;
                        maxidx_y = y;
                        maxidx_x = x;
                    }
                }
            }
            output[index] = maxval;
            argmax_y[index] = maxidx_y;
            argmax_x[index] = maxidx_x;
        }
        else if (pool_mode == 1)
        {
            // We do average pooling inside a bin
            T const count = max(roi_bin_grid_h * roi_bin_grid_w, 1);
            T output_val = 0.;
            for (int iy = 0; iy < roi_bin_grid_h; iy++)
            {
                T const y = roi_start_h + ph * bin_size_h
                    + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h);
                for (int ix = 0; ix < roi_bin_grid_w; ix++)
                {
                    T const x = roi_start_w + pw * bin_size_w
                        + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);
                    T val = bilinear_interpolate(offset_input, height, width, y, x, index);
                    output_val += val;
                }
            }
            output[index] = output_val / count;
        }
    }
}

template <typename scalar_t>
void TRTRoIAlignForwardCUDAKernelLauncher(scalar_t const* input, scalar_t const* rois, scalar_t* output,
    scalar_t* argmax_y, scalar_t* argmax_x, int output_size, int channels, int height, int width, int aligned_height,
    int aligned_width, scalar_t spatial_scale, int sampling_ratio, int pool_mode, bool aligned, cudaStream_t stream)
{
    roi_align_forward_cuda_kernel<scalar_t><<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(output_size,
        input, rois, output, argmax_y, argmax_x, aligned_height, aligned_width, static_cast<scalar_t>(spatial_scale),
        sampling_ratio, pool_mode, aligned, channels, height, width);
}

void TRTRoIAlignForwardCUDAKernelLauncher_float(float const* input, float const* rois, float* output, float* argmax_y,
    float* argmax_x, int output_size, int channels, int height, int width, int aligned_height, int aligned_width,
    float spatial_scale, int sampling_ratio, int pool_mode, bool aligned, cudaStream_t stream)
{
    TRTRoIAlignForwardCUDAKernelLauncher<float>(input, rois, output, argmax_y, argmax_x, output_size, channels, height,
        width, aligned_height, aligned_width, spatial_scale, sampling_ratio, pool_mode, aligned, stream);
}
