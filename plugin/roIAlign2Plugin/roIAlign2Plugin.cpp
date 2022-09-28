/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "roIAlign2Plugin.h"
#include "NvInfer.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::RoIAlign2DynamicPlugin;
using nvinfer1::plugin::RoIAlign2BasePluginCreator;
using nvinfer1::plugin::RoIAlign2DynamicPluginCreator;

// plugin specific constants
namespace
{
static const char* ROIALIGN2_PLUGIN_VERSION{"1"};
static const char* ROIALIGN2_PLUGIN_NAME{"RoIAlign2Dynamic_TRT"};
} // namespace

// Static class fields initialization
PluginFieldCollection RoIAlign2BasePluginCreator::mFC{};
std::vector<PluginField> RoIAlign2BasePluginCreator::mPluginAttributes;

// Helper function for serializing plugin
template <typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

RoIAlign2DynamicPlugin::RoIAlign2DynamicPlugin(const std::string name, int pooledSize, int transformCoords,
    bool absCoords, bool swapCoords, int samplingRatio, bool legacy, int imageSize)
    : mLayerName(name)
    , mPooledSize(pooledSize)
    , mImageSize(imageSize)
    , mTransformCoords(transformCoords)
    , mAbsCoords(absCoords)
    , mSwapCoords(swapCoords)
    , mSamplingRatio(samplingRatio)
    , mIsLegacy(legacy)
{
    ASSERT(pooledSize >= 1);
    ASSERT(samplingRatio >= 0);
}

RoIAlign2DynamicPlugin::RoIAlign2DynamicPlugin(const std::string name, int pooledSize, int transformCoords,
    bool absCoords, bool swapCoords, int samplingRatio, bool legacy, int imageSize, int featureLength, int roiCount,
    int inputWidth, int inputHeight)
    : mLayerName(name)
    , mPooledSize(pooledSize)
    , mImageSize(imageSize)
    , mTransformCoords(transformCoords)
    , mAbsCoords(absCoords)
    , mSwapCoords(swapCoords)
    , mSamplingRatio(samplingRatio)
    , mIsLegacy(legacy)
    , mFeatureLength(featureLength)
    , mROICount(roiCount)
    , mInputWidth(inputWidth)
    , mInputHeight(inputHeight)
{
    ASSERT(pooledSize >= 1);
    ASSERT(samplingRatio >= 0);
}

RoIAlign2DynamicPlugin::RoIAlign2DynamicPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mPooledSize = readFromBuffer<int>(d);
    mImageSize = readFromBuffer<int>(d);
    mTransformCoords = readFromBuffer<int>(d);
    mAbsCoords = readFromBuffer<int>(d);
    mSwapCoords = readFromBuffer<int>(d);
    mSamplingRatio = readFromBuffer<int>(d);
    mIsLegacy = readFromBuffer<int>(d);
    mFeatureLength = readFromBuffer<int>(d);
    mROICount = readFromBuffer<int>(d);
    mInputWidth = readFromBuffer<int>(d);
    mInputHeight = readFromBuffer<int>(d);
    ASSERT(d == a + length);
}

RoIAlign2DynamicPlugin::~RoIAlign2DynamicPlugin() noexcept {}

const char* RoIAlign2DynamicPlugin::getPluginType() const noexcept
{
    return ROIALIGN2_PLUGIN_NAME;
}

const char* RoIAlign2DynamicPlugin::getPluginVersion() const noexcept
{
    return ROIALIGN2_PLUGIN_VERSION;
}

int RoIAlign2DynamicPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DimsExprs RoIAlign2DynamicPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    // Validate input arguments
    ASSERT(outputIndex == 0);
    ASSERT(nbInputs == 2);

    // Shape of feature_map input should be
    // Constant shape: [batch_size, C, W, H] or Dynamic shape: some dimension values may be -1
    ASSERT(inputs[0].nbDims == 4);

    // Shape of roi input should be
    // Constant shape: [batch_size, R, 4] or Dynamic shape: some dimension values may be -1
    ASSERT(inputs[1].nbDims == 3);

    DimsExprs out_dim;
    out_dim.nbDims = 5;
    out_dim.d[0] = inputs[0].d[0];
    // roiCount
    out_dim.d[1] = inputs[1].d[1];
    // featureLength
    out_dim.d[2] = inputs[0].d[1];
    // height
    out_dim.d[3] = exprBuilder.constant(mPooledSize);
    // width
    out_dim.d[4] = exprBuilder.constant(mPooledSize);
    return out_dim;
}

int RoIAlign2DynamicPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

size_t RoIAlign2DynamicPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int RoIAlign2DynamicPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int batchSize = inputDesc[0].dims.d[0];

    const xy_t layerDims = {mInputHeight, mInputWidth};

    const void* feat = inputs[0];
    const void* rois = inputs[1];
    void* output = outputs[0];

    cudaError_t status;

    // Support legacy UFF mode
    if (mIsLegacy)
    {
        // Legacy values
        mTransformCoords = -1;
        mSwapCoords = true;
        mAbsCoords = false;
        mSamplingRatio = 1;
        status = roiAlign(stream, batchSize, mImageSize, mFeatureLength, mROICount, mTransformCoords, mAbsCoords,
            mSwapCoords, mSamplingRatio, rois, feat, layerDims, output, mPooledSize);
    }
    else
    {
        status = roiAlign(stream, batchSize, mImageSize, mFeatureLength, mROICount, mTransformCoords, mAbsCoords,
            mSwapCoords, mSamplingRatio, rois, feat, layerDims, output, mPooledSize);
    }
    ASSERT(status == cudaSuccess)
    return status;
}

size_t RoIAlign2DynamicPlugin::getSerializationSize() const noexcept
{
    return 11 * sizeof(int);
}

void RoIAlign2DynamicPlugin::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    writeToBuffer<int>(d, mPooledSize);
    writeToBuffer<int>(d, mImageSize);
    writeToBuffer<int>(d, mTransformCoords);
    writeToBuffer<int>(d, mAbsCoords);
    writeToBuffer<int>(d, mSwapCoords);
    writeToBuffer<int>(d, mSamplingRatio);
    writeToBuffer<int>(d, mIsLegacy);
    writeToBuffer<int>(d, mFeatureLength);
    writeToBuffer<int>(d, mROICount);
    writeToBuffer<int>(d, mInputWidth);
    writeToBuffer<int>(d, mInputHeight);
    ASSERT(d == a + getSerializationSize());
}

bool RoIAlign2DynamicPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    // 2 inputs, 1 outputs, so 3 input/output in total
    ASSERT(0 <= pos && pos < 3);
    const auto* in = inOut;
    const auto* out = inOut + nbInputs;
    const bool consistentFloatPrecision = (in[0].type == in[pos].type);
    switch (pos)
    {
    case 0: return in[0].type == DataType::kFLOAT && in[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 1: return in[1].type == DataType::kFLOAT && in[1].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 2:
        return out[0].type == DataType::kFLOAT && out[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    }
    return false;
}

void RoIAlign2DynamicPlugin::terminate() noexcept {}

void RoIAlign2DynamicPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt* RoIAlign2DynamicPlugin::clone() const noexcept
{
    auto plugin = new RoIAlign2DynamicPlugin(mLayerName, mPooledSize, mTransformCoords, mAbsCoords, mSwapCoords,
        mSamplingRatio, mIsLegacy, mImageSize, mFeatureLength, mROICount, mInputWidth, mInputHeight);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void RoIAlign2DynamicPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* RoIAlign2DynamicPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

DataType RoIAlign2DynamicPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    ASSERT(index == 0);
    return inputTypes[0];
}

void RoIAlign2DynamicPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    ASSERT(nbInputs == 2);
    ASSERT(nbOutputs == 1);

    mFeatureLength = in[0].desc.dims.d[1];
    mInputHeight = in[0].desc.dims.d[2];
    mInputWidth = in[0].desc.dims.d[3];
    mROICount = in[1].desc.dims.d[1];
}

RoIAlign2BasePluginCreator::RoIAlign2BasePluginCreator() noexcept
{
    mPluginAttributes.clear();

    mPluginAttributes.emplace_back(PluginField("pooled_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("image_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("roi_coords_absolute", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("roi_coords_swap", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("roi_coords_transform", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("sampling_ratio", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("legacy", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

RoIAlign2DynamicPluginCreator::RoIAlign2DynamicPluginCreator() noexcept
{
    mPluginName = ROIALIGN2_PLUGIN_NAME;
}

const char* RoIAlign2BasePluginCreator::getPluginName() const noexcept
{
    return mPluginName.c_str();
}

const char* RoIAlign2BasePluginCreator::getPluginVersion() const noexcept
{
    return ROIALIGN2_PLUGIN_VERSION;
}

const PluginFieldCollection* RoIAlign2BasePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* RoIAlign2DynamicPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;

    // Default values for the plugin creator, these will be used when the corresponding
    // plugin field is not passed, allowing to have defaults for "optional" ONNX attributes.
    int pooledSize = 7;
    int transformCoords = 2;
    bool absCoords = true;
    bool swapCoords = false;
    bool legacy = false;
    int samplingRatio = 0;
    int imageSize = 640;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attrName = fields[i].name;

        if (!strcmp(attrName, "pooled_size"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            pooledSize = *(static_cast<int const*>(fields[i].data));
            ASSERT(pooledSize >= 1);
        }
        if (!strcmp(attrName, "image_size"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            imageSize = *static_cast<int const*>(fields[i].data);
        }
        if (!strcmp(attrName, "roi_coords_absolute"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            absCoords = *(static_cast<int const*>(fields[i].data));
        }
        if (!strcmp(attrName, "roi_coords_swap"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            swapCoords = *(static_cast<int const*>(fields[i].data));
        }
        if (!strcmp(attrName, "roi_coords_transform"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            transformCoords = *(static_cast<int const*>(fields[i].data));
        }
        if (!strcmp(attrName, "sampling_ratio"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            samplingRatio = *(static_cast<int const*>(fields[i].data));
            ASSERT(samplingRatio >= 0);
        }
        if (!strcmp(attrName, "legacy"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            legacy = *(static_cast<int const*>(fields[i].data));
        }
    }

    IPluginV2DynamicExt* plugin = new RoIAlign2DynamicPlugin(
        name, pooledSize, transformCoords, absCoords, swapCoords, samplingRatio, legacy, imageSize);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2DynamicExt* RoIAlign2DynamicPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed,
    IPluginV2DynamicExt* plugin = new RoIAlign2DynamicPlugin(name, serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
