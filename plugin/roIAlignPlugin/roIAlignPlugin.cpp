/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "roIAlignPlugin.h"
#include "NvInfer.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdio.h>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::RoIAlignDynamicPlugin;
using nvinfer1::plugin::RoIAlignBasePluginCreator;
using nvinfer1::plugin::RoIAlignDynamicPluginCreator;

// plugin specific constants
namespace
{
static const char* PROPOSAL_PLUGIN_VERSION{"1"};
static const char* PROPOSAL_PLUGIN_NAMES[] = {"ROIAlign_TRT"};
static const float RPN_STD_SCALING{1.0f};
} // namespace

// Static class fields initialization
PluginFieldCollection RoIAlignBasePluginCreator::mFC{};
std::vector<PluginField> RoIAlignBasePluginCreator::mPluginAttributes;

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

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8: return 1;
    default: throw std::runtime_error("Invalid DataType.");
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

RoIAlignDynamicPlugin::RoIAlignDynamicPlugin(const std::string name, int outWidth, int outHeight, float spatialScale,
    int sampleRatio, int poolMode, bool aligned)
    : mLayerName(name)
    , mOutWidth(outWidth)
    , mOutHeight(outHeight)
    , mSpatialScale(spatialScale)
    , mSampleRatio(sampleRatio)
    , mPoolMode(poolMode)
    , mAligned(aligned)
{
}

RoIAlignDynamicPlugin::RoIAlignDynamicPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mOutWidth = readFromBuffer<int>(d);
    mOutHeight = readFromBuffer<int>(d);
    mSpatialScale = readFromBuffer<float>(d);
    mSampleRatio = readFromBuffer<int>(d);
    mPoolMode = readFromBuffer<int>(d);
    mAligned = readFromBuffer<int>(d);
    assert(d == a + length);
}

RoIAlignDynamicPlugin::~RoIAlignDynamicPlugin() noexcept {}

const char* RoIAlignDynamicPlugin::getPluginType() const noexcept
{
    return PROPOSAL_PLUGIN_NAMES[0];
}

const char* RoIAlignDynamicPlugin::getPluginVersion() const noexcept
{
    return PROPOSAL_PLUGIN_VERSION;
}

int RoIAlignDynamicPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DimsExprs RoIAlignDynamicPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    // Validate input arguments
    ASSERT(outputIndex == 0);
    ASSERT(nbInputs == 2);
    ASSERT(inputs[0].nbDims == 4);
    ASSERT(inputs[1].nbDims == 2);
    DimsExprs out_dim;
    out_dim.nbDims = 4;
    out_dim.d[0] = inputs[1].d[0];
    out_dim.d[1] = inputs[0].d[1];
    out_dim.d[2] = exprBuilder.constant(mOutHeight);
    out_dim.d[3] = exprBuilder.constant(mOutWidth);
    return out_dim;
}

int RoIAlignDynamicPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

size_t RoIAlignDynamicPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    size_t output_size = 0;
    size_t word_size = 0;
    switch (mPoolMode)
    {
    case 0: // max
        output_size = outputs[0].dims.d[0] * outputs[0].dims.d[1] * outputs[0].dims.d[2] * outputs[0].dims.d[3];
        word_size = getElementSize(outputs[0].type);
        return output_size * word_size * 2;
        break;
    case 1: return 0; break;
    default: return 0;
    }
    return 0;
}

int RoIAlignDynamicPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // TODO:
    // int status = -1;
    // // Our plugin outputs only one tensor
    // void* output = outputs[0];
    // int batchSize = inputDesc[0].dims.d[0];
    // status = proposalInference_gpu(stream, inputs[0], inputs[1], batchSize, mInputHeight, mInputWidth, mRpnHeight,
    //     mRpnWidth, mMaxBoxNum, mPreNmsTopN, &mAnchorSizes[0], mAnchorSizeNum, &mAnchorRatios[0], mAnchorRatioNum,
    //     mRpnStdScaling, mRpnStride, mBboxMinSize, mNmsIouThreshold, workspace, output);
    // ASSERT(status == STATUS_SUCCESS);
    // return status;

    int channels = inputDesc[0].dims.d[1];
    int height = inputDesc[0].dims.d[2];
    int width = inputDesc[0].dims.d[3];

    int output_size
        = outputDesc[0].dims.d[0] * outputDesc[0].dims.d[1] * outputDesc[0].dims.d[2] * outputDesc[0].dims.d[3];
    int word_size = getElementSize(outputDesc[0].type);

    const void* feat = inputs[0];
    const void* rois = inputs[1];
    void* output = outputs[0];
    void* argmax_y = nullptr;
    void* argmax_x = nullptr;

    switch (mPoolMode)
    {
    case 0: // max
        argmax_y = workspace;
        argmax_x = argmax_y + output_size * word_size;
        break;
    case 1: // avg
        break;
    }

    switch (outputDesc[0].type)
    {
    case DataType::kFLOAT:
        TRTRoIAlignForwardCUDAKernelLauncher_float((const float*) feat, (const float*) rois, (float*) output,
            (float*) argmax_y, (float*) argmax_x, output_size, channels, height, width, mOutHeight, mOutWidth,
            mSpatialScale, mSampleRatio, mPoolMode, mAligned, stream);
        break;
    case DataType::kHALF:
        // TODO:
        break;
    default: break;
    }

    return 0;
}

size_t RoIAlignDynamicPlugin::getSerializationSize() const noexcept
{
    return sizeof(mOutWidth) + sizeof(mOutHeight) + sizeof(mSpatialScale) + sizeof(mSampleRatio) + sizeof(mPoolMode)
        + sizeof(mAligned);
}

void RoIAlignDynamicPlugin::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    writeToBuffer<int>(d, mOutWidth);
    writeToBuffer<int>(d, mOutHeight);
    writeToBuffer<float>(d, mSpatialScale);
    writeToBuffer<int>(d, mSampleRatio);
    writeToBuffer<int>(d, mPoolMode);
    writeToBuffer<int>(d, mAligned);
    assert(d == a + getSerializationSize());
}

bool RoIAlignDynamicPlugin::supportsFormatCombination(
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

void RoIAlignDynamicPlugin::terminate() noexcept {}

void RoIAlignDynamicPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt* RoIAlignDynamicPlugin::clone() const noexcept
{
    auto* plugin = new RoIAlignDynamicPlugin(
        mLayerName, mOutWidth, mOutHeight, mSpatialScale, mSampleRatio, mPoolMode, mAligned);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void RoIAlignDynamicPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* RoIAlignDynamicPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

DataType RoIAlignDynamicPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[0];
}

void RoIAlignDynamicPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    ASSERT(nbInputs == 2);
    ASSERT(nbOutputs == 1);
}

RoIAlignBasePluginCreator::RoIAlignBasePluginCreator() noexcept
{
    mPluginAttributes.clear();

    mPluginAttributes.emplace_back(PluginField("output_width", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("output_height", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("spatial_scale", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("sampling_ratio", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("mode", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("aligned", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

RoIAlignDynamicPluginCreator::RoIAlignDynamicPluginCreator() noexcept
{
    mPluginName = PROPOSAL_PLUGIN_NAMES[0];
}

const char* RoIAlignBasePluginCreator::getPluginName() const noexcept
{
    return mPluginName.c_str();
}

const char* RoIAlignBasePluginCreator::getPluginVersion() const noexcept
{
    return PROPOSAL_PLUGIN_VERSION;
}

const PluginFieldCollection* RoIAlignBasePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* RoIAlignDynamicPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept
{

    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;

    int outWidth = 7;
    int outHeight = 7;
    float spatialScale = 1.0;
    int sampleRatio = 0;
    int poolMode = -1;
    bool aligned = true;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;

        if (!strcmp(attr_name, "output_height"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            outHeight = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attr_name, "output_width"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            outWidth = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attr_name, "spatial_scale"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            spatialScale = *(static_cast<const float*>(fields[i].data));
        }
        else if (!strcmp(attr_name, "sampling_ratio"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            sampleRatio = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attr_name, "mode"))
        {
            int data_size = fc->fields[i].length;
            const char* data_start = static_cast<const char*>(fc->fields[i].data);
            std::string poolModeStr(data_start, data_size);
            if (poolModeStr == "avg")
            {
                poolMode = 1;
            }
            else if (poolModeStr == "max")
            {
                poolMode = 0;
            }
            else
            {
                std::cout << "Unknown pool mode \"" << poolModeStr << "\"." << std::endl;
            }
            assert(poolMode >= 0);
        }
        else if (!strcmp(attr_name, "aligned"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            int aligned_int = *(static_cast<const int*>(fields[i].data));
            aligned = aligned_int != 0;
        }
    }

    ASSERT(outHeight > 0 && outWidth > 0 && spatialScale > 0.0f && poolMode >= 0);

    IPluginV2DynamicExt* plugin
        = new RoIAlignDynamicPlugin(name, outWidth, outHeight, spatialScale, sampleRatio, poolMode, aligned);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2DynamicExt* RoIAlignDynamicPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed,
    IPluginV2DynamicExt* plugin = new RoIAlignDynamicPlugin(name, serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
