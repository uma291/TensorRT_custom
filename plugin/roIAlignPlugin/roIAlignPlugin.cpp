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
static char const* ROIALIGN_PLUGIN_VERSION{"1"};
static char const* ROIALIGN_PLUGIN_NAME{"RoIAlignDynamic_TRT"};
} // namespace

// Static class fields initialization
PluginFieldCollection RoIAlignBasePluginCreator::mFC{};
std::vector<PluginField> RoIAlignBasePluginCreator::mPluginAttributes;

// Helper function for serializing plugin
template <typename T>
void writeToBuffer(char*& buffer, T const& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
T readFromBuffer(char const*& buffer)
{
    T val = *reinterpret_cast<T const*>(buffer);
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

RoIAlignDynamicPlugin::RoIAlignDynamicPlugin(std::string const name, int outWidth, int outHeight, float spatialScale,
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

RoIAlignDynamicPlugin::RoIAlignDynamicPlugin(std::string const name, void const* data, size_t length)
    : mLayerName(name)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    mOutWidth = readFromBuffer<int>(d);
    mOutHeight = readFromBuffer<int>(d);
    mSpatialScale = readFromBuffer<float>(d);
    mSampleRatio = readFromBuffer<int>(d);
    mPoolMode = readFromBuffer<int>(d);
    mAligned = readFromBuffer<int>(d);
    PLUGIN_VALIDATE(d == a + length);
}

RoIAlignDynamicPlugin::~RoIAlignDynamicPlugin() noexcept {}

char const* RoIAlignDynamicPlugin::getPluginType() const noexcept
{
    return ROIALIGN_PLUGIN_NAME;
}

char const* RoIAlignDynamicPlugin::getPluginVersion() const noexcept
{
    return ROIALIGN_PLUGIN_VERSION;
}

int RoIAlignDynamicPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DimsExprs RoIAlignDynamicPlugin::getOutputDimensions(
    int outputIndex, DimsExprs const* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    // Validate input arguments
    PLUGIN_VALIDATE(outputIndex == 0);
    PLUGIN_VALIDATE(nbInputs == 2);

    // Shape of feature_map input should be
    // Constant shape: [batch_size, C, W, H] or Dynamic shape: some dimension values may be -1
    PLUGIN_VALIDATE(inputs[0].nbDims == 4);

    // Shape of roi input should be
    // Constant shape: [R, 5] or Dynamic shape: some dimension values may be -1
    PLUGIN_VALIDATE(inputs[1].nbDims == 2);

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
    PluginTensorDesc const* inputs, int nbInputs, PluginTensorDesc const* outputs, int nbOutputs) const noexcept
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

int RoIAlignDynamicPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int channels = inputDesc[0].dims.d[1];
    int height = inputDesc[0].dims.d[2];
    int width = inputDesc[0].dims.d[3];

    int output_size
        = outputDesc[0].dims.d[0] * outputDesc[0].dims.d[1] * outputDesc[0].dims.d[2] * outputDesc[0].dims.d[3];
    int word_size = getElementSize(outputDesc[0].type);

    void const* feat = inputs[0];
    void const* rois = inputs[1];
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
        TRTRoIAlignForwardCUDAKernelLauncher_float((float const*) feat, (float const*) rois, (float*) output,
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
    return 5 * sizeof(int) + 1 * sizeof(float);
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
    PLUGIN_VALIDATE(d == a + getSerializationSize());
}

bool RoIAlignDynamicPlugin::supportsFormatCombination(
    int pos, PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    // 2 inputs, 1 outputs, so 3 input/output in total
    PLUGIN_VALIDATE(0 <= pos && pos < 3);
    auto const* in = inOut;
    auto const* out = inOut + nbInputs;
    bool const consistentFloatPrecision = (in[0].type == in[pos].type);
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

void RoIAlignDynamicPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* RoIAlignDynamicPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

DataType RoIAlignDynamicPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[0];
}

void RoIAlignDynamicPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int nbInputs, DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    PLUGIN_VALIDATE(nbInputs == 2);
    PLUGIN_VALIDATE(nbOutputs == 1);
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
    mPluginName = ROIALIGN_PLUGIN_NAME;
}

char const* RoIAlignBasePluginCreator::getPluginName() const noexcept
{
    return mPluginName.c_str();
}

char const* RoIAlignBasePluginCreator::getPluginVersion() const noexcept
{
    return ROIALIGN_PLUGIN_VERSION;
}

PluginFieldCollection const* RoIAlignBasePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* RoIAlignDynamicPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    int nbFields = fc->nbFields;

    int outWidth = 7;
    int outHeight = 7;
    float spatialScale = 1.0;
    int sampleRatio = 0;
    int poolMode = -1;
    bool aligned = true;

    for (int i = 0; i < nbFields; ++i)
    {
        char const* attr_name = fields[i].name;

        if (!strcmp(attr_name, "output_height"))
        {
            PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
            outHeight = *(static_cast<int const*>(fields[i].data));
        }
        else if (!strcmp(attr_name, "output_width"))
        {
            PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
            outWidth = *(static_cast<int const*>(fields[i].data));
        }
        else if (!strcmp(attr_name, "spatial_scale"))
        {
            PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
            spatialScale = *(static_cast<float const*>(fields[i].data));
        }
        else if (!strcmp(attr_name, "sampling_ratio"))
        {
            PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
            sampleRatio = *(static_cast<int const*>(fields[i].data));
        }
        else if (!strcmp(attr_name, "mode"))
        {
            int data_size = fc->fields[i].length;
            char const* data_start = static_cast<char const*>(fc->fields[i].data);
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
            PLUGIN_VALIDATE(poolMode >= 0);
        }
        else if (!strcmp(attr_name, "aligned"))
        {
            PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
            int aligned_int = *(static_cast<int const*>(fields[i].data));
            aligned = aligned_int != 0;
        }
    }

    PLUGIN_VALIDATE(outHeight > 0 && outWidth > 0 && spatialScale > 0.0f && poolMode >= 0);

    IPluginV2DynamicExt* plugin
        = new RoIAlignDynamicPlugin(name, outWidth, outHeight, spatialScale, sampleRatio, poolMode, aligned);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2DynamicExt* RoIAlignDynamicPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed,
    IPluginV2DynamicExt* plugin = new RoIAlignDynamicPlugin(name, serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
