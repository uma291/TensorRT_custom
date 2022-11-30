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

#ifndef ROIALIGN_PLUGIN_H
#define ROIALIGN_PLUGIN_H

#include "NvInferPlugin.h"
#include "common/kernel.h"
#include "common/plugin.h"
#include "roIAlignForward.h"
#include <string>
#include <vector>

using namespace nvinfer1::plugin;

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2Ext and BaseCreator classes.
// For requirements for overriden functions, check TensorRT API docs.
namespace nvinfer1
{
namespace plugin
{

class RoIAlignDynamicPlugin : public IPluginV2DynamicExt
{
public:
    RoIAlignDynamicPlugin(std::string const name);

    RoIAlignDynamicPlugin(std::string const name, int outWidth, int outHeight, float spatialScale, int sampleRatio,
        int poolMode, bool aligned);

    RoIAlignDynamicPlugin(std::string const name, void const* data, size_t length);

    // It doesn't make sense to make RoIAlignDynamicPlugin without arguments, so we delete default constructor.
    RoIAlignDynamicPlugin() noexcept = delete;

    ~RoIAlignDynamicPlugin() noexcept override;

    // IPluginV2 methods
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* libNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

    // IPluginV2Ext methods
    DataType getOutputDataType(int index, nvinfer1::DataType const* inputType, int nbInputs) const noexcept override;

    // IPluginV2DynamicExt methods
    IPluginV2DynamicExt* clone() const noexcept override;
    DimsExprs getOutputDimensions(
        int outputIndex, DimsExprs const* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(DynamicPluginTensorDesc const* in, int nbInputs, DynamicPluginTensorDesc const* out,
        int nbOutputs) noexcept override;
    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int nbInputs, PluginTensorDesc const* outputs,
        int nbOutputs) const noexcept override;
    int enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

private:
    std::string const mLayerName;
    std::string mNamespace;
    int mOutWidth;
    int mOutHeight;
    float mSpatialScale;
    int mSampleRatio;
    int mPoolMode; // 1:avg 0:max
    bool mAligned;
};

class RoIAlignBasePluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    RoIAlignBasePluginCreator() noexcept;
    ~RoIAlignBasePluginCreator() noexcept override = default;
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    PluginFieldCollection const* getFieldNames() noexcept override;

protected:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mPluginName;
};

class RoIAlignPluginCreator : public RoIAlignBasePluginCreator
{
public:
    RoIAlignPluginCreator() noexcept;
    ~RoIAlignPluginCreator() noexcept override = default;
    IPluginV2Ext* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;
    IPluginV2Ext* deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept override;
};

class RoIAlignDynamicPluginCreator : public RoIAlignBasePluginCreator
{
public:
    RoIAlignDynamicPluginCreator() noexcept;
    ~RoIAlignDynamicPluginCreator() noexcept override = default;
    IPluginV2DynamicExt* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;
    IPluginV2DynamicExt* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;
};

} // namespace plugin

} // namespace nvinfer1

#endif // ROIALIGN_PLUGIN_H
