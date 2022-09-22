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

#include "efficientNMSLandmarkPlugin.h"
#include "efficientNMSLandmarkInference.h"

using namespace nvinfer1;
using nvinfer1::plugin::EfficientNMSLandmarkPlugin;
using nvinfer1::plugin::EfficientNMSLandmarkParameters;
using nvinfer1::plugin::EfficientNMSLandmarkPluginCreator;

namespace
{
const char* EFFICIENT_NMS_LANDMARK_PLUGIN_VERSION{"1"};
const char* EFFICIENT_NMS_LANDMARK_PLUGIN_NAME{"EfficientNMSLandmark_TRT"};
} // namespace

EfficientNMSLandmarkPlugin::EfficientNMSLandmarkPlugin(EfficientNMSLandmarkParameters param)
    : mParam(param)
{
}

EfficientNMSLandmarkPlugin::EfficientNMSLandmarkPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mParam = read<EfficientNMSLandmarkParameters>(d);
    ASSERT(d == a + length);
}

const char* EfficientNMSLandmarkPlugin::getPluginType() const noexcept
{
    return EFFICIENT_NMS_LANDMARK_PLUGIN_NAME;
}

const char* EfficientNMSLandmarkPlugin::getPluginVersion() const noexcept
{
    return EFFICIENT_NMS_LANDMARK_PLUGIN_VERSION;
}

int EfficientNMSLandmarkPlugin::getNbOutputs() const noexcept
{
    // Standard Plugin Implementation
    return 5;
}

int EfficientNMSLandmarkPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void EfficientNMSLandmarkPlugin::terminate() noexcept {}

size_t EfficientNMSLandmarkPlugin::getSerializationSize() const noexcept
{
    return sizeof(EfficientNMSLandmarkParameters);
}

void EfficientNMSLandmarkPlugin::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mParam);
    ASSERT(d == a + getSerializationSize());
}

void EfficientNMSLandmarkPlugin::destroy() noexcept
{
    delete this;
}

void EfficientNMSLandmarkPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
}

const char* EfficientNMSLandmarkPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

nvinfer1::DataType EfficientNMSLandmarkPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    // On standard NMS, num_detections and detection_classes use integer outputs
    if (index == 0 || index == 3)
    {
        return nvinfer1::DataType::kINT32;
    }
    // All others should use the same datatype as the input
    return inputTypes[0];
}

IPluginV2DynamicExt* EfficientNMSLandmarkPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new EfficientNMSLandmarkPlugin(mParam);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

DimsExprs EfficientNMSLandmarkPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        DimsExprs out_dim;

        // When pad per class is set, the output size may need to be reduced:
        // i.e.: outputBoxes = min(outputBoxes, outputBoxesPerClass * numClasses)
        // As the number of classes may not be static, numOutputBoxes must be a dynamic
        // expression. The corresponding parameter can not be set at this time, so the
        // value will be calculated again in configurePlugin() and the param overwritten.
        const IDimensionExpr* numOutputBoxes = exprBuilder.constant(mParam.numOutputBoxes);
        if (mParam.padOutputBoxesPerClass && mParam.numOutputBoxesPerClass > 0)
        {
            const IDimensionExpr* numOutputBoxesPerClass = exprBuilder.constant(mParam.numOutputBoxesPerClass);
            const IDimensionExpr* numClasses = inputs[1].d[2];
            numOutputBoxes = exprBuilder.operation(DimensionOperation::kMIN, *numOutputBoxes,
                *exprBuilder.operation(DimensionOperation::kPROD, *numOutputBoxesPerClass, *numClasses));
        }

        // Standard NMS
        ASSERT(outputIndex >= 0 && outputIndex <= 4);

        // num_detections
        if (outputIndex == 0)
        {
            out_dim.nbDims = 2;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = exprBuilder.constant(1);
        }
        // detection_boxes
        else if (outputIndex == 1)
        {
            out_dim.nbDims = 3;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = numOutputBoxes;
            out_dim.d[2] = exprBuilder.constant(4);
        }
        // detection_scores
        else if (outputIndex == 2)
        {
            out_dim.nbDims = 2;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = numOutputBoxes;
        }
        // detection_classes
        else if (outputIndex == 3)
        {
            out_dim.nbDims = 2;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = numOutputBoxes;
        }
        // detection_lmks
        else if (outputIndex == 4)
        {
            out_dim.nbDims = 3;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = numOutputBoxes;
            out_dim.d[2] = exprBuilder.constant(10);
        }

        return out_dim;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool EfficientNMSLandmarkPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (inOut[pos].format != PluginFormat::kLINEAR)
    {
        return false;
    }

    ASSERT(nbInputs == 3 || nbInputs == 4);
    ASSERT(nbOutputs == 5);
    if (nbInputs == 3)
    {
        ASSERT(0 <= pos && pos <= 7);
    }
    if (nbInputs == 4)
    {
        ASSERT(0 <= pos && pos <= 8);
    }

    // num_detections and detection_classes output: int
    const int posOut = pos - nbInputs;
    if (posOut == 0 || posOut == 3)
    {
        return inOut[pos].type == DataType::kINT32 && inOut[pos].format == PluginFormat::kLINEAR;
    }

    // all other inputs/outputs: fp32 or fp16
    return (inOut[pos].type == DataType::kHALF || inOut[pos].type == DataType::kFLOAT)
        && (inOut[0].type == inOut[pos].type);
}

void EfficientNMSLandmarkPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    try
    {
        // Accepts two or three inputs
        // If two inputs: [0] boxes, [1] scores
        // If three inputs: [0] boxes, [1] scores, [2] anchors
        ASSERT(nbInputs == 3 || nbInputs == 4);
        ASSERT(nbOutputs == 5);
        mParam.datatype = in[0].desc.type;

        // Shape of scores input should be
        // [batch_size, num_boxes, num_classes] or [batch_size, num_boxes, num_classes, 1]
        ASSERT(in[1].desc.dims.nbDims == 3 || (in[1].desc.dims.nbDims == 4 && in[1].desc.dims.d[3] == 1));
        mParam.numScoreElements = in[1].desc.dims.d[1] * in[1].desc.dims.d[2];
        mParam.numClasses = in[1].desc.dims.d[2];

        // When pad per class is set, the total ouput boxes size may need to be reduced.
        // This operation is also done in getOutputDimension(), but for dynamic shapes, the
        // numOutputBoxes param can't be set until the number of classes is fully known here.
        if (mParam.padOutputBoxesPerClass && mParam.numOutputBoxesPerClass > 0)
        {
            if (mParam.numOutputBoxesPerClass * mParam.numClasses < mParam.numOutputBoxes)
            {
                mParam.numOutputBoxes = mParam.numOutputBoxesPerClass * mParam.numClasses;
            }
        }

        // Shape of boxes input should be
        // [batch_size, num_boxes, 4] or [batch_size, num_boxes, 1, 4] or [batch_size, num_boxes, num_classes, 4]
        ASSERT(in[0].desc.dims.nbDims == 3 || in[0].desc.dims.nbDims == 4);
        if (in[0].desc.dims.nbDims == 3)
        {
            ASSERT(in[0].desc.dims.d[2] == 4);
            mParam.shareLocation = true;
            mParam.numBoxElements = in[0].desc.dims.d[1] * in[0].desc.dims.d[2];
        }
        else
        {
            mParam.shareLocation = (in[0].desc.dims.d[2] == 1);
            ASSERT(in[0].desc.dims.d[2] == mParam.numClasses || mParam.shareLocation);
            ASSERT(in[0].desc.dims.d[3] == 4);
            mParam.numBoxElements = in[0].desc.dims.d[1] * in[0].desc.dims.d[2] * in[0].desc.dims.d[3];
        }
        mParam.numAnchors = in[0].desc.dims.d[1];

        // Shape of landmarks input should be
        // [batch_size, num_boxes, 10] or [batch_size, num_boxes, 1, 10]
        ASSERT(in[2].desc.dims.nbDims == 3 || in[2].desc.dims.nbDims == 4);
        if (in[2].desc.dims.nbDims == 3)
        {
            ASSERT(in[2].desc.dims.d[2] == 10);
        }
        else
        {
            ASSERT(in[2].desc.dims.d[2] == 1);
            ASSERT(in[2].desc.dims.d[3] == 10);
        }

        if (nbInputs == 3)
        {
            // Only two inputs are used, disable the fused box decoder
            mParam.boxDecoder = false;
        }
        if (nbInputs == 4)
        {
            // All three inputs are used, enable the box decoder
            // Shape of anchors input should be
            // Constant shape: [1, numAnchors, 4] or [batch_size, numAnchors, 4]
            ASSERT(in[3].desc.dims.nbDims == 3);
            mParam.boxDecoder = true;
            mParam.shareAnchors = (in[3].desc.dims.d[0] == 1);
        }
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
}

size_t EfficientNMSLandmarkPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    int batchSize = inputs[1].dims.d[0];
    int numScoreElements = inputs[1].dims.d[1] * inputs[1].dims.d[2];
    int numClasses = inputs[1].dims.d[2];
    return EfficientNMSLandmarkWorkspaceSize(batchSize, numScoreElements, numClasses, mParam.datatype);
}

int EfficientNMSLandmarkPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        mParam.batchSize = inputDesc[0].dims.d[0];

        // Standard NMS Operation
        const void* const boxesInput = inputs[0];
        const void* const scoresInput = inputs[1];
        const void* const landmarksInput = inputs[2];
        const void* const anchorsInput = mParam.boxDecoder ? inputs[3] : nullptr;

        void* numDetectionsOutput = outputs[0];
        void* nmsBoxesOutput = outputs[1];
        void* nmsScoresOutput = outputs[2];
        void* nmsClassesOutput = outputs[3];
        void* nmsLandmarksOutput = outputs[4];

        return EfficientNMSLandmarkInference(mParam, boxesInput, scoresInput, landmarksInput, anchorsInput, numDetectionsOutput,
            nmsBoxesOutput, nmsScoresOutput, nmsClassesOutput, nmsLandmarksOutput, workspace, stream);
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return -1;
}

// Standard NMS Plugin Operation

EfficientNMSLandmarkPluginCreator::EfficientNMSLandmarkPluginCreator()
    : mParam{}
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("score_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("iou_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_output_boxes", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("background_class", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("score_activation", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("box_coding", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* EfficientNMSLandmarkPluginCreator::getPluginName() const noexcept
{
    return EFFICIENT_NMS_LANDMARK_PLUGIN_NAME;
}

const char* EfficientNMSLandmarkPluginCreator::getPluginVersion() const noexcept
{
    return EFFICIENT_NMS_LANDMARK_PLUGIN_VERSION;
}

const PluginFieldCollection* EfficientNMSLandmarkPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* EfficientNMSLandmarkPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        const PluginField* fields = fc->fields;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "score_threshold"))
            {
                ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                mParam.scoreThreshold = *(static_cast<const float*>(fields[i].data));
            }
            if (!strcmp(attrName, "iou_threshold"))
            {
                ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                mParam.iouThreshold = *(static_cast<const float*>(fields[i].data));
            }
            if (!strcmp(attrName, "max_output_boxes"))
            {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.numOutputBoxes = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "background_class"))
            {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.backgroundClass = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "score_activation"))
            {
                mParam.scoreSigmoid = *(static_cast<const bool*>(fields[i].data));
            }
            if (!strcmp(attrName, "box_coding"))
            {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.boxCoding = *(static_cast<const int*>(fields[i].data));
            }
        }

        auto* plugin = new EfficientNMSLandmarkPlugin(mParam);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* EfficientNMSLandmarkPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call EfficientNMSLandmarkPlugin::destroy()
        auto* plugin = new EfficientNMSLandmarkPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
