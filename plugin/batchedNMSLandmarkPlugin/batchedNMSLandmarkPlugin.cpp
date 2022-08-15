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

#include "batchedNMSLandmarkPlugin.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::BatchedNMSLandmarkBasePluginCreator;
using nvinfer1::plugin::BatchedNMSLandmarkDynamicPlugin;
using nvinfer1::plugin::BatchedNMSLandmarkDynamicPluginCreator;
using nvinfer1::plugin::BatchedNMSLandmarkPlugin;
using nvinfer1::plugin::BatchedNMSLandmarkPluginCreator;
using nvinfer1::plugin::NMSParameters;

#define NVBUG_3321606_WAR 1

namespace {
const char *NMS_PLUGIN_VERSION{"1"};
const char *NMS_PLUGIN_NAMES[] = {"BatchedNMSLandmark_TRT", "BatchedNMSLandmarkDynamic_TRT"};
} // namespace

// namespace nvinfer1 {
// namespace plugin {
// template <>
// void write<NMSParameters>(char *&buffer, const NMSParameters &val)
// {
//     auto *param = reinterpret_cast<NMSParameters *>(buffer);
//     std::memset(param, 0, sizeof(NMSParameters));
//     param->shareLocation = val.shareLocation;
//     param->backgroundLabelId = val.backgroundLabelId;
//     param->numClasses = val.numClasses;
//     param->topK = val.topK;
//     param->keepTopK = val.keepTopK;
//     param->scoreThreshold = val.scoreThreshold;
//     param->iouThreshold = val.iouThreshold;
//     param->isNormalized = val.isNormalized;
//     buffer += sizeof(NMSParameters);
// }
// } // namespace plugin
// } // namespace nvinfer1

PluginFieldCollection BatchedNMSLandmarkBasePluginCreator::mFC{};
std::vector<PluginField> BatchedNMSLandmarkBasePluginCreator::mPluginAttributes;

static inline pluginStatus_t checkParams(const NMSParameters &param)
{
    // NMS plugin supports maximum thread blocksize of 512 and upto 8 blocks at once.
    constexpr int32_t maxTopK{512 * 8};
    if (param.topK > maxTopK) {
        gLogError << "Invalid parameter: NMS topK (" << param.topK << ") exceeds limit (" << maxTopK
                  << ")" << std::endl;
        return STATUS_BAD_PARAM;
    }

    return STATUS_SUCCESS;
}

BatchedNMSLandmarkPlugin::BatchedNMSLandmarkPlugin(NMSParameters params) : param(params)
{
    mPluginStatus = checkParams(param);
}

BatchedNMSLandmarkPlugin::BatchedNMSLandmarkPlugin(const void *data, size_t length)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;
    param = read<NMSParameters>(d);
    boxesSize = read<int>(d);
    scoresSize = read<int>(d);
    landmarksSize = read<int>(d);
    numPriors = read<int>(d);
    mClipBoxes = read<bool>(d);
    mPrecision = read<DataType>(d);
    mScoreBits = read<int32_t>(d);
    mCaffeSemantics = read<bool>(d);
    ASSERT(d == a + length);

    mPluginStatus = checkParams(param);
}

BatchedNMSLandmarkDynamicPlugin::BatchedNMSLandmarkDynamicPlugin(NMSParameters params)
    : param(params)
{
    mPluginStatus = checkParams(param);
}

BatchedNMSLandmarkDynamicPlugin::BatchedNMSLandmarkDynamicPlugin(const void *data, size_t length)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;
    param = read<NMSParameters>(d);
    boxesSize = read<int>(d);
    scoresSize = read<int>(d);
    landmarksSize = read<int>(d);
    numPriors = read<int>(d);
    mClipBoxes = read<bool>(d);
    mPrecision = read<DataType>(d);
    mScoreBits = read<int32_t>(d);
    mCaffeSemantics = read<bool>(d);
    ASSERT(d == a + length);

    mPluginStatus = checkParams(param);
}

int BatchedNMSLandmarkPlugin::getNbOutputs() const noexcept
{
    return 5;
}

int BatchedNMSLandmarkDynamicPlugin::getNbOutputs() const noexcept
{
    return 5;
}

int BatchedNMSLandmarkPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

int BatchedNMSLandmarkDynamicPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void BatchedNMSLandmarkPlugin::terminate() noexcept
{
}

void BatchedNMSLandmarkDynamicPlugin::terminate() noexcept
{
}

Dims BatchedNMSLandmarkPlugin::getOutputDimensions(int index,
                                                   const Dims *inputs,
                                                   int nbInputDims) noexcept
{
    try {
        ASSERT(nbInputDims == 3);
        ASSERT(index >= 0 && index < this->getNbOutputs());
        ASSERT(inputs[0].nbDims == 3);
        ASSERT(inputs[1].nbDims == 2 || (inputs[1].nbDims == 3 && inputs[1].d[2] == 1));
        ASSERT(inputs[2].nbDims == 3);
        // boxesSize: number of box coordinates for one sample
        boxesSize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
        // scoresSize: number of scores for one sample
        scoresSize = inputs[1].d[0] * inputs[1].d[1];
        // landmarksSize: number of landmark coordinates for one sample
        landmarksSize = inputs[2].d[0] * inputs[2].d[1] * inputs[2].d[2];
        // num_detections
        if (index == 0) {
            Dims dim0{};
            dim0.nbDims = 0;
            return dim0;
        }
        // nmsed_boxes
        if (index == 1) {
            return DimsHW(param.keepTopK, 4);
        }
        // nmsed_landmarks
        if (index == 4) {
            return DimsHW(param.keepTopK, 10);
        }
        // nmsed_scores or nmsed_classes
        Dims dim1{};
        dim1.nbDims = 1;
        dim1.d[0] = param.keepTopK;
        return dim1;
    } catch (const std::exception &e) {
        caughtError(e);
    }
    return Dims{};
}

DimsExprs BatchedNMSLandmarkDynamicPlugin::getOutputDimensions(int outputIndex,
                                                               const DimsExprs *inputs,
                                                               int nbInputs,
                                                               IExprBuilder &exprBuilder) noexcept
{
    try {
        ASSERT(nbInputs == 3);
        ASSERT(outputIndex >= 0 && outputIndex < this->getNbOutputs());

        // Shape of boxes input should be
        // Constant shape: [batch_size, num_boxes, num_classes, 4] or [batch_size, num_boxes, 1, 4]
        //           shareLocation ==              0               or          1
        // or
        // Dynamic shape: some dimension values may be -1
        ASSERT(inputs[0].nbDims == 4);

        // Shape of scores input should be
        // Constant shape: [batch_size, num_boxes, num_classes] or [batch_size, num_boxes,
        // num_classes, 1] or Dynamic shape: some dimension values may be -1
        ASSERT(inputs[1].nbDims == 3 || inputs[1].nbDims == 4);

        // Shape of landmarks input should be
        // Constant shape: [batch_size, num_boxes, num_classes, 10] or [batch_size, num_boxes, 1,
        // 10]
        //           shareLocation ==              0               or          1
        // or
        // Dynamic shape: some dimension values may be -1
        ASSERT(inputs[2].nbDims == 4);

        if (inputs[0].d[0]->isConstant() && inputs[0].d[1]->isConstant() &&
            inputs[0].d[2]->isConstant() && inputs[0].d[3]->isConstant()) {
            boxesSize = exprBuilder
                            .operation(DimensionOperation::kPROD,
                                       *exprBuilder.operation(DimensionOperation::kPROD,
                                                              *inputs[0].d[1], *inputs[0].d[2]),
                                       *inputs[0].d[3])
                            ->getConstantValue();
        }

        if (inputs[1].d[0]->isConstant() && inputs[1].d[1]->isConstant() &&
            inputs[1].d[2]->isConstant()) {
            scoresSize =
                exprBuilder.operation(DimensionOperation::kPROD, *inputs[1].d[1], *inputs[1].d[2])
                    ->getConstantValue();
        }

        if (inputs[2].d[0]->isConstant() && inputs[2].d[1]->isConstant() &&
            inputs[2].d[2]->isConstant() && inputs[2].d[3]->isConstant()) {
            landmarksSize = exprBuilder
                                .operation(DimensionOperation::kPROD,
                                           *exprBuilder.operation(DimensionOperation::kPROD,
                                                                  *inputs[2].d[1], *inputs[2].d[2]),
                                           *inputs[2].d[3])
                                ->getConstantValue();
        }

        DimsExprs out_dim;
        // num_detections
        if (outputIndex == 0) {
            out_dim.nbDims = 2;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = exprBuilder.constant(1);
        }
        // nmsed_boxes
        else if (outputIndex == 1) {
            out_dim.nbDims = 3;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = exprBuilder.constant(param.keepTopK);
            out_dim.d[2] = exprBuilder.constant(4);
        }
        // nmsed_scores
        else if (outputIndex == 2) {
            out_dim.nbDims = 2;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = exprBuilder.constant(param.keepTopK);
        }
        // nmsed_landmarks
        else if (outputIndex == 4) {
            out_dim.nbDims = 3;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = exprBuilder.constant(param.keepTopK);
            out_dim.d[2] = exprBuilder.constant(10);
        }
        // nmsed_classes
        else {
            out_dim.nbDims = 2;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = exprBuilder.constant(param.keepTopK);
        }

        return out_dim;
    } catch (const std::exception &e) {
        caughtError(e);
    }
    return DimsExprs{};
}

size_t BatchedNMSLandmarkPlugin::getWorkspaceSize(int maxBatchSize) const noexcept
{
    return detectionInferenceWorkspaceSize(param.shareLocation, maxBatchSize, boxesSize, scoresSize,
                                           landmarksSize, param.numClasses, numPriors, param.topK,
                                           mPrecision, mPrecision);
}

size_t BatchedNMSLandmarkDynamicPlugin::getWorkspaceSize(const PluginTensorDesc *inputs,
                                                         int nbInputs,
                                                         const PluginTensorDesc *outputs,
                                                         int nbOutputs) const noexcept
{
    return detectionInferenceWorkspaceSize(param.shareLocation, inputs[0].dims.d[0], boxesSize,
                                           scoresSize, landmarksSize, param.numClasses, numPriors,
                                           param.topK, mPrecision, mPrecision);
}

int BatchedNMSLandmarkPlugin::enqueue(int32_t batchSize,
                                      void const *const *inputs,
                                      void *const *outputs,
                                      void *workspace,
                                      cudaStream_t stream) noexcept
{
    try {
        const void *const locData = inputs[0];
        const void *const confData = inputs[1];
        const void *const landData = inputs[2];

        if (mPluginStatus != STATUS_SUCCESS) {
            return -1;
        }

        void *keepCount = outputs[0];
        void *nmsedBoxes = outputs[1];
        void *nmsedScores = outputs[2];
        void *nmsedClasses = outputs[3];
        void *nmsedLandmarks = outputs[4];

        pluginStatus_t status = nmsInferenceLandmark(
            stream, batchSize, boxesSize, scoresSize, landmarksSize, param.shareLocation,
            param.backgroundLabelId, numPriors, param.numClasses, param.topK, param.keepTopK,
            param.scoreThreshold, param.iouThreshold, mPrecision, locData, mPrecision, confData,
            landData, keepCount, nmsedBoxes, nmsedScores, nmsedClasses, nmsedLandmarks, workspace,
            param.isNormalized, false, mClipBoxes, mScoreBits, mCaffeSemantics);
        return status == STATUS_SUCCESS ? 0 : -1;
    } catch (const std::exception &e) {
        caughtError(e);
    }
    return -1;
}

int BatchedNMSLandmarkDynamicPlugin::enqueue(const PluginTensorDesc *inputDesc,
                                             const PluginTensorDesc *outputDesc,
                                             const void *const *inputs,
                                             void *const *outputs,
                                             void *workspace,
                                             cudaStream_t stream) noexcept
{
    try {
        const void *const locData = inputs[0];
        const void *const confData = inputs[1];
        const void *const landData = inputs[2];

        if (mPluginStatus != STATUS_SUCCESS) {
            return -1;
        }

        void *keepCount = outputs[0];
        void *nmsedBoxes = outputs[1];
        void *nmsedScores = outputs[2];
        void *nmsedClasses = outputs[3];
        void *nmsedLandmarks = outputs[4];

        pluginStatus_t status =
            nmsInferenceLandmark(stream, inputDesc[0].dims.d[0], boxesSize, scoresSize, landmarksSize,
                         param.shareLocation, param.backgroundLabelId, numPriors, param.numClasses,
                         param.topK, param.keepTopK, param.scoreThreshold, param.iouThreshold,
                         mPrecision, locData, mPrecision, confData, landData, keepCount, nmsedBoxes,
                         nmsedScores, nmsedClasses, nmsedLandmarks, workspace, param.isNormalized,
                         false, mClipBoxes, mScoreBits, mCaffeSemantics);
        return status;
    } catch (const std::exception &e) {
        caughtError(e);
    }
    return -1;
}

size_t BatchedNMSLandmarkPlugin::getSerializationSize() const noexcept
{
    // NMSParameters, boxesSize,scoresSize,landmarksSize,numPriors
    return sizeof(NMSParameters) + sizeof(int) * 4 + sizeof(bool) * 2 + sizeof(DataType) +
           sizeof(int32_t);
}

void BatchedNMSLandmarkPlugin::serialize(void *buffer) const noexcept
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, param);
    write(d, boxesSize);
    write(d, scoresSize);
    write(d, landmarksSize);
    write(d, numPriors);
    write(d, mClipBoxes);
    write(d, mPrecision);
    write(d, mScoreBits);
    write(d, mCaffeSemantics);
    ASSERT(d == a + getSerializationSize());
}

size_t BatchedNMSLandmarkDynamicPlugin::getSerializationSize() const noexcept
{
    // NMSParameters, boxesSize,scoresSize,landmarksSize,numPriors
    return sizeof(NMSParameters) + sizeof(int) * 4 + sizeof(bool) * 2 + sizeof(DataType) +
           sizeof(int32_t);
}

void BatchedNMSLandmarkDynamicPlugin::serialize(void *buffer) const noexcept
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, param);
    write(d, boxesSize);
    write(d, scoresSize);
    write(d, landmarksSize);
    write(d, numPriors);
    write(d, mClipBoxes);
    write(d, mPrecision);
    write(d, mScoreBits);
    write(d, mCaffeSemantics);
    ASSERT(d == a + getSerializationSize());
}

void BatchedNMSLandmarkPlugin::configurePlugin(const Dims *inputDims,
                                               int nbInputs,
                                               const Dims *outputDims,
                                               int nbOutputs,
                                               const DataType *inputTypes,
                                               const DataType *outputTypes,
                                               const bool *inputIsBroadcast,
                                               const bool *outputIsBroadcast,
                                               nvinfer1::PluginFormat format,
                                               int maxBatchSize) noexcept
{
    try {
        ASSERT(nbInputs == 3);
        ASSERT(nbOutputs == 5);
        ASSERT(inputDims[0].nbDims == 3);
        ASSERT(inputDims[1].nbDims == 2 || (inputDims[1].nbDims == 3 && inputDims[1].d[2] == 1));
        ASSERT(
            std::none_of(inputIsBroadcast, inputIsBroadcast + nbInputs, [](bool b) { return b; }));
        ASSERT(std::none_of(outputIsBroadcast, outputIsBroadcast + nbInputs,
                            [](bool b) { return b; }));

        boxesSize = inputDims[0].d[0] * inputDims[0].d[1] * inputDims[0].d[2];
        scoresSize = inputDims[1].d[0] * inputDims[1].d[1];
        landmarksSize = inputDims[2].d[0] * inputDims[2].d[1] * inputDims[2].d[2];
        // num_boxes
        numPriors = inputDims[0].d[0];
        const int numLocClasses = param.shareLocation ? 1 : param.numClasses;
        // Third dimension of boxes must be either 1 or num_classes
        ASSERT(inputDims[0].d[1] == numLocClasses);
        ASSERT(inputDims[0].d[2] == 4);
        mPrecision = inputTypes[0];
    } catch (const std::exception &e) {
        caughtError(e);
    }
}

void BatchedNMSLandmarkDynamicPlugin::configurePlugin(const DynamicPluginTensorDesc *in,
                                                      int nbInputs,
                                                      const DynamicPluginTensorDesc *out,
                                                      int nbOutputs) noexcept
{
    try {
        ASSERT(nbInputs == 3);
        ASSERT(nbOutputs == 5);

        // Shape of boxes input should be
        // Constant shape: [batch_size, num_boxes, num_classes, 4] or [batch_size, num_boxes, 1, 4]
        //           shareLocation ==              0               or          1
        const int numLocClasses = param.shareLocation ? 1 : param.numClasses;
        ASSERT(in[0].desc.dims.nbDims == 4);
        ASSERT(in[0].desc.dims.d[2] == numLocClasses);
        ASSERT(in[0].desc.dims.d[3] == 4);

        // Shape of scores input should be
        // Constant shape: [batch_size, num_boxes, num_classes] or [batch_size, num_boxes,
        // num_classes, 1]
        ASSERT(in[1].desc.dims.nbDims == 3 ||
               (in[1].desc.dims.nbDims == 4 && in[1].desc.dims.d[3] == 1));

        boxesSize = in[0].desc.dims.d[1] * in[0].desc.dims.d[2] * in[0].desc.dims.d[3];
        scoresSize = in[1].desc.dims.d[1] * in[1].desc.dims.d[2];
        landmarksSize = in[2].desc.dims.d[1] * in[2].desc.dims.d[2] * in[2].desc.dims.d[3];
        // num_boxes
        numPriors = in[0].desc.dims.d[1];

        mPrecision = in[0].desc.type;
    } catch (const std::exception &e) {
        caughtError(e);
    }
}

bool BatchedNMSLandmarkPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
{
#if NVBUG_3321606_WAR
    return ((type == DataType::kFLOAT || type == DataType::kINT32) &&
            format == PluginFormat::kLINEAR);
#else
    return ((type == DataType::kHALF || type == DataType::kFLOAT || type == DataType::kINT32) &&
            format == PluginFormat::kLINEAR);
#endif // NVBUG_3321606_WAR
}

bool BatchedNMSLandmarkDynamicPlugin::supportsFormatCombination(int pos,
                                                                const PluginTensorDesc *inOut,
                                                                int nbInputs,
                                                                int nbOutputs) noexcept
{
    ASSERT(nbInputs <= 3 && nbInputs >= 0);
    ASSERT(nbOutputs <= 5 && nbOutputs >= 0);
    ASSERT(pos < 8 && pos >= 0);
    const auto *in = inOut;
    const auto *out = inOut + nbInputs;
    const bool consistentFloatPrecision = in[0].type == in[pos].type;
    switch (pos) {
    case 0:
        return (in[0].type == DataType::kHALF || in[0].type == DataType::kFLOAT) &&
               in[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 1:
        return (in[1].type == DataType::kHALF || in[1].type == DataType::kFLOAT) &&
               in[1].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 2:
        return (in[2].type == DataType::kHALF || in[2].type == DataType::kFLOAT) &&
               in[2].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 3:
        return out[0].type == DataType::kINT32 && out[0].format == PluginFormat::kLINEAR;
    case 4:
        return (out[1].type == DataType::kHALF || out[1].type == DataType::kFLOAT) &&
               out[1].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 5:
        return (out[2].type == DataType::kHALF || out[2].type == DataType::kFLOAT) &&
               out[2].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 6:
        return (out[3].type == DataType::kHALF || out[3].type == DataType::kFLOAT) &&
               out[3].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 7:
        return (out[4].type == DataType::kHALF || out[4].type == DataType::kFLOAT) &&
               out[4].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    }
    return false;
}

const char *BatchedNMSLandmarkPlugin::getPluginType() const noexcept
{
    return NMS_PLUGIN_NAMES[0];
}

const char *BatchedNMSLandmarkDynamicPlugin::getPluginType() const noexcept
{
    return NMS_PLUGIN_NAMES[1];
}

const char *BatchedNMSLandmarkPlugin::getPluginVersion() const noexcept
{
    return NMS_PLUGIN_VERSION;
}

const char *BatchedNMSLandmarkDynamicPlugin::getPluginVersion() const noexcept
{
    return NMS_PLUGIN_VERSION;
}

void BatchedNMSLandmarkPlugin::destroy() noexcept
{
    delete this;
}

void BatchedNMSLandmarkDynamicPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2Ext *BatchedNMSLandmarkPlugin::clone() const noexcept
{
    try {
        auto *plugin = new BatchedNMSLandmarkPlugin(param);
        plugin->boxesSize = boxesSize;
        plugin->scoresSize = scoresSize;
        plugin->landmarksSize = landmarksSize;
        plugin->numPriors = numPriors;
        plugin->setPluginNamespace(mNamespace.c_str());
        plugin->setClipParam(mClipBoxes);
        plugin->mPrecision = mPrecision;
        plugin->setScoreBits(mScoreBits);
        plugin->setCaffeSemantics(mCaffeSemantics);
        return plugin;
    } catch (const std::exception &e) {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt *BatchedNMSLandmarkDynamicPlugin::clone() const noexcept
{
    try {
        auto *plugin = new BatchedNMSLandmarkDynamicPlugin(param);
        plugin->boxesSize = boxesSize;
        plugin->scoresSize = scoresSize;
        plugin->landmarksSize = landmarksSize;
        plugin->numPriors = numPriors;
        plugin->setPluginNamespace(mNamespace.c_str());
        plugin->setClipParam(mClipBoxes);
        plugin->mPrecision = mPrecision;
        plugin->setScoreBits(mScoreBits);
        plugin->setCaffeSemantics(mCaffeSemantics);
        return plugin;
    } catch (const std::exception &e) {
        caughtError(e);
    }
    return nullptr;
}

void BatchedNMSLandmarkPlugin::setPluginNamespace(const char *pluginNamespace) noexcept
{
    try {
        mNamespace = pluginNamespace;
    } catch (const std::exception &e) {
        caughtError(e);
    }
}

const char *BatchedNMSLandmarkPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void BatchedNMSLandmarkDynamicPlugin::setPluginNamespace(const char *pluginNamespace) noexcept
{
    try {
        mNamespace = pluginNamespace;
    } catch (const std::exception &e) {
        caughtError(e);
    }
}

const char *BatchedNMSLandmarkDynamicPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

nvinfer1::DataType BatchedNMSLandmarkPlugin::getOutputDataType(int index,
                                                               const nvinfer1::DataType *inputTypes,
                                                               int nbInputs) const noexcept
{
    if (index == 0) {
        return nvinfer1::DataType::kINT32;
    }
    return inputTypes[0];
}

nvinfer1::DataType BatchedNMSLandmarkDynamicPlugin::getOutputDataType(
    int index,
    const nvinfer1::DataType *inputTypes,
    int nbInputs) const noexcept
{
    if (index == 0) {
        return nvinfer1::DataType::kINT32;
    }
    return inputTypes[0];
}

void BatchedNMSLandmarkPlugin::setClipParam(bool clip) noexcept
{
    mClipBoxes = clip;
}

void BatchedNMSLandmarkDynamicPlugin::setClipParam(bool clip) noexcept
{
    mClipBoxes = clip;
}

void BatchedNMSLandmarkPlugin::setScoreBits(int32_t scoreBits) noexcept
{
    mScoreBits = scoreBits;
}

void BatchedNMSLandmarkDynamicPlugin::setScoreBits(int32_t scoreBits) noexcept
{
    mScoreBits = scoreBits;
}

void BatchedNMSLandmarkPlugin::setCaffeSemantics(bool caffeSemantics) noexcept
{
    mCaffeSemantics = caffeSemantics;
}

void BatchedNMSLandmarkDynamicPlugin::setCaffeSemantics(bool caffeSemantics) noexcept
{
    mCaffeSemantics = caffeSemantics;
}

bool BatchedNMSLandmarkPlugin::isOutputBroadcastAcrossBatch(int outputIndex,
                                                            const bool *inputIsBroadcasted,
                                                            int nbInputs) const noexcept
{
    return false;
}

bool BatchedNMSLandmarkPlugin::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

BatchedNMSLandmarkBasePluginCreator::BatchedNMSLandmarkBasePluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(
        PluginField("shareLocation", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(
        PluginField("backgroundLabelId", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("numClasses", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("topK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("keepTopK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(
        PluginField("scoreThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(
        PluginField("iouThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(
        PluginField("isNormalized", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("clipBoxes", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("scoreBits", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(
        PluginField("caffeSemantics", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *BatchedNMSLandmarkPluginCreator::getPluginName() const noexcept
{
    return NMS_PLUGIN_NAMES[0];
}

const char *BatchedNMSLandmarkDynamicPluginCreator::getPluginName() const noexcept
{
    return NMS_PLUGIN_NAMES[1];
}

const char *BatchedNMSLandmarkBasePluginCreator::getPluginVersion() const noexcept
{
    return NMS_PLUGIN_VERSION;
}

const PluginFieldCollection *BatchedNMSLandmarkBasePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext *BatchedNMSLandmarkPluginCreator::createPlugin(
    const char *name,
    const PluginFieldCollection *fc) noexcept
{
    try {
        NMSParameters params;
        const PluginField *fields = fc->fields;
        bool clipBoxes = true;
        int32_t scoreBits = 16;
        bool caffeSemantics = true;

        for (int i = 0; i < fc->nbFields; ++i) {
            const char *attrName = fields[i].name;
            if (!strcmp(attrName, "shareLocation")) {
                params.shareLocation = *(static_cast<const bool *>(fields[i].data));
            } else if (!strcmp(attrName, "backgroundLabelId")) {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                params.backgroundLabelId = *(static_cast<const int *>(fields[i].data));
            } else if (!strcmp(attrName, "numClasses")) {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                params.numClasses = *(static_cast<const int *>(fields[i].data));
            } else if (!strcmp(attrName, "topK")) {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                params.topK = *(static_cast<const int *>(fields[i].data));
            } else if (!strcmp(attrName, "keepTopK")) {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                params.keepTopK = *(static_cast<const int *>(fields[i].data));
            } else if (!strcmp(attrName, "scoreThreshold")) {
                ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                params.scoreThreshold = *(static_cast<const float *>(fields[i].data));
            } else if (!strcmp(attrName, "iouThreshold")) {
                ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                params.iouThreshold = *(static_cast<const float *>(fields[i].data));
            } else if (!strcmp(attrName, "isNormalized")) {
                params.isNormalized = *(static_cast<const bool *>(fields[i].data));
            } else if (!strcmp(attrName, "clipBoxes")) {
                clipBoxes = *(static_cast<const bool *>(fields[i].data));
            } else if (!strcmp(attrName, "scoreBits")) {
                scoreBits = *(static_cast<const int32_t *>(fields[i].data));
            } else if (!strcmp(attrName, "caffeSemantics")) {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                caffeSemantics = *(static_cast<const bool *>(fields[i].data));
            }
        }

        auto *plugin = new BatchedNMSLandmarkPlugin(params);
        plugin->setClipParam(clipBoxes);
        plugin->setScoreBits(scoreBits);
        plugin->setCaffeSemantics(caffeSemantics);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    } catch (const std::exception &e) {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt *BatchedNMSLandmarkDynamicPluginCreator::createPlugin(
    const char *name,
    const PluginFieldCollection *fc) noexcept
{
    try {
        NMSParameters params;
        const PluginField *fields = fc->fields;
        bool clipBoxes = true;
        int32_t scoreBits = 16;
        bool caffeSemantics = true;

        for (int i = 0; i < fc->nbFields; ++i) {
            const char *attrName = fields[i].name;
            if (!strcmp(attrName, "shareLocation")) {
                params.shareLocation = *(static_cast<const bool *>(fields[i].data));
            } else if (!strcmp(attrName, "backgroundLabelId")) {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                params.backgroundLabelId = *(static_cast<const int *>(fields[i].data));
            } else if (!strcmp(attrName, "numClasses")) {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                params.numClasses = *(static_cast<const int *>(fields[i].data));
            } else if (!strcmp(attrName, "topK")) {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                params.topK = *(static_cast<const int *>(fields[i].data));
            } else if (!strcmp(attrName, "keepTopK")) {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                params.keepTopK = *(static_cast<const int *>(fields[i].data));
            } else if (!strcmp(attrName, "scoreThreshold")) {
                ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                params.scoreThreshold = *(static_cast<const float *>(fields[i].data));
            } else if (!strcmp(attrName, "iouThreshold")) {
                ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                params.iouThreshold = *(static_cast<const float *>(fields[i].data));
            } else if (!strcmp(attrName, "isNormalized")) {
                params.isNormalized = *(static_cast<const bool *>(fields[i].data));
            } else if (!strcmp(attrName, "clipBoxes")) {
                clipBoxes = *(static_cast<const bool *>(fields[i].data));
            } else if (!strcmp(attrName, "scoreBits")) {
                scoreBits = *(static_cast<const int32_t *>(fields[i].data));
            } else if (!strcmp(attrName, "caffeSemantics")) {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                caffeSemantics = *(static_cast<const bool *>(fields[i].data));
            }
        }

        auto *plugin = new BatchedNMSLandmarkDynamicPlugin(params);
        plugin->setClipParam(clipBoxes);
        plugin->setScoreBits(scoreBits);
        plugin->setCaffeSemantics(caffeSemantics);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    } catch (const std::exception &e) {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext *BatchedNMSLandmarkPluginCreator::deserializePlugin(const char *name,
                                                                 const void *serialData,
                                                                 size_t serialLength) noexcept
{
    try {
        // This object will be deleted when the network is destroyed, which will
        // call NMS::destroy()
        auto *plugin = new BatchedNMSLandmarkPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    } catch (const std::exception &e) {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt *BatchedNMSLandmarkDynamicPluginCreator::deserializePlugin(
    const char *name,
    const void *serialData,
    size_t serialLength) noexcept
{
    try {
        // This object will be deleted when the network is destroyed, which will
        // call NMS::destroy()
        auto *plugin = new BatchedNMSLandmarkDynamicPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    } catch (const std::exception &e) {
        caughtError(e);
    }
    return nullptr;
}
