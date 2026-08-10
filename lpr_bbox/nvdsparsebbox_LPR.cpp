/*
 * DeepStream custom bbox parser for single-class YOLO license plate detection.
 *
 * Model output format: [1, 5, N]  (channel-first, batch stripped by TRT)
 *   InferDims: numDims=2, d[0]=5, d[1]=N  (N = number of proposals, e.g. 8400)
 *
 * Memory layout:
 *   buf[0*N .. 1*N-1]  cx (center x)    — absolute pixel coords in network input space
 *   buf[1*N .. 2*N-1]  cy (center y)
 *   buf[2*N .. 3*N-1]  w  (width)
 *   buf[3*N .. 4*N-1]  h  (height)
 *   buf[4*N .. 5*N-1]  confidence score
 *
 * Class ID is always 0 — this is a single-class detector (license plate).
 *
 * Function exported: NvDsInferParseCustomYoloLPR
 * Set in nvinfer config:  parse-bbox-func-name=NvDsInferParseCustomYoloLPR
 *                         custom-lib-path=<path>/libnvdsinfer_custom_impl_Yolo_LPR.so
 */

#include <vector>
#include <iostream>
#include <algorithm>

#include "nvdsinfer_custom_impl.h"

static inline float
clamp_val(float val, float lo, float hi)
{
    return std::min(std::max(val, lo), hi);
}

extern "C" bool
NvDsInferParseCustomYoloLPR(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo          const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.empty()) {
        std::cerr << "[LPR parser] ERROR: no output layers\n";
        return false;
    }

    const NvDsInferLayerInfo& layer = outputLayersInfo[0];

    // After TRT strips the batch dim the tensor is [5, N].
    if (layer.inferDims.numDims < 2) {
        std::cerr << "[LPR parser] ERROR: expected 2D tensor [5, N], got "
                  << layer.inferDims.numDims << "D\n";
        return false;
    }

    const uint numChannels  = static_cast<uint>(layer.inferDims.d[0]);
    const uint numProposals = static_cast<uint>(layer.inferDims.d[1]);

    if (numChannels != 5) {
        std::cerr << "[LPR parser] ERROR: expected 5 channels (cx,cy,w,h,score), got "
                  << numChannels << "\n";
        return false;
    }

    const float confThreshold =
        detectionParams.perClassPreclusterThreshold.empty()
            ? 0.25f
            : detectionParams.perClassPreclusterThreshold[0];

    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);

    const float* buf        = static_cast<const float*>(layer.buffer);
    const float* cx_data    = buf + 0 * numProposals;
    const float* cy_data    = buf + 1 * numProposals;
    const float* bw_data    = buf + 2 * numProposals;
    const float* bh_data    = buf + 3 * numProposals;
    const float* score_data = buf + 4 * numProposals;

    for (uint p = 0; p < numProposals; ++p) {
        float score = score_data[p];
        if (score < confThreshold)
            continue;

        float half_w = bw_data[p] * 0.5f;
        float half_h = bh_data[p] * 0.5f;
        float x1 = clamp_val(cx_data[p] - half_w, 0.f, netW);
        float y1 = clamp_val(cy_data[p] - half_h, 0.f, netH);
        float x2 = clamp_val(cx_data[p] + half_w, 0.f, netW);
        float y2 = clamp_val(cy_data[p] + half_h, 0.f, netH);

        float w = x2 - x1;
        float h = y2 - y1;
        if (w < 1.f || h < 1.f)
            continue;

        NvDsInferParseObjectInfo obj;
        obj.left                = x1;
        obj.top                 = y1;
        obj.width               = w;
        obj.height              = h;
        obj.detectionConfidence = score;
        obj.classId             = 0;   // single class: license plate
        objectList.push_back(obj);
    }

    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloLPR);
