// Full-frame classifier parser for a stock classification export: softmax over the raw
// class scores (logits), reporting the best class with its probability.
//
// nvinfer's built-in classifier parser reports the raw maximum score, which for a model
// without a Softmax layer (ResNet50 v1-7 ends in Gemm) is a logit, not a probability.
// This parser applies softmax once, in a numerically stable form, so the attached
// result_prob is the probability eager's ClassificationPostprocess computes.
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferClassiferParseSoftmaxTop1(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList, std::string &descString)
{
    (void)networkInfo;
    for (unsigned int l = 0; l < outputLayersInfo.size(); ++l) {
        const NvDsInferLayerInfo &layer = outputLayersInfo[l];
        const unsigned int numClasses = layer.inferDims.numElements;
        if (layer.dataType != FLOAT || layer.buffer == nullptr || numClasses == 0)
            return false;
        const float *scores = static_cast<const float *>(layer.buffer);

        unsigned int best = 0;
        for (unsigned int c = 1; c < numClasses; ++c)
            if (scores[c] > scores[best])
                best = c;
        // softmax(best) = 1 / sum(exp(x_c - x_best)); subtracting the max keeps exp bounded.
        double sum = 0.0;
        for (unsigned int c = 0; c < numClasses; ++c)
            sum += std::exp(double(scores[c]) - double(scores[best]));
        const float probability = float(1.0 / sum);
        if (probability < classifierThreshold)
            continue;

        NvDsInferAttribute attr;
        attr.attributeIndex = l;
        attr.attributeValue = best;
        attr.attributeConfidence = probability;
        // nvinfer attaches nothing for an empty label, and takes ownership of this string
        // (released with free()). Names come from the framework's label file; emit the id.
        const std::string label = std::to_string(best);
        attr.attributeLabel = strdup(label.c_str());
        attrList.push_back(attr);
        descString.append(label).append(" ");
    }
    return true;
}

CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferClassiferParseSoftmaxTop1);
