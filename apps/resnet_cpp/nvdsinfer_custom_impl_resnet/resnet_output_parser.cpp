#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"

extern "C" 
bool NvDsInferParseResnetOutput(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo  const &networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList,
    std::string &descString 
);

extern "C" 
bool NvDsInferParseResnetOutput(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo  const &networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList,
    std::string &descString 
) {
    std::cout << "NvDsInferParseResnetOutput" << std::endl;
    /* Get the number of attributes supported by the classifier. */
    unsigned int numOutputLayer = outputLayersInfo.size();

    /* Iterate through all the output coverage layers of the classifier.
    */
    for (unsigned int l = 0; l < numOutputLayer; l++)
    {
        /* outputCoverageBuffer for classifiers is usually a softmax layer.
         * The layer is an array of probabilities of the object belonging
         * to each class with each probability being in the range [0,1] and
         * sum all probabilities will be 1.
         */
        NvDsInferDimsCHW dims;

        getDimsCHWFromDims(dims, outputLayersInfo[l].inferDims);
        unsigned int numClasses = dims.c;
        float *outputCoverageBuffer = (float *)outputLayersInfo[l].buffer;
        float maxProbability = 0;
        bool attrFound = false;
        NvDsInferAttribute attr;

        /* Iterate through all the probabilities that the object belongs to
         * each class. Find the maximum probability and the corresponding class
         * which meets the minimum threshold. */
        for (unsigned int c = 0; c < numClasses; c++)
        {
            float probability = outputCoverageBuffer[c];
            if (probability > classifierThreshold
                    && probability > maxProbability)
            {
                maxProbability = probability;
                attrFound = true;
                attr.attributeIndex = l;
                attr.attributeValue = c;
                attr.attributeConfidence = probability;
            }
        }
        if (attrFound)
        {
            descString.append(std::to_string(attr.attributeValue)).append(" ");
            std::cout << "Attribute: " << attr.attributeValue << " Confidence: " << attr.attributeConfidence << std::endl;
        }
    }

    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferParseResnetOutput);