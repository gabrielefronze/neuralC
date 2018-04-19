//
// Created by Filippo Valle on 17/04/2018.
//

#ifndef NEURALNET_LAYER_H
#define NEURALNET_LAYER_H


#include <vector>
#include "Perceptron.h"

class Layer {
public:
    Layer(uint64_t numOfNeurons, uint64_t numOfFeatures);

    std::vector<double> getOutputs();
    std::vector<Perceptron> fNeurons;
    void updateWeigths();

    void toOstream();
private:
    uint64_t fnumOfNeurons;
};

class InputLayer: public Layer{
public:
    InputLayer(uint64_t fnumOfNeurons, uint64_t inputSize);

};

class OutputLayer: public Layer{
public:
    explicit OutputLayer(uint64_t inputSize):Layer(1, inputSize){};
};


#endif //NEURALNET_LAYER_H
