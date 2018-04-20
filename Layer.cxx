//
// Created by Filippo Valle on 17/04/2018.
//

#include "Layer.h"

Layer::Layer(uint64_t numOfNeurons, uint64_t numOfFeatures) : fnumOfNeurons(numOfNeurons) {
    fNeurons.reserve(numOfNeurons);

    for(uint64_t i = 0; i < numOfNeurons; i++){
        fNeurons.emplace_back(Perceptron(i, numOfFeatures, +[](double x) { return tanh(x); }, +[](double x) { return 1.+ tanh(x) * tan(x); }));
    }
}

std::vector<double> Layer::getOutputs() {
    std::vector<double> output;
    output.reserve(fnumOfNeurons);

    for(auto &neuron : fNeurons){
        output.emplace_back(neuron.getOutputX());
    }

    return output;
}

InputLayer::InputLayer(uint64_t numOfNeurons, uint64_t inputSize) : Layer(numOfNeurons, inputSize) {

}

void Layer::toOstream() {
    printf("\n\n*************\nIm a layer\nI have %lu neurons\n", fNeurons.size());
    for(auto &neuron : fNeurons){
        neuron.toOstream();
    }
}

void Layer::updateWeigths() {
    for(auto &neuron : fNeurons){
        neuron.updateWeights();
    }
}
