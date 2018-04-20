//
// Created by Filippo Valle on 17/04/2018.
//

#ifndef NEURALNET_TOPOLOGY_H
#define NEURALNET_TOPOLOGY_H

#include <vector>
#include "Layer.h"

class NeuralNet {
public:
    NeuralNet();

    NeuralNet &firstLayer(uint64_t numOfNeurons, const std::vector<std::vector<double>> &input);
    NeuralNet &addLayer(uint64_t numOfNeurons);
    NeuralNet &lastLayer(const std::vector<double> &y);
    NeuralNet & toOstream();
    NeuralNet & train();

    double infere(std::vector<double> &X);

    uint64_t fmaxiterations;
private:
    uint64_t fDepth;
    std::vector<datatype> fX;
    std::vector<double> fy;
    std::vector<Layer> fLayers;

    void propagate(const datatype &data);
    void backPropagate(int iData);
};


#endif //NEURALNET_TOPOLOGY_H
