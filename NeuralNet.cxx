//
// Created by Filippo Valle on 17/04/2018.
//

#include "NeuralNet.h"

NeuralNet::NeuralNet() : fDepth(0), fmaxiterations(10) {
//TODO
}

NeuralNet &NeuralNet::firstLayer(uint64_t numOfNeurons, const std::vector<datatype> &input) {
    fLayers.emplace_back(InputLayer(numOfNeurons, input[0].size()));
    fX = input;

    return *this;
}

NeuralNet &NeuralNet::addLayer(uint64_t numOfNeurons) {

    //number of input is the number of neurons in previous layer
    uint64_t numOfInputs = fLayers[fDepth].fNeurons.size();

    fLayers.emplace_back(Layer(numOfNeurons, numOfInputs));
    fDepth++;

    return *this;
}

NeuralNet &NeuralNet::lastLayer(const std::vector<double> &y) {
    //number of input is the number of neurons in previous layer
    uint64_t numOfInputs = fLayers[fDepth].fNeurons.size();

    fLayers.emplace_back(OutputLayer(numOfInputs));

    fy = y;
    return *this;
}

NeuralNet & NeuralNet::train() {
    int iData = -1;
    for(size_t step = 0; step <= fmaxiterations; step++) {
        for (auto &data : fX) {
            iData++;
            //fwd propagate

            //set global input as input for first layer
            for (auto &neuron : fLayers[0].fNeurons) {
                neuron.setInput(data);
            }

            for (auto layer = fLayers.begin() + 1; layer != fLayers.end(); layer++) {
                auto outputs = (layer - 1)->getOutputs();

                for (auto &neuron : layer->fNeurons) {
                    neuron.setInput(outputs);
                    neuron.fit();
                }
            }

            //backpropagate
            double deltaLast = 0.;
            double XLast = fLayers[fDepth].fNeurons[0].getOutputX();
            double thetaprimeLast = fLayers[fDepth].fNeurons[0].getOutputtheta_d();
            deltaLast += 2 * (XLast - fy[iData]) * thetaprimeLast;
            fLayers[fDepth].fNeurons[0].setDelta(deltaLast);

            for (size_t l = fDepth - 1; l > 0; l--) {
                auto layer = fLayers[l];
                auto nextLayer = fLayers[l + 1];

                for (auto &neuron : layer.fNeurons) {
                    double delta = 0.;

                    for (auto &nextNeuron : nextLayer.fNeurons) {
                        delta += nextNeuron.fdelta * nextNeuron.fW[neuron.fID] * nextNeuron.getOutputtheta_d();
                    }

                    neuron.setDelta(delta);
                }
            }

//update
            for (auto &layer: fLayers) {
                layer.updateWeigths();
            }


        }
    }

    return *this;
}

NeuralNet & NeuralNet::toOstream() {
    printf("\n\n\n********\nNeural Net\nnum of layers: %llu\nmaxiterations: %llu\n", fDepth, fmaxiterations);
    for(auto &layer:fLayers){
        layer.toOstream();
    }

    return *this;
}

