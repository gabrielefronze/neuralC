//
// Created by Gabriele Gaetano Fronzé on 06/03/2018.
//

#ifndef NEURALC_TOPOLOGY_H
#define NEURALC_TOPOLOGY_H

#include <unordered_map>
#include <vector>
#include "perceptron.h"


namespace topology{


//______________________________________________________________________________________________________________________
// Items
//______________________________________________________________________________________________________________________

struct synapse{
    synapse(uint64_t senderID, uint64_t senderLayer, uint64_t receiverID, uint64_t receiverLayer);
    explicit synapse(const uint64_t IDs[4]);
    uint64_t fSenderID[2];
    uint64_t fReceiverID[2];
};


//______________________________________________________________________________________________________________________
// Layers
//______________________________________________________________________________________________________________________
class neuralLayer{
  public:
    neuralLayer(uint64_t nNeurons, uint64_t depth, uint64_t iNeuron0=1 /*so that ID=0 is protected for the input fase*/);
    std::vector<perceptron> fNeurons;
    std::unordered_map<uint64_t,uint64_t> fNeuronMap;
    uint64_t fDepth;
    inline perceptron & operator[](uint64_t ID);
};

struct synapseLayer{
    synapseLayer(uint64_t nSynapses, uint64_t depth);
    std::vector<synapse> fSynapses;
    uint64_t fDepth;
};

class topology
{
  public:
    topology(std::vector<uint64_t> neuronsNumber, std::vector<uint64_t[4]> connections);

    explicit topology(uint64_t inputSize, uint64_t nNeurons = 10);

    void train();

    topology &addLayer(uint64_t nNeurons);

private:
    std::vector<std::pair<neuralLayer, synapseLayer>> fLayers;

    void fwdPropagate();
    void bckPropagate();

    uint64_t fFwdDepth;
    uint64_t fBckDepth;
    uint64_t fMaxDepth;
};
}

#endif //NEURALC_TOPOLOGY_H
