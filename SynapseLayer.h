//
// Created by Filippo Valle on 28/09/2018.
//

#ifndef NEURALNET_SYNAPSELAYER_H
#define NEURALNET_SYNAPSELAYER_H

#include <vector>

#include "Synapse.h"

class SynapseLayer {
public:
    SynapseLayer(uint64_t nSynapses, uint64_t depth);
    std::vector<Synapse> fSynapses;

    Synapse operator[](uint64_t s){return fSynapses[s];};

    uint64_t fDepth;
};


#endif //NEURALNET_SYNAPSELAYER_H
