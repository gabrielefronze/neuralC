//
// Created by Filippo Valle on 28/09/2018.
//

#ifndef NEURALNET_SYNAPSE_H
#define NEURALNET_SYNAPSE_H


#include <cstdint>

class Synapse {
    Synapse(uint64_t senderID, uint64_t senderLayer, uint64_t receiverID, uint64_t receiverLayer);
    Synapse(const uint64_t IDs[4]);

    inline double getWeight(){return fStream;};

    inline void setWeight(double weight){fStream=weight;};

private:
    double fStream;
    uint64_t fId[4];
    uint64_t fSenderID[2];
    uint64_t fReceiverID[2];

};


#endif //NEURALNET_SYNAPSE_H
