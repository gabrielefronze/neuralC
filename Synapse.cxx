//
// Created by Filippo Valle on 28/09/2018.
//

#include "Synapse.h"

Synapse::Synapse(uint64_t senderID, uint64_t senderLayer, uint64_t receiverID, uint64_t receiverLayer) {
    fSenderID[0]=senderID;
    fSenderID[1]=senderLayer;
    fReceiverID[0]=receiverID;
    fReceiverID[1]=receiverLayer;
}

Synapse::Synapse(const uint64_t *IDs) {
    fSenderID[0]=IDs[0];
    fSenderID[1]=IDs[1];
    fReceiverID[0]=IDs[2];
    fReceiverID[1]=IDs[3];
}
