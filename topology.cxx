//
// Created by Gabriele Gaetano Fronzé on 06/03/2018.
//

#include "topology.h"

topology::neuralLayer::neuralLayer(uint64_t nNeurons, uint64_t depth, uint64_t iNeuron0) : fDepth(depth)
{
  fNeurons.reserve(nNeurons);
  fNeuronMap.reserve(nNeurons);

  uint64_t neuronIndex = 0;
  for (uint64_t iNeuron = iNeuron0; iNeuron < nNeurons+iNeuron0; ++iNeuron) {
    fNeurons.emplace_back(perceptron(iNeuron));
    fNeuronMap.emplace(std::make_pair(iNeuron,neuronIndex));
    neuronIndex++;
  }
}

perceptron &topology::neuralLayer::operator[](uint64_t ID)
{
  return fNeurons[fNeuronMap[ID]];
}

topology::synapse::synapse(uint64_t senderID, uint64_t senderLayer, uint64_t receiverID, uint64_t receiverLayer)
{
  fSenderID[0]=senderID;
  fSenderID[1]=senderLayer;
  fReceiverID[0]=receiverID;
  fReceiverID[1]=receiverLayer;
}

topology::synapse::synapse(const uint64_t *IDs)
{
  fSenderID[0]=IDs[0];
  fSenderID[1]=IDs[1];
  fReceiverID[0]=IDs[2];
  fReceiverID[1]=IDs[3];
}

topology::synapseLayer::synapseLayer(uint64_t nSynapses, uint64_t depth) :  fDepth(depth)
{
  fSynapses.resize(nSynapses);
}

topology::topology::topology(std::vector<uint64_t> neuronsNumber, std::vector<uint64_t[4]> connections)
{
  uint64_t depth=0;

  for(const auto & itNeuronsNumber :  neuronsNumber){
    fLayers.emplace_back(std::make_pair(neuralLayer(itNeuronsNumber,depth+1),synapseLayer(0,depth+2)));
    depth+=2;
  }
  fLayers.emplace_back(std::make_pair(neuralLayer(1,depth+1),synapseLayer(0,depth+2)));
  depth+=2;


  for(const auto & itConnection : connections){
    fLayers[itConnection[0]].second.fSynapses.emplace_back(itConnection);
  }

  uint64_t layerCount = 1;
  for(const auto & itLayers : fLayers){
    printf("Layer %llu contains %lu neurons with %lu synapses.\n",
           layerCount++,
           itLayers.first.fNeurons.size(),
           itLayers.second.fSynapses.size());
  }

  fFwdDepth = 0;
  fBckDepth = fMaxDepth = fLayers.size()-1;
}

void topology::topology::fwdPropagate()
{
  for(auto & itLayers : fLayers){
    for(auto & itNeurons : itLayers.first.fNeurons){
      itNeurons.infere();
    }

    for(auto & itSynapse : itLayers.second.fSynapses){
      fLayers[itSynapse.fReceiverID[0]].first[itSynapse.fReceiverID[1]].setInput(itSynapse.fSenderID[1],fLayers[itSynapse.fSenderID[0]].first[itSynapse.fSenderID[1]].getOutput());
    }
  }
}

void topology::topology::bckPropagate()
{
  for(auto & itLayers : fLayers){
    for(auto & itSynapse : itLayers.second.fSynapses){
      fLayers[itSynapse.fReceiverID[0]].first[itSynapse.fReceiverID[1]].setInput(itSynapse.fSenderID[1],fLayers[itSynapse.fSenderID[0]].first[itSynapse.fSenderID[1]].getOutput());
    }

    for(auto & itNeurons : itLayers.first.fNeurons){

    }
  }
}
