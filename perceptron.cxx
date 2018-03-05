//
// Created by Gabriele Gaetano Fronzé on 05/03/2018.
//

#include "perceptron.h"
#include "potentials.h"

perceptron::perceptron(uint64_t ID,
                       std::vector<uint64_t> InputConnections,
                       std::vector<uint64_t> OutputConnections,
                       uint32_t seed=777,
                       uint32_t stream=1)
  : fID(ID),
    fInputConnections(std::move(InputConnections)),
    fOutputConnections(std::move(OutputConnections)),
    fStatus(kNotReady),
    fIterations(0)
{
  pcg32_fast myRNG(seed,stream);
  std::uniform_real_distribution<double> distribution(0.0,1.0);

  fInputs = std::vector<double>(fInputConnections.size()+1,0);

  fWeights.reserve(fInputConnections.size()+1);
  for(auto & itWeights : fWeights){
    itWeights=distribution(myRNG);
  }
}

void perceptron::setInput(uint64_t id, double value)
{
  auto positionIt = std::find(fInputConnections.begin(),fInputConnections.end(),id);
  auto index = std::distance(fInputConnections.begin(), positionIt);
  fInputs[index] = value;
}

bool perceptron::infere()
{
  if(fStatus==kReady){

    auto product=std::inner_product(fInputs.begin(), fInputs.end(),fWeights.begin()+1, 0.0);
    product+=fWeights[0];

    fOutput=potentials::step(product);

    return true;

  } else return false;
}

void perceptron::update(double expected)
{
    auto tempInputs = fInputs;
    std::for_each(tempInputs.begin(), tempInputs.end(), [](int &el){el *= expected; });
    fWeights[0]+=expected;
    std::transform (fWeights.begin()+1, fWeights.end(), tempInputs.begin(), fWeights.begin(), std::plus<double>());
}

bool perceptron::check(double expected, double variance)
{
  infere();
  bool isOk = abs(expected-fOutput)<variance;

  if(isOk || fIterations>kMaxIterations) return false;
  else {
    update(expected);
    return true;
  }
}

void perceptron::train(std::vector<std::pair<std::vector<double>, double[2]>> dataSet)
{
  for(size_t iData=0; iData<dataSet.size(); iData++){
    if(iData==0) fIterations++;
    fInputs = dataSet[iData].first;
    if (check(dataSet[iData].second[0],dataSet[iData].second[1])) iData = 0;
  }
}

void perceptron::communicate()
{
  for(const auto & itID : fOutputConnections){
    auto i = globalMap[itID];
    i.setInput(fID,fOutput);
  }
}
