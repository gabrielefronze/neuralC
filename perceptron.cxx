//
// Created by Gabriele Gaetano Fronzé on 05/03/2018.
//

#include "perceptron.h"
#include "potentials.h"

perceptron::perceptron(uint64_t ID,
                       double learningRate,
                       uint32_t seed=777,
                       uint32_t stream=1)
  : fID(ID),
    fLearningRate(learningRate),
    fIterations(0),
    fIndex(0),
    fRNG(seed,stream),
    fDistribution(0.0,1.0)
{
  fWeights.push_back(fDistribution(fRNG));
}


void perceptron::setInput(uint64_t senderID, double value)
{

  if(fInputIdCorrelationMap.find(senderID) == fInputIdCorrelationMap.end()){
    fFwdInputs.push_back(value);
    fWeights.push_back(fDistribution(fRNG));
    fInputIdCorrelationMap.emplace(std::make_pair(senderID,fIndex++));
  } else {
    fFwdInputs[fInputIdCorrelationMap[senderID]]=value;
  }
}


void perceptron::setCorrection(uint64_t senderID, double value)
{
  fDeltaWeightSum+=value;
}

bool perceptron::infere()
{
  auto product=std::inner_product(fFwdInputs.begin(), fFwdInputs.end(),fWeights.begin()+1, 0.0);
  product+=fWeights[0];

  fOutput=potentials::step(product);

  return true;
}

void perceptron::update()
{
  double quadFwdInputs=std::accumulate(fFwdInputs.begin(),
                                       fFwdInputs.end(),
                                       0.,
                                       [](double sum_so_far, double x)->double {
                                         return sum_so_far + x * x;
                                       });
  double delta=(1-quadFwdInputs)*fDeltaWeightSum;
  double deltaeta=delta*fLearningRate;


  fCorrections.push_back(1.);
  fCorrections.insert(fFwdInputs.begin(),fFwdInputs.end(),fCorrections.end());
  std::for_each(fCorrections.begin(),fCorrections.end(),[deltaeta](double &el){el *= -deltaeta;});
  std::transform(fWeights.begin(), fWeights.end(), fCorrections.begin(), fWeights.begin(), std::plus<double>());

  fBckInputs=fWeights;
  std::for_each(fBckInputs.begin(),fBckInputs.end(),[delta](double &el){el *= delta;});
}

//void perceptron::update(double expected)
//{
//  fDeltaWeightSum.resize(0);
//  fDeltaWeightSum.push_back(expected);
//  fDeltaWeightSum.insert(fFwdInputs.begin(),fFwdInputs.end(),fDeltaWeightSum.end());
//  std::for_each(fDeltaWeightSum.begin()+1, fDeltaWeightSum.end(), [](double &el){el *= expected;});
//  std::transform(fWeights.begin(), fWeights.end(), fDeltaWeightSum.begin(), fWeights.begin(), std::plus<double>());
//}

