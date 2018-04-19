//
// Created by Filippo Valle on 17/04/2018.
//

#include "Perceptron.h"

Perceptron::Perceptron(uint64_t id, uint64_t numOfFeatures, theta_function theta, theta_function theta_d,
                       double learningRate, uint64_t seed)
        :
        fID(id),
        fNumOfData(numOfFeatures),
        fLearningRate(learningRate),
        fStatus(kReady),
        fNumOfFeatures(numOfFeatures),
        fTheta(theta),
        fTheta_d(theta_d)
{
    pcg32_fast myRng(seed);
    std::uniform_int_distribution<double> distribution(0.,1.);

    fW.reserve(fNumOfFeatures+1);

    //add vapnick dimension and random init w
    for(int i=0;i<fNumOfFeatures+1;i++){
        fW.push_back(distribution(myRng));
    }
}

void Perceptron::setInput(const std::vector<double> &X) {
    fInputs.reserve(fNumOfData);
    for (size_t i = 0; i < fNumOfData; ++i) {
        fInputs.emplace_back(X[i]);
    }

    fStatus = kDataLoaded;
}


void Perceptron::fit() {
    if(fStatus<1){
        std::cerr<<"Data not loaded in Perceptron" << fID;
        return;
    }

    fSignal = std::inner_product(fInputs.begin(),fInputs.end(),fW.begin()+1,fW[0]);
    fX = fTheta(fSignal);
    fThetaprime = fTheta_d(fSignal);
    fStatus = kTrained;
}


double Perceptron::getOutputX() {
    if (fStatus<2) fit();
    return fX;
}


double Perceptron::getOutputtheta_d() {
    return fThetaprime;
}

void Perceptron::predict(std::vector<double> X) {
    if (fStatus < 2) fit();
//TODO
}

void Perceptron::toOstream(){
    printf("neuron id: %llu\n",fID);
}

void Perceptron::updateWeights() {
    fW[0]+=-fLearningRate*fdelta;
    for(size_t i = 1; i< fW.size(); i++){
        fW[i] += -fLearningRate * fdelta * fInputs[i];
    }
}

