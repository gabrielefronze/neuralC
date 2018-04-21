//
// Created by Filippo Valle on 17/04/2018.
//

#ifndef NEURALNET_PERCEPTRON_H
#define NEURALNET_PERCEPTRON_H

#include <iostream>
#include <utility>
#include <vector>
#include <numeric>
#include <random>
#include <cmath>

#include "pcg_random.hpp"

enum PerceptronStatuses{
    kReady,
    kDataLoaded,
    kTrained,
    kBackProp
};

typedef double (*theta_function)(double);
typedef std::vector<double> datatype;


class Perceptron {
public:
    Perceptron(uint64_t id, uint64_t numOfFeatures, theta_function theta, theta_function theta_d,
                   double learningRate = 0.01, uint64_t seed = 42);

    void setInput(const std::vector<double> &X);
    void updateWeights();
    void fit();
    void predict(std::vector<double> X);
    void toOstream();
    inline void setDelta(double delta){fdelta = delta;};


    uint64_t fID;
    std::vector<double> fInputs;
    std::vector<double> fW;
    double fdelta;

    double getOutputX();
    double getOutputtheta_d();



private:
    PerceptronStatuses fStatus;

    double fSignal;
    double fX; //theta of (fSignal)
    double fThetaprime; //theta_d of (fSignal)
    double fLearningRate;

    uint64_t fNumOfData;
    uint64_t fNumOfFeatures;

    theta_function fTheta;
    theta_function fTheta_d;

};

#endif //NEURALNET_PERCEPTRON_H
