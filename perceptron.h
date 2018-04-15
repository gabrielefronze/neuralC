//
// Created by Gabriele Gaetano Fronzé on 05/03/2018.
//

#ifndef NEURALC_PERCEPTRON_H
#define NEURALC_PERCEPTRON_H

enum status{
    kNotReady,
    kReady,
    kRunning,
    kDone
};

#include <utility>
#include <vector>
#include <random>
#include <unordered_map>
#include "pcg_random.hpp"

uint32_t kMaxIterations = 10000;

class perceptron
{
  public:
    explicit perceptron(uint64_t ID,
                        double learningRate = 0.01,
                        uint32_t seed = 777,
                        uint32_t stream = 1);

    //Getters
    inline uint64_t getID() const { return fID; }
    inline const std::vector<double> &getWeights() const { return fWeights; }
    inline double getOutput() const { return fOutput; }
    inline double getCorrection(uint64_t senderIndex) const { return fCorrections[senderIndex];}

    inline double getWi(uint64_t ID) { return fBckInputs; }
    inline std::vector<double> & getInputs() const { return fFwdInputs; }

    //Setters
    void setInput(uint64_t senderID, double value);
    void setCorrection(uint64_t senderID, double value);

    //Operative methods
    bool infere();

    //Train methods
    void update();

  private:
    uint64_t fID;
    std::vector<double> fFwdInputs;
    std::vector<double> fBckInputs;
    std::vector<double> fWeights;
    std::vector<double> fCorrections;
    double fLearningRate;
    double fDeltaWeightSum;
    double fOutput;


    pcg32_fast fRNG;
    std::uniform_real_distribution<double> fDistribution;

    std::unordered_map<uint64_t,uint64_t> fInputIdCorrelationMap;
    uint64_t fIndex;

    uint32_t fIterations;
};

#endif //NEURALC_PERCEPTRON_H
