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
#include "potentials.h"
#include "pcg_random.hpp"

uint32_t kMaxIterations = 10000;

class perceptron
{
  public:
    perceptron(uint64_t ID,
               std::vector<uint64_t> InputConnections,
               std::vector<uint64_t> OutputConnections,
               uint32_t seed=777,
               uint32_t stream=1);

    //Setters
    void setInput(uint64_t id, double value);

    //Getters
    inline uint64_t getFID() const { return fID; }
    inline const std::vector<uint64_t> &getFInputConnections() const { return fInputConnections; }
    inline const std::vector<uint64_t> &getFOutputConnections() const { return fOutputConnections; }
    inline const std::vector<double> &getFWeights() const { return fWeights; }
    inline status getFStatus() const { return fStatus; }
    inline double getFOutput() const { return fOutput; }

    //Operative methods
    bool infere();
    void communicate();

    //Train methods
    void update(double expected);
    bool check(double expected, double variance);
    void train(std::vector<std::pair<std::vector<double>,double[2]>> dataSet);

  private:
    uint64_t fID;
    std::vector<uint64_t> fInputConnections;
    std::vector<double> fInputs;
    double fOutput;
    std::vector<uint64_t> fOutputConnections;
    std::vector<double> fWeights;
    status fStatus;
    uint32_t fIterations;
};

std::unordered_map<uint64_t,perceptron> globalMap;

#endif //NEURALC_PERCEPTRON_H
