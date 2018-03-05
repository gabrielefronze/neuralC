//
// Created by Gabriele Gaetano Fronzé on 05/03/2018.
//

#ifndef NEURALC_POTENTIALS_H
#define NEURALC_POTENTIALS_H

#include <math.h>

namespace potentials{
  double step(double value) const{
    return (double)((value>0.)*2.-1.);
  };

  double sigmoid(double value) const{
    auto ex = exp(value);
    return ex/(1+ex);
  };

  double arctan(double value) const{
    return arctan(value);
  };
}

#endif //NEURALC_POTENTIALS_H
