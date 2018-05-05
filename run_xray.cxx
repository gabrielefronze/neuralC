//
// Created by Filippo Valle on 05/05/2018.
//

#include "NeuralNet.h"
#include "XRayMachine/XRayMachine.h"

void run_xray(){
    NeuralNet net;
    net.firstLayer(3, "dataset.csv").addLayer(5).addLayer(8).addLayer(10).addLayer(7).addLayer(3).lastLayer("target.csv");
    auto cx =  XRayMachine(net).fCanvas;
}