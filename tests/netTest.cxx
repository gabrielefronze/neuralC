#include "gtest/gtest.h"
#include "NeuralNet.h"

using std::vector;

void initdata(std::vector<std::vector<double>> &X, std::vector<double> &y) {
    for(size_t i=0; i<10; i++){
        std::vector<double> x;
        for (int j = 0; j < 5; ++j) {
            x.push_back(j*i);
        }
        X.push_back(x);
        y.push_back(i<5);
    }
}

TEST(net_Test, net_test_createNet) {
    NeuralNet net;

    std::vector<std::vector<double>> X;
    std::vector<double> y;
    initdata(X,y);

    net.firstLayer(4, X).addLayer(3).addLayer(3).addLayer(6).lastLayer(y).toOstream();

    ASSERT_EQ(0, 0);
}

TEST(net_Test, net_test_train) {
    NeuralNet net;

    std::vector<std::vector<double>> X;
    std::vector<double> y;

    for(size_t i=0; i<10; i++){
        std::vector<double> x;
        for (int j = 0; j < 5; ++j) {
            x.push_back(j*i);
        }

        X.push_back(x);
        y.push_back(i<5);
    }
    net.firstLayer(3,X).addLayer(2).lastLayer(y).toOstream();
    net.train();

    EXPECT_EQ(0, 0);
}