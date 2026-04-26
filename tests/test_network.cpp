#include <vector>

#include "check.h"
#include "mlp/Network.h"

using mlp::Activation;
using mlp::Network;

namespace
{

// XOR(a, b) = AND(OR(a, b), NAND(a, b)), with hand-picked weights.
Network handmadeXor()
{
    Network net(2);
    net.addLayer(2, Activation(mlp::STEP)).addLayer(1, Activation(mlp::STEP));

    mlp::Layer& hidden = net.layer(0);
    hidden.setWeight(0, 0, 1.0); // OR
    hidden.setWeight(0, 1, 1.0);
    hidden.setBias(0, -0.5);
    hidden.setWeight(1, 0, -1.0); // NAND
    hidden.setWeight(1, 1, -1.0);
    hidden.setBias(1, 1.5);

    mlp::Layer& output = net.layer(1);
    output.setWeight(0, 0, 1.0); // AND
    output.setWeight(0, 1, 1.0);
    output.setBias(0, -1.5);
    return net;
}

double run(const Network& net, double a, double b)
{
    std::vector<double> inputs(2);
    inputs[0] = a;
    inputs[1] = b;
    return net.predict(inputs)[0];
}

} // namespace

void testNetwork()
{
    Network empty(2);
    CHECK(empty.outputCount() == 0);
    CHECK_THROWS(empty.predict(std::vector<double>(2)));
    CHECK_THROWS(Network(0));

    Network xorNet = handmadeXor();
    CHECK(xorNet.inputCount() == 2);
    CHECK(xorNet.layerCount() == 2);
    CHECK(xorNet.outputCount() == 1);
    CHECK(run(xorNet, 0, 0) == 0.0);
    CHECK(run(xorNet, 0, 1) == 1.0);
    CHECK(run(xorNet, 1, 0) == 1.0);
    CHECK(run(xorNet, 1, 1) == 0.0);
    CHECK_THROWS(xorNet.predict(std::vector<double>(3)));

    // Same seed, same initial weights.
    Network a(3, 5), b(3, 5);
    a.addLayer(4).addLayer(2);
    b.addLayer(4).addLayer(2);
    CHECK(a.layer(1).size() == 2);
    CHECK(a.layer(1).inputCount() == 4);
    CHECK(a.layer(0).weight(3, 2) == b.layer(0).weight(3, 2));
    CHECK(a.layer(1).bias(1) == b.layer(1).bias(1));
}
