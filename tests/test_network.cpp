#include <sstream>
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

namespace
{

double halfSquaredError(const Network& net, const std::vector<double>& inputs,
                        const std::vector<double>& targets)
{
    std::vector<double> outputs = net.predict(inputs);
    double sum = 0.0;
    for (std::size_t k = 0; k < outputs.size(); k++)
        sum += 0.5 * (outputs[k] - targets[k]) * (outputs[k] - targets[k]);
    return sum;
}

} // namespace

// Check backpropagation against finite differences: after one call to
// trainExample(), every weight must have moved by -rate * dE/dw.
void testBackpropagation()
{
    Network net(3, 11);
    net.addLayer(4).addLayer(3).addLayer(2);

    std::vector<double> inputs(3), targets(2);
    inputs[0] = 0.3;
    inputs[1] = -0.7;
    inputs[2] = 0.9;
    targets[0] = 1.0;
    targets[1] = 0.0;

    const double rate = 0.1, h = 1e-6;
    Network trained = net;
    trained.trainExample(inputs, targets, rate);

    for (std::size_t k = 0; k < net.layerCount(); k++)
    {
        mlp::Layer& layer = net.layer(k);
        for (std::size_t j = 0; j < layer.size(); j++)
        {
            for (std::size_t i = 0; i <= layer.inputCount(); i++)
            {
                // i == inputCount() stands for the bias.
                bool isBias = i == layer.inputCount();
                double original = isBias ? layer.bias(j) : layer.weight(j, i);

                isBias ? layer.setBias(j, original + h) : layer.setWeight(j, i, original + h);
                double plus = halfSquaredError(net, inputs, targets);
                isBias ? layer.setBias(j, original - h) : layer.setWeight(j, i, original - h);
                double minus = halfSquaredError(net, inputs, targets);
                isBias ? layer.setBias(j, original) : layer.setWeight(j, i, original);

                double numeric = (plus - minus) / (2 * h);
                double updated = isBias ? trained.layer(k).bias(j) : trained.layer(k).weight(j, i);
                CHECK_CLOSE((original - updated) / rate, numeric, 1e-7);
            }
        }
    }
}

void testTraining()
{
    mlp::Dataset data = mlp::Dataset::load("examples/data/xor2.txt");

    Network net(2, 1);
    net.addLayer(3).addLayer(1);
    CHECK_THROWS(net.train(mlp::Dataset::load("data/ET3.txt")));

    mlp::TrainingResult result = net.train(data);
    CHECK(result.converged);
    CHECK(result.error <= 0.001);
    CHECK(result.epochs > 0 && result.epochs < 10000);
    CHECK_CLOSE(net.error(data), result.error, 1e-12);
    for (std::size_t i = 0; i < data.size(); i++)
        CHECK_CLOSE(net.predict(data[i].inputs)[0], data[i].targets[0], 0.1);

    // Hidden tanh layer and linear output, on a regression problem:
    // y = x1 - 2.x2 + 0.5
    mlp::Dataset linear(2, 1);
    for (int a = -2; a <= 2; a++)
    {
        for (int b = -2; b <= 2; b++)
        {
            std::vector<double> inputs(2), target(1);
            inputs[0] = a / 2.0;
            inputs[1] = b / 2.0;
            target[0] = inputs[0] - 2 * inputs[1] + 0.5;
            linear.add(inputs, target);
        }
    }
    Network regression(2, 3);
    regression.addLayer(4, Activation(mlp::TANH)).addLayer(1, Activation(mlp::LINEAR));
    mlp::TrainingOptions regressionOptions;
    regressionOptions.learningRate = 0.05;
    regressionOptions.targetError = 1e-4;
    result = regression.train(linear, regressionOptions);
    CHECK(result.converged);

    // Training stops at maxEpochs when the target cannot be reached.
    Network small(2, 1);
    small.addLayer(1);
    mlp::TrainingOptions options;
    options.maxEpochs = 50;
    result = small.train(data, options);
    CHECK(!result.converged);
    CHECK(result.epochs == 50);
}

void testSaveLoad()
{
    Network net(3, 9);
    net.addLayer(4, Activation(mlp::TANH, 0.5)).addLayer(2, Activation(mlp::SIGMOID));

    std::stringstream file;
    net.save(file);
    Network loaded = Network::load(file);

    CHECK(loaded.inputCount() == 3);
    CHECK(loaded.layerCount() == 2);
    CHECK(loaded.layer(0).activation().type() == mlp::TANH);
    CHECK(loaded.layer(0).activation().steepness() == 0.5);
    CHECK(loaded.layer(1).activation().type() == mlp::SIGMOID);
    for (std::size_t k = 0; k < net.layerCount(); k++)
    {
        for (std::size_t j = 0; j < net.layer(k).size(); j++)
        {
            // Saved with enough digits to read back the exact same values.
            CHECK(loaded.layer(k).bias(j) == net.layer(k).bias(j));
            for (std::size_t i = 0; i < net.layer(k).inputCount(); i++)
                CHECK(loaded.layer(k).weight(j, i) == net.layer(k).weight(j, i));
        }
    }

    std::istringstream truncated("mlp 1\ninputs 2\nlayers 1\nlayer 1 sigmoid 1\n0.5 0.5\n");
    CHECK_THROWS(Network::load(truncated));
    std::istringstream unknown("mlp 1\ninputs 1\nlayers 1\nlayer 1 relu 1\n0.5 0.5\n");
    CHECK_THROWS(Network::load(unknown));
    std::istringstream version("mlp 2\n");
    CHECK_THROWS(Network::load(version));
    CHECK_THROWS(Network::load("does/not/exist.mlp"));
}
