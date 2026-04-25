#include <vector>

#include "check.h"
#include "mlp/Layer.h"
#include "mlp/Random.h"

using mlp::Activation;
using mlp::Layer;

void testLayer()
{
    // Two step neurons computing AND and OR of their two inputs.
    Layer layer(2, 2, Activation(mlp::STEP));
    CHECK(layer.inputCount() == 2);
    CHECK(layer.size() == 2);
    layer.setWeight(0, 0, 1.0);
    layer.setWeight(0, 1, 1.0);
    layer.setBias(0, -1.5);
    layer.setWeight(1, 0, 1.0);
    layer.setWeight(1, 1, 1.0);
    layer.setBias(1, -0.5);

    std::vector<double> inputs(2), outputs;
    inputs[0] = 1.0;
    inputs[1] = 0.0;
    layer.compute(inputs, outputs);
    CHECK(outputs.size() == 2);
    CHECK(outputs[0] == 0.0);
    CHECK(outputs[1] == 1.0);

    inputs[1] = 1.0;
    layer.compute(inputs, outputs);
    CHECK(outputs[0] == 1.0);
    CHECK(outputs[1] == 1.0);

    CHECK_THROWS(layer.compute(std::vector<double>(3), outputs));
    CHECK_THROWS(layer.weight(2, 0));
    CHECK_THROWS(Layer(0, 1));

    mlp::Random random(42);
    layer.randomize(random, 0.5);
    for (std::size_t j = 0; j < layer.size(); j++)
    {
        CHECK(layer.bias(j) >= -0.5 && layer.bias(j) <= 0.5);
        for (std::size_t i = 0; i < layer.inputCount(); i++)
            CHECK(layer.weight(j, i) >= -0.5 && layer.weight(j, i) <= 0.5);
    }
}
