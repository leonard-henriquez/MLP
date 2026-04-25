#include <string>

#include "check.h"
#include "mlp/Dataset.h"
#include "mlp/Perceptron.h"

using mlp::Dataset;
using mlp::Perceptron;

namespace
{

void checkLearns(const std::string& path)
{
    Dataset data = Dataset::load(path);
    Perceptron perceptron(data.inputCount());
    CHECK(perceptron.train(data));
    for (std::size_t i = 0; i < data.size(); i++)
        CHECK(perceptron.output(data[i].inputs) == data[i].targets[0]);
}

} // namespace

void testPerceptron()
{
    checkLearns("data/ET3.txt");
    checkLearns("data/OU2.txt");
    checkLearns("data/NON2.txt");

    // XOR is not linearly separable: a single perceptron cannot learn it.
    Perceptron xorPerceptron(2);
    CHECK(!xorPerceptron.train(Dataset::load("examples/data/xor2.txt"), 200));
    CHECK(xorPerceptron.epochs() == 200);

    Perceptron perceptron(2);
    std::vector<double> inputs(3, 0.0);
    CHECK_THROWS(perceptron.output(inputs));
    CHECK_THROWS(perceptron.train(Dataset::load("data/ET3.txt")));
}
