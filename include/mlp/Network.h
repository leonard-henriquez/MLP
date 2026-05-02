#ifndef MLP_NETWORK_H
#define MLP_NETWORK_H

#include <cstddef>
#include <vector>

#include "mlp/Activation.h"
#include "mlp/Dataset.h"
#include "mlp/Layer.h"
#include "mlp/Random.h"

namespace mlp
{

struct TrainingOptions
{
    TrainingOptions();

    double learningRate; // gradient descent step (default 0.5)
    unsigned maxEpochs;  // give up after this many passes over the data (default 10000)
    double targetError;  // stop once the mean squared error is below (default 0.001)
    bool shuffle;        // present the examples in a random order each epoch (default true)
};

struct TrainingResult
{
    unsigned epochs;
    double error; // mean squared error at the end of training
    bool converged;
};

// Multilayer perceptron: a stack of fully connected layers.
//
//     mlp::Network net(2);          // 2 inputs
//     net.addLayer(3);              // hidden layer of 3 neurons
//     net.addLayer(1);              // output layer
//     net.train(dataset);
//     std::vector<double> y = net.predict(x);
class Network
{
public:
    explicit Network(std::size_t inputCount, unsigned long seed = 1);

    // Append a layer whose weights are drawn uniformly in
    // [-1/sqrt(n), 1/sqrt(n)], n being the number of inputs of the layer.
    Network& addLayer(std::size_t neuronCount, const Activation& activation = Activation());

    std::size_t inputCount() const { return m_inputCount; }
    std::size_t outputCount() const;
    std::size_t layerCount() const { return m_layers.size(); }
    Layer& layer(std::size_t i) { return m_layers.at(i); }
    const Layer& layer(std::size_t i) const { return m_layers.at(i); }

    std::vector<double> predict(const std::vector<double>& inputs) const;

    // One step of stochastic gradient descent on the squared error of a
    // single example, using backpropagation. Return the squared error
    // sum_k (target_k - output_k)^2 before the update.
    double trainExample(const std::vector<double>& inputs, const std::vector<double>& targets,
                        double learningRate);

    // Train on every example of the dataset, epoch after epoch, until the
    // mean squared error goes below options.targetError.
    TrainingResult train(const Dataset& data, const TrainingOptions& options = TrainingOptions());

    // Mean over the examples and outputs of (target - output)^2.
    double error(const Dataset& data) const;

private:
    void checkDataset(const Dataset& data) const;

    std::size_t m_inputCount;
    std::vector<Layer> m_layers;
    Random m_random;
};

} // namespace mlp

#endif // MLP_NETWORK_H
