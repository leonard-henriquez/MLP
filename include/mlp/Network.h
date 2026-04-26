#ifndef MLP_NETWORK_H
#define MLP_NETWORK_H

#include <cstddef>
#include <vector>

#include "mlp/Activation.h"
#include "mlp/Layer.h"
#include "mlp/Random.h"

namespace mlp
{

// Multilayer perceptron: a stack of fully connected layers.
//
//     mlp::Network net(2);          // 2 inputs
//     net.addLayer(3);              // hidden layer of 3 neurons
//     net.addLayer(1);              // output layer
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

private:
    std::size_t m_inputCount;
    std::vector<Layer> m_layers;
    Random m_random;
};

} // namespace mlp

#endif // MLP_NETWORK_H
