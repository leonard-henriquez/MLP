#ifndef MLP_LAYER_H
#define MLP_LAYER_H

#include <cstddef>
#include <vector>

#include "mlp/Activation.h"
#include "mlp/Random.h"

namespace mlp
{

// Fully connected layer of neurons sharing the same activation function.
// Neuron j computes f(sum_i weight(j, i) * input_i + bias(j)).
class Layer
{
public:
    Layer(std::size_t inputCount, std::size_t neuronCount,
          const Activation& activation = Activation());

    std::size_t inputCount() const { return m_inputCount; }
    std::size_t size() const { return m_biases.size(); }
    const Activation& activation() const { return m_activation; }

    double weight(std::size_t neuron, std::size_t input) const;
    void setWeight(std::size_t neuron, std::size_t input, double value);
    double bias(std::size_t neuron) const;
    void setBias(std::size_t neuron, double value);

    // Draw every weight and bias uniformly in [-range, range].
    void randomize(Random& random, double range);

    void compute(const std::vector<double>& inputs, std::vector<double>& outputs) const;

private:
    std::size_t m_inputCount;
    std::vector<double> m_weights; // m_weights[neuron * m_inputCount + input]
    std::vector<double> m_biases;
    Activation m_activation;
};

} // namespace mlp

#endif // MLP_LAYER_H
