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

    // Same as compute(), but remember the inputs and weighted sums so that
    // backward() can be called afterwards.
    const std::vector<double>& forward(const std::vector<double>& inputs);

    // Backpropagation step for the last forward() call. Given the gradient of
    // the error with respect to the outputs of this layer, compute the
    // gradient with respect to its inputs (using the weights as they were
    // during forward()), then move the weights and biases one gradient
    // descent step of the given learning rate.
    void backward(const std::vector<double>& outputGradient, double learningRate,
                  std::vector<double>& inputGradient);

private:
    double weightedSum(std::size_t neuron, const std::vector<double>& inputs) const;

    std::size_t m_inputCount;
    std::vector<double> m_weights; // m_weights[neuron * m_inputCount + input]
    std::vector<double> m_biases;
    Activation m_activation;

    // State of the last forward() call.
    std::vector<double> m_inputs;
    std::vector<double> m_sums;
    std::vector<double> m_outputs;
};

} // namespace mlp

#endif // MLP_LAYER_H
