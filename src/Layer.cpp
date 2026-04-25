#include "mlp/Layer.h"

#include <stdexcept>

namespace mlp
{

Layer::Layer(std::size_t inputCount, std::size_t neuronCount, const Activation& activation)
    : m_inputCount(inputCount),
      m_weights(inputCount * neuronCount, 0.0),
      m_biases(neuronCount, 0.0),
      m_activation(activation)
{
    if (inputCount == 0 || neuronCount == 0)
        throw std::invalid_argument("a layer needs at least one input and one neuron");
}

double Layer::weight(std::size_t neuron, std::size_t input) const
{
    return m_weights.at(neuron * m_inputCount + input);
}

void Layer::setWeight(std::size_t neuron, std::size_t input, double value)
{
    m_weights.at(neuron * m_inputCount + input) = value;
}

double Layer::bias(std::size_t neuron) const
{
    return m_biases.at(neuron);
}

void Layer::setBias(std::size_t neuron, double value)
{
    m_biases.at(neuron) = value;
}

void Layer::randomize(Random& random, double range)
{
    for (std::size_t i = 0; i < m_weights.size(); i++)
        m_weights[i] = random.uniform(-range, range);
    for (std::size_t j = 0; j < m_biases.size(); j++)
        m_biases[j] = random.uniform(-range, range);
}

void Layer::compute(const std::vector<double>& inputs, std::vector<double>& outputs) const
{
    if (inputs.size() != m_inputCount)
        throw std::invalid_argument("wrong number of inputs");

    outputs.resize(size());
    for (std::size_t j = 0; j < size(); j++)
    {
        const double* w = &m_weights[j * m_inputCount];
        double sum = m_biases[j];
        for (std::size_t i = 0; i < m_inputCount; i++)
            sum += w[i] * inputs[i];
        outputs[j] = m_activation.value(sum);
    }
}

} // namespace mlp
