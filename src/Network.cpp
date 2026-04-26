#include "mlp/Network.h"

#include <cmath>
#include <stdexcept>

namespace mlp
{

Network::Network(std::size_t inputCount, unsigned long seed)
    : m_inputCount(inputCount), m_random(seed)
{
    if (inputCount == 0)
        throw std::invalid_argument("a network needs at least one input");
}

Network& Network::addLayer(std::size_t neuronCount, const Activation& activation)
{
    std::size_t inputs = m_layers.empty() ? m_inputCount : m_layers.back().size();
    Layer layer(inputs, neuronCount, activation);
    layer.randomize(m_random, 1.0 / std::sqrt(static_cast<double>(inputs)));
    m_layers.push_back(layer);
    return *this;
}

std::size_t Network::outputCount() const
{
    return m_layers.empty() ? 0 : m_layers.back().size();
}

std::vector<double> Network::predict(const std::vector<double>& inputs) const
{
    if (m_layers.empty())
        throw std::logic_error("the network has no layer");

    std::vector<double> values = inputs, next;
    for (std::size_t k = 0; k < m_layers.size(); k++)
    {
        m_layers[k].compute(values, next);
        values.swap(next);
    }
    return values;
}

} // namespace mlp
